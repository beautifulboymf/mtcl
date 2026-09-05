# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import time
from contextlib import nullcontext
from functools import partial
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor
from torch.utils import _pytree

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    PARAM_GROUP_SLOT_A,
    PARAM_GROUP_SLOT_B,
    FSDPModelManager,
)
from rlinf.hybrid_engines.fsdp.utils import (
    pack_fsdp_input,
    prepare_pack_fsdp,
    unpack_fsdp_logprobs,
    unpack_sequences,
)
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    all_reduce_dict,
    all_reduce_int,
    masked_normalization,
)
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.rollout.utils import RankMapper

# ==========================================================================================
# Per-phase timing + in-step progress logging for run_training  (DIAGNOSTIC, OFF BY DEFAULT)
# ==========================================================================================
# THE PROBLEM. A routed multi-teacher OPD step at global_batch_size 192 / micro_batch_size 2
# on two ranks runs 48 micro-batches per optimizer update, each one a student forward, a
# frozen-teacher forward, a KL-distillation loss and a backward that (with gradient
# checkpointing) re-runs the forward. That step took over three hours and `run_training`
# printed NOTHING between step boundaries -- there was no way to tell slow from hung, and no
# way to say which of those five things was the expensive one.
#
# THE MEASUREMENT. CUDA is asynchronous: a host-side `time.perf_counter()` around
# `self.model(...)` measures how long it took to QUEUE the forward, not to run it, and all
# the real time then lands on whichever later call happens to block. So the phases are timed
# with `torch.cuda.Event` pairs -- recorded into the stream, read back with `elapsed_time`.
#
# ONE MOVING CURSOR, NOT START/STOP PAIRS. Each `mark(phase)` records one event and charges
# the interval since the PREVIOUS event to `phase`. Consequence: no interval can be dropped.
# With start/stop pairs, any statement outside a bracket silently vanishes from the
# breakdown while the totals still look plausible; here the phases always sum to the wall
# clock, and `unaccounted` on the summary line is the check that says so.
#
# WHAT IT COSTS, AND WHY THAT IS NOT THE 5.6x TRAP. Recording an event is a few microseconds
# of host time and does not stall. Reading it needs the event to have completed, so the
# flush calls `event.synchronize()` -- normally the expensive part. It is free HERE because
# of exactly where the flush is placed: `micro_done()` is called immediately after
# `metrics_data["actor/total_loss"] = loss.detach().item()`, which already drains the stream
# on every micro-batch, profiling or not. The events being read are therefore already
# complete and the synchronize returns at once. This distinction is load-bearing: this repo
# measured run_training go from 14.3 min to 80.0 min (5.6x) from roughly six device syncs
# per micro-batch placed MID-GRAPH, where they serialize the host against the GPU. A sync
# after an existing sync costs nothing; a sync in the middle of loss construction costs
# everything. Do not move the flush earlier.
#
# It is still gated OFF by default (`algorithm.profile_train_phases`) and documented as
# diagnostic-only, because that argument depends on a `.item()` that is not this feature's
# to guarantee.
_PROFILE_PHASES = (
    "prep",  # device copy + before_micro_batch + slot routing (_route_prepare)
    "student_fwd",  # self.model(...)
    "teacher_fwd",  # self._teacher_forward(...)  (absent on non-OPD runs -> folds into loss)
    "loss",  # advantage + KL/anchor/loss construction, up to the backward
    "backward",  # grad_scaler.scale(loss).backward()
    "empty_cache",  # per-update torch_platform.empty_cache() + the slot lr alternation
    "optim",  # self.optimizer_step()
)


def _fmt_dur(seconds: float) -> str:
    """Seconds as ``41.2s`` / ``7m12s`` / ``3h04m05s``.

    A 3-hour step is the thing being measured, so raw seconds are unreadable exactly
    where it matters most.
    """
    s = float(seconds)
    if s != s or s in (float("inf"), float("-inf")):
        return "n/a"
    s = max(0.0, s)
    if s < 60.0:
        return f"{s:.1f}s"
    total = int(round(s))
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    return f"{m}m{sec:02d}s"


def _eta_seconds(elapsed_s: float, done: int, total: int) -> float:
    """Remaining step time projected from the rate observed SO FAR IN THIS STEP.

    Deliberately the crudest possible model -- ``elapsed / done`` extrapolated over what
    is left. It is honest about the optimizer step (amortised across the update's
    micro-batches rather than pretended away) and it needs no calibration. Returns 0.0
    rather than a negative or infinite number for the degenerate inputs, so a progress
    line can never be the thing that raises inside a training loop.
    """
    if done <= 0 or total <= done:
        return 0.0
    return float(elapsed_s) * float(total - done) / float(done)


def _phase_breakdown(totals_ms: dict, wall_s: Optional[float] = None) -> str:
    """The per-phase times, always in the same order and ALWAYS all of them.

    A fixed key set means the line is greppable and two runs are diffable; a phase that
    is genuinely zero (``teacher_fwd`` off the OPD path) says so instead of disappearing.
    """
    parts = []
    for name in _PROFILE_PHASES:
        secs = float(totals_ms.get(name, 0.0)) / 1000.0
        if wall_s is not None and wall_s > 0:
            parts.append(f"{name} {_fmt_dur(secs)} ({100.0 * secs / wall_s:.1f}%)")
        else:
            parts.append(f"{name} {_fmt_dur(secs)}")
    return " ".join(parts)


def _progress_line(
    *,
    rank,
    step,
    update_index,
    updates_per_step,
    mb_in_update,
    mbs_per_update,
    mb_done,
    mb_total,
    elapsed_s,
    totals_ms,
) -> str:
    pct = (100.0 * mb_done / mb_total) if mb_total else 0.0
    eta = _eta_seconds(elapsed_s, mb_done, mb_total)
    return (
        f"[prof][r{rank}] step {step} | update {update_index}/{updates_per_step} | "
        f"micro {mb_in_update}/{mbs_per_update} (update) "
        f"{mb_done}/{mb_total} (step, {pct:.1f}%) | "
        f"elapsed {_fmt_dur(elapsed_s)} | eta {_fmt_dur(eta)} | "
        f"{_phase_breakdown(totals_ms)}"
    )


def _summary_line(
    *, rank, step, updates_per_step, mbs_per_update, mb_total, elapsed_s, totals_ms
) -> str:
    accounted = sum(float(totals_ms.get(p, 0.0)) for p in _PROFILE_PHASES) / 1000.0
    other = max(0.0, float(elapsed_s) - accounted)
    pct = (100.0 * other / elapsed_s) if elapsed_s else 0.0
    return (
        f"[prof][r{rank}] step {step} DONE | {updates_per_step} updates x "
        f"{mbs_per_update} micro-batches = {mb_total} | wall {_fmt_dur(elapsed_s)} | "
        f"{_phase_breakdown(totals_ms, elapsed_s)} | "
        f"unaccounted {_fmt_dur(other)} ({pct:.1f}%)"
    )


class _CudaEventClock:
    """Timing source backed by ``torch.cuda.Event`` -- the accurate one."""

    __slots__ = ()

    def event(self):
        return torch.cuda.Event(enable_timing=True)

    def record(self, ev):
        ev.record()

    def sync(self, ev):
        ev.synchronize()

    def elapsed_ms(self, a, b):
        return a.elapsed_time(b)


class _PerfCounterClock:
    """Host-clock fallback for CPU-only runs.

    On a GPU run this would be WRONG (it measures queueing, not execution), which is why
    it is only ever selected when there is no CUDA device to be wrong about.
    """

    __slots__ = ()

    def event(self):
        return [0.0]

    def record(self, ev):
        ev[0] = time.perf_counter()

    def sync(self, ev):
        pass

    def elapsed_ms(self, a, b):
        return (b[0] - a[0]) * 1000.0


class TrainPhaseProfiler:
    """Per-phase timing and in-step progress logging for one training step.

    Protocol, mirroring ``run_training``'s loop structure::

        begin_step(updates_per_step)
          begin_update(len(train_micro_batch))          # once per optimizer update
            mark("prep"); mark("student_fwd"); ...      # once per micro-batch
            micro_done()                                # flush + maybe a progress line
          mark("empty_cache"); mark("optim")
          update_done()                                 # per-update breakdown
        step_done()                                     # per-step summary

    Every method is a no-op when ``enabled`` is False: no clock is chosen, no event is
    allocated, nothing is synchronized and nothing is logged.
    """

    PHASES = _PROFILE_PHASES

    def __init__(
        self,
        *,
        enabled: bool = False,
        log_every: int = 0,
        log_fn=None,
        warn_fn=None,
        rank: int = 0,
        step=None,
        clock=None,
        now_fn=time.perf_counter,
    ):
        self.enabled = bool(enabled)
        self._cfg_log_every = max(0, int(log_every or 0))
        self._log = log_fn if log_fn is not None else (lambda _msg: None)
        self._warn = warn_fn if warn_fn is not None else self._log
        self._rank = rank
        self._step = step if step is not None else "?"
        self._now = now_fn
        self._clock = clock
        self.totals = dict.fromkeys(_PROFILE_PHASES, 0.0)  # ms, this step
        self.update_totals = dict.fromkeys(_PROFILE_PHASES, 0.0)  # ms, this update
        self.updates_per_step = 0
        self.mbs_per_update = 0
        self.update_index = 0
        self.mb_in_update = 0
        self.mb_done = 0
        self.log_every = 0
        self._pending = []  # [(event, phase or None)] -- [0] is always the cursor
        self._pool = []
        self._t0 = None
        self._t_update0 = None

    # -- interval policy ----------------------------------------------------------------
    @staticmethod
    def auto_log_every(mbs_per_update: int) -> int:
        """Roughly ten progress lines per optimizer update, from the REAL bound.

        48 micro-batches -> every 5th; 12 -> every one. Never 0 (that would mean silence,
        which is the bug this feature fixes).
        """
        if mbs_per_update <= 0:
            return 1
        # Half-up on purpose: Python's round() is banker's, so round(2.5) is 2 and a
        # 25-micro-batch update would quietly get 12 lines where 24 gets 10.
        return max(1, int(mbs_per_update / 10.0 + 0.5))

    # -- lifecycle ----------------------------------------------------------------------
    def begin_step(self, updates_per_step: int) -> None:
        if not self.enabled:
            return
        if self._clock is None:
            self._clock = (
                _CudaEventClock() if torch.cuda.is_available() else _PerfCounterClock()
            )
        self.updates_per_step = int(updates_per_step)
        self.mbs_per_update = 0
        self.update_index = 0
        self.mb_in_update = 0
        self.mb_done = 0
        self.log_every = 0
        self.totals = dict.fromkeys(_PROFILE_PHASES, 0.0)
        self.update_totals = dict.fromkeys(_PROFILE_PHASES, 0.0)
        self._pool.extend(ev for ev, _ in self._pending)
        self._pending = []
        self._t0 = self._now()
        self._t_update0 = self._t0
        self._record(None)

    def begin_update(self, micro_batches_per_update: int) -> None:
        if not self.enabled or self._t0 is None:
            return
        self.update_index += 1
        self.mbs_per_update = int(micro_batches_per_update)
        self.mb_in_update = 0
        self.update_totals = dict.fromkeys(_PROFILE_PHASES, 0.0)
        self._t_update0 = self._now()
        self.log_every = self._cfg_log_every or self.auto_log_every(self.mbs_per_update)

    def mark(self, phase: str) -> None:
        """Close the current interval and charge it to ``phase``."""
        if not self.enabled or self._t0 is None:
            return
        self._record(phase)

    def micro_done(self) -> None:
        """End of one micro-batch: read the events back, maybe log a progress line."""
        if not self.enabled or self._t0 is None:
            return
        self._flush()
        if not self.enabled:
            return
        self.mb_in_update += 1
        self.mb_done += 1
        n = self.log_every
        # The first micro-batch of the step ALWAYS logs: on a 3-hour step, waiting for
        # the 5th one to learn that the step started is the same silence as before.
        if n > 0 and (self.mb_done == 1 or self.mb_in_update % n == 0):
            self._log(
                _progress_line(
                    rank=self._rank,
                    step=self._step,
                    update_index=self.update_index,
                    updates_per_step=self.updates_per_step,
                    mb_in_update=self.mb_in_update,
                    mbs_per_update=self.mbs_per_update,
                    mb_done=self.mb_done,
                    mb_total=self.updates_per_step * self.mbs_per_update,
                    elapsed_s=self._now() - self._t0,
                    totals_ms=self.totals,
                )
            )

    def update_done(self) -> None:
        if not self.enabled or self._t0 is None:
            return
        self._flush()
        if not self.enabled:
            return
        base = self._t_update0 if self._t_update0 is not None else self._t0
        wall = self._now() - base
        self._log(
            f"[prof][r{self._rank}] step {self._step} | "
            f"update {self.update_index}/{self.updates_per_step} DONE | "
            f"{self.mb_in_update} micro-batches in {_fmt_dur(wall)} | "
            f"{_phase_breakdown(self.update_totals, wall)}"
        )

    def step_done(self) -> dict:
        """Log the step summary; return the per-phase totals in SECONDS."""
        if not self.enabled or self._t0 is None:
            return {}
        self._flush()
        if not self.enabled:
            return {}
        wall = self._now() - self._t0
        self._log(
            _summary_line(
                rank=self._rank,
                step=self._step,
                updates_per_step=self.updates_per_step,
                mbs_per_update=self.mbs_per_update,
                mb_total=self.mb_done,
                elapsed_s=wall,
                totals_ms=self.totals,
            )
        )
        self._t0 = None
        return {p: self.totals[p] / 1000.0 for p in _PROFILE_PHASES}

    # -- internals ----------------------------------------------------------------------
    def _record(self, phase) -> None:
        try:
            ev = self._pool.pop() if self._pool else self._clock.event()
            self._clock.record(ev)
            self._pending.append((ev, phase))
        except Exception as e:  # a diagnostic must never take the run down with it
            self._disable(e)

    def _flush(self) -> None:
        if len(self._pending) < 2:
            return
        try:
            self._clock.sync(self._pending[-1][0])
            prev = self._pending[0][0]
            for ev, phase in self._pending[1:]:
                dt = self._clock.elapsed_ms(prev, ev)
                if phase in self.totals:
                    self.totals[phase] += dt
                    self.update_totals[phase] += dt
                prev = ev
        except Exception as e:
            self._disable(e)
            return
        # The last event becomes the next cursor, so the timeline stays continuous across
        # flushes and no interval can fall between two micro-batches. Everything else goes
        # back to the pool: the pool therefore stays at ~8 events for the whole run instead
        # of allocating 7 per micro-batch.
        self._pool.extend(ev for ev, _ in self._pending[:-1])
        self._pending = [(self._pending[-1][0], None)]

    def _disable(self, exc) -> None:
        self.enabled = False
        self._pending = []
        self._pool = []
        self._warn(
            f"[prof] phase profiling disabled after an internal error: {exc!r} -- "
            "training continues, only the timing is lost"
        )


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


# slot-LoRI alternating schedule: the only two factors there are to train. "A" is Z
# (the free matrix behind Ā), "B" is the readout.
SLOT_ALT_FACTORS = ("A", "B")


def parse_slot_alt_schedule(value) -> Optional[str]:
    """Normalize ``actor.model.slot_lora.alt_schedule`` into a cycle string, or ``None``.

    Args:
        value: The config value. ``None`` means "train both factors jointly" -- the
            ablation the alternation is measured against. A string is a cycle over the
            factors, one character per OPTIMIZER UPDATE, case-insensitive: ``"BBA"``
            (the default) trains B, B, then A, then repeats.

    Returns:
        The upper-cased schedule, or ``None`` for the joint ablation.

    Raises:
        ValueError: for anything else. Loudly, at construction: a schedule string with
            a typo in it has no safe reading -- silently dropping the bad character
            would run a different experiment than the config describes, and silently
            falling back to joint training would run the ABLATION under the
            alternation's name, which is the same result with the wrong label on it.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"actor.model.slot_lora.alt_schedule must be a string over "
            f"{SLOT_ALT_FACTORS} (e.g. 'BBA') or null; got "
            f"{type(value).__name__} ({value!r})."
        )
    schedule = value.strip().upper()
    if not schedule:
        raise ValueError(
            "actor.model.slot_lora.alt_schedule is empty. Write null to train both "
            "factors jointly (the ablation); an empty string is a typo, and the two "
            "must not look alike in a config that decides which experiment ran."
        )
    unknown = sorted({c for c in schedule if c not in SLOT_ALT_FACTORS})
    if unknown:
        raise ValueError(
            f"actor.model.slot_lora.alt_schedule {value!r} contains {unknown}; only "
            f"{SLOT_ALT_FACTORS} are factors of a slot ('A' = Z, 'B' = the readout). "
            "One character per optimizer update, cycled, e.g. 'BBA'."
        )
    return schedule


def parse_slot_alt_anneal(value) -> Optional[list[tuple[int, Optional[str]]]]:
    """Normalize ``actor.model.slot_lora.alt_anneal`` into sorted ``(from_step, schedule)``.

    WHY THIS EXISTS. ``alt_schedule`` is one cycle for the whole run, and R1 ran it at
    ``BBA`` -- A free on a third of every step's updates, at the SAME lr as B, from the
    first step to the last. Measured consequence (2026-08-22, four steps): ``dw_norm``
    grew monotonically on all four slots while ``opd_kl`` moved 1.1325 -> 1.1158, i.e.
    the weights moved and the function did not. mt4, the control, moved its KL eleven
    times further over the same four steps AND bounced twenty times harder step to
    step. The mechanism that fits: every A update re-orthogonalizes the WHOLE frame
    (measured: moving one slot's Z rotates the other slots' subspaces by as much as, and
    sometimes more than, its own), so a B fitted to the previous frame is repeatedly
    re-projected onto a rotated one.

    The fix is a two-timescale schedule: let the frame find its allocation early, then
    hand the whole budget to B and stop moving the coordinate system underneath it. At
    the ``"B"`` end this degenerates EXACTLY to LoRI (a frozen orthogonal frame), which
    makes the anneal the natural ablation axis for "are learned orthogonal subspaces
    better than random frozen ones".

    Args:
        value: ``None`` to keep ``alt_schedule`` constant for the whole run (the old
            behaviour, unchanged). Otherwise a list of ``[from_step, schedule]`` pairs;
            the entry with the largest ``from_step`` that is ``<= version`` wins, so
            ``[[0, "BBBBBBBA"], [9, "B"]]`` reads "A on one update in eight until step
            9, then never again". ``schedule`` follows :func:`parse_slot_alt_schedule`,
            so ``null`` there means the joint ablation for that stretch.

    Returns:
        The stages sorted by ``from_step``, or ``None`` when annealing is off.

    Raises:
        ValueError: on a malformed entry, a negative step, or a duplicate ``from_step``
            -- loudly, at construction. A stage table with two entries claiming the
            same step has no defined winner, and picking one silently would run a
            different experiment than the config describes.
    """
    if value is None:
        return None
    stages: list[tuple[int, Optional[str]]] = []
    seen: set[int] = set()
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(
            "actor.model.slot_lora.alt_anneal must be a list of [from_step, schedule] "
            f"pairs or null; got {type(value).__name__} ({value!r})."
        ) from exc
    if not items:
        raise ValueError(
            "actor.model.slot_lora.alt_anneal is an empty list. Write null to keep "
            "alt_schedule constant; an empty list is a typo, and the two must not look "
            "alike in a config that decides which experiment ran."
        )
    for entry in items:
        pair = list(entry) if not isinstance(entry, str) else None
        if pair is None or len(pair) != 2:
            raise ValueError(
                "each actor.model.slot_lora.alt_anneal entry must be "
                f"[from_step, schedule]; got {entry!r}."
            )
        step_raw, sched_raw = pair
        try:
            from_step = int(step_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"actor.model.slot_lora.alt_anneal from_step must be an int; got "
                f"{step_raw!r}."
            ) from exc
        if from_step < 0:
            raise ValueError(
                f"actor.model.slot_lora.alt_anneal from_step must be >= 0; got "
                f"{from_step}."
            )
        if from_step in seen:
            raise ValueError(
                f"actor.model.slot_lora.alt_anneal has two entries at from_step "
                f"{from_step}; which one wins is undefined."
            )
        seen.add(from_step)
        stages.append((from_step, parse_slot_alt_schedule(sched_raw)))
    stages.sort(key=lambda item: item[0])
    if stages[0][0] != 0:
        raise ValueError(
            "actor.model.slot_lora.alt_anneal must define a stage at from_step 0; "
            f"the earliest given is {stages[0][0]}, which leaves steps before it "
            "without a schedule."
        )
    return stages


def slot_alt_schedule_for_step(
    stages: Optional[list[tuple[int, Optional[str]]]],
    default_schedule: Optional[str],
    step,
) -> Optional[str]:
    """The schedule in force at ``step``: the last stage whose ``from_step <= step``.

    Args:
        stages: From :func:`parse_slot_alt_anneal`; ``None`` means no annealing.
        default_schedule: The constant ``alt_schedule``, used when annealing is off or
            when ``step`` is not a number (a resumed run whose counter is not wired
            through yet would otherwise silently pick stage 0 -- the WRONG stage, and
            the one that looks most like the old behaviour).
        step: The training step, normally ``self.version``.

    Returns:
        The cycle string, or ``None`` for the joint ablation.
    """
    if not stages:
        return default_schedule
    try:
        step_i = int(step)
    except (TypeError, ValueError):
        return default_schedule
    chosen = stages[0][1]
    for from_step, schedule in stages:
        if from_step <= step_i:
            chosen = schedule
        else:
            break
    return chosen


def slot_alt_a_fraction(schedule: Optional[str]) -> float:
    """Share of a cycle spent on A -- the configured value the metric is checked against.

    ``slot/phase_is_A`` reading 0.0 is a healthy reading under a pure-``B`` stage and a
    dead mechanism under any other, and the two are indistinguishable without this.
    """
    if not schedule:
        return float("nan")
    return schedule.count("A") / len(schedule)


def dance_build_order(
    suite_ids,
    batch_size_per_rank: int,
    keep_frac: float,
    num_suites: int,
    seed: int,
    suites_in_rotation=None,
    rotations=None,
):
    """The DanceOPD-transfer sampling order: suite-blocked updates over thinned trajectories.

    Two mechanisms from DanceOPD (2606.27377), adapted from flow-field distillation to
    routed AR-token OPD:

    * UPDATE-LEVEL SUITE ISOLATION. Their stress test: even with per-sample hard routing,
      summing three capabilities' gradients in ONE optimizer step costs 22.8% (46% on the
      most conflicting one). Here every consecutive ``batch_size_per_rank`` slice -- one
      update's per-rank share -- holds samples of a SINGLE suite, and the suite of update
      ``k`` is ``k % num_suites`` (a fixed global rotation, so every FSDP rank sums the
      same suite on the same update; a per-rank choice would re-mix the gradients across
      ranks and silently undo the whole mechanism).

    * TRAJECTORY THINNING. Their finding: dense targets along one rollout overcount
      correlated supervision (shared prompt, seed, path history); one query per rollout
      is best. 16 envs cannot afford K=1, so this keeps a ``keep_frac`` random subset of
      each suite's samples instead -- the honest compromise, recorded as such.

    Ranks hold different suite mixtures (each rank owns its own envs' rollouts), so a
    rank whose pool for suite ``s`` is smaller than its share WRAPS AROUND that pool
    (reuse) rather than borrowing from another suite -- purity of the update outranks
    sample freshness. A suite with NO samples on this rank contributes updates drawn
    from ... nothing; it is skipped in the rotation and the schedule recorded in the
    returned ``update_suites`` says so.

    Args:
        suite_ids: 1-D LongTensor/list, one entry per flattened rollout sample, the
            suite index in ``[0, num_suites)``; ``-1`` = unrouted (dropped).
        batch_size_per_rank: This rank's slice of one optimizer update.
        keep_frac: Fraction of each suite's samples to keep, in ``(0, 1]``.
        num_suites: How many suites rotate.
        seed: Per-rank seed (caller passes ``cfg.actor.seed + rank``).

    Returns:
        ``(order, update_suites)`` -- ``order`` a 1-D LongTensor of sample indices whose
        length is a multiple of ``batch_size_per_rank``, and ``update_suites`` the suite
        index of every update, in order.

    Raises:
        ValueError: on an empty routing (every sample -1), a non-positive
            batch_size_per_rank, or keep_frac outside (0, 1].
    """
    if batch_size_per_rank <= 0:
        raise ValueError(f"batch_size_per_rank must be positive; got {batch_size_per_rank}")
    if not 0.0 < keep_frac <= 1.0:
        raise ValueError(f"keep_frac must be in (0, 1]; got {keep_frac}")
    ids = torch.as_tensor(suite_ids, dtype=torch.long)
    g = torch.Generator()
    g.manual_seed(int(seed))
    pools = []
    for k in range(num_suites):
        idx = (ids == k).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            pools.append(idx)
            continue
        keep = max(1, int(round(idx.numel() * keep_frac)))
        perm = torch.randperm(idx.numel(), generator=g)[:keep]
        pools.append(idx[perm])
    total_kept = sum(int(p.numel()) for p in pools)
    if total_kept == 0:
        raise ValueError(
            "dance_build_order: no sample matched any suite -- the routing table is "
            "empty or every sample decoded to an unknown prompt. A silent fallback "
            "here would train on an arbitrary mixture, which is exactly what this "
            "order exists to prevent."
        )
    # one update per suite per rotation; rotations sized so the kept samples are seen
    # roughly once (wraparound covers per-suite imbalance).
    # suites_in_rotation / rotations: when given, they were decided by the DISTRIBUTED
    # caller from ALL-REDUCED per-suite counts, so every rank runs the SAME schedule --
    # the mechanism's whole point (the first smoke showed rank0 rotating goal while
    # rank1 rotated object: a locally-derived rotation silently re-mixes gradients
    # across ranks, and a locally-derived rotation COUNT can desync the number of
    # optimizer updates entirely, which deadlocks FSDP).
    if suites_in_rotation is not None:
        empty = [k for k in suites_in_rotation if pools[k].numel() == 0]
        if empty:
            raise ValueError(
                f"dance_build_order: suites {empty} are in the agreed rotation but this "
                "rank holds ZERO of their samples. The caller must rotate only suites "
                "whose all-rank MINIMUM kept-count is positive; anything else either "
                "fabricates an update from nothing or desyncs the update count."
            )
        non_empty = list(suites_in_rotation)
    else:
        non_empty = [k for k in range(num_suites) if pools[k].numel() > 0]
    if rotations is None:
        rotations = max(1, round(total_kept / (batch_size_per_rank * len(non_empty))))
    order_parts, update_suites = [], []
    cursors = {k: 0 for k in non_empty}
    for _ in range(rotations):
        for k in non_empty:
            pool = pools[k]
            take, c = [], cursors[k]
            need = batch_size_per_rank
            while need > 0:
                if c >= pool.numel():
                    c = 0  # wraparound: purity of the update outranks freshness
                n = min(need, pool.numel() - c)
                take.append(pool[c : c + n])
                c += n
                need -= n
            cursors[k] = c
            order_parts.append(torch.cat(take))
            update_suites.append(k)
    return torch.cat(order_parts), update_suites


def anchor_obs_embed(
    pixel_values,
    texts,
    img_hw: int = 16,
    txt_bins: int = 128,
    img_weight: float = 0.5,
):
    """Observation-space embedding for Memory-Anchor retrieval (ANCHORER, arXiv:2608.26545).

    The paper's Step 1 measures state overlap in the POLICY's latent space. Here the
    embedding is an observation-space proxy instead -- contrast-normalized downsampled
    image + hashed bag-of-words of the instruction -- because extracting the FSDP-wrapped
    student's hidden states mid-training would add a forward pass and new plumbing
    through a loss path that has bitten us twice before. The paper's own ablation
    (Table 2, "Ours-ActionDis" .13 vs full .11 vs random .18) shows retrieval geometry
    carries most of the effect; a policy-latent version is the v2 upgrade if this pays.

    Args:
        pixel_values: float tensor ``(N, C, H, W)`` (any C; wrist-concat 6-channel ok),
            ``(N, K, C, H, W)`` (first image used), or ``None`` (image part skipped).
        texts: list of N decoded instruction strings.
        img_hw: image is average-pooled to ``img_hw x img_hw``.
        txt_bins: hash-bucket count for the instruction bag-of-words.
        img_weight: weight of the image part in the combined cosine (0..1). With both
            parts L2-normalized and scaled by sqrt(w) / sqrt(1-w), the cosine of the
            concatenation is exactly ``w*cos_img + (1-w)*cos_txt``.

    Returns:
        float32 CPU tensor ``(N, D)``, rows L2-normalized.
    """
    import zlib

    n = len(texts)
    parts = []
    w_img = float(min(max(img_weight, 0.0), 1.0))
    if pixel_values is None:
        w_img = 0.0
    if w_img > 0.0:
        pv = pixel_values
        if pv.dim() == 5:
            pv = pv[:, 0]
        if pv.dim() != 4 or pv.shape[0] != n:
            raise ValueError(
                f"anchor_obs_embed: pixel_values shape {tuple(pixel_values.shape)} "
                f"does not flatten to (N={n}, C, H, W); refusing to guess."
            )
        with torch.no_grad():
            g = pv.float().mean(dim=1, keepdim=True)  # grayscale
            g = torch.nn.functional.adaptive_avg_pool2d(g, (img_hw, img_hw))
            g = g.reshape(n, -1)
            g = g - g.mean(dim=1, keepdim=True)
            g = g / g.std(dim=1, keepdim=True).clamp_min(1e-6)  # contrast-normalize
            g = torch.nn.functional.normalize(g, dim=1).cpu() * (w_img**0.5)
        parts.append(g)
    if w_img < 1.0:
        t = torch.zeros(n, txt_bins)
        for i, s in enumerate(texts):
            for tok in s.lower().split():
                t[i, zlib.crc32(tok.encode()) % txt_bins] += 1.0
        t = torch.nn.functional.normalize(t, dim=1) * ((1.0 - w_img) ** 0.5)
        parts.append(t)
    return torch.cat(parts, dim=1).float()


def dance_anchor_augment(
    order,
    update_suites,
    suite_ids,
    batch_size_per_rank: int,
    anchor_frac: float,
    embed,
    suite_w=None,
):
    """Fill the tail of every single-suite dance update with cross-suite Memory Anchors.

    ANCHORER (arXiv:2608.26545) transferred to joint routed OPD: while the paper
    rehearses OLD-task data most similar to the NEW task's conflict region during
    sequential training, here every single-suite update k IS the "new task" for the
    other suites, and the fresh on-policy samples of those other suites are the "old
    data" pool. The last ``round(bspr*anchor_frac)`` positions of each update are
    replaced by the other-suite samples most similar (cosine in ``embed`` space) to the
    update's own centroid -- each anchor keeps its OWN routed teacher target downstream,
    so the anchor is a targeted rehearsal term, exactly the ER-with-anchors semantics.

    This deliberately relaxes dance's pure update-level isolation by a small, targeted
    fraction -- that is the method, not an accident (paper: 10-20%% of the buffer).
    Ranks pick anchors locally (each holds different envs); the update COUNT and the
    majority suite of every update stay rank-identical, so FSDP stays in sync.

    Args:
        order: flat LongTensor from :func:`dance_build_order` (len = n_updates * bspr).
        update_suites: suite index per update, same length as ``len(order)//bspr``.
        suite_ids: 1-D LongTensor over ALL rollout samples (-1 = unrouted). Candidates
            are drawn from the FULL other-suite sample set, not the thinned pools:
            thinning exists to de-correlate the majority suite's dense supervision,
            while anchors want the best-matching states available.
        batch_size_per_rank: per-rank samples of one update.
        anchor_frac: fraction of each update to replace, in ``[0, 0.5]``. 0 = no-op.
        embed: ``(N, D)`` row-normalized embeddings from :func:`anchor_obs_embed`.
        suite_w: optional per-suite weight tensor (Step-2 proxy: e.g. normalized
            student-teacher KL EMA ** beta) multiplied into candidate scores.

    Returns:
        ``(new_order, stats)`` -- stats holds ``mean_sim`` and per-suite anchor counts.
    """
    if not 0.0 <= anchor_frac <= 0.5:
        raise ValueError(
            f"anchor_frac must be in [0, 0.5] (majority suite must stay the majority); "
            f"got {anchor_frac}"
        )
    bspr = batch_size_per_rank
    n_a = int(round(bspr * anchor_frac))
    if n_a == 0:
        return order, {"mean_sim": 0.0, "anchor_counts": {}}
    ids = torch.as_tensor(suite_ids, dtype=torch.long)
    order = order.clone()
    sims_all, counts = [], {}
    for u, k in enumerate(update_suites):
        lo, hi = u * bspr, (u + 1) * bspr
        cand = ((ids >= 0) & (ids != k)).nonzero(as_tuple=True)[0]
        if cand.numel() == 0:
            continue  # single-suite rollout: nothing to anchor with
        centroid = torch.nn.functional.normalize(
            embed[order[lo : hi - n_a]].mean(dim=0), dim=0
        )
        score = embed[cand] @ centroid
        sims = score.clone()
        if suite_w is not None:
            score = score * suite_w[ids[cand]]
        top = torch.topk(score, k=min(n_a, cand.numel())).indices
        chosen = cand[top]
        order[hi - chosen.numel() : hi] = chosen
        sims_all.append(sims[top])
        for s in ids[chosen].tolist():
            counts[s] = counts.get(s, 0) + 1
    mean_sim = torch.cat(sims_all).mean().item() if sims_all else 0.0
    return order, {"mean_sim": mean_sim, "anchor_counts": counts}


def parse_slot_round_table(value) -> Optional[list[tuple[int, str]]]:
    """Validate ``slot_lora.round_table``: the sequential-round plan, or ``None``.

    The table drives the seqslot mode: ``[[0, "libero_object"], [6, "libero_10"],
    ...]`` means "from step 0 train ONLY object's slot against object's expert while
    anchoring every other suite to M_(k-1); from step 6 switch to libero_10's slot",
    and so on. It shares :func:`slot_alt_schedule_for_step`'s last-stage-at-or-below
    lookup, so a resumed run lands in the right round from its step number alone --
    no extra state to checkpoint.

    Validated with the same strictness (and for the same reason) as
    :func:`parse_slot_alt_anneal`: every failure below would otherwise surface ~25
    minutes into a run, after the rollout, as a wrong-round training step that LOOKS
    fine. Suite names are checked for non-emptiness only -- whether each names a real
    slot needs ``slot_order``, which lives on the model; the actor checks that pairing
    at init, where both are in hand.

    Args:
        value: The raw config value. ``None``/empty -> ``None`` (mode off).

    Returns:
        ``[(from_step, suite), ...]`` sorted ascending, or ``None``.

    Raises:
        ValueError: on a malformed table -- not a sequence of pairs, a negative or
            duplicated ``from_step``, no stage at step 0 (the steps before the first
            stage would silently train NO round), or a non-string/empty suite.
    """
    if value is None:
        return None
    try:
        items = list(value)
    except TypeError:
        raise ValueError(
            f"slot_lora.round_table must be a sequence of [from_step, suite] pairs; "
            f"got {type(value).__name__} ({value!r})."
        )
    if not items:
        return None
    stages: list[tuple[int, str]] = []
    for i, item in enumerate(items):
        try:
            from_step, suite = item
        except (TypeError, ValueError):
            raise ValueError(
                f"slot_lora.round_table entry {i} must be a [from_step, suite] pair; "
                f"got {item!r}."
            )
        if isinstance(from_step, bool) or not isinstance(from_step, int):
            raise ValueError(
                f"slot_lora.round_table entry {i}: from_step must be a plain int; "
                f"got {type(from_step).__name__} ({from_step!r})."
            )
        if from_step < 0:
            raise ValueError(
                f"slot_lora.round_table entry {i}: from_step {from_step} is negative; "
                "steps count from 0."
            )
        if not isinstance(suite, str) or not suite:
            raise ValueError(
                f"slot_lora.round_table entry {i}: suite must be a non-empty string "
                f"naming a routed suite; got {suite!r}."
            )
        stages.append((from_step, suite))
    stages.sort(key=lambda pair: pair[0])
    steps = [f for f, _ in stages]
    if len(set(steps)) != len(steps):
        dup = sorted({f for f in steps if steps.count(f) > 1})
        raise ValueError(
            f"slot_lora.round_table repeats from_step {dup}: two rounds at the same "
            "step -- one of them can never run, and which one wins depends on sort "
            "stability rather than on anything the config says."
        )
    if stages[0][0] != 0:
        raise ValueError(
            f"slot_lora.round_table must define the round at step 0; the earliest "
            f"stage starts at {stages[0][0]}. Steps before the first stage would "
            "otherwise train NO round -- gradient nowhere, anchor nowhere -- and look "
            "exactly like a slow warm-up."
        )
    return stages


def slot_alt_phase(
    schedule: Optional[str], cycle_pos: int, is_last_update: bool
) -> tuple[Optional[str], int]:
    """Which factor this optimizer update trains, and where the cycle stands afterwards.

    ALTERNATION IS PER OPTIMIZER UPDATE, NOT PER TRAINING STEP. One rollout yields
    several updates (global_batch_size 192 at micro_batch_size 8 over 6 ranks is 4
    micro-batches per update; three global batches per step is three updates), and it
    is the update that is the atom here: with B fixed, ΔW = B Ā is linear in A and vice
    versa, so each subproblem is better conditioned than chasing both at once.
    Orthogonality does NOT depend on any of this -- it comes from the Z
    parameterization and holds whatever the schedule says.

    THE LAST UPDATE OF EVERY STEP IS FORCED TO B. B is the readout computed on the
    current Ā, so ending each step on B means every checkpoint that step could save
    carries a B that matches its Ā. Doing it per step rather than once at the end of
    the run is strictly stronger and needs no knowledge of when the run ends.

    THE FORCED B DOES NOT CONSUME THE CYCLE POSITION, and that is not a detail: with
    three updates per step and a three-character schedule, advancing the cycle on the
    override would lock it in phase with the step boundary, the ``A`` would land on the
    forced update every single time, and A would never train -- for the whole run, with
    ``slot/phase_is_A`` reading a perfectly plausible 0.0 and nothing else to say so.
    Deferring the position instead retries that A on the next step's first update.

    Args:
        schedule: The cycle from :func:`parse_slot_alt_schedule`; ``None`` (or empty)
            means train both factors jointly.
        cycle_pos: Where the cycle stands, carried across training steps by the caller.
        is_last_update: Whether this is the final optimizer update of the training step.

    Returns:
        ``(phase, next_cycle_pos)``. ``phase`` is ``"A"``, ``"B"``, or ``None`` for
        "train both" -- the joint ablation, and the only case in which nothing is
        frozen.
    """
    if not schedule:
        return None, cycle_pos
    if is_last_update:
        return "B", cycle_pos
    return schedule[cycle_pos % len(schedule)], (cycle_pos + 1) % len(schedule)


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        cfg_fsdp: Optional[DictConfig] = None,
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        if cfg_fsdp is None:
            cfg_fsdp = cfg.actor
        Worker.__init__(self)
        super().__init__(cfg_fsdp, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            cfg.actor.model.encoder_seq_length - cfg.data.max_prompt_length
        )
        self.calculate_entropy = cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = cfg.algorithm.kl_beta
        self.kl_penalty_type = cfg.algorithm.kl_penalty_type
        self.reinpp_kl_beta = cfg.algorithm.get("reinpp_kl_beta", 0.0)
        self.combine_reference_model = cfg.actor.get("combine_reference_model", True)

        self.total_batch_size_per_dp = (
            cfg.data.rollout_batch_size * cfg.algorithm.group_size // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(cfg.algorithm.loss_agg_func)
        self.enable_offload = not self.is_pipeline and cfg.actor.get(
            "enable_offload", False
        )
        self.micro_batch_size = cfg.actor.micro_batch_size
        self.n_mini_batches = cfg.algorithm.n_minibatches
        self.task_type = cfg.runner.task_type
        self.entropy_op_type = cfg.algorithm.get("entropy_op_type", "flash_attn")
        self.enable_dp_load_balance = cfg.actor.get("enable_dp_load_balance", False)
        self.lr_sched_sync_with_optim = cfg.actor.get("lr_sched_sync_with_optim", True)
        self.enable_dynamic_batch_size = cfg.runner.get(
            "enable_dynamic_batch_size", False
        )
        if self.is_pipeline:
            assert not self.enable_dp_load_balance, (
                "DP load balance is not supported in pipeline mode."
            )
            assert not self.enable_dynamic_batch_size, (
                "Dynamic batch size is not supported in pipeline mode."
            )
        self.max_tokens_per_mbs = cfg.runner.get("max_tokens_per_mbs", 2048)

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if (
            self.kl_beta > 0 or self.reinpp_kl_beta > 0
        ) and self.combine_reference_model:
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)
            self.offload_model_buffer = {}

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        pass

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    def sync_model_to_rollout(self):
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        rollout_dtype = None
        if self._cfg.get("sync_precision", None) is not None:
            rollout_dtype = torch_dtype_from_precision(self._cfg.sync_precision)

        rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        has_visual = any("visual." in k for k in rollout_state_dict.keys())
        model_bucket_list = self.divide_model_to_bucket(rollout_state_dict, has_visual)
        del rollout_state_dict
        send_handles = []
        buffer = {}
        for bucket_idx, model_bucket in enumerate(model_bucket_list):
            for k, v in model_bucket.items():
                if isinstance(v, DTensor):
                    v = v.full_tensor()
                if rollout_dtype is not None:
                    v = v.to(rollout_dtype)
                if not self.is_pipeline:
                    v = reduce_tensor(v)
                buffer[k] = v
            if bucket_idx == 0:
                buffer["bucket_length"] = len(model_bucket_list)

            for send_handle in send_handles:
                send_handle.wait()
            send_handles = []

            if not self.is_pipeline:
                send_handle = self.send(
                    buffer,
                    self._rollout_group_name,
                    self._weight_dst_rank_in_rollout,
                    async_op=True,
                )
                send_handles.append(send_handle)
            else:
                for rank in self._weight_dst_rank_in_rollout:
                    send_handle = self.send(
                        buffer,
                        self._rollout_group_name,
                        rank,
                        async_op=True,
                    )
                    send_handles.append(send_handle)
            buffer = {}

        for send_handle in send_handles:
            send_handle.wait()

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad()

        clear_memory(sync=False)

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def get_dynamic_batch_as_much(
        self,
        input_channel: Channel,
        min_result_len: int,
        max_result_len: int,
        cliped_results=[],
        unfinished_result=None,
    ):
        assert not input_channel.is_local
        rollout_results = cliped_results
        # get min_result_len
        while len(rollout_results) < min_result_len:
            if unfinished_result is not None:
                rollout_result: RolloutResult = unfinished_result.wait()
                unfinished_result = None
            else:
                rollout_result: RolloutResult = input_channel.get()
            rollout_results.append(rollout_result)

        # try to get result as much
        # get result in every 0.1s and do all reduce to get the min result between dp (result_len)
        # stop at: the min result between dp (result_len) is same as the last min result
        last_result_len = 0
        result_len = len(rollout_results)
        time_until = time.time() + 0.1
        while last_result_len < result_len:
            if len(rollout_results) < max_result_len:
                if unfinished_result is None:
                    unfinished_result = input_channel.get(async_op=True)
                else:
                    time.sleep(0.001)
                if unfinished_result.done():
                    rollout_results.append(unfinished_result.wait())
                    unfinished_result = None
                if time.time() >= time_until:
                    last_result_len = result_len
                    result_len = all_reduce_int(len(rollout_results))
                    if last_result_len < result_len:
                        time_until = time.time() + 0.1
            else:
                last_result_len = result_len
                result_len = all_reduce_int(len(rollout_results))

        cliped_results = list(rollout_results[result_len:])
        rollout_results = rollout_results[:result_len]

        batches = []
        for rollout_result in rollout_results:
            batch = rollout_result.to_actor_batch(
                self.cfg.data.max_prompt_length,
                self.cfg.actor.model.encoder_seq_length,
                self.tokenizer.eos_token_id,
            )
            batches.append(batch)

        batch = RolloutResult.merge_batches(batches)
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        return batch, rollout_result, result_len, cliped_results, unfinished_result

    @staticmethod
    def _split_to_micro_batch(
        batch,
        enable_dynamic_batch_size: bool,
        *,
        max_tokens_per_mbs: Optional[int] = None,
        split_num,
    ):
        if enable_dynamic_batch_size:
            (
                micro_batches_iter,
                _,
                micro_batch_cnt,
                dbs_indices,
            ) = split_dynamic_batch_size(
                batch=batch,
                cp_world_size=1,
                vpp_world_size=1,
                max_tokens_per_mbs=max_tokens_per_mbs,
                microbatch_group_size_per_vp_stage=1,
            )
        else:
            micro_batch_cnt = split_num
            micro_batches_iter = get_iterator_k_split(batch, micro_batch_cnt)
            dbs_indices = None
        return micro_batches_iter, micro_batch_cnt, dbs_indices

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    def compute_logprobs(self, logits, target):
        return compute_logprobs_from_logits(
            logits,
            target,
            op_type=self.entropy_op_type,
        )

    def forward_batch(
        self, m_batch: dict[str, torch.Tensor], calculate_entropy: bool = False
    ) -> torch.Tensor:
        input_ids = m_batch["input_ids"]
        attention_mask = m_batch["attention_mask"]
        position_ids = m_batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in m_batch.keys():
            for key in m_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                    dim=0,
                ).to(Worker.torch_device_type)

        if self.enable_dynamic_batch_size:
            max_seq_len_pack = self.max_tokens_per_mbs
            max_seq_len_unpack = self.cfg.actor.model.encoder_seq_length
            max_prompt_len = self.cfg.data.max_prompt_length
            max_response_len = max_seq_len_unpack - max_prompt_len
            idx_starts, idx_ends = prepare_pack_fsdp(m_batch, max_prompt_len)

            input_ids, position_ids, attention_mask = pack_fsdp_input(
                input_ids,
                position_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_pack=max_seq_len_pack,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **multi_modal_inputs,
            )

        logits: torch.Tensor = outputs.logits
        logits.div_(self.cfg.algorithm.sampling_params.temperature)
        if self.enable_dynamic_batch_size:
            logprobs = unpack_fsdp_logprobs(
                logits,
                input_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_unpack=max_seq_len_unpack,
                eos_token_id=self.tokenizer.eos_token_id,
                compute_logprobs_fn=self.compute_logprobs,
            )
            logprobs = logprobs[:, -max_response_len:]
        else:
            # (bsz, response_length, vocab_size)
            logits = logits[:, -self.response_len - 1 : -1, :]
            responses = input_ids[:, -self.response_len :]
            logprobs = self.compute_logprobs(logits, responses)
        if calculate_entropy:
            entropy = compute_entropy_from_logits(logits)
            if self.enable_dynamic_batch_size:
                entropy = unpack_sequences(
                    entropy, idx_starts, idx_ends, max_seq_len_unpack, pad_val=0
                )[:, -self.response_len :]
            return logprobs, entropy
        return logprobs

    def inference_step(
        self,
        batch: dict[str, torch.Tensor],
        rollout_result: RolloutResult,
        compute_ref_logprobs: bool,
    ):
        micro_batches_iter, _, dbs_indices = self._split_to_micro_batch(
            batch,
            self.enable_dynamic_batch_size,
            max_tokens_per_mbs=self.max_tokens_per_mbs,
            split_num=rollout_result.num_sequence
            // self.cfg.algorithm.logprob_forward_micro_batch_size,
        )
        if self.enable_dynamic_batch_size:
            indices = sum(dbs_indices, [])
            revert_indices = torch.tensor(
                get_reverse_idx(indices),
                dtype=torch.long,
            )
        micro_batches = list(micro_batches_iter)

        prev_logprobs, ref_logprobs = None, None

        # Prev logprobs
        prev_logprobs = torch.cat(
            [self.forward_batch(batch) for batch in micro_batches]
        ).cpu()

        if self.enable_dynamic_batch_size:
            assert len(indices) == prev_logprobs.size(0), (
                f"Dynamic batch size indices length {len(indices)} does not equal "
                f"output length {prev_logprobs.size(0)}"
            )
            prev_logprobs = prev_logprobs[revert_indices]

        # Ref logprobs
        if compute_ref_logprobs:
            assert self.ref_policy_state_dict is not None, (
                "Reference policy state dict is None but compute_ref_logprobs is True"
            )
            with cpu_weight_swap(
                self.model,
                self.ref_policy_state_dict,
                self.offload_model_buffer,
            ):
                ref_logprobs = torch.cat(
                    [self.forward_batch(batch) for batch in micro_batches]
                ).cpu()

                if self.enable_dynamic_batch_size:
                    assert len(indices) == ref_logprobs.size(0), (
                        f"Dynamic batch size indices length {len(indices)} does not equal "
                        f"output length {ref_logprobs.size(0)}"
                    )
                    ref_logprobs = ref_logprobs[revert_indices]

        return prev_logprobs, ref_logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
        do_offload=False,
    ):
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
            do_offload: Whether offload weights after inference is done
        """
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        inference_split = self.cfg.actor.get("inference_split", None)
        if inference_split is None:
            if not self.is_pipeline:
                inference_split = 1
            else:
                inference_split = self.cfg.algorithm.n_minibatches
        assert self.total_batch_size_per_dp % inference_split == 0, (
            f"FSDPActor: total_batch_size_per_dp[{self.total_batch_size_per_dp}] should be divisible by inference_split[{inference_split}]"
        )

        min_result_len = 1
        max_result_len = (
            self.cfg.data.rollout_batch_size // self._world_size // inference_split
        )
        if not self.is_pipeline:
            min_result_len = max_result_len
            coll_rollout_results = []
        total_result_len = 0
        total_result_len_per_dp = self.cfg.data.rollout_batch_size // self._world_size
        cliped_results, unfinished_result = [], None
        while total_result_len < total_result_len_per_dp:
            batch, rollout_result, result_len, cliped_results, unfinished_result = (
                self.get_dynamic_batch_as_much(
                    input_channel,
                    min(min_result_len, total_result_len_per_dp - total_result_len),
                    min(max_result_len, total_result_len_per_dp - total_result_len),
                    cliped_results,
                    unfinished_result,
                )
            )
            total_result_len += result_len
            self.log_debug(
                f"[dynamic inference rank-{self._rank}] inference result_len={result_len}, total_result_len={total_result_len}/{total_result_len_per_dp}"
            )
            self._load_weight_and_optimizer()
            self.model.eval()

            with self.worker_timer():
                with torch.no_grad():
                    prev_logprobs, ref_logprobs = self.inference_step(
                        batch, rollout_result, compute_ref_logprobs
                    )

                if rollout_result.rollout_logprobs is not None:
                    # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                    rollout_result.recompute_prev_logprobs = prev_logprobs
                else:
                    # Otherwise, directly store the logprobs in prev_logprobs (the final logprobs used for training)
                    rollout_result.prev_logprobs = prev_logprobs

                # Ref logprobs
                if compute_ref_logprobs:
                    rollout_result.ref_logprobs = ref_logprobs

            if self.is_pipeline:
                # for pipeline mode, send after inference to reduce latency.
                # should do split to ensure actor won't get too much batches.
                split_results = RolloutResult.split_results(rollout_result, result_len)
                for split_result in split_results:
                    output_channel.put(split_result, async_op=True)
            else:
                coll_rollout_results.append(rollout_result)

        if not self.is_pipeline:
            # for coll mode, merge results to reduce send time.
            rollout_result = RolloutResult.merge_result_list(coll_rollout_results)
            split_results = RolloutResult.split_results(
                rollout_result,
                min(total_result_len, self.cfg.algorithm.n_minibatches),
            )
            for split_result in split_results:
                output_channel.put(split_result)
        assert total_result_len == total_result_len_per_dp, (
            f"Expected {total_result_len_per_dp} sequences from channel, but got {total_result_len}"
        )

    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batches_iter, micro_batch_cnt, _ = self._split_to_micro_batch(
                batch,
                self.enable_dynamic_batch_size,
                max_tokens_per_mbs=self.max_tokens_per_mbs,
                split_num=global_batch_size // self.micro_batch_size,
            )
            self.gradient_accumulation = micro_batch_cnt
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == micro_batch_cnt,
            )
            for k, v in m_batch.items():
                m_batch[k] = (
                    v.to(Worker.torch_device_type) if isinstance(v, torch.Tensor) else v
                )

            # batch for forward
            logprobs, entropy = self.forward_batch(m_batch, True)

            # batch for backward
            prev_logprobs = m_batch["prev_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            if self.cfg.algorithm.get("importance_sampling_fix", False):
                rollout_prev_logprobs = prev_logprobs
                recompute_prev_logprobs = m_batch["recompute_prev_logprobs"]
                advantages = advantages * torch.clamp(
                    (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                    min=self.cfg.algorithm.importance_sampling_clip,
                )

            loss, mbs_metrics_data = policy_loss(
                task_type=self.task_type,
                loss_type=self.cfg.algorithm.loss_type,
                loss_agg_func=self.loss_agg_func,
                logprobs=logprobs,
                old_logprobs=prev_logprobs,
                advantages=advantages,
                clip_ratio_c=clip_ratio_c,
                clip_ratio_low=clip_ratio_low,
                clip_ratio_high=clip_ratio_high,
                loss_mask=loss_mask,
                clip_log_ratio_min=self.cfg.algorithm.get("clip_log_ratio_min", None),
                clip_log_ratio_max=self.cfg.algorithm.get("clip_log_ratio_max", None),
                fast_path_zero_loss_mask=True,
            )

            entropy_loss = torch.tensor(
                0.0, device=Worker.torch_platform.current_device()
            )
            if self.calculate_entropy:
                entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                if self.calculate_entropy_loss:
                    loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

            kl_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
            if self.kl_beta > 0 and ref_logprobs is not None:
                kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                kl_loss = self.loss_agg_func(kld, loss_mask)
                loss = loss + kl_loss * self.kl_beta

            # add to log
            # scale loss for gradient accumulation and backprop
            final_loss_metric = loss.detach()
            loss = loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "actor/final_loss": final_loss_metric,
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()

        if self.lr_sched_sync_with_optim:
            self.lr_scheduler.step()

        # aggregate metrics across micro-batches
        mean_metric_dict = {
            key: torch.mean(torch.stack(value))
            for key, value in mbs_metrics_list.items()
        }
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        mean_metric_dict["actor/grad_norm"] = float(grad_norm)
        mean_metric_dict["actor/lr"] = lr_list[0]
        return mean_metric_dict

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer("run_training"):
            for _ in range(self.n_mini_batches):
                mean_metric_dict = self.training_step(batch=train_batch_iterator)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def _dp_load_balance(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        assert batch_size == self.total_batch_size_per_dp, (
            f"DP Load balance is only available when a single batch contains all data, e.g., in collocated mode. But got {batch_size=} and {self.total_batch_size_per_dp=}."
        )
        batch = RolloutDataBalance.from_rollout_batches(
            rollout_batches=batch,
            dp_world_size=torch.distributed.get_world_size(),
            dp_rank=torch.distributed.get_rank(),
            dp_group=torch.distributed.group.WORLD,
            partitioning_tool=get_seqlen_balanced_partitions,
        )
        return batch

    def run_training(
        self, input_channel: Channel, do_offload=False
    ) -> tuple[dict, list]:
        # Get all batches for this DP
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        if self.is_pipeline:
            return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.enable_dp_load_balance:
            global_batch = self._dp_load_balance(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["response_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mini_batch in mini_batches:
                mean_metric_dict = self.training_step(batch=mini_batch)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"][:, -self.response_len :]
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].to(Worker.torch_device_type),
                    loss_mask=mask.to(Worker.torch_device_type),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.reinpp_kl_beta,
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=batch["prev_logprobs"].to(Worker.torch_device_type)
                    if "prev_logprobs" in batch
                    else None,
                    ref_logprob=batch["ref_logprobs"].to(Worker.torch_device_type)
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages

        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )

        self.enable_sft_co_train = cfg.actor.get("enable_sft_co_train", False)
        self.version = 0
        if self.enable_sft_co_train:
            self._build_sft_data_loader()

        # create weight syncer
        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer")
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)

        # ---- slot-LoRI ------------------------------------------------------------
        # K per-suite LoRA slots on orthogonal input subspaces instead of one shared
        # block, so K distillation teachers stop overwriting each other. Read from
        # CONFIG, which makes it identical on every rank -- the routing metric below is
        # emitted on this condition, and all_reduce_dict sizes its packed tensor by the
        # key count, so a rank-dependent metric key set deadlocks the collective.
        self._slot_enabled = bool(
            OmegaConf.select(cfg, "actor.model.slot_lora.enabled", default=False)
        )
        self._slot_gate = None  # the student's SlotGate; resolved in init_worker
        self._slot_order = ()  # slot index -> suite name, read off the model
        self._slot_gate_ids = None  # THIS micro-batch's routing (set by _route_prepare)
        self._slot_fallback = 0.0  # fraction of samples that matched no suite
        self._route_ready = False  # was the routing prepared for THIS micro-batch?
        self._slot_alt_init()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """
        Setup destination ranks for weight communication.
        It can support any topology between actor and rollout workers.
        Assuming there are M actor ranks and N rollout ranks, each actor rank
        will send weights to most ceil(N/M) rollout ranks according to the modulo rule.
        """
        rollout_world_size = self._component_placement.get_world_size("rollout")
        actor_world_size = self._world_size
        rank = self._rank
        self._weight_dst_rank_in_rollout = []
        rollout_ranks_per_actor = (
            rollout_world_size + actor_world_size - 1
        ) // actor_world_size
        for i in range(rollout_ranks_per_actor):
            if i * actor_world_size + rank < rollout_world_size:
                self._weight_dst_rank_in_rollout.append(i * actor_world_size + rank)

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        # VLA-OPD: load a frozen teacher (separate bf16 model, no LoRA, no grad) that
        # scores the student's on-policy rollouts. Kept resident on GPU alongside the
        # (LoRA) student; only used for no_grad forward -> teacher logprobs.
        self.teacher_model = None
        if self.cfg.actor.get("use_teacher_distill", False):
            self._load_teacher_model()

        # dual-KL BASE anchor: frozen copy of the base/generalist (= student init) whose
        # broad behavior we preserve via a mode-covering forward-KL term (data-free).
        # Only loaded when the anchor is active (anchor_lambda>0), so default runs unchanged.
        self.base_model = None
        # anchor_mode=self_masked replaces the loaded base with the student's own
        # slot-k-excluded forward (see the anchor block in run_training): the 15G
        # frozen copy would sit in memory scoring nothing. shift_beta still needs the
        # real base -- it anchors to the ORIGINAL model, which the masked self is not
        # once any slot has trained -- so it keeps loading one.
        _needs_base_anchor = (
            float(self.cfg.algorithm.get("anchor_lambda", 0.0)) > 0.0
            or float(self.cfg.algorithm.get("visual_anchor_lambda", 0.0)) > 0.0
        ) and str(self.cfg.algorithm.get("anchor_mode", "base")) != "self_masked"
        if (
            _needs_base_anchor
            or float(self.cfg.algorithm.get("shift_beta", 0.0)) > 0.0
        ):
            self._load_base_model()

        # slot-LoRI: resolve the gate ONCE (find_slot_gate walks the module tree when
        # the explicit handle is missing, and this is read once per micro-batch), and
        # check the teacher routing table can actually fill every slot. AFTER the
        # teachers, because that check needs their routing table.
        self._slot_routing_setup()

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

        self._setup_rollout_weight_dst_ranks()

    def _load_teacher_model(self) -> None:
        """VLA-OPD frozen teacher(s). Two forms:

          * single  : ``actor.teacher_model_path`` (one teacher for everything).
          * routed  : ``actor.teacher_map`` = {suite: ckpt_path} (sequential CL). Teachers
            are DE-DUPLICATED by path, so mapping every suite to the one 130 generalist
            (the current setup) loads exactly ONE model. Point a suite at its own expert
            ckpt later to go true multi-teacher -- only the map changes.

        Each teacher is full (non-LoRA), eval + requires_grad_(False), resident on the
        training device, and only SCORES the student's rollouts (never acts)."""
        from copy import deepcopy

        from omegaconf import OmegaConf, open_dict

        def _load_one(path: str):
            # A teacher may be given either as a full HF dir, or as "<base_dir>::<adapter_dir>"
            # (base + PEFT LoRA adapter). The latter lets N per-suite expert teachers that all
            # sit on the SAME base be expressed as N small adapters (our spatial/goal/object
            # teachers are exactly this); get_model applies the adapter via is_lora/lora_path.
            _adapter = None
            if "::" in str(path):
                path, _adapter = str(path).split("::", 1)
            tcfg = deepcopy(self.cfg.actor.model)
            with open_dict(tcfg):
                tcfg.model_path = path
                tcfg.is_lora = _adapter is not None
                tcfg.lora_path = _adapter
                # A teacher is a PEFT/full model, NEVER a slot-LoRI student: it is
                # frozen and only scores. On a slot run actor.model carries
                # slot_lora.enabled=true and this deepcopy drags it along, and
                # get_model would then take the slot path -- which refuses to run
                # together with lora_path (it would silently ignore the adapter), so
                # every adapter-form teacher dies at load. Drop the key here instead.
                if tcfg.get("slot_lora", None) is not None:
                    tcfg.slot_lora = None
                # teacher may have DIFFERENT native norm_stats than the student. Teacher
                # only SCORES (never acts), so its unnorm_key is a load-time validation
                # only. Optional override; default (None) = inherit student's key.
                _tuk = self.cfg.actor.get("teacher_unnorm_key", None)
                if _tuk:
                    tcfg.unnorm_key = _tuk
            m = get_model(tcfg)
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            return m

        _tmap = self.cfg.actor.get("teacher_map", None)
        if _tmap:
            if OmegaConf.is_config(_tmap):
                _tmap = OmegaConf.to_container(_tmap, resolve=True)
            _tmap = dict(_tmap)
            # Only keep suites we actually roll out on. The stock config maps ALL FOUR suites to
            # the 130 generalist; a run over a subset would otherwise load teachers for suites
            # that never appear in a batch (wasting a full 7B each -> OOM) and would also break
            # the shared-base fast path below (one stale non-adapter entry disables it).
            try:
                _act = self.cfg.env.train.get("active_suites", None)
                if _act:
                    _act = set(OmegaConf.to_container(_act, resolve=True)
                               if OmegaConf.is_config(_act) else _act)
                    _drop = [s for s in _tmap if s not in _act]
                    if _drop and len(_act & set(_tmap)) > 0:
                        for s in _drop:
                            _tmap.pop(s)
                        self.log_info(
                            f"[VLA-OPD] teacher_map restricted to active suites "
                            f"{sorted(_act)}; dropped {sorted(_drop)}"
                        )
            except Exception:
                pass
            self.teacher_models = {}  # unique ckpt path -> model (loaded once)
            self.teacher_suite_to_path = {}  # suite name -> ckpt path (routing table)
            # SHARED-BASE FAST PATH: when every teacher is "<same base>::<adapter>", load the
            # 7B base ONCE and attach the adapters to it (PEFT multi-adapter). N experts then
            # cost 1 base + N small adapters instead of N full models -- without this, 3
            # teachers + the student = 4x7B and the run OOMs on 2 GPUs.
            _paths = list(dict.fromkeys(_tmap.values()))
            _all_adapters = all("::" in str(p) for p in _paths)
            self.teacher_adapter_of_path = None
            if _all_adapters and len(_paths) > 1:
                # GROUP BY BASE, rather than demanding a single base for all teachers. The 4-teacher
                # set has two lineages -- spatial/object/goal sit on base_stats130 while the long
                # teacher sits on the CL student's own merged model -- and the old "len(bases)==1"
                # test failed that outright, silently falling back to loading FOUR full 7B teachers
                # (the 78.5 GiB OOM). Per base we pay one 7B and attach that base's adapters, so
                # here it is 2 bases + 4 adapters instead of 4 full models.
                # Adapter names only have to be unique WITHIN one model, so each group may reuse
                # "default"; the routing code looks up teacher_models[path] first, then switches to
                # teacher_adapter_of_path[path] on THAT model.
                _by_base = {}
                for _p in _paths:
                    _by_base.setdefault(_p.split("::", 1)[0], []).append(_p)
                self.teacher_adapter_of_path = {}
                for _gi, (_b, _ps) in enumerate(_by_base.items()):
                    _shared = _load_one(_ps[0])         # base + its first adapter (name "default")
                    _pm = _shared if hasattr(_shared, "load_adapter") else getattr(_shared, "model", None)
                    self.teacher_adapter_of_path[_ps[0]] = "default"
                    for _i, _p in enumerate(_ps[1:], start=1):
                        _name = f"g{_gi}t{_i}"
                        _pm.load_adapter(_p.split("::", 1)[1], adapter_name=_name)
                        self.teacher_adapter_of_path[_p] = _name
                    for _p in _ps:
                        self.teacher_models[_p] = _shared  # one object per BASE; adapter per suite
                self.log_info(
                    f"[VLA-OPD] SHARED-BASE teachers: {len(_by_base)} base(s) + {len(_paths)} adapters "
                    f"{list(self.teacher_adapter_of_path.values())}"
                )
                for suite, path in _tmap.items():
                    self.teacher_suite_to_path[suite] = path
            else:
                for suite, path in _tmap.items():
                    self.teacher_suite_to_path[suite] = path
                    if path not in self.teacher_models:
                        self.teacher_models[path] = _load_one(path)
            # Single handle used by the OPD loss. With one generalist teacher for all
            # suites this IS that teacher. For true multi-teacher, select per suite via
            # self.teacher_suite_to_path at the teacher-forward site (OPD block).
            self.teacher_model = next(iter(self.teacher_models.values()))
            self.log_info(
                f"[VLA-OPD] teacher_map: loaded {len(self.teacher_models)} unique "
                f"teacher(s) for {len(self.teacher_suite_to_path)} suite(s): "
                f"{sorted(set(self.teacher_suite_to_path.values()))}"
            )
            # TRUE multi-teacher routing table. The batch reaching the actor carries the
            # tokenized task prompt (input_ids); each LIBERO task's language instruction is
            # unique, so prompt -> task -> suite -> teacher is an exact lookup. We key on the
            # LOWERCASED instruction text (matching how the prompt is built) and resolve at
            # forward time by decoding input_ids once per micro-batch.
            self.teacher_prompt_to_suite = None
            # >= 1, not > 1: a SINGLE-entry teacher_map is how the LoRI-independent runs
            # train exactly one slot (active_suites restricts the env to that suite and
            # this map to that expert). The slot machinery still needs prompt->suite to
            # build its masks and gate ids, and "one routed suite" is a perfectly good
            # routing -- the true single-teacher form (teacher_model_path, no map) still
            # skips the table below, unchanged.
            if len(self.teacher_suite_to_path) >= 1:
                try:
                    # Build prompt->suite the SAME way get_libero130_task_id_to_suite()
                    # builds task_id->suite (iterate benchmark.libero_suites -> task_maps,
                    # de-dup by task name) so the two are guaranteed consistent.
                    from libero.libero import benchmark as _lb

                    # Only the suites we actually route (the teacher_map keys). LIBERO's 130
                    # tasks have 112 unique instructions; the 2 cross-suite duplicates both
                    # involve libero_90, which is never a teacher_map key -> restricting to
                    # the mapped suites makes the prompt key EXACT for our routing.
                    _want = set(self.teacher_suite_to_path.keys())
                    _p2s, _seen, _dupe = {}, set(), 0
                    for _suite_name in getattr(_lb, "libero_suites", []):
                        if _suite_name not in _want:
                            continue
                        for _tname, _task in _lb.task_maps.get(_suite_name, {}).items():
                            if _tname in _seen:
                                continue
                            _seen.add(_tname)
                            _lang = getattr(_task, "language", None)
                            if not _lang:
                                continue
                            _k = _lang.strip().lower()
                            if _k in _p2s and _p2s[_k] != _suite_name:
                                _dupe += 1  # ambiguous across ROUTED suites -> would misroute
                            _p2s[_k] = _suite_name
                    if not _p2s:
                        raise RuntimeError("empty prompt->suite map")
                    if _dupe:
                        self.log_warning(
                            f"[VLA-OPD] {_dupe} instruction(s) are ambiguous across routed "
                            "suites; those samples may be routed to the wrong expert."
                        )
                    self.teacher_prompt_to_suite = _p2s
                    self.log_info(
                        f"[VLA-OPD] multi-teacher routing ON: {len(_p2s)} task prompts -> "
                        f"suites {sorted(set(_p2s.values()))}"
                    )
                except Exception as e:  # routing table optional; fall back to single teacher
                    self.log_warning(
                        f"[VLA-OPD] could not build prompt->suite routing table ({e}); "
                        "falling back to the FIRST teacher for all suites."
                    )
        else:
            self.teacher_models = None
            self.teacher_suite_to_path = None
            self.teacher_model = _load_one(self.cfg.actor.teacher_model_path)
            self.log_info(
                f"[VLA-OPD] loaded frozen teacher from "
                f"{self.cfg.actor.teacher_model_path}"
            )

    def _slot_routing_setup(self) -> None:
        """Resolve the student's :class:`SlotGate` and check the routing can drive it.

        Called once, from :meth:`init_worker`, after the model and the teachers exist.
        Every failure below is a config error whose symptom appears a long way from its
        cause:

        * no gate on a slot run -> every one of the 200-400 gated linears reads a
          routing that was never installed (a strict gate raises from inside the model;
          a non-strict one runs ungated and trains every slot on every sample, with no
          error and a normal-looking loss curve);
        * no prompt->suite table -> there is nothing to build a routing FROM, so every
          sample would land on slot ``-1`` and the whole run would train nothing;
        * a teacher-routed suite with no slot -> those samples ARE scored by their
          expert and folded into the loss, but their gradient reaches no slot at all.

        Doing it here costs ~two minutes of startup instead of one rollout (~25 min)
        plus a forward.

        Raises:
            RuntimeError: on any of the three above.
        """
        if not self._slot_enabled:
            return

        from rlinf.models import find_slot_gate

        self._slot_gate = find_slot_gate(self.model)
        if self._slot_gate is None:
            raise RuntimeError(
                "actor.model.slot_lora.enabled=true but the built student has no "
                "SlotGate. get_model only takes the slot path when actor.model.is_lora "
                "is also true (see rlinf/models/__init__.py), so this usually means "
                "is_lora=false, a checkpoint reload that rebuilt the modules, or a "
                "model built by some other path entirely. Routing through a missing "
                "gate is exactly the ungated forward the strict gate exists to prevent."
            )
        order = getattr(self.model, "_slot_order", None)
        if not order:
            raise RuntimeError(
                "the student has a SlotGate but no _slot_order. The slot INDEX order "
                "is what routing produces (match_suite_ids returns "
                "suite_order.index(suite)), and it must be the same value the slots "
                "were built from -- it is stashed on the model by _apply_slot_lora, "
                "not re-read from config here, so that the two cannot drift apart."
            )
        self._slot_order = tuple(str(s) for s in order)

        route = getattr(self, "teacher_prompt_to_suite", None)
        if not route:
            raise RuntimeError(
                f"slot-LoRI is on with slots {list(self._slot_order)}, but there is no "
                "prompt->suite routing table, so no sample can be assigned to a slot. "
                "The table is built in _load_teacher_model from actor.teacher_map (any "
                "number of entries); reaching this means the run uses the mapless "
                "single-teacher form (actor.teacher_model_path) or the table build "
                "failed -- see the warning above it. Either give the run a teacher_map "
                "or turn actor.model.slot_lora.enabled off."
            )
        suite_to_path = getattr(self, "teacher_suite_to_path", None) or {}
        unslotted = sorted(set(suite_to_path) - set(self._slot_order))
        if unslotted:
            raise RuntimeError(
                f"suites {unslotted} have a teacher but no slot "
                f"(actor.model.slot_lora.slot_order={list(self._slot_order)}). Their "
                "samples would still be SCORED by their expert and folded into the "
                "distillation loss while their gradient reached no slot at all -- a "
                "silent one-way loss of exactly those suites' signal."
            )
        dead = [s for s in self._slot_order if s not in suite_to_path]
        if dead:
            self.log_warning(
                f"[slot-lora] slots {dead} have no teacher in teacher_map; no sample "
                "can be routed to them, so they stay at their zero initialization for "
                "the whole run while still costing their rank."
            )
        self.log_info(
            f"[slot-lora] routing ON: slots {list(self._slot_order)} <- "
            f"{len(route)} task prompts over suites {sorted(suite_to_path)}"
        )

    def _route_match_order(self):
        """Suite names the router matches against: the slots' order first.

        A suite that has a TEACHER but no slot is appended rather than dropped, so it
        still reaches its own expert instead of whichever one happens to be first.
        (:meth:`_slot_routing_setup` refuses that combination on a slot run; this keeps
        the plain multi-teacher path, where there are no slots at all, unchanged.)

        Returns:
            The list ``match_suite_ids`` indexes into. Cached: it is derived from
            config and is identical on every rank and every micro-batch.
        """
        order = getattr(self, "_route_match_order_cache", None)
        if order is None:
            order = list(self._slot_order)
            extra = set(getattr(self, "teacher_suite_to_path", None) or {})
            route = getattr(self, "teacher_prompt_to_suite", None) or {}
            extra |= set(route.values())
            order += sorted(s for s in extra if s not in order)
            self._route_match_order_cache = order
        return order

    def _route_prepare(self, forward_inputs) -> None:
        """Decode THIS micro-batch's prompts ONCE and derive BOTH routings from them.

        Sets, for the current micro-batch:

        * ``self._last_groups`` -- teacher ckpt path -> sample indices, what
          :meth:`_teacher_forward` splits and scatters its per-expert forwards with.
        * ``self._slot_gate_ids`` -- ``LongTensor[B]`` giving each sample's slot, with
          ``-1`` for "no slot owns this sample", on the input's device (which is the
          activations' device). ``None`` when slot-LoRI is off.
        * ``self._slot_fallback`` -- the fraction of samples that matched no suite.

        WHY THIS IS NOT PART OF _teacher_forward. The STUDENT forward runs first and
        the teacher forward second, so a gate built from ``self._last_groups`` at the
        student forward would carry the PREVIOUS micro-batch's routing: every sample's
        slot shifted by one micro-batch, with no error, no NaN and a normal-looking
        loss curve. This runs before the student forward; _teacher_forward then reuses
        what it produced instead of decoding a second time.

        WHY BOTH ROUTINGS COME FROM ONE DECODE AND ONE MATCH. If the slot router and
        the teacher router could disagree about a sample, that sample would be scored
        by one suite's expert while its gradient was written into another suite's slot,
        and nothing would raise. So the suite is resolved ONCE, by
        :func:`~rlinf.models.slot_lora.match_suite_ids` -- whose matching semantics are
        the verbatim mirror of the loop this replaced -- and the teacher path and the
        slot index are both read off that one answer.

        Args:
            forward_inputs: This micro-batch's model inputs; ``input_ids`` carries the
                tokenized task prompt and fixes the device the ids are built on.

        Raises:
            RuntimeError: if the share of samples matching no suite exceeds
                ``algorithm.slot_route_fallback_tol`` (default ``0.0``). See the
                message for why a single unmatched sample is worth stopping for.
        """
        from rlinf.models.slot_lora import match_suite_ids
        from rlinf.models.slot_lora.modules import make_slot_ids

        self._last_groups = None
        self._slot_gate_ids = None
        self._slot_fallback = 0.0
        # Set BEFORE the early returns: every exit from here is a fully prepared
        # micro-batch, and _teacher_forward keys its "compute it myself" fallback off
        # this flag -- an exit that left it False would decode the prompts twice.
        self._route_ready = True

        route = getattr(self, "teacher_prompt_to_suite", None)
        models = getattr(self, "teacher_models", None)
        if not route or not models:
            return  # no routing table / no teachers: nothing to route
        if len(models) <= 1 and not self._slot_enabled:
            # A single teacher with no slots really has nothing to route. WITH slots the
            # routing must still run -- the gate ids and the per-sample suites come from
            # here, and skipping it left _slot_gate_ids None and killed the run at the
            # first gated forward (the LoRI-independent single-expert runs, 2026-08-25).
            return
        # The embodied actor has no self.tokenizer; the (frozen) teacher model carries
        # the OFT input_processor, whose .tokenizer decodes the rollout prompts.
        tok = getattr(self, "_route_tokenizer", None)
        if tok is None:
            proc = getattr(self.teacher_model, "input_processor", None)
            tok = getattr(proc, "tokenizer", None) if proc is not None else None
            self._route_tokenizer = tok
        if tok is None:
            return

        ids = forward_inputs["input_ids"]
        bsz = ids.shape[0]
        texts = tok.batch_decode(ids, skip_special_tokens=True)
        match_order = self._route_match_order()
        matched = match_suite_ids(texts, route, match_order)
        suites = [match_order[m] if m >= 0 else None for m in matched]

        # Expose the teacher routing so the OPD loss can (a) build per-suite masks and
        # (b) CROSS-SCORE: re-score suite i's states with suite j's expert.
        # Cross-scoring is the functional-space test of "do the teachers fight" -- if
        # KL(expert_j || student) RISES on suite i's states while KL(expert_i ||
        # student) falls, the student is paying for one teacher with another. Note the
        # teachers never meet in the loss itself (each scores only its own suite), so
        # any conflict has to be parameter-level interference; this measures its
        # behavioural shadow.
        default_path = next(iter(models))
        groups: dict = {}
        for i, suite in enumerate(suites):
            path = (
                self.teacher_suite_to_path.get(suite, default_path)
                if suite
                else default_path
            )
            groups.setdefault(path, []).append(i)
        self._last_groups = groups
        # Per-sample suite of THIS micro-batch, for the seqslot loss masks: the OPD
        # loss keeps only the current round's suite, the anchor keeps the complement.
        # A list of the same length the gate ids have, built from the same `suites`.
        self._last_suites = suites
        self._slot_fallback = sum(1 for s in suites if s is None) / max(bsz, 1)

        if not self._slot_enabled:
            return
        slot_of = self._slot_index_of()
        # make_slot_ids, not torch.as_tensor: it range-checks the plain Python list
        # HERE, before the host-to-device copy, and stamps the tensor so the gate does
        # not repeat the check on the device. The guarantee is identical (an id that
        # matches no slot still raises, loudly -- it is indistinguishable from "no slot
        # owns this sample" downstream, so it would train nothing for a whole suite in
        # silence); what disappears is one device-to-host sync per micro-batch.
        self._slot_gate_ids = make_slot_ids(
            [slot_of.get(s, -1) if s is not None else -1 for s in suites],
            len(self._slot_order),
            device=ids.device,
        )
        # getattr, not a bare read: the routing tests (and any caller composing this
        # method onto a partial actor) run without _slot_alt_init having set the
        # seqslot fields, and "attribute missing" must mean "mode off", not a crash.
        if getattr(self, "_seqslot_suite", None) is not None:
            # seqslot: EVERY sample owns the current round's slot -- the anchor
            # samples included, deliberately. The anchor loss exists to shape what
            # B_k does ON the other suites' states (push its contribution there to
            # zero), and a gradient gated to each sample's own suite-slot would send
            # that signal to the frozen B_j instead of to the one factor that is
            # training. The per-suite routing above still ran: `suites` feeds the
            # loss masks, and the teacher router still groups by suite.
            _round_slot = self._slot_index_of().get(self._seqslot_suite)
            if _round_slot is None:
                raise RuntimeError(
                    f"seqslot round {self._seqslot_suite!r} names no slot in "
                    f"slot_order {sorted(self._slot_index_of())}; _slot_step_begin "
                    "checks this at the round boundary, so reaching it here means "
                    "the round changed without going through _slot_step_begin."
                )
            self._slot_gate_ids = make_slot_ids(
                [_round_slot] * len(suites),
                len(self._slot_order),
                device=ids.device,
            )
        if self._slot_fallback > 0.0:
            unmatched = [t for t, s in zip(texts, suites) if s is None]
            msg = (
                f"[slot-lora] route_fallback: {len(unmatched)}/{bsz} samples of this "
                f"micro-batch match no suite in the {len(route)}-prompt routing table, "
                f"e.g. {unmatched[:3]!r}. Such a sample is ASYMMETRIC: the teacher "
                f"router hands it to {default_path} (whichever expert loaded first) "
                "and its KL is folded into the distillation loss, while the slot "
                "router gives it -1, so its gradient reaches no slot at all -- it only "
                "dilutes the loss denominator for the samples that ARE routed. All 40 "
                "task prompts of this experiment are in the table, so the expected "
                "value is EXACTLY 0 and any other value means the table has a hole "
                "(and that other samples may be misrouted too, which is not visible "
                "here). Fix the table; set algorithm.slot_route_fallback_tol above "
                f"{self._slot_fallback:.4f} only if unrouted samples are deliberate."
            )
            if self._slot_fallback > float(
                self.cfg.algorithm.get("slot_route_fallback_tol", 0.0)
            ):
                raise RuntimeError(msg)
            if not getattr(self, "_slot_fallback_warned", False):
                # Once per run, not once per micro-batch: it is tolerated by config
                # here, so it is news exactly once.
                self._slot_fallback_warned = True
                self.log_warning(msg)

    def _slot_index_of(self):
        """``{suite name: slot index}``, cached. Empty when slot-LoRI is off."""
        cache = getattr(self, "_slot_index_cache", None)
        if cache is None:
            cache = {s: i for i, s in enumerate(self._slot_order)}
            self._slot_index_cache = cache
        return cache

    def _slot_scope(self):
        """Install THIS micro-batch's slot routing, for the forward AND its backward.

        Returned as a context manager rather than applied here because of what the
        scope has to COVER. Gradient checkpointing (fsdp_model_manager.py) re-executes
        the wrapped forward during ``backward()``, on the autograd engine's per-device
        worker thread; a scope that closed after the forward leaves that recomputation
        with no routing installed at all. A strict gate raises there (from inside a
        backward, on another thread); a non-strict one silently runs ungated, every
        slot takes gradient from every sample, and the isolation mechanism is entirely
        off with no error and a normal-looking loss curve. So the block must enclose
        the student forward, the loss, and ``scale(loss).backward()``.

        Returns:
            ``gate.scoped(ids)`` on a slot run, else :func:`contextlib.nullcontext` --
            so with slot-LoRI off this is the code path it was before.

        Raises:
            RuntimeError: on a slot run whose routing was never prepared. The strict
                gate would raise anyway, but 200-400 gated linears deep and with
                nothing naming the micro-batch that skipped _route_prepare.
        """
        gate = self._slot_gate
        if gate is None:
            return nullcontext()
        ids = self._slot_gate_ids
        if ids is None:
            raise RuntimeError(
                "slot-LoRI is enabled but this micro-batch has no slot routing: "
                "_route_prepare either was not called or found nothing to route "
                "(no prompt->suite table, no tokenizer, or no forward_inputs). "
                "Routing must be prepared from THIS micro-batch, before the student "
                "forward -- reusing the previous one shifts every sample's slot by a "
                "micro-batch, silently."
            )
        return gate.scoped(ids)

    # ---- slot-LoRI: the B,B,A alternating schedule and the per-step diagnostics ------

    def _slot_alt_init(self) -> None:
        """Read the alternation config and reset the per-step counters.

        A method called from ``__init__`` rather than inline code there, because it is
        also the only way anything can reach this configuration without ray, LIBERO and
        a 7B checkpoint -- which is to say, the only way it can be tested at all.

        Raises:
            ValueError: on a malformed ``alt_schedule`` (see
                :func:`parse_slot_alt_schedule`) -- at construction, not at the first
                optimizer update ~25 minutes into the run.
        """
        # Only a slot run has factors to alternate. Reading the key only when slots are
        # on also keeps a stale alt_schedule in some inherited config from raising on
        # runs where the whole slot_lora block is ignored anyway.
        self._slot_alt_schedule = (
            parse_slot_alt_schedule(
                OmegaConf.select(
                    self.cfg, "actor.model.slot_lora.alt_schedule", default="BBA"
                )
            )
            if self._slot_enabled
            else None
        )
        # alt_anneal, when set, OVERRIDES alt_schedule per step (see
        # parse_slot_alt_anneal). alt_schedule stays the fallback so a config without
        # the table behaves exactly as before.
        self._slot_alt_anneal = (
            parse_slot_alt_anneal(
                OmegaConf.select(
                    self.cfg, "actor.model.slot_lora.alt_anneal", default=None
                )
            )
            if self._slot_enabled
            else None
        )
        self._slot_alt_base_schedule = self._slot_alt_schedule
        # seqslot: the sequential-round plan. When set, each step trains exactly ONE
        # suite's slot (every sample of the batch owns it) against that suite's expert,
        # and anchors every other suite's samples to M_(k-1) = the student minus that
        # slot -- see _route_prepare and the anchor block in run_training.
        self._seqslot_table = (
            parse_slot_round_table(
                OmegaConf.select(
                    self.cfg, "actor.model.slot_lora.round_table", default=None
                )
            )
            if self._slot_enabled
            else None
        )
        self._seqslot_suite: Optional[str] = None  # round in force; set per step
        if self._seqslot_table is not None:
            # A must be FROZEN for the whole run: the anchor identity "student minus
            # slot k == M_(k-1)" holds only while base, A and the other slots' B are
            # all constant, and frozen_orth (if on) additionally stops re-projecting
            # A. Any schedule that lets A move breaks both, silently.
            offending = None
            if self._slot_alt_anneal is not None:
                offending = "alt_anneal"
            elif self._slot_alt_schedule is not None and "A" in self._slot_alt_schedule:
                offending = f"alt_schedule={self._slot_alt_schedule!r}"
            if offending:
                raise ValueError(
                    f"slot_lora.round_table (seqslot) requires A frozen for the whole "
                    f"run, but {offending} lets A train. Set alt_schedule to 'B' and "
                    "remove alt_anneal: a moving A silently invalidates the "
                    "masked-self anchor (student-minus-slot-k is only M_(k-1) while "
                    "everything but B_k is constant)."
                )
        if bool(
            OmegaConf.select(
                self.cfg, "actor.model.slot_lora.frozen_orth", default=False
            )
        ) and self._slot_enabled:
            moving = None
            if self._slot_alt_anneal is not None and any(
                sched is None or "A" in sched for _, sched in self._slot_alt_anneal
            ):
                moving = "alt_anneal"
            elif self._slot_alt_schedule is None or "A" in self._slot_alt_schedule:
                moving = f"alt_schedule={self._slot_alt_schedule!r}"
            if moving:
                raise ValueError(
                    f"slot_lora.frozen_orth=True but {moving} lets A train. "
                    "frozen_orth skips the per-forward re-orthonormalization, so a "
                    "trained A would drift off the orthonormal manifold with nothing "
                    "re-projecting it -- every slot would silently start overlapping "
                    "every other. Freeze A (alt_schedule='B') or drop frozen_orth."
                )
        self._slot_alt_pos = 0  # cycle position, carried ACROSS training steps
        self._slot_alt_lr: dict[str, float] = {}  # THIS step's scheduler lr per group
        self._slot_alt_frozen = None  # group frozen for the update in flight
        self._slot_alt_updates = 0  # optimizer updates so far this step
        self._slot_alt_a_updates = 0  # ... of which A was free to move
        self._slot_alt_expected = 0  # updates this step was told to expect
        self._slot_alt_warned: set[str] = set()  # one-shot warning keys

    def _slot_warn_once(self, key: str, message: str) -> None:
        """Warn the first time only: these fire per training step, and would be noise."""
        if key in self._slot_alt_warned:
            return
        self._slot_alt_warned.add(key)
        self.log_warning(message)

    def _emit_slot_stage(self, schedule: Optional[str]) -> None:
        """Announce an anneal stage change on stderr.

        stderr, not ``log_info``: a ray worker's ``log_info`` writes into the session
        tmpdir that ``run_iso.sh`` deletes, so it never reaches the driver log -- which
        is where anyone reading this run will look for "when did A stop moving".
        """
        import sys as _sys

        frac = slot_alt_a_fraction(schedule)
        msg = (
            f"[slot-lora] alt stage -> {schedule!r} at step "
            f"{getattr(self, 'version', '?')} (A on {frac:.4g} of updates)"
        )
        _sys.stderr.write(msg + "\n")
        self.log_info(msg)

    def _slot_step_begin(self, updates_per_step: int) -> None:
        """Open a training step: snapshot the lr, reset the counters, arm the diagnostics.

        THE LR SNAPSHOT IS TAKEN HERE, EVERY STEP, and that is the whole point. The
        alternation freezes a factor by writing ``0.0`` over its group's ``lr``, so it
        needs the value to put back -- and the value to put back is whatever the LR
        SCHEDULER set for THIS step (it rewrites every group once per step, at the end
        of ``run_training``). A snapshot taken once at build time would restore the
        warm-up lr for the rest of the run.

        THE DIAGNOSTICS ARE ARMED HERE, BEFORE THE FIRST FORWARD, for the same reason
        they are collected after the last update: ``enable_slot_diag`` arms ONE layer
        and the numbers are computed inside that layer's forward, where its parameters
        are already all-gathered by FSDP and cost no extra collective.

        Args:
            updates_per_step: How many optimizer updates this step will run. The "is
                this the last one" decision is the caller's; this copy is what
                :meth:`_slot_step_metrics` checks the realized count against, because
                the two are computed from the same loop bounds in two places and a
                drift between them would break the forced B silently -- either firing
                it early (the step then ends on whatever the cycle says, possibly A,
                and a checkpoint saved there carries a B that does not match its Ā) or
                never firing it at all.

        Raises:
            RuntimeError: if the schedule is on but the optimizer has no slot groups to
                alternate between -- i.e. the mechanism is configured and dead.
        """
        if not self._slot_enabled:
            return
        self._slot_alt_updates = 0
        self._slot_alt_a_updates = 0
        self._slot_alt_expected = int(updates_per_step)
        self._slot_alt_frozen = None
        # Pick THIS step's schedule off the anneal table. The cycle position is carried
        # across steps on purpose (see slot_alt_phase), but carrying it across a CHANGE
        # of schedule is meaningless -- position 7 of "BBBBBBBA" is not position 7 of
        # "B" -- so a stage boundary resets it.
        if self._slot_alt_anneal:
            picked = slot_alt_schedule_for_step(
                self._slot_alt_anneal,
                self._slot_alt_base_schedule,
                getattr(self, "version", None),
            )
            if picked != self._slot_alt_schedule:
                self._emit_slot_stage(picked)
                self._slot_alt_schedule = picked
                self._slot_alt_pos = 0
        # seqslot: resolve THIS step's round off the table. Same last-stage-at-or-below
        # lookup as the anneal, so a resumed run lands in the right round from its
        # step number alone; an unresolvable step (version not wired) keeps the
        # PREVIOUS round rather than silently snapping back to round 0.
        if self._seqslot_table is not None:
            picked_suite = slot_alt_schedule_for_step(
                self._seqslot_table,
                self._seqslot_suite,
                getattr(self, "version", None),
            )
            if picked_suite is None:
                raise RuntimeError(
                    "seqslot: no round resolvable for this step -- self.version is "
                    "not readable and no previous round is in force. Refusing to "
                    "train: with no round there is no owner slot and no anchor, and "
                    "the step would look like a normal one while training nothing."
                )
            if picked_suite != self._seqslot_suite:
                slot_of = self._slot_index_of()
                if picked_suite not in slot_of:
                    raise RuntimeError(
                        f"seqslot round_table names suite {picked_suite!r}, which is "
                        f"not in slot_order {sorted(slot_of)}. The round would own "
                        "no slot; every sample of every step of this round would "
                        "train nothing, silently."
                    )
                import sys as _sys

                _msg = (
                    f"[slot-lora] seqslot round -> {picked_suite!r} "
                    f"(slot {slot_of[picked_suite]}) at step "
                    f"{getattr(self, 'version', '?')}"
                )
                _sys.stderr.write(_msg + "\n")
                self.log_info(_msg)
                self._seqslot_suite = picked_suite
        self._slot_alt_lr = {
            group["name"]: float(group["lr"])
            for group in self.optimizer.param_groups
            if group.get("name") in (PARAM_GROUP_SLOT_A, PARAM_GROUP_SLOT_B)
        }
        if self._slot_alt_schedule and len(self._slot_alt_lr) < 2:
            if getattr(self, "critic_warmup_steps", 0) > 0:
                # Critic warmup rebuilds the optimizer with ONLY the value head, so
                # there is nothing to alternate and nothing to restore. Tolerated,
                # because the slot factors are not training at all during it.
                self._slot_warn_once(
                    "alt_critic_warmup",
                    "[slot-lora] critic warmup is active, so the optimizer has no "
                    "slot_A/slot_B groups; the alternating schedule is inert until "
                    "warmup ends and the optimizer is rebuilt.",
                )
                self._slot_alt_lr = {}
            else:
                raise RuntimeError(
                    "actor.model.slot_lora.alt_schedule is set, but the optimizer has "
                    f"no {PARAM_GROUP_SLOT_A}/{PARAM_GROUP_SLOT_B} groups to alternate "
                    f"between (found {sorted(self._slot_alt_lr)}). The factors are "
                    "grouped by parameter NAME in build_slot_aware_param_groups, so "
                    "this means the slot parameters are not in this optimizer at all "
                    "-- the schedule would silently do nothing for the whole run while "
                    "slot/phase_is_A still reported a plausible number."
                )
        if self._slot_alt_schedule and updates_per_step < 2:
            self._slot_warn_once(
                "alt_one_update",
                f"[slot-lora] this step has {updates_per_step} optimizer update(s) and "
                "the last update of every step is forced to B, so A never trains. "
                "Give the step more than one update (a smaller global_batch_size or a "
                "larger update_epoch), or set alt_schedule: null to train both factors "
                "jointly.",
            )

        from rlinf.models.slot_lora import enable_slot_diag

        if not enable_slot_diag(self.model):
            self._slot_warn_once(
                "diag_no_slots",
                "[slot-lora] slot metrics are on but the model has no SlotLoRALinear "
                "to arm; every slot/* metric will read NaN.",
            )

    def _slot_alt_before_update(self, is_last_update: bool) -> Optional[str]:
        """Freeze the factor this optimizer update is not training. Call BEFORE the step.

        TWO THINGS, AND BOTH ARE NECESSARY.

        ``lr = 0.0`` on the frozen group, rather than ``requires_grad_(False)`` on its
        parameters: FSDP with ``use_orig_params=False`` (this repo's default) requires
        uniform ``requires_grad`` within a flat parameter, and flipping it mid-run
        breaks the flattening. The lr is the only knob that stops the update without
        touching the graph.

        ``p.grad = None`` on the frozen group, because a zeroed lr does NOT stop AdamW
        from folding that parameter's gradient into ``exp_avg``/``exp_avg_sq``: it
        would keep accumulating momentum through every update it was supposed to sit
        out, and the first step after it thawed would move it by all of it. AdamW skips
        a parameter whose ``grad`` is ``None`` entirely -- no moments, no step count.
        Dropping the grads before the step also keeps them out of ``clip_grad_norm_``,
        which is right: a gradient that will not be applied should not spend clip
        budget belonging to the factor that is training.

        Args:
            is_last_update: Whether this is the final optimizer update of this training
                step (which is forced to B; see :func:`slot_alt_phase`).

        Returns:
            The phase actually run: ``"A"``, ``"B"``, or ``None`` when both factors
            train (slot-LoRI off, or the joint ablation).
        """
        if not self._slot_enabled:
            return None
        # An empty snapshot means there are no slot groups to alternate between (see
        # _slot_step_begin); running joint is the only honest thing left to do.
        schedule = self._slot_alt_schedule if len(self._slot_alt_lr) == 2 else None
        phase, self._slot_alt_pos = slot_alt_phase(
            schedule, self._slot_alt_pos, is_last_update
        )
        self._slot_alt_updates += 1
        self._slot_alt_a_updates += int(phase != "B")
        self._slot_alt_frozen = None
        if phase is None:
            return None
        frozen = PARAM_GROUP_SLOT_A if phase == "B" else PARAM_GROUP_SLOT_B
        for group in self.optimizer.param_groups:
            if group.get("name") != frozen:
                continue
            group["lr"] = 0.0
            for param in group["params"]:
                param.grad = None
        self._slot_alt_frozen = frozen
        return phase

    def _slot_alt_after_update(self) -> None:
        """Put the scheduler's lr back on both slot groups. Call AFTER the step.

        Both groups, not just the frozen one: restoring what this step's scheduler set
        is idempotent, and a zero that somehow survived into the next step would be
        restored to zero forever -- the group's lr is the only record of it.
        """
        if not self._slot_alt_frozen:
            return
        for group in self.optimizer.param_groups:
            name = group.get("name")
            if name in self._slot_alt_lr:
                group["lr"] = self._slot_alt_lr[name]
        self._slot_alt_frozen = None

    def _slot_step_metrics(self) -> dict:
        """Close a training step: read the armed diagnostics. Call AFTER the last update.

        THE METRICS AND WHAT THEY MEAN:

        * ``slot/orth_err`` -- ``‖Ā Āᵀ − I‖_F``, read off the fp32 recomputation inside
          :class:`~rlinf.models.slot_lora.modules.SlotProj`, NEVER off the bf16 ``Ā``
          the model consumes. Below 1e-3 is healthy; 1e-3 to 5e-2 means Z is degrading;
          above 5e-2 orthogonality has collapsed and the run's results are invalid.
          (Measured: as cond(Z) goes 10 -> 20 the fp32 error degrades 16x, 6.8e-5 ->
          1.07e-3, while a bf16 reading moves 0.01887 -> 0.01894 -- the rounding floor
          hides the entire slide, and by the time bf16 moves the result is garbage.)
        * ``slot/cos_{s}_{t}`` -- cosine between two slots' ΔW, for every s < t. Below
          1e-3 healthy, above 5e-2 collapsed.
        * ``slot/dw_norm_{k}`` -- ``‖ΔW_k‖_F``. Expected to grow, largest for the suite
          furthest from its teacher; a slot pinned at 0 was never routed to.
        * ``slot/phase_is_A`` -- the share of THIS step's optimizer updates on which the
          A factor was free to move. Under the default "BBA" at three updates per step
          each step reads 0.0 or 1/3 and the RUN averages 2/9 (a third of the updates
          that are not the forced step end); a flat 0.0 across steps means the
          alternation is dead, and 1.0 is the joint ablation (nothing is ever frozen).

        THE KEY SET IS FIXED BY CONFIG, always, including the NaNs. ``all_reduce_dict``
        packs the metric dict into one tensor sized by the key count, so a rank that
        emits a key another rank does not deadlocks the collective -- a bug this repo
        has already hit once. Every key here is derived from ``len(self._slot_order)``,
        which comes from config and is identical on every rank; a reading that is
        missing is emitted as NaN rather than dropped (and NaN rather than 0.0, because
        a plausible-looking constant is exactly the failure these plots must not
        produce). ``slot/route_fallback_frac`` is deliberately NOT here: it is emitted
        per micro-batch elsewhere, and a per-step copy would average two different
        sample counts into one number.

        Returns:
            ``{metric: float}``, empty when slot-LoRI is off.
        """
        if not self._slot_enabled:
            return {}

        from rlinf.models.slot_lora import collect_slot_diag

        n_slots = len(self._slot_order)
        keys = ["slot/orth_err"]
        keys += [f"slot/dw_norm_{k}" for k in range(n_slots)]
        keys += [
            f"slot/cos_{s}_{t}" for s in range(n_slots) for t in range(s + 1, n_slots)
        ]
        collected = collect_slot_diag(self.model)
        unexpected = sorted(set(collected) - set(keys))
        if unexpected:
            # Dropped, not emitted: an extra key on one rank is a hang, not a metric.
            self._slot_warn_once(
                "diag_extra_keys",
                f"[slot-lora] collect_slot_diag returned {unexpected}, which "
                f"actor.model.slot_lora.slot_order ({n_slots} slots) does not predict. "
                "Those readings are DROPPED -- a metric key that depends on the model "
                "rather than on config deadlocks the metric all-reduce.",
            )
        if self._slot_alt_updates and self._slot_alt_updates != self._slot_alt_expected:
            self._slot_warn_once(
                "alt_update_count",
                f"[slot-lora] this step ran {self._slot_alt_updates} optimizer "
                f"update(s) but was opened expecting {self._slot_alt_expected}, so the "
                "'last update of the step is forced to B' override fired at the wrong "
                "update. A step that ends on A saves a checkpoint whose B does not "
                "match its Ā. The expected count and the loop bounds are computed in "
                "two places in run_training and have drifted apart.",
            )
        metrics = {key: float(collected.get(key, float("nan"))) for key in keys}
        metrics["slot/alt_a_frac_cfg"] = slot_alt_a_fraction(self._slot_alt_schedule)
        metrics["slot/phase_is_A"] = (
            self._slot_alt_a_updates / self._slot_alt_updates
            if self._slot_alt_updates
            else float("nan")
        )
        return metrics

    def _teacher_forward(self, forward_inputs, kwargs):
        """Score the student's rollout with the teacher(s).

        Single teacher (or no routing table) -> one forward, unchanged behaviour.
        TRUE multi-teacher -> split the micro-batch by the sample's SUITE (derived from its
        task instruction in ``forward_inputs['input_ids']``) and run each group through its
        own expert, then scatter the per-group outputs back into full-batch tensors. This is
        what makes "one student, N per-suite expert teachers" work: every sample is scored by
        the expert for ITS suite, never by another suite's expert.
        """
        # OPD_DUMP_STATES=<path>: save ONE micro-batch of the student's OWN on-policy rollout
        # states, then exit. Offline teacher-comparison studies otherwise have to use expert DEMO
        # states, which is the wrong distribution -- OPD's whole point is that the teacher scores
        # the states the STUDENT actually visits (and the measured teacher-student gap there was
        # far larger than on demo states). Env-gated, off by default, writes once.
        _dump = os.environ.get("OPD_DUMP_STATES", "")
        if _dump and not getattr(self, "_states_dumped", False):
            self._states_dumped = True
            try:
                torch.save(
                    {k: v.detach().cpu() for k, v in forward_inputs.items() if torch.is_tensor(v)},
                    _dump,
                )
                self.log_info(f"[VLA-OPD] dumped on-policy states -> {_dump}")
                print(f"OPD_STATES_DUMPED={_dump}", flush=True)
            except Exception as e:
                print(f"OPD_STATES_DUMP_FAILED {e}", flush=True)

        _route = getattr(self, "teacher_prompt_to_suite", None)
        _models = getattr(self, "teacher_models", None)
        if not _route or not _models or len(_models) <= 1:
            return self.teacher_model(
                forward_inputs=forward_inputs, compute_logprobs=True,
                use_cache=False, **kwargs,
            )

        ids = forward_inputs["input_ids"]
        bsz = ids.shape[0]
        # The grouping is computed in _route_prepare, BEFORE the student forward, so
        # that the slot gate and this split come from the same decode and the same
        # match (see that docstring). Recompute only when this is called standalone --
        # i.e. nothing prepared a routing for this micro-batch.
        if not getattr(self, "_route_ready", False):
            self._route_prepare(forward_inputs)
        groups = self._last_groups
        if not groups:
            # No routing available (no tokenizer): one teacher for the whole
            # micro-batch, exactly as before.
            return self.teacher_model(
                forward_inputs=forward_inputs, compute_logprobs=True,
                use_cache=False, **kwargs,
            )

        if len(groups) == 1:  # whole micro-batch is one suite -> single forward
            only_path = next(iter(groups))
            _m1 = _models[only_path]
            _ad1 = getattr(self, "teacher_adapter_of_path", None)
            if _ad1:  # shared base -> must still select THIS suite's adapter
                _pm1 = _m1 if hasattr(_m1, "set_adapter") else getattr(_m1, "model", None)
                _pm1.set_adapter(_ad1[only_path])
            return _m1(
                forward_inputs=forward_inputs, compute_logprobs=True,
                use_cache=False, **kwargs,
            )

        out: dict = {}
        _ad_of = getattr(self, "teacher_adapter_of_path", None)
        for path, idxs in groups.items():
            sel = torch.as_tensor(idxs, device=ids.device, dtype=torch.long)
            sub_inputs = {
                k: (v[sel] if torch.is_tensor(v) and v.shape[:1] == (bsz,) else v)
                for k, v in forward_inputs.items()
            }
            _m = _models[path]
            if _ad_of:  # shared base: all paths map to ONE model -> switch the adapter
                _pm = _m if hasattr(_m, "set_adapter") else getattr(_m, "model", None)
                _pm.set_adapter(_ad_of[path])
            sub_out = _m(
                forward_inputs=sub_inputs, compute_logprobs=True,
                use_cache=False, **kwargs,
            )
            for k, v in sub_out.items():
                if not torch.is_tensor(v) or v.shape[:1] != (len(idxs),):
                    continue
                if k not in out:
                    out[k] = v.new_zeros((bsz,) + tuple(v.shape[1:]))
                out[k][sel] = v
        if not hasattr(self, "_mt_logged"):
            self._mt_logged = True
            self.log_info(
                f"[VLA-OPD] multi-teacher forward: micro-batch split across "
                f"{len(groups)} expert(s)"
            )
        return out

    @torch.no_grad()
    def _cross_score(self, forward_inputs, kwargs, ls, loss_mask, sad):
        """DIAGNOSTIC ONLY (no grad, no effect on the loss): on each suite's own rollout states,
        measure forward-KL(expert_j || student) for EVERY expert j, not just that suite's own.

        Returns {"actor/xkl_<states_suite>_by_<expert_suite>": float}. The diagonal
        (states_suite == expert_suite) is the quantity training actually minimises; the
        off-diagonal is the one nobody optimises. Reading them together over training answers
        "are the teachers fighting" in behaviour space:
          diagonal down + off-diagonal UP   -> the student buys one teacher by selling another
          both down                          -> the experts are compatible, conflict is not the story
          off-diagonal flat                  -> the experts simply live in disjoint state regions

        Requires the shared-base setup (all experts = one base + different LoRA adapters), which is
        what makes this cheap: swapping an adapter costs nothing next to loading another 7B.
        """
        out: dict = {}
        groups = getattr(self, "_last_groups", None)
        ad_of = getattr(self, "teacher_adapter_of_path", None)
        if not groups or not ad_of or len(groups) < 2:
            return out
        suite_of_path = {v: k for k, v in getattr(self, "teacher_suite_to_path", {}).items()}
        models = getattr(self, "teacher_models", None) or {}

        for st_path, idxs in groups.items():
            st_name = str(suite_of_path.get(st_path, "unk")).replace("libero_", "")
            sel = torch.as_tensor(idxs, device=ls.device, dtype=torch.long)
            sub_inputs = {
                k: (v[sel] if torch.is_tensor(v) and v.shape[0] == ls.shape[0] else v)
                for k, v in forward_inputs.items()
            }
            ls_sub = ls[sel]
            if loss_mask is not None:
                m = (
                    loss_mask[sel]
                    .to(ls.dtype)
                    .unsqueeze(-1)
                    .expand(-1, -1, sad)
                    .reshape(ls_sub.shape[0], -1)
                )
            else:
                m = torch.ones(ls_sub.shape[:2], dtype=ls.dtype, device=ls.device)
            for ex_path, ex_ad in ad_of.items():
                ex_name = str(suite_of_path.get(ex_path, "unk")).replace("libero_", "")
                mdl = models.get(ex_path, None)
                if mdl is None:
                    continue
                pm = mdl if hasattr(mdl, "set_adapter") else getattr(mdl, "model", None)
                if pm is None:
                    continue
                pm.set_adapter(ex_ad)
                o = mdl(
                    forward_inputs=sub_inputs, compute_logprobs=True, use_cache=False, **kwargs
                )
                if "action_logits" not in o:
                    continue
                # Skip when this group has NO valid positions: dividing by clamp_min(1.0) would
                # emit a literal 0.0 that _probe_emit then reports as a MEASURED zero (n=1),
                # dragging the cross-rank average down with a value that means "nothing to
                # measure". Observed on 2026-08-17 as xkl_object_by_object=0.0 at n=0.25.
                denom = m.sum()
                if denom.item() <= 0:
                    continue
                lt_x = torch.log_softmax(o["action_logits"].float(), dim=-1)
                kl = (lt_x.exp() * (lt_x - ls_sub)).sum(dim=-1)  # forward KL, per action token
                out[f"actor/xkl_{st_name}_by_{ex_name}"] = ((kl * m).sum() / denom).item()
        return out

    # ---- probe metric plumbing -------------------------------------------------------------
    # all_reduce_dict (rlinf/utils/distributed.py) packs the metric dict into ONE tensor whose
    # length is the NUMBER OF KEYS, then all_reduces it. Every rank must therefore emit the
    # IDENTICAL key set, or the collective is called with mismatched sizes and NCCL hangs until
    # the 30-minute watchdog fires. That is exactly what killed the 2026-08-17 diagnostic run:
    # the cross-scoring probe only fires on micro-batches containing >=2 suites, which is
    # data-dependent and therefore rank-dependent, so rank 1 packed a different-length tensor
    # than ranks 0/2/3 and all four deadlocked.
    #
    # Fix, made structural rather than careful: the key list is computed ONCE from teacher_map
    # (identical on every rank) and EVERY key is emitted on EVERY rank, every step. A probe that
    # did not run contributes 0.0 plus a companion "<key>__n"=0.0. Since the reduction is AVG,
    # the true mean over the ranks that measured is reduced("<key>")/reduced("<key>__n") -- the
    # 1/world_size factor cancels between the two.
    def _probe_key_list(self):
        if getattr(self, "_probe_keys", None) is not None:
            return self._probe_keys
        suites = sorted(
            str(s).replace("libero_", "")
            for s in (getattr(self, "teacher_suite_to_path", None) or {})
        )
        if not suites:
            # called before the teacher map exists -> do NOT cache an empty list, or the probe
            # keys would be permanently missing (silently, which is how the last two bugs hid)
            return []
        keys: list[str] = []
        if self.cfg.algorithm.get("cross_score", False):
            keys += [f"actor/xkl_{a}_by_{b}" for a in suites for b in suites]
        if self.cfg.algorithm.get("signal_stats", False):
            keys += [
                "actor/stu_entropy",
                "actor/tea_entropy",
                "actor/topk5_overlap",
                "actor/tea_top1_rank_in_stu",
            ]
            keys += [
                f"actor/{p}_{q}"
                for p in ("frac", "klmass")
                for q in ("hiH_hiKL", "hiH_loKL", "loH_hiKL", "loH_loKL")
            ]
        if self.cfg.algorithm.get("grad_conflict", False):
            keys += [f"actor/gnorm_{s}" for s in suites]
            keys += [
                f"actor/gcos_{suites[i]}_{suites[j]}"
                for i in range(len(suites))
                for j in range(i + 1, len(suites))
            ]
        self._probe_keys = sorted(keys)
        return self._probe_keys

    def _probe_emit(self):
        """Fixed-shape probe metrics for THIS rank. Always the same keys, on every rank."""
        measured: dict = {}
        for d in (
            getattr(self, "_last_xkl", None),
            getattr(self, "_last_sig", None),
            getattr(self, "_last_gconf", None),
        ):
            if d:
                measured.update(d)
        out: dict = {}
        for k in self._probe_key_list():
            v = measured.get(k, None)
            out[k] = float(v) if v is not None else 0.0
            out[f"{k}__n"] = 1.0 if v is not None else 0.0
        return out

    @torch.no_grad()
    def _signal_stats(self, ls, lt, kl_tok, mtok):
        """DIAGNOSTIC ONLY: is this teacher's signal even ABSORBABLE, and where does it live?

        (a) ABSORBABILITY. If the teacher's preferred action bin sits deep in the student's tail,
            the student cannot move there in reasonable steps and the whole distillation target is
            out of reach -- that would make every reweighting scheme moot, so it must be checked
            BEFORE tuning any of them.
              topk5_overlap        share of the teacher's top-5 bins that are also in the student's
              tea_top1_rank_in_stu rank of the teacher's argmax under the student (0 = same choice;
                                   large = the teacher is pointing somewhere the student ignores)

        (b) WHERE THE SIGNAL IS. Split positions by student entropy and by KL, and report both the
            share of POSITIONS and the share of total KL MASS in each quadrant. The mass share is
            the one that matters: it says which region actually drives the gradient.
              loH_hiKL = "confidently wrong" -- student is sure and disagrees with the teacher
              hiH_*    = "unsure"            -- student has no opinion yet
            Splits are at the batch median, so no threshold needs tuning.

        No extra forward pass: ls/lt/kl_tok are already computed for the loss.
        """
        out: dict = {}
        m = mtok > 0
        if m.sum() < 8:
            return out
        # ls/lt are [B, tokens, V]; kl_tok/mtok are [B, tokens]
        ps, pt = ls.exp(), lt.exp()
        H_s = -(ps * ls).sum(-1)
        H_t = -(pt * lt).sum(-1)
        out["actor/stu_entropy"] = H_s[m].mean().item()
        out["actor/tea_entropy"] = H_t[m].mean().item()

        k = 5
        t_top = lt.topk(k, dim=-1).indices
        s_top = ls.topk(k, dim=-1).indices
        inboth = (t_top.unsqueeze(-1) == s_top.unsqueeze(-2)).any(-1).float().mean(-1)
        out[f"actor/topk{k}_overlap"] = inboth[m].mean().item()

        t_arg = lt.argmax(-1, keepdim=True)
        # rank of the teacher's argmax under the student = #bins the student prefers over it
        rank = (ls > ls.gather(-1, t_arg)).sum(-1).float()
        out["actor/tea_top1_rank_in_stu"] = rank[m].mean().item()

        h, kv = H_s[m], kl_tok[m]
        hm, km = h.median(), kv.median()
        tot = kv.sum().clamp_min(1e-12)
        n = float(h.numel())
        for tag, sel in (
            ("hiH_hiKL", (h > hm) & (kv > km)),
            ("hiH_loKL", (h > hm) & (kv <= km)),
            ("loH_hiKL", (h <= hm) & (kv > km)),
            ("loH_loKL", (h <= hm) & (kv <= km)),
        ):
            out[f"actor/frac_{tag}"] = (sel.sum().item() / n) if n > 0 else 0.0
            out[f"actor/klmass_{tag}"] = (kv[sel].sum() / tot).item()
        return out

    def _grad_conflict(self, kl_tok, mtok):
        """DIAGNOSTIC ONLY: per-suite gradients of the distill loss, then pairwise cosine + norms.

        Answers two DIFFERENT questions that "the teachers fight" conflates:
          cos < 0            -> genuine directional conflict; gradient surgery (PCGrad) is justified
          cos ~ 0            -> the experts are simply orthogonal; conflict is NOT the mechanism
          |g_i| >> |g_j|     -> not conflict but DOMINANCE; a per-suite weight fixes it, and no
                                amount of gradient surgery would
        Cosine is a local first-order quantity and does NOT by itself explain the final SR gap --
        it is used here to RULE OUT mechanisms, not to prove one.

        Cost: one extra backward per suite on the probe micro-batch (retain_graph). Intended for
        the small single-GPU debug run: at world_size=1 FSDP does no gradient sharding, so the
        numbers are exact without any cross-rank reduction. On a sharded multi-GPU run these are
        LOCAL-SHARD cosines and would need an all_reduce of the dot products to be meaningful.
        """
        out: dict = {}
        groups = getattr(self, "_last_groups", None)
        if not groups or len(groups) < 2:
            return out
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return out
        suite_of_path = {v: k for k, v in getattr(self, "teacher_suite_to_path", {}).items()}
        grads: dict = {}
        for path, idxs in groups.items():
            name = str(suite_of_path.get(path, "unk")).replace("libero_", "")
            rows = torch.zeros(kl_tok.shape[0], device=kl_tok.device, dtype=kl_tok.dtype)
            rows[torch.as_tensor(idxs, device=kl_tok.device, dtype=torch.long)] = 1.0
            m_s = mtok * rows.unsqueeze(-1)
            den = m_s.sum()
            if den.item() <= 0:
                continue
            loss_s = (kl_tok * m_s).sum() / den
            g = torch.autograd.grad(
                loss_s, params, retain_graph=True, allow_unused=True
            )
            # keep per-parameter (no torch.cat) -- concatenating would add a full extra copy
            grads[name] = [None if gi is None else gi.detach().float() for gi in g]

        names = sorted(grads)
        norms = {}
        for n in names:
            sq = sum(float((gi * gi).sum()) for gi in grads[n] if gi is not None)
            norms[n] = sq**0.5
            out[f"actor/gnorm_{n}"] = norms[n]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = grads[names[i]], grads[names[j]]
                dot = sum(
                    float((x * y).sum())
                    for x, y in zip(a, b)
                    if x is not None and y is not None
                )
                den = norms[names[i]] * norms[names[j]]
                out[f"actor/gcos_{names[i]}_{names[j]}"] = dot / den if den > 0 else 0.0
        del grads
        return out

    def _load_base_model(self) -> None:
        """Dual-KL anchor: frozen BASE (= student's init / the generalist) loaded full
        (non-LoRA), eval + requires_grad_(False). Only SCORES anchor states; never acts.
        Defaults to the student's own model_path when actor.base_model_path is unset."""
        from copy import deepcopy

        from omegaconf import open_dict

        bcfg = deepcopy(self.cfg.actor.model)
        with open_dict(bcfg):
            bcfg.model_path = self.cfg.actor.get(
                "base_model_path", self.cfg.actor.model.model_path
            )
            bcfg.is_lora = False
            bcfg.lora_path = None
            _buk = self.cfg.actor.get("base_unnorm_key", None)
            if _buk:
                bcfg.unnorm_key = _buk
        self.base_model = get_model(bcfg)
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.log_info(f"[dual-KL] loaded frozen BASE anchor from {bcfg.model_path}")

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def get_rollout_state_dict(self) -> dict:
        return self.get_model_state_dict(cpu_offload=False, full_state_dict=False)

    async def sync_model_to_rollout(self) -> None:
        if not self._weight_dst_rank_in_rollout:
            self.log_debug(
                f"Actor rank {self._rank} has no rollout weight-sync destination."
            )
            if self.enable_offload:
                if not self.is_optimizer_offloaded:
                    self.offload_optimizer()
                if not self.is_weight_offloaded:
                    self.offload_param_and_grad(True)
            return

        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        state_dict = self.get_rollout_state_dict()

        async def send_func(data):
            handle = []
            for rank in self._weight_dst_rank_in_rollout:
                handle.append(
                    self.send(
                        data,
                        dst_group_name=self._rollout_group_name,
                        dst_rank=rank,
                        async_op=True,
                        options=self._sync_weight_comm_options,
                    ).async_wait()
                )
            await asyncio.gather(*handle)

        async def recv_func():
            handle = []
            for rank in self._weight_dst_rank_in_rollout:
                handle.append(
                    self.recv(
                        src_group_name=self._rollout_group_name,
                        src_rank=rank,
                        async_op=True,
                        options=self._sync_weight_comm_options,
                    ).async_wait()
                )
            metadata_list = await asyncio.gather(*handle)
            metadata = metadata_list[0]
            for other_metadata in metadata_list[1:]:
                if other_metadata != metadata:
                    raise ValueError("Patch metadata differs across rollout ranks")
            return metadata

        if not self.weight_syncer.sender_initialized():
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
            )

        await self.weight_syncer.sync(state_dict, send_func, version=self.version)

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad(True)

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.rollout_batch = convert_trajectories_to_batch(recv_list)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # FAILURE-ONLY distillation support (algorithm.distill_on_failure).
        # "traj_fail" = 1 where the RETURN-TO-GO of this trajectory is zero, i.e. from this state
        # onward the rollout never succeeded again. Computed HERE because this is the only place
        # the data still carries its trajectory structure -- the shape is documented above as
        # [n_chunk_step, rollout_epoch x bsz, num_action_chunks], so a trajectory is a fixed index
        # in dim 1 running along dim 0. Doing it later (in the OPD loss, on a micro-batch) is what
        # broke the first attempt: there each row is a single 8-action chunk, LIBERO's reward is
        # ~1 only at the success instant, so "this row earned no reward" was true for 98% of rows
        # regardless of whether its episode succeeded -- it just deleted the 2% of chunks that
        # carried the success, the exact opposite of the intent (measured fail_frac=0.977 against
        # success_once=0.736).
        # env.train has auto_reset=False and ignore_terminations=False, so one trajectory slot
        # holds exactly ONE episode and a plain reverse cumsum needs no per-done segment reset.
        # Arm the cross-scoring probe ONCE per step. _process_received_rollout_batch runs exactly
        # once per training step, whereas the loss body runs once per micro-batch (dozens of times
        # a step) -- gating on "first micro-batch" there would fire once per global batch, not once
        # per step. A latch set here and cleared by the first micro-batch is the cheap correct gate.
        if self.cfg.algorithm.get("cross_score", False):
            self._xscore_armed = True
        if self.cfg.algorithm.get("grad_conflict", False):
            self._gconf_armed = True
        if self.cfg.algorithm.get("signal_stats", False):
            self._sig_armed = True

        if (
            self.cfg.algorithm.get("distill_on_failure", False)
            or float(self.cfg.algorithm.get("distill_fail_alpha", 0.0)) > 0.0
        ):
            with torch.no_grad():
                rw = rollout_batch["rewards"]  # [T, B, C]
                T, B, C = rw.shape
                # -> [B, T*C] laid out in trajectory time order, reverse-cumsum, back to [T, B, C]
                r = rw.transpose(0, 1).reshape(B, T * C)
                rtg = r.flip(-1).cumsum(-1).flip(-1)
                fail = (rtg <= 0).to(rw.dtype)
                rollout_batch["traj_fail"] = (
                    fail.reshape(B, T, C).transpose(0, 1).contiguous()
                )

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        if self.cfg.algorithm.adv_type == "opd":
            # VLA-OPD: the advantage is the per-token reverse-KL reward
            # r_t = log pi_teacher(a_t) - log pi_student(a_t), computed PER MICRO-BATCH inside
            # run_training (the teacher forward needs the flattened/processed batch + matching
            # forward_inputs). Here we only place a same-shape placeholder so
            # process_nested_dict_for_train and the training loop find an "advantages" tensor.
            self.rollout_batch["advantages"] = torch.zeros_like(
                self.rollout_batch["prev_logprobs"]
            )
            return compute_rollout_metrics(self.rollout_batch)

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "opd_center": self.cfg.algorithm.get("opd_center", False),
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    def _build_sft_data_loader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            # NOTE: This must be set before importing openpi.training.data_loader
            if self.cfg.actor.get("sft_data_path", None):
                os.environ["HF_LEROBOT_HOME"] = self.cfg.actor.sft_data_path

            import openpi.training.data_loader as _data

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            if "config_name" not in self.cfg.actor:
                raise ValueError(
                    "config_name is required when enable_sft_co_train=True"
                )
            training_config_name = self.cfg.actor.config_name
            data_loader_config = get_openpi_config(
                training_config_name,
                model_path=self.cfg.actor.model.model_path,
                data_kwargs=getattr(self.cfg.actor, "openpi_data", None),
            )
            self.data_loader = _data.create_data_loader(
                data_loader_config, framework="pytorch", shuffle=True
            )
            self.sft_iterator = iter(self.data_loader)
            self.train_epoch = 0
            self.sft_loss_weight = self.cfg.actor.get("sft_loss_weight", 0.1)
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _train_sft_epoch(
        self, metrics_data: dict[str, torch.Tensor], loss: torch.Tensor
    ):
        """
        Train one epoch of SFT.
        """
        metrics_data["ppo_loss"] = loss.clone().detach().item()

        # Get next data batch
        try:
            observation, actions = next(self.sft_iterator)
        except StopIteration:
            self.train_epoch += 1
            self.data_loader.set_epoch(self.train_epoch)
            self.sft_iterator = iter(self.data_loader)
            observation, actions = next(self.sft_iterator)

        register_pytree_dataclasses(observation)
        observation = _pytree.tree_map(
            lambda x: x.to(self.device) if x is not None else x,
            observation,
        )
        actions = actions.to(torch.float32)
        actions = actions.to(self.device)

        sft_losses = self.model(
            data={"observation": observation, "actions": actions},
            forward_type=ForwardType.SFT,
        )
        # Ensure losses is a tensor and handle different return types
        if isinstance(sft_losses, list | tuple):
            sft_losses = torch.stack(sft_losses)
        elif not isinstance(sft_losses, torch.Tensor):
            sft_losses = torch.tensor(
                sft_losses, device=self.device, dtype=torch.float32
            )

        sft_loss = sft_losses.mean()
        metrics_data["sft_loss"] = sft_loss.clone().detach().item()
        total_loss = loss + self.sft_loss_weight * sft_loss
        loss = total_loss

        metrics_data["loss_ratio"] = (
            np.abs(metrics_data["sft_loss"]) / np.abs(metrics_data["ppo_loss"])
            if np.abs(metrics_data["ppo_loss"]) > 0
            else float("inf")
        )
        if metrics_data["loss_ratio"] > 1e5:
            self.logger.warning(
                "SFT/PPO loss imbalance detected: "
                f"ratio={metrics_data['loss_ratio']:.3e}, "
                f"sft_loss={metrics_data['sft_loss']:.6f}, "
                f"ppo_loss={metrics_data['ppo_loss']:.6f}, "
                f"sft_loss_weight={self.sft_loss_weight:.6f}"
            )

    def _dw_refresh_weights(self):
        """Recompute per-teacher distillation weights ONCE per training step.

        Every sync-forcing op (`.tolist()`, boolean-mask indexing) lives here and nowhere
        else. Weights lag the KL they came from by one step, which is harmless -- an EMA
        already smooths over steps -- and buys back the 5.6x that per-micro-batch syncs
        cost (80.0 min vs a 14.3 min baseline, measured on an identical config).

        Step 1 runs with uniform weights because no KL has been accumulated yet; the
        weighting becomes active from step 2 on.
        """
        _dynw = float(self.cfg.algorithm.get("distill_dyn_weight", 0.0))
        if _dynw <= 0.0 or getattr(self, "_dw_n", None) is None:
            return
        with torch.no_grad():
            _seen = self._dw_d > 0
            _cur = self._dw_n / self._dw_d.clamp_min(1.0)
            _beta = float(self.cfg.algorithm.get("distill_w_ema", 0.9))
            self._dw_ema = torch.where(
                _seen, _beta * self._dw_ema + (1.0 - _beta) * _cur, self._dw_ema
            )
            # mean over suites that have been seen at least once, without boolean indexing
            _posf = (self._dw_ema > 0).to(self._dw_ema.dtype)
            _m = (self._dw_ema * _posf).sum() / _posf.sum().clamp_min(1.0)
            _w = (self._dw_ema / _m.clamp_min(1e-8)).clamp_min(1e-8).pow(_dynw)
            _w = (_w * _posf + (1.0 - _posf)).clamp(
                float(self.cfg.algorithm.get("distill_w_min", 0.25)),
                float(self.cfg.algorithm.get("distill_w_max", 4.0)),
            )
            self._dw_w = _w / _w.mean().clamp_min(1e-8)
            # plain floats for the metric path, so IT never syncs either
            self._dw_w_list = self._dw_w.tolist()
            self._dw_kl_list = self._dw_ema.tolist()
            self._dw_n.zero_()
            self._dw_d.zero_()

    def make_phase_profiler(self) -> TrainPhaseProfiler:
        """Build this step's phase profiler from config. OFF unless asked for.

        Both knobs are read the way every other diagnostic in this method is read --
        ``self.cfg.algorithm.get(...)`` with a default, next to ``distill_dyn_weight``,
        ``cross_score``, ``signal_stats`` and ``grad_conflict`` -- so a config that names
        neither key (which is every config in the repo, including the one the production
        run is using) gets a profiler that does nothing at all.

        ONE boolean, not two. Splitting "time the phases" from "print progress" would
        create two useless combinations: progress lines without the breakdown answer
        "is it alive" but not "where is the time going", and the breakdown without the
        progress lines only prints once the step is over -- three hours late, which is
        the exact silence this exists to remove. The interval is a separate NUMBER
        because it is a volume knob, not a feature:

          algorithm.profile_train_phases  bool, default False -- the gate.
          algorithm.profile_log_every     int, default 0 -- micro-batches between
                                          progress lines. 0 means derive it from the
                                          real loop bound, ~10 lines per optimizer
                                          update. Set it huge to keep only the
                                          per-update and per-step summaries.

        DIAGNOSTIC ONLY. The flush synchronizes on CUDA events; it is free only because
        it sits right after a ``.item()`` that already drained the stream (see the
        TrainPhaseProfiler header). Leave it off for production runs.
        """
        enabled = bool(self.cfg.algorithm.get("profile_train_phases", False))
        # WHY stderr AND log_info. self.log_info writes into ray's per-worker log files
        # under the session tmpdir, which run_iso.sh places in /tmp/rayiso_* and which is
        # deleted when the session ends. Measured: [VLA-OPD] lines (log_info) appear ZERO
        # times in any driver log, including mt4's own successful run, while [slot-lora]
        # lines (sys.stderr.write) all survive. A profile that lands only in a directory
        # about to be removed is a profile that does not exist -- which is exactly what
        # happened on the first profiling run: the flag was on, three steps ran, and not
        # one [prof] line reached the log.
        def _emit(_msg: str) -> None:
            import sys as _sys

            _sys.stderr.write(_msg + "\n")
            self.log_info(_msg)

        def _emit_warn(_msg: str) -> None:
            import sys as _sys

            _sys.stderr.write(_msg + "\n")
            self.log_warning(_msg)

        return TrainPhaseProfiler(
            enabled=enabled,
            log_every=int(self.cfg.algorithm.get("profile_log_every", 0)),
            log_fn=_emit,
            warn_fn=_emit_warn,
            rank=getattr(self, "_rank", 0),
            step=getattr(self, "version", None),
        )

    @Worker.timer("run_training")
    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)
        # ---- DanceOPD-transfer sampling (algorithm.dance_updates) ------------------------
        # Replaces the uniform shuffle with suite-blocked, trajectory-thinned updates:
        # every optimizer update is SINGLE-suite (same suite on every rank -- the rotation
        # is a function of the update index alone), and only dance_keep_frac of the samples
        # survive thinning. See dance_build_order for the two mechanisms and their source.
        self._dance_update_suites = None
        if bool(self.cfg.algorithm.get("dance_updates", False)):
            route = getattr(self, "teacher_prompt_to_suite", None)
            if not route:
                raise RuntimeError(
                    "algorithm.dance_updates=true needs the prompt->suite table "
                    "(actor.teacher_map); without it no sample can be assigned a suite "
                    "and the 'single-suite update' contract cannot be honored."
                )
            from rlinf.models.slot_lora import match_suite_ids

            proc = getattr(self.teacher_model, "input_processor", None)
            tok = getattr(proc, "tokenizer", None) if proc is not None else None
            if tok is None:
                raise RuntimeError(
                    "dance_updates: no tokenizer reachable via the teacher's "
                    "input_processor; cannot decode prompts to suites."
                )
            # nested under forward_inputs -- the same sub-dict the micro-batch loss
            # reads its prompts from, so the two decodes can never disagree.
            _fi = self.rollout_batch.get("forward_inputs", None)
            if _fi is None or "input_ids" not in _fi:
                raise RuntimeError(
                    "dance_updates: rollout_batch carries no "
                    "forward_inputs['input_ids']; cannot decode prompts to suites."
                )
            flat_ids = _fi["input_ids"].reshape(
                rollout_size, *_fi["input_ids"].shape[2:]
            )
            match_order = self._route_match_order()
            texts = tok.batch_decode(flat_ids, skip_special_tokens=True)
            suite_ids = match_suite_ids(texts, route, match_order)
            bspr = self.cfg.actor.global_batch_size // self._world_size
            _kf = float(self.cfg.algorithm.get("dance_keep_frac", 0.25))
            # THE ROTATION IS A GLOBAL DECISION. Each rank holds different envs, so its
            # per-suite pools differ; the first smoke run showed rank0 rotating goal
            # where rank1 rotated object -- gradients re-mixed across ranks -- and a
            # locally-computed rotation COUNT can differ too, which desyncs the number
            # of optimizer updates and deadlocks FSDP. So: all-reduce the per-suite
            # KEPT counts (MIN and SUM), rotate only suites every rank can fill, and
            # size the rotation from the all-reduced totals. Deterministic given the
            # data -- no extra seed coordination needed.
            import torch.distributed as _dist

            _ids_t = torch.as_tensor(suite_ids, dtype=torch.long)
            _counts = torch.stack(
                [
                    torch.clamp(
                        (torch.round((_ids_t == k).sum() * _kf)).long(),
                        min=0 if (_ids_t == k).sum() == 0 else 1,
                    )
                    for k in range(len(match_order))
                ]
            )
            _min_c = _counts.clone()
            _sum_c = _counts.clone()
            if _dist.is_available() and _dist.is_initialized():
                # NCCL: the reduce must ride a CUDA tensor ("No backend type
                # associated with device type cpu" otherwise -- smoke4, 22:52).
                _dev_c = torch.device("cuda", torch.cuda.current_device())
                _min_g = _min_c.to(_dev_c)
                _sum_g = _sum_c.to(_dev_c)
                _dist.all_reduce(_min_g, op=_dist.ReduceOp.MIN)
                _dist.all_reduce(_sum_g, op=_dist.ReduceOp.SUM)
                _min_c = _min_g.cpu()
                _sum_c = _sum_g.cpu()
            _rot_suites = [k for k in range(len(match_order)) if _min_c[k].item() > 0]
            if not _rot_suites:
                raise RuntimeError(
                    "dance_updates: no suite has samples on EVERY rank this step; "
                    "cannot build a rank-consistent rotation. With this few envs the "
                    "batch composition is degenerate -- raise total_num_envs."
                )
            _rots = max(
                1,
                round(
                    _sum_c[_rot_suites].sum().item()
                    / (bspr * max(1, getattr(self, "_world_size", 1)) * len(_rot_suites))
                ),
            )
            shuffle_id, self._dance_update_suites = dance_build_order(
                suite_ids,
                bspr,
                _kf,
                len(match_order),
                self.cfg.actor.seed + self._rank,
                suites_in_rotation=_rot_suites,
                rotations=_rots,
            )
            import sys as _sys

            _msg = (
                f"[dance] step {getattr(self, 'version', '?')}: kept "
                f"{shuffle_id.numel()}/{rollout_size} samples, "
                f"{len(self._dance_update_suites)} single-suite updates, rotation "
                f"{[match_order[k] for k in self._dance_update_suites[: len(match_order)]]}"
            )
            _sys.stderr.write(_msg + "\n")
            # ---- Memory Anchors (algorithm.anchor_frac, ANCHORER arXiv:2608.26545) ----
            # Fill the tail anchor_frac of every single-suite update with the other-suite
            # samples most similar to that update's observations (the confusion region),
            # weighted by which suite currently drifts furthest from its teacher (the
            # per-suite KL EMA maintained in the loss path). Each anchor distills toward
            # its OWN routed teacher downstream -- a targeted rehearsal term inside the
            # OPD update, not a new loss.
            _af = float(self.cfg.algorithm.get("anchor_frac", 0.0))
            if _af > 0.0:
                _pv = _fi.get("pixel_values", None)
                if _pv is not None:
                    _pv = _pv.reshape(rollout_size, *_pv.shape[2:])
                _emb = anchor_obs_embed(
                    _pv,
                    texts,
                    img_weight=float(self.cfg.algorithm.get("anchor_img_weight", 0.5)),
                )
                _beta = float(self.cfg.algorithm.get("anchor_beta", 0.5))
                _ema = getattr(self, "_anchor_kl_ema", {}) or {}
                _klv = torch.tensor(
                    [float(_ema.get(s, 1.0)) for s in match_order], dtype=torch.float32
                )
                _sw = (
                    (_klv / _klv.mean().clamp_min(1e-8)).pow(_beta).clamp(0.5, 2.0)
                )
                shuffle_id, _ast = dance_anchor_augment(
                    shuffle_id,
                    self._dance_update_suites,
                    _ids_t,
                    bspr,
                    _af,
                    _emb,
                    suite_w=_sw,
                )
                _sys.stderr.write(
                    f"[anchor] step {getattr(self, 'version', '?')}: frac={_af} "
                    f"mean_sim={_ast['mean_sim']:.3f} per-suite "
                    f"{ {match_order[s]: c for s, c in sorted(_ast['anchor_counts'].items())} } "
                    f"suite_w={ {match_order[i]: round(float(_sw[i]), 3) for i in range(len(match_order))} }\n"
                )

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        # Refresh the per-teacher distillation weights ONCE per training step, from the KL
        # this rank accumulated during the PREVIOUS step. This call MUST live in THIS method:
        # the class defines run_training twice and the later definition shadows the earlier,
        # so run_training_pipeline is dead code for the OpenVLA-OFT path -- putting the call
        # there left the weights pinned at their all-ones init for a whole 8-step run while
        # the metrics' `or` fallback made it look like a healthy uniform start (2026-08-20).
        self._dw_refresh_weights()
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        # slot-LoRI: open the step. Snapshots the lr the scheduler set for THIS step
        # (the alternation zeroes and restores it), resets the phase counters and arms
        # the once-per-step orthogonality diagnostics on one layer -- BEFORE the first
        # forward, which is what fills them. The update count is computed here and not
        # inside, because "the LAST update of this step is forced to B" has to be known
        # at the FIRST one.
        updates_per_step = update_epoch * (rollout_size // batch_size_per_rank)
        self._slot_step_begin(updates_per_step)
        # Per-phase timing + in-step progress logging. Inert unless
        # algorithm.profile_train_phases is set; see make_phase_profiler.
        prof = self.make_phase_profiler()
        prof.begin_step(updates_per_step)
        update_index = 0
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                prof.begin_update(len(train_micro_batch))
                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    batch = put_tensor_device(
                        batch,
                        f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}",
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )
                    advantages = batch["advantages"]
                    prev_logprobs = batch["prev_logprobs"]
                    returns = batch.get("returns", None)
                    prev_values = batch.get("prev_values", None)
                    loss_mask = batch.get("loss_mask", None)
                    loss_mask_sum = batch.get("loss_mask_sum", None)

                    forward_inputs = batch.get("forward_inputs", None)
                    opd_kl = None   # KL(student||teacher) diagnostic, filled in the opd block
                    opd_gap = None  # mean(teacher_lp - student_rollout_lp) on executed actions
                    opd_distill_loss = None  # differentiable KL-distill loss (opd_mode=distill)
                    anchor_loss = None  # dual-KL BASE anchor (forward-KL to frozen base)
                    visual_loss = None  # visual-representation anchor (cosine to base mid-layer)

                    kwargs = {}
                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        kwargs["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        kwargs["top_k"] = self.cfg.algorithm.sampling_params.top_k
                        # request 256-bin action logits for the OPD student-teacher KL diagnostic
                        kwargs["return_action_logits"] = (
                            self.cfg.algorithm.adv_type == "opd"
                        )
                        # visual-representation anchor: request mid-layer vision+prompt features
                        if float(self.cfg.algorithm.get("visual_anchor_lambda", 0.0)) > 0.0:
                            kwargs["return_mid_features"] = True
                            kwargs["mid_layer"] = int(
                                self.cfg.algorithm.get("visual_anchor_layer", 16)
                            )
                    elif (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        kwargs["prev_logprobs"] = prev_logprobs

                    compute_values = (
                        True if self.cfg.algorithm.adv_type == "gae" else False
                    )

                    # ---- slot-LoRI: route BEFORE the student forward --------------
                    # _teacher_forward runs AFTER this forward and is what used to
                    # compute the prompt->suite grouping, so a gate built from
                    # self._last_groups here would carry the PREVIOUS micro-batch's
                    # routing -- every sample's slot shifted by one micro-batch, with
                    # no error and a normal-looking loss curve. _route_prepare derives
                    # the slot ids AND that grouping from one decode and one match.
                    # Cleared UNCONDITIONALLY, before the call that may not happen:
                    # leaving the previous micro-batch's ids in place is the exact
                    # stale-routing failure this method exists to prevent, and it is
                    # invisible (same shape, plausible values). With them cleared, a
                    # micro-batch that reaches the forward unprepared raises instead.
                    self._route_ready = False
                    self._slot_gate_ids = None
                    self._slot_fallback = 0.0
                    if forward_inputs is not None and (
                        self._slot_enabled or self.cfg.algorithm.adv_type == "opd"
                    ):
                        self._route_prepare(forward_inputs)
                    # The scope must reach the BACKWARD too: gradient checkpointing
                    # re-runs the wrapped forward during backward(), on the autograd
                    # engine's worker thread, and a scope closed after the forward
                    # leaves that recomputation ungated. nullcontext when slot-LoRI is
                    # off, so that path is unchanged. NOTE: an enable_sft_co_train run
                    # would put _train_sft_epoch's forward (a DIFFERENT batch size)
                    # inside this scope; SlotOut refuses a routing that does not match
                    # its batch, loudly, which is the right outcome -- that combination
                    # has no defined per-sample routing.
                    with self._slot_scope():
                        prof.mark("prep")
                        with self.amp_context:
                            output_dict = self.model(
                                forward_inputs=forward_inputs,
                                compute_logprobs=True,
                                compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                                compute_values=compute_values,
                                use_cache=False,
                                **kwargs,
                            )
                        prof.mark("student_fwd")

                        if (
                            SupportedModel(self.cfg.actor.model.model_type)
                            == SupportedModel.GR00T
                        ):
                            prev_logprobs = output_dict["prev_logprobs"]

                        if self.cfg.algorithm.adv_type == "opd":
                            # VLA-OPD: frozen teacher scores the SAME actions the student executed
                            # (forward_inputs holds the rollout action tokens); the reverse-KL
                            # log-ratio is the advantage (detached -> constant reward).
                            # Sequential CL uses ONE generalist teacher for all suites, so
                            # self.teacher_model is used directly here. For TRUE multi-teacher
                            # (per-suite experts) route with self.teacher_suite_to_path using
                            # the sample's suite -- the rest of this block is unchanged.
                            with torch.no_grad(), self.amp_context:
                                teacher_out = self._teacher_forward(
                                    forward_inputs, kwargs
                                )
                            prof.mark("teacher_fwd")
                            t_lp = teacher_out["logprobs"].detach()
                            # per-token reverse-KL -> aggregate to per-chunk (num_action_chunks) so it
                            # matches RLinf's action-granularity advantages. logprobs are
                            # [B, num_action_chunks * action_dim]; the loss preprocessing reduces
                            # logprobs over action_dim, and expects advantages already at [B, num_action_chunks].
                            rkl = (t_lp - prev_logprobs).detach()  # [B, chunks*action_dim]
                            sad = self.cfg.actor.model.get("action_dim", 7)
                            # MEAN over the action_dim (not sum) -> per-chunk RKL, avoids one outlier
                            # token dominating the whole chunk's advantage.
                            adv = rkl.reshape(rkl.shape[0], -1, sad).mean(dim=-1)  # [B, chunks]
                            # STANDARDIZE the advantage (masked) — the EmbodiedFSDPActor path has no
                            # normalize_advantages, so raw RKL gave grad_norm 200-670 -> divergence.
                            # Bring it to ~O(1) so PPO updates are stable.
                            if self.cfg.algorithm.get("normalize_advantages", False):
                                m = (loss_mask.to(adv.dtype) if loss_mask is not None
                                     else torch.ones_like(adv))
                                cnt = m.sum().clamp_min(1.0)
                                mean = (adv * m).sum() / cnt
                                var = (((adv - mean) ** 2) * m).sum() / cnt
                                adv = ((adv - mean) / (var.sqrt() + 1e-6)) * m
                            advantages = adv
                            opd_gap = rkl.mean().item()  # mean(log pi_tea - log pi_stu) on executed actions
                            # DIFFERENTIABLE on-policy distillation (pi0 op_distill analog): directly
                            # minimize KL(student||teacher) over the 256 action bins on the student's
                            # rollout states. Teacher detached (frozen); student side carries gradient.
                            # This realizes OPD's reverse-KL objective as a DIFFERENTIABLE loss (GKD-style),
                            # NOT the REINFORCE/advantage route (which wouldn't converge here).
                            if "action_logits" in output_dict and "action_logits" in teacher_out:
                                ls = torch.log_softmax(output_dict["action_logits"].float(), dim=-1)
                                lt = torch.log_softmax(
                                    teacher_out["action_logits"].float().detach(), dim=-1
                                )
                                # Base-Centered Policy-Shift MOPD (BCF-MOPD): distill toward
                                #   q ∝ π_0 · exp((log π_tea − log π_0)/β)  i.e.
                                #   log q = log_softmax( lb + (lt − lb)/β )
                                # instead of the full teacher lt. Only the teacher's SHIFT relative
                                # to base is transferred, base-anchored: β>1 → q between base and
                                # expert (preserves base generalization); β=1 → q=teacher (vanilla).
                                # ONE coherent target -> no dual-KL anchor conflict.
                                _sbeta = float(self.cfg.algorithm.get("shift_beta", 0.0))
                                if _sbeta > 0.0 and getattr(self, "base_model", None) is not None:
                                    with torch.no_grad(), self.amp_context:
                                        _shift_base_out = self.base_model(
                                            forward_inputs=forward_inputs,
                                            compute_logprobs=True,
                                            use_cache=False,
                                            **kwargs,
                                        )
                                    if "action_logits" in _shift_base_out:
                                        _lb_s = torch.log_softmax(
                                            _shift_base_out["action_logits"].float().detach(),
                                            dim=-1,
                                        )
                                        lt = torch.log_softmax(
                                            _lb_s + (lt - _lb_s) / _sbeta, dim=-1
                                        )
                                # diagnostic: reverse KL(student||teacher) (comparable across runs)
                                opd_kl = (ls.exp() * (ls - lt)).sum(dim=-1).detach().mean().item()
                                # LOSS direction (distill_kl):
                                #   forward  = KL(teacher||student), teacher-weighted, MODE-COVERING
                                #     (student covers teacher's good actions without deleting its own ->
                                #      FORGETS LESS; matches pi0 velocity-MSE that worked).
                                #   reverse  = KL(student||teacher), MODE-SEEKING (zero-forces student
                                #      onto teacher's OOD flatness -> catastrophic forgetting; what failed).
                                _dkl = self.cfg.algorithm.get("distill_kl", "forward")
                                if _dkl == "reverse":
                                    kl_tok = (ls.exp() * (ls - lt)).sum(dim=-1)
                                elif _dkl == "jsd":
                                    # Generalized JSD (GKD, arXiv:2306.13649): beta->0 = forward
                                    # (mode-covering), beta->1 = reverse; bounded by log2 so NO off-support
                                    # blow-up, and closed-form so NO dropped state-visitation bias / no
                                    # REINFORCE variance. NOTE: GKD's own finding is that for WEAK students
                                    # mode-SEEKING (large beta / reverse) wins for CAPABILITY TRANSFER; we
                                    # default the other way (small beta / forward) on purpose because our
                                    # objective is PRESERVATION, where mode-covering forgets less. The
                                    # direction rationale rests on the preservation argument, NOT on GKD.
                                    beta = float(self.cfg.algorithm.get("jsd_beta", 0.3))
                                    pt = lt.exp()
                                    ps = ls.exp()
                                    m = (beta * pt + (1.0 - beta) * ps).clamp_min(1e-8)
                                    lm = m.log()
                                    kl_tok = (
                                        beta * (pt * (lt - lm)).sum(dim=-1)
                                        + (1.0 - beta) * (ps * (ls - lm)).sum(dim=-1)
                                    )
                                elif _dkl == "entropy_adaptive":
                                    # ENTROPY-ADAPTIVE forward+reverse (user design 2026-09-05): switch
                                    # the divergence by the TEACHER's per-token entropy, so each of the two
                                    # objectives the field is split over is used where it is RIGHT:
                                    #   teacher CONFIDENT (low entropy) -> reverse KL(student||teacher),
                                    #     mode-seeking = precisely INHERIT the capability the teacher is
                                    #     sure of (this is what capability-transfer labs use, and it is
                                    #     what breaks the "mode-covering learns everything a little,
                                    #     nothing well" conservation we keep hitting).
                                    #   teacher UNSURE (high entropy) -> ADD forward KL(teacher||student),
                                    #     mode-covering = do NOT zero-force the student onto the teacher's
                                    #     flat off-support region -> preserve what the student already has.
                                    # w = normalized teacher entropy in [0,1]; kl = reverse + w*scale*fwd.
                                    # This FUSES the confidence filter: forward's own P(x) weighting already
                                    # damps high-entropy teacher tokens, so distill_conf_tau is redundant
                                    # under this mode (leave it 0).
                                    pt = lt.exp()
                                    ps = ls.exp()
                                    _kl_rev = (ps * (ls - lt)).sum(dim=-1)  # grad via ls
                                    _kl_fwd = (pt * (lt - ls)).sum(dim=-1)  # grad via ls
                                    with torch.no_grad():
                                        _t_ent = -(pt * lt).sum(dim=-1)  # teacher entropy per token
                                        _ent_max = torch.log(
                                            torch.tensor(float(pt.shape[-1]), device=pt.device)
                                        )
                                        _w_ent = (_t_ent / _ent_max).clamp(0.0, 1.0)
                                    _fwd_scale = float(
                                        self.cfg.algorithm.get("ent_adaptive_fwd_scale", 1.0)
                                    )
                                    kl_tok = _kl_rev + _w_ent * _fwd_scale * _kl_fwd
                                else:
                                    pt = lt.exp()  # teacher probs (detached)
                                    kl_tok = (pt * (lt - ls)).sum(dim=-1)  # forward KL, grad via ls
                                # CONFIDENCE FILTER (kit): down-weight tokens where the TEACHER itself is
                                # uncertain (high entropy = OOD / off-support state) so we don't distill the
                                # teacher's garbage on the weak student's own drifted states.
                                _conf_tau = float(self.cfg.algorithm.get("distill_conf_tau", 0.0))
                                if _conf_tau > 0.0:
                                    with torch.no_grad():
                                        pt_d = lt.exp()
                                        t_ent = -(pt_d * lt).sum(dim=-1)  # teacher entropy per token
                                        ent_max = torch.log(
                                            torch.tensor(float(pt_d.shape[-1]), device=pt_d.device)
                                        )
                                        conf_w = (1.0 - t_ent / ent_max).clamp_min(0.0).pow(_conf_tau)
                                    kl_tok = kl_tok * conf_w
                                # FAILURE-ONLY distillation (distill_on_failure=True): only distill where
                                # the student's own rollout FAILED. Where it already succeeds, the
                                # teacher's disagreement is style, not substance -- we measured that
                                # three models which ALL solve object still disagree on 25-42% of
                                # actions, i.e. as much as the student-teacher gap itself, so copying it
                                # wastes the shared LoRA capacity and is what makes several per-suite
                                # experts fight each other.
                                # The flag is "traj_fail", precomputed in _process_received_rollout_batch
                                # where the trajectory structure still exists (see the comment there for
                                # why computing it from this micro-batch's rewards is WRONG).
                                # Two ways to emphasise what the student got WRONG:
                                #   distill_on_failure=True  -> HARD filter, successful positions are
                                #                               dropped entirely (mtok *= fail).
                                #   distill_fail_alpha=a>0   -> SOFT weight, failed positions count
                                #                               (1+a)x and successful ones still count 1.
                                # Soft is the default choice: the hard filter throws away every state the
                                # student already handles, which is also where "don't break what works"
                                # has to be learned. Hard wins ties (both set = hard).
                                _fail_only = bool(
                                    self.cfg.algorithm.get("distill_on_failure", False)
                                )
                                _fail_alpha = float(
                                    self.cfg.algorithm.get("distill_fail_alpha", 0.0)
                                )
                                fail_m = None
                                if _fail_only or _fail_alpha > 0.0:
                                    fail_m = batch.get("traj_fail", None)
                                    if fail_m is None:
                                        raise RuntimeError(
                                            "distill_on_failure=True but 'traj_fail' is missing from the "
                                            "batch -- it must be built in _process_received_rollout_batch; "
                                            "refusing to silently fall back to distilling everything."
                                        )
                                    fail_m = fail_m.to(kl_tok.dtype)
                                if loss_mask is not None:
                                    # loss_mask is per-chunk [B, chunks]; expand to per-token to mask kl_tok
                                    mtok = (
                                        loss_mask.to(kl_tok.dtype)
                                        .unsqueeze(-1)
                                        .expand(-1, -1, sad)
                                        .reshape(kl_tok.shape[0], -1)
                                    )
                                else:
                                    mtok = torch.ones_like(kl_tok)
                                # ---- CHUNK IMPORTANCE (chunk_select, user design 2026-09-05) ----
                                # Our design: UP-WEIGHT the action chunks that matter most, per the two
                                # cases -- (a) student CONFIDENT but DISAGREES with the teacher (confidently
                                # wrong), (b) student UNSURE. The loss stays a SOFT multiplicative weight on
                                # mtok (mean-1, no samples dropped) so it composes with dance's single-suite
                                # mask, the anchor bookkeeping, and the failure/dyn-weight terms.
                                #
                                # For the "how to combine the two signals" step ONLY, we borrow TIP's
                                # parameter-free soft-OR (arXiv:2604.14084, Eq.5) -- it is a clean,
                                # tuning-free way to say "important if EITHER uncertain OR divergent" and
                                # it provably recovers the confident-but-wrong case (h~0, d>0) that a naive
                                # sum/product would miss. TIP itself uses this score for HARD TopK token
                                # DROPPING; we deliberately do NOT adopt that -- our design keeps every
                                # sample and only reweights, so TIP's selection step defers to our scheme.
                                #   h = norm student entropy  H(P_S)/log|V|      (uncertainty)
                                #   d = KL(student||teacher)                      (disagreement)
                                #   s = 1 - (1-h_hat)(1-d_hat)                     (soft-OR, per-batch min-max)
                                #   weight = 1 + kappa * (s / mean(s))            (kappa scales our emphasis)
                                # VLA note: h,d are per-token (TIP is LLM per-token); OpenVLA-OFT actions
                                # are chunks of `sad` tokens, so both are MEAN-aggregated to per-chunk.
                                _cs_mode = str(self.cfg.algorithm.get("chunk_select", "off")).lower()
                                if _cs_mode in ("false", "0", "none"):
                                    _cs_mode = "off"
                                elif _cs_mode in ("true", "1", "soft"):
                                    _cs_mode = "on"
                                if _cs_mode == "on":
                                    with torch.no_grad():
                                        _bC = kl_tok.shape[0]
                                        _psd = ls.exp()
                                        _s_ent = -(_psd * ls).sum(dim=-1)  # student entropy per token
                                        _emax_s = torch.log(
                                            torch.tensor(float(ls.shape[-1]), device=ls.device)
                                        )
                                        _h = _s_ent / _emax_s  # normalized student entropy
                                        _d = kl_tok.detach()   # KL(student||teacher) disagreement
                                        # per-batch min-max to [0,1]; guard all-equal batch (->0 not NaN)
                                        _h = (_h - _h.min()) / (_h.max() - _h.min()).clamp_min(1e-6)
                                        _d = (_d - _d.min()) / (_d.max() - _d.min()).clamp_min(1e-6)
                                        _h_c = _h.reshape(_bC, -1, sad).mean(dim=-1)  # [B, chunks]
                                        _d_c = _d.reshape(_bC, -1, sad).mean(dim=-1)
                                        _s = _h_c + _d_c - _h_c * _d_c  # soft-OR (TIP Eq.5), borrowed
                                        # OUR soft reweight (not TIP's TopK drop): emphasis kappa,
                                        # mean-1 normalized so loss scale / usable lr do not drift.
                                        _kappa = float(
                                            self.cfg.algorithm.get("chunk_select_kappa", 1.0)
                                        )
                                        _cw = 1.0 + _kappa * (_s / _s.mean().clamp_min(1e-6))
                                        _cw = _cw / _cw.mean().clamp_min(1e-6)
                                        _cw_tok = (
                                            _cw.unsqueeze(-1).expand(-1, -1, sad)
                                            .reshape(_bC, -1)
                                        )
                                    mtok = mtok * _cw_tok
                                # seqslot: the expert-KL trains ONLY the current
                                # round's suite; every other sample belongs to the
                                # anchor term below. One (B,1) 0/1 mask from the
                                # routing this micro-batch already ran -- a single
                                # small H2D copy, no sync. A micro-batch with no
                                # round-suite samples yields an OPD term of exactly 0
                                # through the clamp_min(1.0) denominator: correct
                                # (nothing to distill here), and the anchor term
                                # still trains B_k on what the batch does hold.
                                _seq_smask = None
                                if getattr(self, "_seqslot_suite", None) is not None:
                                    _seq_smask = torch.tensor(
                                        [
                                            1.0 if _s == self._seqslot_suite else 0.0
                                            for _s in self._last_suites
                                        ],
                                        device=kl_tok.device,
                                        dtype=kl_tok.dtype,
                                    ).view(-1, 1)
                                    mtok = mtok * _seq_smask
                                if fail_m is not None:
                                    # traj_fail is per-chunk [B, chunks] like loss_mask -> expand the same way
                                    fm = (
                                        fail_m.unsqueeze(-1)
                                        .expand(-1, -1, sad)
                                        .reshape(kl_tok.shape[0], -1)
                                    )
                                    if _fail_only:
                                        mtok = mtok * fm
                                    else:
                                        mtok = mtok * (1.0 + _fail_alpha * fm)
                                    # fraction of the VALID (loss_mask'd) positions we actually distil on.
                                    # SELF-CHECK: this must land near (1 - success_rate), NOT ~0.98.
                                    with torch.no_grad():
                                        _base = (
                                            loss_mask.to(kl_tok.dtype)
                                            .unsqueeze(-1)
                                            .expand(-1, -1, sad)
                                            .reshape(kl_tok.shape[0], -1)
                                            if loss_mask is not None
                                            else torch.ones_like(kl_tok)
                                        )
                                        # fraction of VALID positions that are failures -- identical in
                                        # both modes, so the "must land near (1 - success_rate)" check
                                        # still applies when soft weighting rescales mtok.
                                        self._last_fail_frac = (
                                            (_base * fm).sum() / _base.sum().clamp_min(1.0)
                                        ).item()
                                # ---- DYNAMIC PER-SUITE DISTILL STRENGTH ----------------------
                                # "push harder where the student is further from its teacher." The
                                # distance is measured by the KL itself, computed on this very forward
                                # pass -- NOT by the per-suite success rate seen during training, which
                                # was measured wrong by +0.32 (long) and -0.33 (goal) against a post-hoc
                                # 50-env eval and would have weighted exactly backwards.
                                #
                                # w_s = clip((ema_kl_s / mean_ema_kl) ** alpha, w_min, w_max), then
                                # renormalised to mean 1 so the loss scale (and the usable lr) does not
                                # drift. w_min > 0 on purpose: a suite the student already matches still
                                # needs a nonzero pull or the other suites' gradients walk it back.
                                #
                                # TWO THINGS THIS VERSION GETS RIGHT AND THE FIRST ONE DID NOT:
                                #  1. NO GPU->CPU SYNC IN THE HOT PATH. The first cut called .item() per
                                #     suite per micro-batch (plus a host->device copy per suite for the
                                #     index list) -- ~12 syncs per micro-batch, which across 6 FSDP ranks
                                #     stalls every rank at the next collective and took the update phase
                                #     from ~15 min to ~39 min. Everything below stays on the GPU:
                                #     index_add_ for the per-suite means, EMA as a device tensor.
                                #  2. A FIXED SUITE KEY SET. The metrics emitted at the bottom must not
                                #     depend on which suites this rank's micro-batch happened to contain:
                                #     all_reduce_dict packs the metric dict into ONE tensor sized by key
                                #     count, so a rank that saw 3 suites and a rank that saw 4 would
                                #     all-reduce different-sized tensors and hang forever. Same failure
                                #     that was fixed in libero_env.py earlier; do not reintroduce it.
                                # Per-suite distillation strength. The WEIGHTS themselves are
                                # recomputed once per training step in _dw_refresh_weights(); this
                                # path only gathers them and accumulates the KL that feeds the next
                                # refresh. NOTHING here may force a device sync -- `.any()`, `.item()`
                                # and boolean-mask indexing all do, and 32 micro-batches x ~6 syncs
                                # measured run_training at 80.0 min against a 14.3 min baseline on an
                                # otherwise identical config (2026-08-20). Keep it gather-only.
                                _dynw = float(self.cfg.algorithm.get("distill_dyn_weight", 0.0))
                                _wrow = None
                                if _dynw > 0.0 and getattr(self, "_last_groups", None):
                                    with torch.no_grad():
                                        if not hasattr(self, "_dw_paths"):
                                            _s2p = getattr(self, "teacher_suite_to_path", {}) or {}
                                            self._dw_paths = sorted(set(_s2p.values()))
                                            self._dw_w = None
                                        _paths = self._dw_paths
                                        _np = len(_paths)
                                        if _np > 1:
                                            _dev, _dt = kl_tok.device, kl_tok.dtype
                                            if getattr(self, "_dw_w", None) is None:
                                                self._dw_w = torch.ones(_np, device=_dev, dtype=_dt)
                                                self._dw_ema = torch.zeros(_np, device=_dev, dtype=_dt)
                                                self._dw_n = torch.zeros(_np, device=_dev, dtype=_dt)
                                                self._dw_d = torch.zeros(_np, device=_dev, dtype=_dt)
                                                self._dw_w_list = None
                                                self._dw_kl_list = None
                                            _pi = {p: i for i, p in enumerate(_paths)}
                                            _B = kl_tok.shape[0]
                                            _g = [-1] * _B
                                            for _p, _idxs in self._last_groups.items():
                                                _k = _pi.get(_p)
                                                if _k is not None:
                                                    for _i in _idxs:
                                                        if 0 <= _i < _B:
                                                            _g[_i] = _k
                                            # pinned staging buffer -> the H2D copy is async and does
                                            # NOT drain the compute stream the way a pageable copy does
                                            _hb = getattr(self, "_dw_hostbuf", None)
                                            if _hb is None or _hb.numel() < _B:
                                                self._dw_hostbuf = torch.empty(
                                                    _B, dtype=torch.long, pin_memory=True
                                                )
                                                _hb = self._dw_hostbuf
                                            _hb = _hb[:_B]
                                            _hb.copy_(torch.as_tensor(_g, dtype=torch.long))
                                            _gidx = _hb.to(_dev, non_blocking=True)
                                            _valid = (_gidx >= 0).to(_dt)
                                            _safe = _gidx.clamp_min(0)
                                            # feed the NEXT refresh (index_add_ never syncs); rows with
                                            # no routed teacher contribute exactly 0 via _valid
                                            self._dw_n.index_add_(
                                                0, _safe, (kl_tok * mtok).sum(-1) * _valid
                                            )
                                            self._dw_d.index_add_(0, _safe, mtok.sum(-1) * _valid)
                                            # weights from the last refresh; unrouted rows get 1.0
                                            _wrow = self._dw_w[_safe] * _valid + (1.0 - _valid)
                                if _wrow is not None:
                                    _wt = _wrow.unsqueeze(-1)
                                    opd_distill_loss = (kl_tok * mtok * _wt).sum() / (mtok * _wt).sum().clamp_min(1.0)
                                else:
                                    opd_distill_loss = (kl_tok * mtok).sum() / mtok.sum().clamp_min(1.0)
                                # Memory-Anchor Step-2 proxy (anchor_frac > 0): per-suite
                                # student-teacher KL EMA, read NEXT step by
                                # dance_anchor_augment as the suite weight -- "prefer
                                # anchors from the suite currently drifting furthest from
                                # its teacher". Suite names come from the routing this
                                # micro-batch already ran (_last_suites); NOT from the
                                # in-training success rate, which mis-measured by +-0.33.
                                if (
                                    float(self.cfg.algorithm.get("anchor_frac", 0.0)) > 0.0
                                    and getattr(self, "_last_suites", None) is not None
                                ):
                                    with torch.no_grad():
                                        _ks = (kl_tok * mtok).sum(dim=1) / mtok.sum(
                                            dim=1
                                        ).clamp_min(1.0)
                                        _ema = getattr(self, "_anchor_kl_ema", None)
                                        if _ema is None:
                                            _ema = {}
                                            self._anchor_kl_ema = _ema
                                        for _ai, _as in enumerate(self._last_suites):
                                            if _as is None:
                                                continue
                                            _av = float(_ks[_ai])
                                            _ema[_as] = 0.9 * _ema.get(_as, _av) + 0.1 * _av

                                # How many suites this micro-batch actually contains. Both probes are
                                # meaningless on a single-suite micro-batch (there is no other expert
                                # to compare against), and whether the data mixes suites at all is an
                                # empirical question about the rollout/shuffle pipeline -- so MEASURE
                                # it instead of assuming. If this sits at 1.0 the probes are silently
                                # inert and the routing/group_size has to change first.
                                _ngrp = len(getattr(self, "_last_groups", {}) or {})
                                self._last_nsuites = float(_ngrp)

                                # absorbability / signal-location probe. NOT gated on multi-suite:
                                # it asks about ONE teacher-student pair, so a single-suite micro-batch
                                # is perfectly valid input.
                                if getattr(self, "_sig_armed", False):
                                    self._sig_armed = False
                                    try:
                                        self._last_sig = self._signal_stats(
                                            ls.detach(), lt, kl_tok.detach(), mtok
                                        )
                                    except Exception as _e:
                                        self._last_sig = {}
                                        self.log_warning(f"[VLA-OPD] signal_stats failed: {_e}")

                                # Probes run on the first MIXED micro-batch of the step. The latch is
                                # NOT cleared on a single-suite batch: clearing it there would spend
                                # the step's one probe on a batch that can produce nothing.
                                if getattr(self, "_xscore_armed", False) and _ngrp >= 2:
                                    try:
                                        self._last_xkl = self._cross_score(
                                            forward_inputs, kwargs, ls.detach(), loss_mask, sad
                                        )
                                        self._xscore_armed = False
                                    except Exception as _e:  # never let a probe kill a training run
                                        self._last_xkl = {}
                                        self._xscore_armed = False
                                        self.log_warning(f"[VLA-OPD] cross_score failed: {_e}")

                                # per-suite gradient conflict probe: first mixed micro-batch of the
                                # step. MUST run before the real backward frees the graph.
                                if getattr(self, "_gconf_armed", False) and _ngrp >= 2:
                                    try:
                                        self._last_gconf = self._grad_conflict(kl_tok, mtok)
                                        self._gconf_armed = False
                                    except Exception as _e:
                                        self._last_gconf = {}
                                        self._gconf_armed = False
                                        self.log_warning(f"[VLA-OPD] grad_conflict failed: {_e}")

                            # ---- data-free BASE anchors (action-KL + visual-representation) ----
                            # (a) action anchor = mode-covering forward-KL to base on rollout states
                            #     (preserve task behavior); (b) visual anchor = cosine of mid-layer
                            #     vision+prompt features to base (preserve BROAD OOD generalization).
                            _alam = float(self.cfg.algorithm.get("anchor_lambda", 0.0))
                            _vlam = float(self.cfg.algorithm.get("visual_anchor_lambda", 0.0))
                            _amode = str(self.cfg.algorithm.get("anchor_mode", "base"))
                            if (
                                _amode == "self_masked"
                                and _alam > 0.0
                                and getattr(self, "_seqslot_suite", None) is not None
                            ):
                                # seqslot anchor teacher = the student MINUS the
                                # current round's slot. With base, every A and the
                                # other slots' B frozen, that forward IS M_(k-1) --
                                # the model as it stood before this round -- so no
                                # merged checkpoint is ever written, read or held in
                                # memory for the anchor. no_grad: teacher side only;
                                # the student side of the KL comes from output_dict,
                                # which already carries gradient. excluding() also
                                # runs ungated and restores the routing on exit, so
                                # the gated recomputation that gradient checkpointing
                                # performs during backward still sees this
                                # micro-batch's routing.
                                if _vlam > 0.0:
                                    raise RuntimeError(
                                        "anchor_mode=self_masked has no base model to "
                                        "take mid-layer features from, so "
                                        "visual_anchor_lambda>0 cannot be honored. "
                                        "Set it to 0 or use anchor_mode=base."
                                    )
                                _ex_slot = self._slot_index_of()[self._seqslot_suite]
                                with (
                                    torch.no_grad(),
                                    self.amp_context,
                                    self._slot_gate.excluding(_ex_slot),
                                ):
                                    base_out = self.model(
                                        forward_inputs=forward_inputs,
                                        compute_logprobs=True,
                                        use_cache=False,
                                        **kwargs,
                                    )
                            elif (_alam > 0.0 or _vlam > 0.0) and getattr(
                                self, "base_model", None
                            ) is not None:
                                with torch.no_grad(), self.amp_context:
                                    base_out = self.base_model(
                                        forward_inputs=forward_inputs,
                                        compute_logprobs=True,
                                        use_cache=False,
                                        **kwargs,
                                    )
                            else:
                                base_out = None
                            if base_out is not None:
                                # (a) ACTION anchor
                                if (
                                    _alam > 0.0
                                    and "action_logits" in output_dict
                                    and "action_logits" in base_out
                                ):
                                    lb = torch.log_softmax(
                                        base_out["action_logits"].float().detach(), dim=-1
                                    )
                                    ls_a = torch.log_softmax(
                                        output_dict["action_logits"].float(), dim=-1
                                    )
                                    pb = lb.exp()
                                    a_tok = (pb * (lb - ls_a)).sum(dim=-1)  # forward-KL, mode-covering
                                    _agate = self.cfg.algorithm.get("anchor_gate", "none")
                                    _atau = float(self.cfg.algorithm.get("anchor_gate_tau", 1.0))
                                    if _agate in ("low_ent", "high_ent"):
                                        with torch.no_grad():
                                            b_ent = -(pb * lb).sum(dim=-1)
                                            _emax = torch.log(
                                                torch.tensor(float(pb.shape[-1]), device=pb.device)
                                            )
                                            conf = (1.0 - b_ent / _emax).clamp(0.0, 1.0)
                                            gate = (
                                                conf.pow(_atau)
                                                if _agate == "low_ent"
                                                else (1.0 - conf).pow(_atau)
                                            )
                                        a_tok = a_tok * gate
                                    if loss_mask is not None:
                                        _amt = (
                                            loss_mask.to(a_tok.dtype)
                                            .unsqueeze(-1)
                                            .expand(-1, -1, sad)
                                            .reshape(a_tok.shape[0], -1)
                                        )
                                    else:
                                        _amt = torch.ones_like(a_tok)
                                    if getattr(self, "_seqslot_suite", None) is not None:
                                        # seqslot: the anchor holds the COMPLEMENT of
                                        # the expert-KL's samples -- old suites only.
                                        # On the round suite the expert term already
                                        # says what B_k should do; anchoring it there
                                        # too would pull the same logits toward
                                        # M_(k-1) and the expert at once. Rebuilt
                                        # here rather than reusing the OPD block's
                                        # mask: that block is behind its own
                                        # action_logits guard, and a NameError from a
                                        # path that skipped it would be this loss's
                                        # only failure mode.
                                        _seq_amask = torch.tensor(
                                            [
                                                0.0
                                                if _s == self._seqslot_suite
                                                else 1.0
                                                for _s in self._last_suites
                                            ],
                                            device=a_tok.device,
                                            dtype=a_tok.dtype,
                                        ).view(-1, 1)
                                        _amt = _amt * _seq_amask
                                    anchor_loss = (a_tok * _amt).sum() / _amt.sum().clamp_min(1.0)
                                # (b) VISUAL anchor: patch-wise cosine of mid-layer features to base
                                if (
                                    _vlam > 0.0
                                    and "mid_features" in output_dict
                                    and "mid_features" in base_out
                                ):
                                    fs = output_dict["mid_features"].float()
                                    fb = base_out["mid_features"].float().detach()
                                    cos = torch.nn.functional.cosine_similarity(fs, fb, dim=-1)
                                    visual_loss = (1.0 - cos).mean()

                        kwargs = {
                            "loss_type": self.cfg.algorithm.loss_type,
                            "logprob_type": self.cfg.algorithm.logprob_type,
                            "reward_type": self.cfg.algorithm.reward_type,
                            "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                            "logprobs": output_dict["logprobs"],
                            "values": output_dict.get("values", None),
                            "old_logprobs": prev_logprobs,
                            "advantages": advantages,
                            "returns": returns,
                            "prev_values": prev_values,
                            "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                            "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                            "value_clip": self.cfg.algorithm.get("value_clip", None),
                            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                            "loss_mask": loss_mask,
                            "loss_mask_sum": loss_mask_sum,
                            "max_episode_steps": self.cfg.env.train.max_episode_steps,
                            "task_type": self.cfg.runner.task_type,
                            "critic_warmup": self.optimizer_steps
                            < self.critic_warmup_steps,
                        }
                        if (
                            self.cfg.algorithm.adv_type == "opd"
                            and self.cfg.algorithm.get("opd_mode", "distill") == "distill"
                            and opd_distill_loss is not None
                        ):
                            # differentiable KL-distillation: minimize KL(student||teacher) directly,
                            # skip the PPO/REINFORCE loss entirely (pi0 op_distill style).
                            loss = opd_distill_loss
                            if anchor_loss is not None:
                                loss = loss + float(
                                    self.cfg.algorithm.get("anchor_lambda", 0.0)
                                ) * anchor_loss
                            if visual_loss is not None:
                                loss = loss + float(
                                    self.cfg.algorithm.get("visual_anchor_lambda", 0.0)
                                ) * visual_loss
                            metrics_data = {
                                "actor/distill_loss": opd_distill_loss.detach().item()
                            }
                            if anchor_loss is not None:
                                metrics_data["actor/anchor_loss"] = (
                                    anchor_loss.detach().item()
                                )
                            if getattr(self, "_seqslot_suite", None) is not None:
                                # Which round this step trained, as the slot index --
                                # the ONE curve that says where every round boundary
                                # actually fell, against which dw_norm_k of the
                                # supposedly frozen slots is read.
                                metrics_data["slot/round_slot"] = float(
                                    self._slot_index_of()[self._seqslot_suite]
                                )
                            # Surface the dynamic weights and the per-suite KL they came from --
                            # without them an adaptive run is indistinguishable from a uniform one and
                            # the mechanism is unfalsifiable.
                            # EMIT A FIXED KEY SET: one entry per ROUTED SUITE, always, defaulting to
                            # 1.0/0.0 for suites this rank's micro-batch did not contain. all_reduce_dict
                            # sizes its packed tensor by the key count, so rank-dependent keys deadlock
                            # the collective (the bug already fixed once in libero_env.py). One .tolist()
                            # here is the ONLY host sync in this path.
                            _s2p = getattr(self, "teacher_suite_to_path", {}) or {}
                            if _s2p and float(self.cfg.algorithm.get("distill_dyn_weight", 0.0)) > 0.0:
                                _paths = getattr(self, "_dw_paths", None) or sorted(set(_s2p.values()))
                                _wl = getattr(self, "_dw_w_list", None) or [1.0] * len(_paths)
                                _kl_ = getattr(self, "_dw_kl_list", None) or [0.0] * len(_paths)
                                _idx = {p: i for i, p in enumerate(_paths)}
                                for _s in sorted(_s2p):
                                    _i = _idx.get(_s2p[_s])
                                    metrics_data[f"actor/dynw_{_s}"] = (
                                        float(_wl[_i]) if _i is not None and _i < len(_wl) else 1.0
                                    )
                                    metrics_data[f"actor/suitekl_{_s}"] = (
                                        float(_kl_[_i]) if _i is not None and _i < len(_kl_) else 0.0
                                    )
                            if getattr(self, "_last_fail_frac", None) is not None:
                                # share of rollout samples that never succeeded = what we distill on
                                metrics_data["actor/fail_frac"] = self._last_fail_frac
                            if anchor_loss is not None:
                                metrics_data["actor/anchor_loss"] = anchor_loss.detach().item()
                            if visual_loss is not None:
                                metrics_data["actor/visual_loss"] = visual_loss.detach().item()
                            # cross-suite KL probe (diagonal = what training minimises, off-diagonal =
                            # what nobody optimises); emitted on the probe micro-batch only, so it is
                            # carried on self and re-emitted for the rest of the step's micro-batches.
                            # ALWAYS emit, ALWAYS the same keys -- see _probe_key_list for why a
                            # rank-dependent key set deadlocks the metric all_reduce.
                            # 1.0 => micro-batches are single-suite => the cross-suite probes are inert
                            metrics_data["actor/n_suites_in_batch"] = float(
                                getattr(self, "_last_nsuites", 0.0) or 0.0
                            )
                            metrics_data.update(self._probe_emit())
                        else:
                            loss, metrics_data = policy_loss(**kwargs)

                        if opd_kl is not None:
                            metrics_data["actor/opd_kl_stu_tea"] = opd_kl
                        if opd_gap is not None:
                            metrics_data["actor/opd_gap_raw"] = opd_gap
                        if self._slot_enabled:
                            # Share of this micro-batch that matched no suite, so it
                            # was scored by an arbitrary expert and folded into the
                            # loss while its gradient reached no slot. EXPECTED EXACTLY
                            # 0 -- every task prompt of this experiment is in the
                            # routing table -- and _route_prepare raises above
                            # algorithm.slot_route_fallback_tol (default 0.0), so a
                            # non-zero reading here only happens on a run that
                            # deliberately tolerates it. The key is config-derived and
                            # therefore identical on every rank, which all_reduce_dict
                            # requires.
                            metrics_data["slot/route_fallback_frac"] = float(
                                self._slot_fallback
                            )

                        entropy_loss = torch.tensor(
                            0.0, device=Worker.torch_platform.current_device()
                        )
                        if (
                            self.cfg.algorithm.entropy_bonus > 0
                            and not kwargs["critic_warmup"]
                        ):
                            entropy = output_dict["entropy"]
                            entropy = reshape_entropy(
                                entropy,
                                entropy_type=self.cfg.algorithm.entropy_type,
                                action_dim=self.cfg.actor.model.get("action_dim", 7),
                                batch_size=output_dict["logprobs"].shape[0],
                            )
                            entropy_loss = masked_mean(entropy, mask=loss_mask)
                            loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                        metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()

                        if self.enable_sft_co_train:
                            self._train_sft_epoch(metrics_data, loss)

                        loss /= self.gradient_accumulation
                        prof.mark("loss")
                        with backward_ctx:
                            self.grad_scaler.scale(loss).backward()
                        prof.mark("backward")

                    metrics_data["actor/total_loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)
                    # Read the events back HERE and nowhere earlier: the .item() above has
                    # already drained the stream, so the synchronize this does is free.
                    prof.micro_done()

                self.torch_platform.empty_cache()

                # slot-LoRI B,B,A: freeze one factor for THIS update by zeroing its
                # optimizer group's lr and dropping its grads, then put the lr back
                # afterwards. Both calls must bracket optimizer_step: the freeze has to
                # be in place before AdamW runs, and the restore has to happen before
                # the next update reads the group. Note that on a slot run the
                # `actor/lr` key below reads whichever group comes first, so it shows
                # the momentary per-group lr the alternation writes; `slot/phase_is_A`
                # is the metric that reports the schedule.
                update_index += 1
                self._slot_alt_before_update(
                    is_last_update=(update_index == updates_per_step)
                )
                prof.mark("empty_cache")
                grad_norm, lr_list = self.optimizer_step()
                prof.mark("optim")
                self._slot_alt_after_update()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
                prof.update_done()
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        prof.step_done()
        # slot-LoRI: close the step. Reads what the armed forward stashed (orthogonality
        # error, per-slot ΔW norms, cross-slot cosines) plus the phase counter, as a
        # key set fixed by config so the metric all-reduce cannot deadlock.
        append_to_dict(metrics, self._slot_step_metrics())
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        return mean_metric_dict

    def set_global_step(self, global_step: int) -> None:
        """
        Set the global step for the model, if needed.
        """
        self.version = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
