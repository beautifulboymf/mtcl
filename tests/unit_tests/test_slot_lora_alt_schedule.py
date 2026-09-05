# Copyright 2026 The RLinf Authors.
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

"""The B,B,A alternating schedule, the per-step slot metrics, and the gate's sync fix.

WHY THESE TESTS LOOK LIKE THIS. The real actor needs ray, LIBERO and a 7B checkpoint,
so nothing here instantiates one. Instead, in descending order of strength:

* the schedule DECISION is a pure function (``slot_alt_phase``) and is tested directly;
* the freeze/thaw is driven through the REAL unbound actor methods against a
  hand-built ``EmbodiedFSDPActor.__new__`` carrying a REAL ``torch.optim.AdamW`` over a
  REAL injected slot model, so "the frozen factor's Adam state does not advance" is
  measured on the optimizer that runs in production, not asserted about a mock;
* the metrics come out of the real ``collect_slot_diag`` on that same real model;
* two ``ast`` tests pin the ORDER of the calls inside ``run_training``, which is the
  one thing no CPU test can execute (it needs a rollout batch) and whose failure is
  silent: a freeze applied after ``optimizer_step`` would train both factors every
  update while ``slot/phase_is_A`` still read a healthy 1/3.

The sync fix is tested with a ``TorchFunctionMode`` that records every torch function
the validation runs. On CUDA the sync is the ``.any()``/``bool()`` pair; on CPU those
calls are cheap, so what is asserted is that they no longer HAPPEN -- which is the
same fact, observable without a GPU.
"""

import ast
import inspect
import math
import textwrap

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.overrides import TorchFunctionMode

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    PARAM_GROUP_SLOT_A,
    PARAM_GROUP_SLOT_B,
    build_slot_aware_param_groups,
)
from rlinf.models.slot_lora import SlotGate, inject_slot_lora
from rlinf.models.slot_lora.modules import make_slot_ids
from rlinf.utils.utils import warmup_optimizer_state
from rlinf.workers.actor.fsdp_actor_worker import (
    EmbodiedFSDPActor,
    parse_slot_alt_schedule,
    slot_alt_phase,
)

LR = 1e-4


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


class _Tiny(nn.Module):
    """One targetable linear -- the smallest model injection accepts."""

    def __init__(self, d_in=16, d_out=8):
        super().__init__()
        self.q_proj = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.q_proj(x)


def _toy_slot_model(ranks=(2, 2, 2), dtype=torch.float32):
    """A real injected slot model plus its gate. ``B`` is randomized on purpose.

    ``B`` is zero at init, so ``dΔW/dZ ∝ B`` is exactly zero and every gradient the
    A side could take would be zero for a reason that has nothing to do with the
    schedule -- which would make "the frozen factor did not move" vacuously true.
    """
    model = _Tiny().to(dtype)
    injection = inject_slot_lora(model, ranks, ["q_proj"], scale_mode="unit")
    for module in model.modules():
        if hasattr(module, "slot_B"):
            nn.init.normal_(module.slot_B.weight, std=0.1)
    return model, injection.gate


def _optimizer_for(model, lr=LR):
    """The optimizer the real path builds: named groups, then AdamW, then the warmup."""
    groups = build_slot_aware_param_groups(
        [(n, p) for n, p in model.named_parameters() if p.requires_grad],
        lr=lr,
        value_lr=lr,
        betas=(0.9, 0.95),
        weight_decay=1e-2,
    )
    optimizer = torch.optim.AdamW(groups, eps=1e-8, weight_decay=1e-2)
    warmup_optimizer_state(optimizer)
    return optimizer


def _make_actor(
    schedule="BBA",
    *,
    slot_enabled=True,
    ranks=(2, 2, 2),
    dtype=torch.float32,
    lr=LR,
):
    """A bare EmbodiedFSDPActor carrying only what the schedule/metric methods read."""
    model, gate = _toy_slot_model(ranks, dtype=dtype)
    actor = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "actor": {"model": {"slot_lora": {"alt_schedule": schedule}}},
            "algorithm": {},
        }
    )
    actor.model = model
    actor.optimizer = _optimizer_for(model, lr=lr)
    actor._slot_enabled = slot_enabled
    actor._slot_order = (
        tuple(f"suite_{i}" for i in range(len(ranks))) if slot_enabled else ()
    )
    actor._slot_gate = gate
    actor.warnings = []
    actor.infos = []
    actor.log_warning = actor.warnings.append
    actor.log_info = actor.infos.append
    actor._slot_alt_init()
    return actor


def _group(actor, name):
    for group in actor.optimizer.param_groups:
        if group.get("name") == name:
            return group
    raise AssertionError(f"no {name} group")


def _backward(actor, ids=(0, 1)):
    """One forward+backward through the real gated layer, so both factors have grads."""
    n_slots = len(actor._slot_order) or 3
    routing = make_slot_ids(list(ids), n_slots)
    x = torch.randn(len(ids), 16, dtype=next(actor.model.parameters()).dtype)
    with actor._slot_gate.scoped(routing):
        actor.model(x).square().sum().backward()


def _adam_state(actor, name):
    """``(step, exp_avg, exp_avg_sq)`` clones for every parameter of one group."""
    out = []
    for p in _group(actor, name)["params"]:
        state = actor.optimizer.state[p]
        out.append(
            (
                float(state["step"]),
                state["exp_avg"].clone(),
                state["exp_avg_sq"].clone(),
            )
        )
    return out


def _same_state(a, b):
    return len(a) == len(b) and all(
        x[0] == y[0] and torch.equal(x[1], y[1]) and torch.equal(x[2], y[2])
        for x, y in zip(a, b)
    )


def _run_update(actor, is_last_update):
    """What run_training does around one optimizer update, minus the loss."""
    actor.optimizer.zero_grad()
    _backward(actor)
    phase = actor._slot_alt_before_update(is_last_update=is_last_update)
    actor.optimizer.step()
    actor._slot_alt_after_update()
    return phase


def _run_step(actor, updates=3):
    actor._slot_step_begin(updates)
    return [
        _run_update(actor, is_last_update=(i + 1) == updates) for i in range(updates)
    ]


def _step_metrics(actor, updates=3):
    """One whole training step's worth of updates, then the step's metrics."""
    _run_step(actor, updates=updates)
    return actor._slot_step_metrics()


# --------------------------------------------------------------------------------------
# the schedule decision, as a pure function
# --------------------------------------------------------------------------------------


def test_bba_cycles_when_no_update_is_the_step_boundary():
    pos, phases = 0, []
    for _ in range(9):
        phase, pos = slot_alt_phase("BBA", pos, is_last_update=False)
        phases.append(phase)
    assert phases == list("BBABBABBA")


def test_the_last_update_of_every_step_is_forced_to_b():
    pos = 0
    for _ in range(6):  # six training steps of three updates each
        step = []
        for i in range(3):
            phase, pos = slot_alt_phase("BBA", pos, is_last_update=(i == 2))
            step.append(phase)
        assert step[-1] == "B", (
            "the step ended on A: the saved checkpoint's B would not match its Ā"
        )


def test_the_forced_b_does_not_consume_the_cycle_position():
    """The regression that would make the whole schedule dead.

    Three updates per step and a three-character schedule: if the forced B advanced
    the cycle like a scheduled update does, the cycle would stay locked in phase with
    the step boundary and the A that lands on the last update would be overwritten
    every single step -- A would never run, with `slot/phase_is_A` reading a
    perfectly plausible 0.0 and nothing else to say so.
    """
    pos, phases = 0, []
    for _ in range(6):
        for i in range(3):
            phase, pos = slot_alt_phase("BBA", pos, is_last_update=(i == 2))
            phases.append(phase)
    assert phases.count("A") > 0, "A never trains: the cycle is locked to the step"
    # exactly what the deferral produces: the overwritten A is retried next step
    assert phases == list("BBBABBBABBBBABBBAB")


def test_a_one_update_step_is_all_b():
    """Inherent to forcing the step end: worth pinning so it is not mistaken for a bug."""
    pos = 0
    phases = [slot_alt_phase("BBA", pos, is_last_update=True)[0] for _ in range(4)]
    assert phases == ["B"] * 4


def test_no_schedule_means_joint_and_never_moves_the_cycle():
    for last in (False, True):
        phase, pos = slot_alt_phase(None, 2, is_last_update=last)
        assert phase is None and pos == 2


@pytest.mark.parametrize(
    "raw,expected",
    [("BBA", "BBA"), ("bba", "BBA"), (" BA ", "BA"), ("A", "A"), (None, None)],
)
def test_parse_accepts_what_it_should(raw, expected):
    assert parse_slot_alt_schedule(raw) == expected


@pytest.mark.parametrize("raw", ["BBC", "B B", "", "  ", 3, ["B", "A"]])
def test_parse_rejects_everything_else(raw):
    with pytest.raises(ValueError):
        parse_slot_alt_schedule(raw)


def test_config_default_is_bba_and_null_means_joint():
    assert _make_actor()._slot_alt_schedule == "BBA"
    assert _make_actor(schedule=None)._slot_alt_schedule is None
    actor = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create({"actor": {"model": {}}, "algorithm": {}})
    actor._slot_enabled = True
    actor._slot_alt_init()
    assert actor._slot_alt_schedule == "BBA", "a missing key must take the default"


# --------------------------------------------------------------------------------------
# freezing: lr 0 AND grad None, on the real optimizer
# --------------------------------------------------------------------------------------


def test_b_phase_zeroes_the_a_group_lr_and_drops_its_grads():
    actor = _make_actor()
    actor._slot_step_begin(3)
    _backward(actor)
    assert all(p.grad is not None for p in _group(actor, PARAM_GROUP_SLOT_A)["params"])

    phase = actor._slot_alt_before_update(is_last_update=False)
    assert phase == "B"
    assert _group(actor, PARAM_GROUP_SLOT_A)["lr"] == 0.0
    assert _group(actor, PARAM_GROUP_SLOT_B)["lr"] == pytest.approx(LR)
    assert all(p.grad is None for p in _group(actor, PARAM_GROUP_SLOT_A)["params"])
    assert all(p.grad is not None for p in _group(actor, PARAM_GROUP_SLOT_B)["params"])


def test_a_phase_freezes_the_other_factor():
    actor = _make_actor(schedule="A")
    actor._slot_step_begin(3)
    _backward(actor)
    assert actor._slot_alt_before_update(is_last_update=False) == "A"
    assert _group(actor, PARAM_GROUP_SLOT_B)["lr"] == 0.0
    assert _group(actor, PARAM_GROUP_SLOT_A)["lr"] == pytest.approx(LR)
    assert all(p.grad is None for p in _group(actor, PARAM_GROUP_SLOT_B)["params"])


def test_the_frozen_factors_parameters_do_not_move():
    actor = _make_actor()
    actor._slot_step_begin(3)
    before = [p.clone() for p in _group(actor, PARAM_GROUP_SLOT_A)["params"]]
    _run_update(actor, is_last_update=False)
    after = _group(actor, PARAM_GROUP_SLOT_A)["params"]
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    # and the factor that WAS training really did move, or this proves nothing
    assert any(
        torch.linalg.vector_norm(actor.optimizer.state[p]["exp_avg"]) > 0
        for p in _group(actor, PARAM_GROUP_SLOT_B)["params"]
    )


def test_the_frozen_factors_adam_state_does_not_advance():
    """The reason lr=0 alone is not enough.

    AdamW folds a parameter's gradient into ``exp_avg``/``exp_avg_sq`` whatever the
    learning rate is; only ``p.grad is None`` keeps it out of the update entirely. With
    the momentum still accumulating, the first step after the factor thaws would move
    it by everything it collected while it was supposed to be still.
    """
    actor = _make_actor()
    actor._slot_step_begin(3)
    _run_update(actor, is_last_update=False)  # A frozen
    frozen_state = _adam_state(actor, PARAM_GROUP_SLOT_A)
    trained_state = _adam_state(actor, PARAM_GROUP_SLOT_B)

    _run_update(actor, is_last_update=False)  # A frozen again
    assert _same_state(frozen_state, _adam_state(actor, PARAM_GROUP_SLOT_A)), (
        "the frozen factor's Adam moments advanced while its lr was zero"
    )
    assert not _same_state(trained_state, _adam_state(actor, PARAM_GROUP_SLOT_B))


def test_zeroing_the_lr_alone_would_have_advanced_the_adam_state():
    """The counter-test: without the grad=None, the state DOES move.

    This is what makes the assertion above meaningful rather than a tautology about a
    model whose gradients happen to be zero.
    """
    actor = _make_actor()
    actor._slot_step_begin(3)
    _backward(actor)
    _group(actor, PARAM_GROUP_SLOT_A)["lr"] = 0.0  # the lr half of the freeze only
    before = _adam_state(actor, PARAM_GROUP_SLOT_A)
    actor.optimizer.step()
    assert not _same_state(before, _adam_state(actor, PARAM_GROUP_SLOT_A))


def test_thawing_after_frozen_updates_moves_only_on_fresh_gradient():
    """End to end over one step: A's first real update must not carry stale momentum."""
    actor = _make_actor()
    actor._slot_step_begin(9)
    for i in range(2):  # B, B
        assert _run_update(actor, is_last_update=False) == "B"
    a_params = _group(actor, PARAM_GROUP_SLOT_A)["params"]
    assert all(float(actor.optimizer.state[p]["step"]) == 1 for p in a_params), (
        "the frozen factor took optimizer steps it should have sat out"
    )
    assert all(
        torch.linalg.vector_norm(actor.optimizer.state[p]["exp_avg"]) == 0
        for p in a_params
    )
    assert _run_update(actor, is_last_update=False) == "A"
    assert all(float(actor.optimizer.state[p]["step"]) == 2 for p in a_params)


# --------------------------------------------------------------------------------------
# restoring the learning rate
# --------------------------------------------------------------------------------------


def test_lr_is_restored_after_every_update():
    actor = _make_actor()
    for phase in _run_step(actor, updates=3):
        assert phase in ("A", "B")
    assert _group(actor, PARAM_GROUP_SLOT_A)["lr"] == pytest.approx(LR)
    assert _group(actor, PARAM_GROUP_SLOT_B)["lr"] == pytest.approx(LR)


def test_restored_lr_is_the_schedulers_current_value_not_a_stale_one():
    """A snapshot taken once at build time would re-install last step's warmup lr.

    The LR scheduler rewrites every group's lr once per training step; the value the
    alternation restores has to be THAT one. Here the scheduler's move is simulated by
    writing the groups directly, exactly as ``LambdaLR.step()`` does.
    """
    actor = _make_actor()
    _run_step(actor, updates=3)

    scheduled = LR * 7.0  # the scheduler's new value for the next step
    for name in (PARAM_GROUP_SLOT_A, PARAM_GROUP_SLOT_B):
        _group(actor, name)["lr"] = scheduled

    _run_step(actor, updates=3)
    assert _group(actor, PARAM_GROUP_SLOT_A)["lr"] == pytest.approx(scheduled)
    assert _group(actor, PARAM_GROUP_SLOT_B)["lr"] == pytest.approx(scheduled)


def test_a_zeroed_lr_never_survives_into_the_next_step():
    actor = _make_actor()
    _run_step(actor, updates=1)  # the single update is the forced B -> A is zeroed
    assert _group(actor, PARAM_GROUP_SLOT_A)["lr"] == pytest.approx(LR)
    actor._slot_step_begin(3)
    assert actor._slot_alt_lr[PARAM_GROUP_SLOT_A] == pytest.approx(LR), (
        "the new step snapshotted a zero, so the zero would be restored forever"
    )


def test_missing_slot_groups_with_an_active_schedule_are_loud():
    actor = _make_actor()
    actor.optimizer.param_groups = [
        g for g in actor.optimizer.param_groups if g.get("name") != PARAM_GROUP_SLOT_A
    ]
    with pytest.raises(RuntimeError, match="slot_A"):
        actor._slot_step_begin(3)


def test_running_a_different_number_of_updates_than_promised_is_reported():
    """The forced B is aimed by a count computed separately from the loop bounds.

    If the two ever drift, the override fires at the wrong update and the step can end
    on A -- a checkpoint whose B does not match its Ā -- with every other metric
    looking exactly as healthy as before.
    """
    actor = _make_actor()
    actor._slot_step_begin(5)  # promised five
    for i in range(3):  # ran three
        _run_update(actor, is_last_update=(i == 2))
    actor._slot_step_metrics()
    assert any("forced to B" in w for w in actor.warnings), actor.warnings


def test_the_promised_update_count_is_not_flagged():
    actor = _make_actor()
    _run_step(actor, updates=3)
    actor._slot_step_metrics()
    assert actor.warnings == []


# --------------------------------------------------------------------------------------
# a model with no slots is completely unaffected
# --------------------------------------------------------------------------------------


def test_no_slots_means_no_freeze_no_metrics_no_touching_the_optimizer():
    actor = _make_actor(slot_enabled=False)
    lrs = [g["lr"] for g in actor.optimizer.param_groups]
    actor._slot_step_begin(3)
    _backward(actor)
    grads_before = [
        None if p.grad is None else p.grad.clone()
        for g in actor.optimizer.param_groups
        for p in g["params"]
    ]
    assert actor._slot_alt_before_update(is_last_update=False) is None
    actor._slot_alt_after_update()
    assert [g["lr"] for g in actor.optimizer.param_groups] == lrs
    grads_after = [
        None if p.grad is None else p.grad
        for g in actor.optimizer.param_groups
        for p in g["params"]
    ]
    assert all(
        (a is None and b is None) or torch.equal(a, b)
        for a, b in zip(grads_before, grads_after)
    )
    assert actor._slot_step_metrics() == {}
    assert actor._slot_alt_schedule is None


def test_joint_ablation_trains_both_factors_every_update():
    actor = _make_actor(schedule=None)
    phases = _run_step(actor, updates=3)
    assert phases == [None, None, None]
    assert all(
        g["lr"] == pytest.approx(LR)
        for g in actor.optimizer.param_groups
        if g.get("name") in (PARAM_GROUP_SLOT_A, PARAM_GROUP_SLOT_B)
    )
    assert actor._slot_step_metrics()["slot/phase_is_A"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# the once-per-step metrics
# --------------------------------------------------------------------------------------


def _expected_keys(n_slots):
    # slot/alt_a_frac_cfg is the CONFIGURED A share for the step. It is here for the
    # same reason every other key is: the set must be identical on every rank, and this
    # one is derived from the schedule string, which comes from config. It is also what
    # makes a phase_is_A of 0.0 readable -- healthy under a pure-"B" anneal stage, dead
    # mechanism under anything else.
    keys = {"slot/orth_err", "slot/phase_is_A", "slot/alt_a_frac_cfg"}
    keys |= {f"slot/dw_norm_{k}" for k in range(n_slots)}
    keys |= {f"slot/cos_{s}_{t}" for s in range(n_slots) for t in range(s + 1, n_slots)}
    return keys


def test_collected_diagnostics_land_in_the_metrics_dict():
    actor = _make_actor(ranks=(2, 3, 4))
    metrics = _step_metrics(actor, updates=3)
    assert set(metrics) == _expected_keys(3)
    assert metrics["slot/orth_err"] < 1e-3, "Ā is not orthonormal"
    assert all(metrics[f"slot/dw_norm_{k}"] > 0 for k in range(3))
    assert all(
        abs(metrics[f"slot/cos_{s}_{t}"]) < 1.0 for s, t in [(0, 1), (0, 2), (1, 2)]
    )
    assert not any(math.isnan(v) for v in metrics.values())


def test_route_fallback_frac_is_not_emitted_a_second_time():
    """Task 9 already emits it per micro-batch; a per-step copy would mix two means."""
    actor = _make_actor()
    metrics = _step_metrics(actor, updates=3)
    assert "slot/route_fallback_frac" not in metrics


@pytest.mark.parametrize("routing", [(0, 0), (1, 2), (-1, -1), (2, 0)])
def test_the_key_set_does_not_depend_on_what_the_batch_contained(routing):
    """A rank-dependent key set deadlocks all_reduce_dict; this repo has hit that."""
    actor = _make_actor()
    actor._slot_step_begin(2)
    actor.optimizer.zero_grad()
    _backward(actor, ids=routing)
    actor._slot_alt_before_update(is_last_update=False)
    actor.optimizer.step()
    actor._slot_alt_after_update()
    assert set(actor._slot_step_metrics()) == _expected_keys(3)


def test_an_uncollected_reading_is_nan_but_the_keys_stay():
    """Second read with no re-arm: the values go missing, the key set must not."""
    actor = _make_actor()
    first = _step_metrics(actor, updates=3)
    second = actor._slot_step_metrics()
    assert set(second) == set(first)
    assert math.isnan(second["slot/orth_err"])
    assert math.isnan(second["slot/dw_norm_0"])
    assert not math.isnan(second["slot/phase_is_A"]), (
        "the phase is counted by the actor, not read off the model"
    )


def test_phase_is_a_reports_the_share_of_updates_that_trained_a():
    actor = _make_actor()
    phases = _run_step(actor, updates=3)
    metrics = actor._slot_step_metrics()
    expected = sum(p != "B" for p in phases) / len(phases)
    assert metrics["slot/phase_is_A"] == pytest.approx(expected)
    # over several steps it settles near the documented 2/9 (a third of the updates
    # that are not the forced step end)
    seen = []
    for _ in range(6):
        seen.extend(_run_step(actor, updates=3))
    assert 0.1 < sum(p == "A" for p in seen) / len(seen) < 0.4


def test_orth_err_is_read_on_the_fp32_side_even_for_a_bf16_model():
    """A bf16 reading of a healthy Ā floors around 1.9e-2 and cannot resolve anything.

    Measured in SlotProj: as cond(Z) goes 10 -> 20 the fp32 error moves 6.8e-5 ->
    1.07e-3 while the bf16 one moves 0.01887 -> 0.01894. Anything above the bf16 floor
    here means the metric is reading the model's own low-precision Ā.
    """
    actor = _make_actor(dtype=torch.bfloat16)
    metrics = _step_metrics(actor, updates=1)
    assert metrics["slot/orth_err"] < 1e-3


def test_metrics_are_absent_before_any_step_but_the_actor_survives():
    actor = _make_actor()
    metrics = actor._slot_step_metrics()
    assert set(metrics) == _expected_keys(3)
    assert math.isnan(metrics["slot/phase_is_A"])


# --------------------------------------------------------------------------------------
# the gate's range check: same guarantee, no device sync on the list path
# --------------------------------------------------------------------------------------

_SYNCING = {"any", "all", "__bool__", "item", "tolist", "nonzero", "equal"}


class _SyncSpy(TorchFunctionMode):
    """Records every torch function called inside the block."""

    def __init__(self):
        self.seen = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.seen.append(getattr(func, "__name__", str(func)))
        return func(*args, **(kwargs or {}))

    @property
    def syncs(self):
        return [name for name in self.seen if name in _SYNCING]


def test_make_slot_ids_rejects_an_out_of_range_id_on_the_host():
    with pytest.raises(ValueError, match=r"trains nothing"):
        make_slot_ids([0, 1, 7], 2)
    with pytest.raises(ValueError, match=r"trains nothing"):
        make_slot_ids([-2], 2)
    assert make_slot_ids([0, 1, -1], 2).tolist() == [0, 1, -1]
    assert make_slot_ids([], 2).shape == (0,)


def test_make_slot_ids_refuses_a_tensor():
    with pytest.raises(ValueError, match="list"):
        make_slot_ids(torch.zeros(3, dtype=torch.long), 2)


def test_a_checked_routing_installs_without_a_single_sync():
    gate = SlotGate(4, strict=True)
    ids = make_slot_ids([0, 3, -1, 2], 4)
    spy = _SyncSpy()
    with spy:
        with gate.scoped(ids):
            pass
    assert spy.syncs == [], (
        f"the routing install still forces a device sync: {spy.syncs}. On CUDA this "
        "fires once per micro-batch, on the path this repo has measured at 5.6x."
    )


def test_a_bare_tensor_is_still_range_checked():
    gate = SlotGate(2, strict=True)
    spy = _SyncSpy()
    with pytest.raises(ValueError, match=r"trains nothing"):
        with spy:
            with gate.scoped(torch.tensor([0, 7], dtype=torch.long)):
                pass
    good = torch.tensor([0, 1, -1], dtype=torch.long)
    spy = _SyncSpy()
    with spy:
        with gate.scoped(good):
            pass
    assert "any" in spy.syncs, "the tensor path must keep paying for its own check"


def test_a_mark_from_a_different_slot_count_is_not_trusted():
    ids = make_slot_ids([0, 3], 4)
    gate = SlotGate(2, strict=True)  # the model really has two slots
    with pytest.raises(ValueError, match=r"trains nothing"):
        with gate.scoped(ids):
            pass


def test_a_derived_tensor_loses_the_mark_and_pays_the_check():
    ids = make_slot_ids([0, 1, 0, 1], 2)
    gate = SlotGate(2, strict=True)
    spy = _SyncSpy()
    with spy:
        with gate.scoped(ids[1:]):
            pass
    assert "any" in spy.syncs, (
        "a sliced routing kept the mark; the mark must never outlive the exact tensor "
        "whose values were checked"
    )


def test_the_other_contract_checks_still_fire():
    gate = SlotGate(2, strict=True)
    for bad in (torch.zeros(2), torch.zeros((2, 2), dtype=torch.long), [0, 1]):
        with pytest.raises(ValueError):
            with gate.scoped(bad):
                pass
    with pytest.raises(ValueError, match="num_slots"):
        with SlotGate(None, strict=False).scoped(make_slot_ids([0], 2)):
            pass


# --------------------------------------------------------------------------------------
# the actor's own routing produces checked ids (the normal path pays nothing)
# --------------------------------------------------------------------------------------


class _FakeTokenizer:
    def __init__(self, texts):
        self.texts = list(texts)

    def batch_decode(self, ids, skip_special_tokens=True):
        return list(self.texts)


class _FakeTeacher:
    def __call__(self, forward_inputs, **kwargs):
        n = forward_inputs["input_ids"].shape[0]
        return {"logprobs": torch.zeros(n, 3)}


def _routing_actor(texts):
    suites = ["libero_spatial", "libero_object"]
    actor = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create({"algorithm": {"adv_type": "opd"}})
    actor.teacher_prompt_to_suite = {
        "pick up the black bowl": "libero_spatial",
        "pick up the alphabet soup": "libero_object",
    }
    actor.teacher_suite_to_path = {s: f"/ckpt/{s}" for s in suites}
    actor.teacher_models = {f"/ckpt/{s}": _FakeTeacher() for s in suites}
    actor.teacher_model = next(iter(actor.teacher_models.values()))
    actor._route_tokenizer = _FakeTokenizer(texts)
    actor._slot_enabled = True
    actor._slot_order = tuple(suites)
    actor._slot_gate = SlotGate(len(suites), strict=True)
    actor._slot_gate_ids = None
    actor._slot_fallback = 0.0
    actor._route_ready = False
    actor.warnings = []
    actor.log_warning = actor.warnings.append
    actor.log_info = actor.warnings.append
    return actor


def test_the_actors_routing_reaches_the_gate_without_a_sync():
    actor = _routing_actor(["pick up the black bowl", "pick up the alphabet soup"])
    actor._route_prepare({"input_ids": torch.arange(8).reshape(2, 4)})
    assert actor._slot_gate_ids.tolist() == [0, 1]
    spy = _SyncSpy()
    with spy:
        with actor._slot_scope():
            pass
    assert spy.syncs == [], f"the hot path still syncs: {spy.syncs}"


# --------------------------------------------------------------------------------------
# structural: where the calls sit inside run_training
# --------------------------------------------------------------------------------------


def _run_training_ast():
    src = textwrap.dedent(inspect.getsource(EmbodiedFSDPActor))
    cls = ast.parse(src).body[0]
    defs = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_training"
    ]
    assert defs, "run_training not found"
    return defs[-1]


def _call_lines(fn, attr):
    return [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == attr
    ]


def test_the_freeze_brackets_the_optimizer_step():
    fn = _run_training_ast()
    steps = _call_lines(fn, "optimizer_step")
    before = _call_lines(fn, "_slot_alt_before_update")
    after = _call_lines(fn, "_slot_alt_after_update")
    assert steps and before and after
    assert max(before) < min(steps), (
        "the freeze runs after the optimizer step: both factors would update every "
        "time, and the frozen one's momentum would keep accumulating"
    )
    assert min(after) > max(steps), (
        "the lr is restored before the step it was zeroed for"
    )


def test_the_diagnostics_are_armed_before_the_forward_and_read_after_the_updates():
    fn = _run_training_ast()
    begin = _call_lines(fn, "_slot_step_begin")
    collect = _call_lines(fn, "_slot_step_metrics")
    forwards = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "model"
        and any(kw.arg == "compute_logprobs" for kw in n.keywords)
    ]
    steps = _call_lines(fn, "optimizer_step")
    assert begin and collect and forwards and steps
    assert max(begin) < min(forwards), (
        "the diagnostics are armed after the forward that was supposed to fill them"
    )
    assert min(collect) > max(steps), (
        "the metrics are read before the step's last update"
    )
