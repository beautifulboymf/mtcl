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

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ``_pinned_precision`` and ``_compute_dtype`` are imported across modules WITHIN this
# package on purpose. The diagnostic Gram below, and the merged ΔW further down, must be
# formed under the same "no autocast, no TF32" guard and the same fp32 FLOOR that
# :func:`orthogonalize` computes under, and a second copy of either rule living here
# could drift from the one it has to agree with. They stay underscored because they are
# package-internal precision plumbing, not something the rest of the repo should reach
# for -- and promoting them would mean editing ``orth.py`` to widen an API that has
# exactly two other callers.
from rlinf.models.slot_lora.orth import (
    _DEFAULT_NS_ITERS,
    _compute_dtype,
    _pinned_precision,
    orthogonalize,
)

# Stamp `make_slot_ids` leaves on the tensor it validated, naming the slot count it
# validated AGAINST. It is a plain Python attribute on that one tensor object, so
# nothing derived from it (a slice, a `.to(device)`, a clone) carries it -- which is
# the point: the only way to be trusted is to be the exact tensor whose values were
# read on the host. Forgetting the stamp costs a device sync; it can never cost the
# check itself.
_RANGE_CHECKED_ATTR = "_slot_ids_range_checked"


def _range_error(offenders: list[int], positions: list[int], num_slots: int) -> str:
    """The one message both range checks raise, so the two cannot drift apart."""
    more = "" if len(positions) <= 8 else f" (+{len(positions) - 8} more)"
    return (
        f"slot gate ids must be -1 (unrouted) or in [0, {num_slots}); got "
        f"{offenders[:8]} at positions {positions[:8]}{more}. An out-of-range id "
        "is NOT a no-op: it matches no slot, so the sample trains nothing at all "
        "-- byte for byte what -1 means -- and a router off-by-one or a "
        "task-id-to-slot map missing an entry would silently train nothing for a "
        "whole suite. Fix the router, or widen num_slots if the model really has "
        "that many slots."
    )


def make_slot_ids(ids, num_slots: int, device=None) -> torch.Tensor:
    """Range-check a routing on the HOST, then copy it to ``device``. THE way to build one.

    ``match_suite_ids`` produces a plain ``list[int]``, and this is the only moment at
    which its values are readable for free: once the list is a CUDA tensor, the same
    check costs a device-to-host sync, and it would pay it once per micro-batch
    forever. This repo has measured ``run_training`` going 14.3 -> 80.0 min from
    roughly six syncs per micro-batch, so the check moves to the list and the tensor
    carries a stamp saying it was already done (see ``_RANGE_CHECKED_ATTR``).

    The GUARANTEE is unchanged, and deliberately so: an id that matches no slot is
    indistinguishable from ``-1`` ("no slot owns this sample") once it reaches
    :meth:`SlotOut.forward`, so a router off-by-one would silently train nothing for a
    whole suite. It is rejected here instead, one step earlier and one sync cheaper.

    Args:
        ids: The routing as plain Python ints, one per sample of THIS micro-batch;
            ``-1`` for a sample no slot owns. A ``torch.Tensor`` is refused -- reading
            one element-wise here is exactly the sync this exists to avoid, and a
            caller that already has a tensor should hand it straight to
            :meth:`SlotGate.scoped`, which checks it (with the sync) on the spot.
        num_slots: How many slots the model has. Every id must be in ``[-1, num_slots)``.
        device: Where the activations live; passed straight to ``torch.as_tensor``.
            ``None`` leaves it on the host.

    Returns:
        A 1-D ``torch.long`` tensor on ``device``, stamped as range-checked against
        ``num_slots``.

    Raises:
        ValueError: if ``num_slots`` is not a positive int, if ``ids`` is a tensor, or
            if any id is outside ``[-1, num_slots)``.
    """
    if isinstance(ids, torch.Tensor):
        raise ValueError(
            "make_slot_ids takes the plain list of ints the router produced, not a "
            "tensor: checking a tensor element-wise here is the device-to-host sync "
            "this function exists to remove. Pass a tensor to SlotGate.scoped "
            "directly -- it range-checks it there, sync and all."
        )
    if isinstance(num_slots, bool) or not isinstance(num_slots, int) or num_slots <= 0:
        raise ValueError(
            f"make_slot_ids needs a positive int num_slots; got {num_slots!r}. It is "
            "the bound every id is checked against, so there is no meaningful "
            "'unknown' here."
        )
    values = [int(v) for v in ids]
    bad = [i for i, v in enumerate(values) if v < -1 or v >= num_slots]
    if bad:
        offenders = sorted({values[i] for i in bad})
        raise ValueError(_range_error(offenders, bad, num_slots))
    tensor = torch.as_tensor(values, dtype=torch.long, device=device)
    setattr(tensor, _RANGE_CHECKED_ATTR, int(num_slots))
    return tensor


def _check_gate_ids(ids: Optional[torch.Tensor], num_slots: Optional[int]) -> None:
    """Validate a routing tensor against the contract in :class:`SlotGate`.

    Checked once per installation (i.e. once per micro-batch), never in the read
    path, which runs 200-400 times per forward. The RANGE check reads ``ids`` on the
    host, so on CUDA it costs one device-to-host sync -- unless the routing came from
    :func:`make_slot_ids`, which did that check on the plain Python list BEFORE the
    host-to-device copy and stamped the tensor to say so. The stamped path is the hot
    one (the actor's router); the sync below is what any other caller pays.

    Args:
        ids: The candidate routing, or ``None`` for ungated.
        num_slots: How many slots the model has, or ``None`` when the gate was built
            without one -- in which case a routing cannot be installed at all (see
            :class:`SlotGate`).

    Raises:
        ValueError: if ``ids`` is neither ``None`` nor a 1-D ``torch.long`` tensor, if
            a routing is offered to a gate with no slot count, or if any id is outside
            ``[-1, num_slots)``.
    """
    if ids is None:
        return
    if not isinstance(ids, torch.Tensor):
        raise ValueError(
            f"slot gate ids must be a LongTensor[B] or None; got "
            f"{type(ids).__name__}. Build it with torch.as_tensor(..., "
            "dtype=torch.long) on the device the activations live on."
        )
    if ids.dtype != torch.long:
        raise ValueError(
            f"slot gate ids must have dtype torch.long; got {ids.dtype}. Slot "
            "indices are compared for equality against integer slot numbers and "
            "-1 means 'unrouted', so a float or bool tensor cannot express the "
            "contract."
        )
    if ids.ndim != 1:
        raise ValueError(
            f"slot gate ids must be 1-D, one entry per sample in the micro-batch; "
            f"got shape {tuple(ids.shape)}."
        )
    if num_slots is None:
        raise ValueError(
            "cannot install a slot routing on a gate built without num_slots. A gate "
            "with no slot count cannot tell a valid slot id from a typo, and an "
            "out-of-range id behaves EXACTLY like the deliberate -1: the sample "
            "trains nothing, with no error, no NaN and a normal-looking loss curve. "
            "Pass the model's slot count when you create the gate "
            "(`SlotGate(num_slots=len(slot_ranks))`). A run that genuinely has no "
            "routing does not need one -- it installs None or uses `gate.ungated()`."
        )
    # Already checked on the host, against THIS slot count, by make_slot_ids. Compared
    # to num_slots and not merely present: a routing validated against a 4-slot model
    # says nothing about a 2-slot one, and trusting it there would reopen exactly the
    # hole (id 3 in a 2-slot model trains nothing, silently) that this check closes.
    # Checked AFTER the num_slots-is-None raise above, so a gate that cannot verify a
    # routing still refuses one rather than deferring to the stamp.
    checked = getattr(ids, _RANGE_CHECKED_ATTR, None)
    if isinstance(checked, int) and checked == num_slots:
        return

    # One host sync per micro-batch. `.any()` first so the happy path pays exactly one,
    # and the (rare) failure path pays a second to name the offenders.
    out_of_range = (ids < -1) | (ids >= num_slots)
    if bool(out_of_range.any()):
        positions = out_of_range.nonzero().flatten().tolist()
        offenders = sorted({int(v) for v in ids[out_of_range].tolist()})
        raise ValueError(_range_error(offenders, positions, num_slots))


class SlotGate:
    """Shared, mutable holder for the per-sample slot routing of one student model.

    WHAT IT CARRIES. ``ids`` is a ``LongTensor[B]`` giving the slot index that owns
    each sample. ``B`` is the MICRO-batch that reaches the gated forward -- the first
    dimension of the activation the module is handed -- NOT the global batch: with
    gradient accumulation the global batch is split, and each micro-batch has to
    install its own routing. ``-1`` marks a sample no slot owns: it still contributes
    to the forward VALUE (the acting policy is the fully merged model, so the value is
    always the full sum over slots) but sends gradient to no slot. dtype must be
    ``torch.long`` and the tensor must be 1-D; both are checked when the routing is
    installed. Putting ``ids`` on the same device as the activations is the CALLER's
    job -- this class never moves it, because a ``.to(device)`` in the read path would
    fire once per gated linear (200-400 times per forward) and each one is a fresh
    host-to-device copy.

    WHY AN OBJECT AND NOT A ContextVar. The routing has to reach hundreds of LoRA'd
    linears the caller never touches directly, so it travels out of band. It used to
    travel in a ``ContextVar``, and that is broken here: a fresh thread starts with an
    empty contextvars context and reads the DEFAULT (measured: ``None``), not the value
    the caller installed. The autograd engine runs backward nodes for CUDA tensors on
    per-device worker threads, and this repo enables gradient checkpointing
    (``fsdp_model_manager.py:253``), which RE-EXECUTES the wrapped forward during
    backward -- on one of those threads. The recomputed forward would therefore see no
    gate, every slot would take gradient from every sample, the isolation mechanism
    would be entirely off, and nothing would say so: no error, no NaN, normal-looking
    loss curves. Attribute access on an object the modules already hold a reference to
    crosses thread boundaries and recomputation unchanged.

    ONE INSTANCE PER MODEL. Injection creates a single :class:`SlotGate` and hands the
    SAME object to every gated module, so installing a routing is one attribute write
    rather than a walk over 200-400 modules.

    THE SLOT COUNT. ``num_slots`` is how many slots the model has; it is what makes an
    out-of-range id an ERROR instead of a silent no-op. Nothing downstream can catch
    one: :meth:`SlotOut.forward` compares ``ids == k`` for each existing slot, so an id
    of 7 in a 2-slot model matches nothing and the sample trains nothing -- byte for
    byte the behaviour of the deliberate ``-1``. Measured before this check existed, a
    2-slot model handed ``[0, 1, 7, 99]`` raised nothing and quietly trained on half its
    batch. Strict mode catches "no routing installed"; only the count catches "WRONG
    routing installed". It is validated in :meth:`scoped`, once per micro-batch, and
    never in the read path: the check reads ``ids`` on the host, and doing that inside
    the forward would force a device-to-host sync 200-400 times per step.

    A gate built WITHOUT a count (``num_slots=None``) refuses to install a routing at
    all. Skipping the check instead would be a silently reduced guarantee, which is the
    failure this class exists to prevent; and the case that legitimately has no count --
    a single-suite run with no routing -- never installs one, so it pays nothing. In
    other words: if you route, you must say how many slots you route among.

    STRICT MODE. ``strict=True`` (the default) makes :meth:`current` RAISE when no
    routing is installed, instead of quietly returning ``None`` and running ungated.
    That is the whole point: an ungated forward in a run that meant to be gated is a
    silent degradation, and this project has already lost a multi-hour run to a
    mechanism that was dead while its metric read a healthy-looking constant. A run
    that legitimately has no routing (single-suite distillation) constructs the gate
    with ``strict=False``; a run that needs one ungated forward inside an otherwise
    gated model uses :meth:`ungated`.

    THE SCOPE MUST COVER THE BACKWARD. Under gradient checkpointing the forward runs
    again during ``backward()``, so the routing has to still be installed then::

        with gate.scoped(ids):
            loss = student(**batch)
            loss.backward()

    Closing the scope before the backward leaves the recomputed forward with no
    routing. Under ``strict`` that raises; it is not silently ungated.
    """

    __slots__ = ("_ids", "num_slots", "strict")

    def __init__(self, num_slots: Optional[int] = None, strict: bool = True) -> None:
        """Create the holder for one model.

        Args:
            num_slots: How many slots the model has. Every installed routing is checked
                against it, so passing it is what turns an out-of-range slot id into an
                error rather than a sample that silently trains nothing. ``None`` means
                "unknown", and a gate that does not know cannot install a routing at
                all -- use it only for runs that never route.
            strict: When true, reading through :meth:`current` with no routing
                installed raises instead of returning ``None``. Turn it off only for
                runs that genuinely have no routing.

        Raises:
            ValueError: if ``num_slots`` is neither ``None`` nor a positive int.
        """
        # bool is an int subclass, so `SlotGate(True)` -- which READS as the strict
        # flag -- would otherwise bind to num_slots and quietly build a ONE-slot gate.
        if isinstance(num_slots, bool):
            raise ValueError(
                "SlotGate num_slots must be an int or None; got a bool. "
                "`SlotGate(True)` reads as the strict flag but binds to num_slots, "
                "where True would become a one-slot gate. Pass `strict=` by keyword."
            )
        if num_slots is not None:
            if not isinstance(num_slots, int):
                raise ValueError(
                    f"SlotGate num_slots must be an int or None; got "
                    f"{type(num_slots).__name__} ({num_slots!r})."
                )
            if num_slots <= 0:
                raise ValueError(
                    f"SlotGate num_slots must be positive; got {num_slots}. A gate "
                    "with no slots can own no sample, so every routing it accepted "
                    "would be entirely unrouted."
                )
        self._ids: Optional[torch.Tensor] = None
        self.num_slots = num_slots
        self.strict = bool(strict)

    def current(self) -> Optional[torch.Tensor]:
        """The routing for the forward running right now. THE accessor gated code uses.

        Returns:
            The installed ``LongTensor[B]``, or ``None`` when the gate is not strict
            and nothing is installed (the ungated case).

        Raises:
            RuntimeError: if the gate is strict and no routing is installed.
        """
        ids = self._ids
        if ids is None and self.strict:
            raise RuntimeError(
                "slot gate read with no slot routing installed. This gate is strict, "
                "so an unset read is an error rather than a silently UNGATED forward "
                "in which every slot takes gradient from every sample -- a failure "
                "that produces no error, no NaN and a normal-looking loss curve. "
                "Either keep the routing installed across the forward AND its "
                "backward (`with gate.scoped(ids): loss = model(...); "
                "loss.backward()`) -- gradient checkpointing re-runs the forward "
                "during backward -- or say the run is ungated on purpose, with "
                "`SlotGate(strict=False)` or a `gate.ungated()` window."
            )
        return ids

    def current_unchecked(self) -> Optional[torch.Tensor]:
        """The installed routing, or ``None``, never raising.

        The deliberate escape hatch, named so that reading ungated is a visible
        decision in a diff. Gated modules must call :meth:`current` instead; this is
        for code that only reports on the gate (metrics, logging, assertions) and must
        not blow up when there is nothing to report.

        Returns:
            The installed ``LongTensor[B]``, or ``None``.
        """
        return self._ids

    @contextmanager
    def scoped(self, ids: Optional[torch.Tensor]) -> Iterator["SlotGate"]:
        """Install a routing for the duration of the block, then restore the previous one.

        Restoring on the way out -- including on an exception -- is what stops a
        crashed forward from leaking its routing into the next one.

        Args:
            ids: ``LongTensor[B]`` naming the slot that owns each sample of THIS
                micro-batch, with ``-1`` for unrouted samples; or ``None``. Note that
                ``None`` does not mean "ungated" on a strict gate: reads still raise,
                because a router that produced nothing is a bug, not a decision. Use
                :meth:`ungated` to actually run ungated. Build it with
                :func:`make_slot_ids`, which range-checks the router's plain list
                before the host-to-device copy: the checks below then cost no device
                sync at all. A tensor from anywhere else is checked here instead, at
                one sync per micro-batch.

        Yields:
            This gate, for convenience.

        Raises:
            ValueError: if ``ids`` violates the contract (see :class:`SlotGate`) --
                wrong type, dtype or rank, an id outside ``[-1, num_slots)``, or any
                routing at all on a gate built without ``num_slots``. The previously
                installed routing is left untouched in every one of those cases.
        """
        _check_gate_ids(ids, self.num_slots)
        previous = self._ids
        self._ids = ids
        try:
            yield self
        finally:
            self._ids = previous

    @contextmanager
    def ungated(self) -> Iterator["SlotGate"]:
        """Run one block with no routing, even on a strict gate.

        For forwards that legitimately have nothing to route -- an eval or rollout
        pass, a warm-up -- inside a model whose training forwards are gated. Both the
        routing and the strict flag are restored on the way out, including on an
        exception.

        Yields:
            This gate, for convenience.
        """
        previous_ids, previous_strict = self._ids, self.strict
        self._ids, self.strict = None, False
        try:
            yield self
        finally:
            self._ids, self.strict = previous_ids, previous_strict


class SlotProj(nn.Module):
    """Holds Z; produces the row-orthonormal Ā = (Z Zᵀ)^(-1/2) Z shared by every slot.

    Deliberately a LEAF module (no child modules, one ``.weight``): that is the exact
    test in ``rlinf/hybrid_engines/fsdp/utils.py:306``, so FSDP gives it its own flat
    parameter. Two consequences we depend on: requires_grad is uniform inside that flat
    param (required with ``use_orig_params=False``), and the FULL Z is all-gathered when
    this module's forward runs -- :func:`orthogonalize` needs every row at once.

    THE PRICE OF THAT WRAP is that this module may not carry EXTRA STATE. Torch 2.6's
    ``checkpoint.state_dict._get_fqns`` cannot walk to an ``_extra_state`` key through
    an FSDP unit -- it raises ``AttributeError`` on the first weight sync -- so the
    ``scale`` this module owns is RECORDED one level up, by
    :meth:`SlotLoRALinear.get_extra_state`, which explains the whole constraint.

    The orthogonality of the slots lives entirely here. ``ΔW = Σ_k B_k Ā_k`` gives
    ``⟨ΔW_s, ΔW_t⟩_F = tr(B_sᵀB_t · Ā_tĀ_sᵀ)``, so mutually orthogonal ROW BLOCKS of Ā
    make the cross terms vanish whatever the B side does. This module owns Ā in one
    piece; the row blocks are carved out of it downstream.
    """

    def __init__(
        self,
        in_features: int,
        total_rank: int,
        scale: float,
        eps: float = 1e-6,
        iters: int = _DEFAULT_NS_ITERS,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """Build the Z parameter for one LoRA'd linear.

        Args:
            in_features: ``d_in`` of the linear this projects the input of.
            total_rank: ``R``, the summed rank of all slots. Must be
                ``<= in_features`` or no row-orthonormal Ā exists.
            scale: LoRA scaling applied to the projection output. Must be finite and
                strictly positive -- see the check below for why zero is not merely a
                degenerate value but a silently dead run. See also the note on the
                attribute below.
            eps: Passed through to :func:`orthogonalize` as the floor on the
                Frobenius norm of ``Z Zᵀ``.
            iters: Newton-Schulz iteration count, passed through to
                :func:`orthogonalize`. The default is converged at the production
                shape; the ONE documented remediation for a ``slot/orth_err`` that
                drifts into the 1e-3..5e-2 band is to raise it to 16 and change
                nothing else (measured at cond(Z)=20: 11 -> 6.8e-2, 12 -> 1.07e-3,
                13 -> 1.62e-4). It is bounded ABOVE as well: on a rank-deficient Z
                the null-space component grows 1.5x per iteration and reaches NaN
                around 50, so 16 is safe and 40+ is not.
            dtype: Parameter dtype; ``None`` uses the torch default.
            device: Parameter device; ``None`` uses the torch default.

        Raises:
            ValueError: if ``0 < total_rank <= in_features`` does not hold, if
                ``scale`` is not finite and strictly positive, or if ``iters`` is not
                strictly positive.
        """
        super().__init__()
        self.in_features = int(in_features)
        self.total_rank = int(total_rank)
        # Checked HERE, not at the first forward. orthogonalize enforces rows <= cols
        # too, but that fires inside the forward of one of the 200-400 instances of a
        # 7B model -- after every instance has been built, FSDP-wrapped and moved to
        # device -- for a fact that was already decided at construction. A bad rank is
        # a config error and has to cost a construction, not a model build.
        if not 0 < self.total_rank <= self.in_features:
            raise ValueError(
                f"SlotProj needs 0 < total_rank <= in_features; got "
                f"total_rank={self.total_rank}, in_features={self.in_features}. "
                "Ā = (Z Zᵀ)^(-1/2) Z has ORTHONORMAL ROWS, and with more rows than "
                "columns Z Zᵀ is rank-deficient, so no such Ā exists; a non-positive "
                "rank leaves no subspace for any slot at all."
            )
        # A plain float, not a buffer: it must stay assignable (``p.scale = 2.5``)
        # and must not acquire tensor semantics or be sharded/synchronized by FSDP.
        # It IS persisted -- but by the PARENT, through
        # :meth:`SlotLoRALinear.get_extra_state`, not from here: this module is
        # individually FSDP-wrapped, and torch 2.6 cannot walk to an ``_extra_state``
        # key through an FSDP unit at all (see there for the exact limitation).
        self.scale = float(scale)
        # Checked HERE, and not left to the caller, because s = 0 is the quietest
        # possible failure in this package. Measured with scale=0.0: the wrapper's
        # forward is BITWISE the frozen base's, both weight.grad tensors come back with
        # zero non-zero elements, and |ΔW|max is 0.0 -- forever. Nothing errors, nothing
        # goes NaN, and the loss curve is simply the base model's. The injection pass
        # derives s = sqrt(d_in) / ref_rank from config, so a missing key or a zero
        # ref_rank produces exactly that: a whole distillation run that is a no-op and
        # looks fine. SlotOut already refuses a non-positive slot rank and SlotGate a
        # non-positive slot count with the same reasoning -- "trains nothing while
        # still looking routed" -- and scale is the argument that got missed. NaN and
        # inf are refused for the mirror-image reason: they DO show up, but only as
        # a NaN loss 200-400 gated linears deep, where the config key that caused it
        # is no longer visible. Negative s is not rejected as meaningless (it is just a
        # sign absorbed by B) but as unreachable from any policy that computes it.
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(
                f"SlotProj needs a finite, strictly positive scale; got "
                f"scale={self.scale}. s multiplies every slot's contribution to ΔW, so "
                "s=0 makes the adapter output exactly zero and its gradients exactly "
                "zero for the whole run -- no error, no NaN, and the frozen base's own "
                "loss curve. The injection pass computes s = sqrt(d_in) / ref_rank, so "
                "check that ref_rank is set and non-zero in the config; a non-finite s "
                "means that arithmetic already produced nan/inf."
            )
        self.eps = float(eps)
        # Checked HERE for the same reason scale is: a non-positive count is not a
        # degenerate setting, it is a SILENT one. `_ns_loop` runs `range(iters)`, so at
        # iters <= 0 it never executes and returns X = I unchanged, making Ā =
        # Z / sqrt(‖Z Zᵀ‖_F) -- a scaled Z, with no orthogonality at all, no error and
        # no NaN. Every slot would then share subspace with every other and the one
        # mechanism this package exists to provide would be off while `slot/dw_norm_k`
        # kept moving for all K slots. There is no upper guard, because the failure at
        # the other end is LOUD either way -- a NaN, or an orth_err that blows past the
        # 5e-2 collapse threshold the training loop already watches -- and because the
        # safe ceiling depends on cond(Z) rather than on any constant this class could
        # hard-code; see the docstring for the measured 1.5^iters growth.
        self.iters = int(iters)
        if self.iters <= 0:
            raise ValueError(
                f"SlotProj needs a strictly positive Newton-Schulz iters; got "
                f"iters={self.iters}. The recurrence would not run at all, so Ā would "
                "come back as a rescaled Z -- NOT row-orthonormal -- with nothing "
                "raised and every slot silently overlapping every other. Set "
                "actor.model.slot_lora.orth_iters to 12 (the default) or, if "
                "slot/orth_err has entered the 1e-3..5e-2 band, to 16."
            )
        self.weight = nn.Parameter(
            torch.empty(self.total_rank, self.in_features, dtype=dtype, device=device)
        )
        # Gaussian, NOT orthogonal: see test_init_is_gaussian_not_orthogonal. std =
        # 1/sqrt(d_in) gives unit-ish row norms and a well-conditioned Z Zᵀ. The actual
        # scale is irrelevant to Ā (orthogonalize is scale-invariant).
        nn.init.normal_(self.weight, mean=0.0, std=self.in_features**-0.5)
        # Armed externally (by the training loop, on ONE module, once per step); the
        # forward disarms it again so a single arm costs a single extra NS call.
        self._collect_diag = False
        self._diag: Optional[dict] = None

    def arm_diag(self) -> None:
        """Ask the next :meth:`orth_weight` to record diagnostics; clear the old ones.

        Clearing is the point, and it is why arming has a method instead of being a
        bare ``_collect_diag = True``. ``_diag`` is only ever refreshed by a forward,
        so a module that is armed but never REACHED by one -- a frozen layer, a
        rollout-only step, a branch this batch did not take -- would keep LAST step's
        numbers, and a reader has no way to tell a stale reading from a fresh one.
        Measured before this existed: arm, forward, record, mutate the weight, re-arm
        without a forward, and ``_diag`` still held the old values. These numbers are
        plotted to answer "is this slot dead?", so a stale constant is the exact
        failure they must not produce. After this call, ``_diag is None`` means "not
        collected", full stop -- which is why every non-forward caller of
        :meth:`orth_weight` (there is one, :meth:`SlotLoRALinear.delta_weight`) passes
        ``collect=False``: a merge that happened to run between the arming and the
        forward would otherwise write ``_diag`` from a moment that is not the forward's,
        and disarm the A side so the forward never corrected it.
        """
        self._collect_diag = True
        self._diag = None

    def orth_weight(self, collect: bool = True) -> torch.Tensor:
        """Ā = (Z Zᵀ)^(-1/2) Z, differentiable w.r.t. Z, in the parameter's dtype.

        When armed (see :meth:`arm_diag`) this also refreshes ``self._diag`` with
        ``{"gram": Ā Āᵀ as (R, R) fp32, "scale": float}`` and disarms itself. ``gram``
        is the WHOLE ``(R, R)`` matrix -- this module does not know ``slot_ranks``, so
        the consumer carves out the block it wants; see :meth:`SlotOut._stash_diag`
        for the exact slicing and the ``scale**2`` factor that pair it with the B side.

        Args:
            collect: Whether this call may CONSUME an arming. ``True`` (the default) is
                the forward's behaviour. ``False`` is for callers that need ``Ā`` but
                are not the forward -- :meth:`SlotLoRALinear.delta_weight` -- and it
                touches neither ``_collect_diag`` nor ``_diag``.

                A parameter rather than a save/restore at the call site, because the
                two invariants at stake are both about state that must not be WRITTEN,
                and the only unbreakable way to promise that is to not write it: there
                is no window for an exception mid-merge to leave half-restored, and
                nothing to forget to unwind on a new call path. Restoring
                ``_collect_diag`` afterwards would not be enough anyway -- ``_diag``
                would already hold a reading taken outside a forward, which is exactly
                what :meth:`arm_diag` promises cannot happen.

        Returns:
            The row-orthonormal ``(R, d_in)`` matrix.
        """
        a = orthogonalize(self.weight, iters=self.iters, eps=self.eps)
        if collect and self._collect_diag:
            # The precision guard has to wrap the GRAM as well, not just the
            # orthogonalization. orthogonalize pins autocast and TF32 off internally,
            # but autocast intercepts per op, so an ambient bf16 autocast demotes the
            # bare matmul that FORMS the Gram right back down: measured on
            # SlotProj(64, 16, 1.0, dtype=bfloat16), ‖G - I‖_F 1.46e-6 (fp32) outside
            # autocast versus 4.61e-3 (bf16) inside it, ~3000x worse. TF32 leaks the
            # same way and is process-global. Both would land in the one number whose
            # entire job is to resolve a 6.8e-5 -> 1.07e-3 slide.
            with torch.no_grad(), _pinned_precision(self.weight.device.type):
                # The Gram MUST come from an fp32 recomputation, not from `a` cast down
                # to the forward dtype. Measured: as cond(Z) goes 10 -> 20 the fp32
                # orthogonality error degrades 16x (6.8e-5 -> 1.07e-3) while a bf16
                # reading moves only 0.01887 -> 0.01894 -- the rounding floor hides the
                # entire slide, and by the time bf16 moves (cond 30 -> 0.054) the result
                # is already garbage. Failure is a cliff, not a slope, so the sentinel
                # has to watch the slope. One extra NS call (~1.3 ms) on ONE module per
                # training step.
                a32 = orthogonalize(
                    self.weight.detach().float(), iters=self.iters, eps=self.eps
                )
                self._diag = {"gram": a32 @ a32.transpose(-2, -1), "scale": self.scale}
            self._collect_diag = False
        return a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the layer input onto the shared orthonormal basis.

        Args:
            x: Input of shape ``(..., d_in)``.

        Returns:
            ``s · Ā x`` of shape ``(..., R)`` -- the coordinates every slot's B
            side reads, sliced by row block downstream.
        """
        return F.linear(x, self.orth_weight() * self.scale)


class SlotOut(nn.Module):
    """Holds B; sums every slot's contribution, but routes gradient to one slot per sample.

    ``ΔW = Σ_k B_k Ā_k``. This module owns the B side as ONE ``(d_out, R)`` parameter
    whose k-th COLUMN block is ``B_k``, so the stacking already performs the sum: with
    ``h = s·Āx`` from :class:`SlotProj`, ``B h = Σ_k B_k (s·Ā_k x)``. Like
    :class:`SlotProj` it is deliberately a LEAF module -- no child modules, exactly one
    ``.weight`` -- because that is the test at ``rlinf/hybrid_engines/fsdp/utils.py:306``
    which gives it its own flat parameter. ``gate`` is stored as a plain attribute for
    the same reason: it is not a Parameter, Module or buffer, so it adds nothing FSDP
    would see.

    WHY THE GATE IS APPLIED TO THE CONTRIBUTION AND NOT TO ``h``. The obvious
    implementation -- zero out the row blocks of ``h`` that a sample's slot does not own
    -- does not isolate anything. ``B`` is a single parameter block, and for the j-th
    column block ``∂L/∂B_j = Σ_i outer(grad_out_i, h_{i,j})``: EVERY slot's B takes a
    gradient contribution from every sample, and multiplying ``h`` by a mask only
    changes the VALUE that lands in ``∂L/∂B_j``, never the fact that it lands there.
    Worse, the masked entries are exactly the ones set to zero, so the leak is
    ``outer(grad_out_i, 0) = 0`` for the masked blocks and full-strength for the owned
    one -- which looks correct for ``B`` and is still wrong for anything downstream, and
    it destroys the forward value, because a masked ``h`` no longer sums over all slots.

    So the gate is applied one level up, to the whole per-slot CONTRIBUTION
    ``c_k = B_k h_k``. ``torch.where(owns, c_k, c_k.detach())`` selects between two
    tensors holding the SAME BITS, so it is a no-op on the value and a cut on the
    graph: the owner's gradient is untouched -- not merely larger than the others' --
    and a sample that does not own slot k sends it nothing.

    WHAT THE ``where`` DOES NOT DO is remove the backward edge. It saves only its
    CONDITION, the edge to ``c_k`` survives, and every slot's backward matmul still runs
    over the whole micro-batch; what happens is that grad_output is SELECTED to zero at
    non-owner positions. So a non-owner's contribution is an exact NUMERICAL zero, not a
    structural one, and the gating saves no compute (measured: after an all-``-1``
    backward, ``weight.grad`` is an allocated all-zero tensor, not ``None``).

    BOTH CONSEQUENCES OF THAT SURVIVING EDGE ARE LOAD-BEARING. (a) ``weight.grad`` is
    ALWAYS allocated, so FSDP's post-backward hook always fires: a micro-batch that
    happens to contain no samples for some slot cannot desync the ranks. Measured with
    rank0 routed entirely to ``-1`` and rank1 routed normally, the backward completed
    with no desync and no hang; a structurally cut edge would have fired the hook on one
    rank and not the other. (b) Because the backward is a select rather than a multiply,
    a NaN sitting in grad_output at a non-owner position cannot leak into that slot
    (measured: injected NaN, slot 1's gradient stayed exactly zero and NaN-free).

    THE FORWARD VALUE IS ALWAYS THE FULL SUM OVER SLOTS. The acting policy is the fully
    merged model: there is no inference-time routing, no per-slot forward, and so no
    train/inference mismatch in the ROUTING to reason about. The gate exists only to
    decide which slot's B is allowed to LEARN from a given sample. The two code paths
    are not bit-identical, though: the gated path accumulates K per-slot terms where the
    ungated one does a single matmul, so they differ by floating-point summation order.
    Measured max|Δ| 8.882e-16 in fp64 and 9.375e-2 in bf16 against max|out| 2.05e1
    (relative ~4.6e-3, the bf16 rounding floor). Do not compare the two outputs for
    bitwise equality.

    Zero-initialized, so ``ΔW = 0`` at init and the student starts numerically identical
    to its base model.
    """

    def __init__(
        self,
        out_features: int,
        slot_ranks: "tuple[int, ...]",
        gate: SlotGate,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """Build the B parameter and the column layout of its slots.

        Args:
            out_features: ``d_out`` of the linear this is the B side of.
            slot_ranks: Per-slot rank, one entry per slot, in slot-index order. Slot
                ``k`` owns columns ``offsets[k] : offsets[k] + slot_ranks[k]``.
            gate: The model's single :class:`SlotGate`. Stored by reference and read
                inside every forward, so a routing installed once reaches every
                instance without walking the module tree.
            dtype: Parameter dtype; ``None`` uses the torch default.
            device: Parameter device; ``None`` uses the torch default.

        Raises:
            ValueError: if ``slot_ranks`` is empty, contains a non-positive rank, or
                has a different number of slots than ``gate.num_slots``.
        """
        super().__init__()
        self.out_features = int(out_features)
        ranks = tuple(int(r) for r in slot_ranks)
        # Checked HERE rather than at the first forward, for the same reason
        # SlotProj checks its rank at construction: a bad rank list is a config
        # error, and left to blow up in a forward it costs a full 7B model build,
        # an FSDP wrap and a device transfer before it says so.
        if not ranks:
            raise ValueError(
                "SlotOut needs at least one slot; got an empty slot_ranks. With no "
                "slots there are no columns in B, so ΔW is identically zero and the "
                "adapter cannot learn anything."
            )
        if any(r <= 0 for r in ranks):
            raise ValueError(
                f"SlotOut needs every slot rank to be strictly positive; got "
                f"{ranks}. A zero- or negative-rank slot owns no columns of B, so "
                "the samples routed to it would train nothing while still looking "
                "routed."
            )
        # The gate's range check is only as good as its slot count, so the count has to
        # AGREE with the module that consumes it. A gate saying 4 against a 3-slot B
        # would happily install an id of 3, which then matches no slot and trains the
        # sample nothing -- the exact silent failure num_slots exists to close, walking
        # back in through a different door. Build time, host side, once per module.
        if gate.num_slots is not None and gate.num_slots != len(ranks):
            raise ValueError(
                f"SlotOut was built with {len(ranks)} slots {ranks} but its gate was "
                f"built with num_slots={gate.num_slots}. The gate validates every "
                "routing against ITS count, so a disagreement means ids the gate "
                "accepts can match no slot here (training that sample nothing, "
                "silently) or slots here can never be routed to at all. One of the "
                "two is a config error; this is not reconciled here."
            )
        self.slot_ranks = ranks
        offsets, acc = [], 0
        for rank in ranks:
            offsets.append(acc)
            acc += rank
        # Plain tuples/ints, not tensors: they are layout metadata read on every
        # forward, and as buffers they would be sharded and synchronized by FSDP.
        self.offsets = tuple(offsets)
        self.total_rank = acc
        # A plain attribute. nn.Module.__setattr__ only intercepts Parameters,
        # Modules and buffers, and a SlotGate is none of those, so this adds no
        # child and no parameter and the FSDP leaf predicate still fires.
        self.gate = gate
        # ZERO, not Gaussian: ΔW = B Ā must be exactly zero at init so the student
        # starts identical to its base model. The randomness that keeps the two
        # factors from being stuck at a saddle lives on the Z side (SlotProj).
        self.weight = nn.Parameter(
            torch.zeros(self.out_features, self.total_rank, dtype=dtype, device=device)
        )
        # Armed externally (by the training loop, on ONE module, once per step); the
        # forward disarms it again, so a single arm costs a single extra measurement.
        self._collect_diag = False
        self._diag: Optional[dict] = None

    def arm_diag(self) -> None:
        """Ask the next :meth:`forward` to record diagnostics; clear the old ones.

        Same contract, and the same reason, as :meth:`SlotProj.arm_diag`: ``_diag`` is
        only refreshed by a forward, so an armed module the forward never reaches would
        otherwise keep last step's numbers with nothing to mark them stale -- in plots
        whose entire job is to answer "is this slot dead?". After this call,
        ``_diag is None`` means "not collected", full stop.
        """
        self._collect_diag = True
        self._diag = None

    def _stash_diag(self) -> None:
        """Record the B-side halves of the cross-slot interference measure.

        ``⟨ΔW_s, ΔW_t⟩_F = tr(B_sᵀB_t · Ā_tĀ_sᵀ) = Σ (B_tᵀB_s) ⊙ (Ā_tĀ_sᵀ)``, so the
        only thing needed from this side is the small ``(R_t, R_s)`` block ``B_tᵀB_s``,
        laid out to pair elementwise with the ``Ā_tĀ_sᵀ`` block :class:`SlotProj`
        contributes. ``ΔW`` itself is ``d_out x d_in`` and is NEVER materialized:
        forming it to measure it would cost more memory than the adapter saves, per
        module, per step.

        HOW A CONSUMER COMBINES THE TWO HALVES -- with the ``scale``, which neither half
        carries, since :meth:`SlotProj.forward` folds it into Ā::

            g = proj._diag["gram"]  # (R, R), the WHOLE matrix
            t_slice = slice(out.offsets[t], out.offsets[t] + out.slot_ranks[t])
            s_slice = slice(out.offsets[s], out.offsets[s] + out.slot_ranks[s])
            interference = (out._diag["cross"][(s, t)] * g[t_slice, s_slice]).sum()
            interference *= proj._diag["scale"] ** 2  # == ⟨ΔW_s, ΔW_t⟩_F

        Two things there are easy to get wrong and neither fails loudly. The Gram must
        be sliced ``[t_slice, s_slice]``, NOT ``[s_slice, t_slice]``: with equal
        per-slot ranks both slicings have the SAME SHAPE, so the wrong one yields a
        wrong number instead of a shape error. And the ``scale**2`` is not optional --
        it is the only place the LoRA scaling enters, and dropping it rescales every
        reported interference by a constant. :class:`SlotProj` does not know
        ``slot_ranks``, which is why the consumer does the slicing.

        Computed under ``no_grad`` and under the same ``_pinned_precision`` guard
        :meth:`SlotProj.orth_weight` uses. The guard is not optional: autocast
        intercepts per op, so an ambient bf16 autocast demotes the bare matmul that
        forms ``B_tᵀB_s`` even when the operands were cast to fp32 by hand -- measured
        on SlotProj's Gram at ~3000x worse resolution -- and TF32 leaks the same way
        and is process-global.
        """
        with torch.no_grad(), _pinned_precision(self.weight.device.type):
            blocks = [
                self.weight.detach()[:, s : s + r].float()
                for s, r in zip(self.offsets, self.slot_ranks)
            ]
            self._diag = {
                "b_norms": torch.stack([b.norm() for b in blocks]),
                "cross": {
                    (s, t): blocks[t].transpose(-2, -1) @ blocks[s]
                    for s in range(len(blocks))
                    for t in range(s + 1, len(blocks))
                },
            }
        self._collect_diag = False

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Sum every slot's contribution, keeping only the owner's on the gradient path.

        Args:
            h: The projected coordinates ``s·Āx`` from :class:`SlotProj`, of shape
                ``(B, ..., R)``. The last dim must be exactly ``total_rank`` and ``B``
                is the micro-batch, which must match the routing.

        Returns:
            ``Σ_k B_k h_k`` of shape ``(B, ..., d_out)`` -- the full sum over slots
            whatever the routing says, since the acting policy is the merged model.

        Raises:
            ValueError: if ``h`` is not ``total_rank`` wide, or if the installed
                routing does not have one entry per sample of ``h``. Both are host-side
                shape comparisons: no device sync, no measurable cost.
        """
        # Shared by BOTH paths on purpose. The gated path slices h[..., start:stop] per
        # slot, so it always reads the FIRST total_rank columns: an over-wide h ran fine
        # and silently DROPPED the surplus, while the ungated matmul raised. A SlotProj
        # whose total_rank disagrees with sum(slot_ranks) would then train to completion
        # and only blow up at eval, with the checkpoint's ΔW already wrong.
        if h.shape[-1] != self.total_rank:
            raise ValueError(
                f"SlotOut was handed h of width {h.shape[-1]} but its slots span "
                f"{self.total_rank} columns {self.slot_ranks}. h comes from SlotProj, "
                "so this means SlotProj.total_rank and sum(SlotOut.slot_ranks) "
                "disagree -- a build-time config error."
            )
        ids = self.gate.current()
        if self._collect_diag:
            self._stash_diag()
        if ids is None:
            # Ungated (eval, rollout, or a single-slot run): one matmul, no masks,
            # no graph surgery -- and bit-identical to the merged adapter.
            return F.linear(h, self.weight)
        # A raise, not an assert: `python -O` / PYTHONOPTIMIZE=1 STRIPS asserts, and
        # measured with them stripped, a length-1 routing against a batch of 4 assigned
        # all four samples to slot 0 while a length-4 routing against a batch of 1
        # broadcast the output from (1, d_out) to (4, d_out). A host-side shape compare
        # costs the same either way.
        if ids.shape[0] != h.shape[0]:
            raise ValueError(
                f"slot routing has {ids.shape[0]} entries but this forward was handed "
                f"{h.shape[0]} samples. The gate must be built from THIS micro-batch "
                "and installed before the student forward: with gradient accumulation "
                "the global batch is split, and each micro-batch needs its own routing."
            )
        # (B,) -> (B, 1, ..., 1) so it broadcasts against (B, ..., d_out) for any
        # number of intervening dims (a sequence axis, in practice).
        owns_shape = (-1,) + (1,) * (h.dim() - 1)
        out = None
        for k, (start, rank) in enumerate(zip(self.offsets, self.slot_ranks)):
            stop = start + rank
            contribution = F.linear(h[..., start:stop], self.weight[:, start:stop])
            owns = (ids == k).view(owns_shape)
            # Both branches hold the same bits, so this is a no-op on the VALUE. It
            # does not CUT the edge -- torch.where saves only its condition -- it
            # selects grad_output to zero at non-owner positions, so B_k and h_k
            # receive an exact numerical zero from those samples. See the class
            # docstring for why masking h instead does not work, and for why the
            # surviving edge is what keeps FSDP's post-backward hook firing.
            contribution = torch.where(owns, contribution, contribution.detach())
            out = contribution if out is None else out + contribution
        return out


class SlotLoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus K per-suite LoRA slots on mutually orthogonal subspaces.

    This is the only thing the rest of the model sees. It exists because the two slot
    modules above are each deliberately a LEAF -- that is what earns them their own FSDP
    flat parameter (``rlinf/hybrid_engines/fsdp/utils.py:306``) -- and a leaf cannot also
    own a base weight and a merge routine. So the composition lives here, one level up,
    where it costs nothing: this class has CHILDREN and no ``.weight``, so that predicate
    skips it, ``slot_A`` and ``slot_B`` are still wrapped individually, and the frozen
    ``base`` is left to fold into the enclosing transformer layer's flat parameter. That
    is the arrangement ``use_orig_params=False`` (this repo's default,
    ``rlinf/config.py:419``) needs, since it requires uniform ``requires_grad`` within a
    flat parameter: everything trainable sits in the two slot units, and what folds into
    the transformer layer is uniformly FROZEN -- as long as the injection pass freezes
    every non-slot parameter, which is exactly what it does.

    WHY IT REPLACES PEFT'S ``lora.Linear`` AT ALL. K distillation teachers writing into
    ONE shared LoRA block overwrite each other: the block has no structure that keeps
    suite k's update from moving suite j's function. Here ``ΔW = Σ_k B_k Ā_k`` with the
    row blocks of ``Ā`` mutually orthonormal, so ``⟨ΔW_s, ΔW_t⟩_F = tr(B_sᵀB_t ·
    Ā_tĀ_sᵀ)`` vanishes whatever the B side does, and :class:`SlotGate` keeps each
    sample's gradient inside its own slot. Only the STUDENT is built this way; the
    teachers and the dual-KL anchor keep using PEFT and are untouched.

    IT IS THE IDENTITY AT INITIALIZATION. ``B`` is zero, so ``ΔW`` is exactly zero and
    the student's first forward is its base model's, bit for bit -- not approximately.
    That is what makes injection safe to switch on mid-project without moving any
    baseline.

    THE MERGE IS THE OTHER HALF OF THIS CLASS. A checkpoint is worthless if it cannot be
    turned back into a plain HF model, and the conversion is
    ``W <- W + delta_weight().to(W.dtype)`` followed by swapping this module out for its
    ``base``; the cast is part of the contract, not tidiness (see :meth:`delta_weight`).
    That path is only correct if :meth:`delta_weight` reproduces what the forward
    actually computed, which is why it recomputes ``Ā`` from the stored ``Z`` through
    :meth:`SlotProj.orth_weight` -- the same call the forward makes -- rather than
    re-deriving it some other way. ``Z`` is what the checkpoint carries; ``Ā`` is never
    stored, and it is never a random seed to be replayed.
    """

    def __init__(
        self,
        base: nn.Linear,
        slot_ranks: "tuple[int, ...]",
        scale: float,
        gate: SlotGate,
        eps: float = 1e-6,
        iters: int = _DEFAULT_NS_ITERS,
    ):
        """Wrap one linear, freeze it, and give it its slots.

        Args:
            base: The ``nn.Linear`` to adapt. Taken over by reference and frozen in
                place; the slots inherit its dtype and device, so it must already be
                where and what the model wants it to be.
            slot_ranks: Per-slot rank, one entry per slot, in slot-index order. Their
                sum is the width of the shared orthonormal basis and must not exceed
                ``base.in_features``.
            scale: LoRA scaling ``s``, applied once, on the ``Ā`` side. The POLICY that
                picks the number (matching the row norm of PEFT's gaussian-initialized
                A, so ``ΔW`` moves at the same rate on step 1 and the learning rate is
                not a hidden variable in the comparison) lives in the injection pass;
                this class applies whatever it is handed and records it in the
                checkpoint via :meth:`get_extra_state`.
            gate: The model's single :class:`SlotGate`, shared by every gated module.
                Passed to :class:`SlotOut`, which reads it inside every forward.
            eps: Passed through to :func:`orthogonalize`.
            iters: Newton-Schulz iteration count, passed through to
                :func:`orthogonalize`. See :class:`SlotProj` for the escalation ladder
                and why it is bounded on both sides.

        Raises:
            ValueError: if ``base`` is not a childless ``nn.Linear`` with a
                materialized floating-point weight, if ``slot_ranks`` is empty or does
                not fit the input width, if ``scale`` is not finite and positive, if
                ``iters`` is not strictly positive, or if ``gate``'s slot count
                disagrees with ``slot_ranks``.
        """
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise ValueError(
                f"SlotLoRALinear wraps an nn.Linear; got {type(base).__name__}. The "
                "merge path assumes a (d_out, d_in) `weight` and an optional `bias`, "
                "so a module that stores its weight transposed (HF Conv1D) would merge "
                "into a silently wrong ΔW."
            )
        if list(base.named_children()):
            raise ValueError(
                f"SlotLoRALinear was handed a base with child module(s) "
                f"{[n for n, _ in base.named_children()]}. An nn.Linear SUBCLASS that "
                "keeps its adaptation in a child would hide a SECOND ΔW inside `base` "
                "that delta_weight() knows nothing about: training would work, and the "
                "merged checkpoint would silently drop it. Wrap the plain linear, "
                "before any other adapter. (The peft this repo pins, 0.11.1, does not "
                "reach here: its `lora.Linear` is not an nn.Linear subclass at all -- "
                "MRO Linear, Module, LoraLayer, BaseTunerLayer -- so it is refused by "
                "the isinstance check above. That was not true before peft 0.4, and "
                "any wrapper that subclasses nn.Linear still lands here.)"
            )
        # in_features is read below to size Z, and nn.LazyLinear -- also an nn.Linear
        # subclass, also childless -- reports 0 for it until its first forward. Left to
        # SlotProj that surfaced as "total_rank=8, in_features=0", which is true and
        # tells the caller nothing about what to fix.
        if base.in_features <= 0:
            raise ValueError(
                f"SlotLoRALinear needs a base with in_features > 0; got "
                f"in_features={base.in_features}. An nn.LazyLinear reports 0 until its "
                "first forward materializes its weight, and the slots have to be sized "
                "from the real width at construction. Run one forward through the base "
                "(or build a plain nn.Linear) before wrapping it."
            )
        ref = base.weight
        # bitsandbytes' Linear8bitLt and Linear4bit ARE nn.Linear subclasses with no
        # children, so they pass both guards above. Measured: construction then got as
        # far as nn.Parameter(torch.empty(..., dtype=uint8)) and died with an unrelated
        # RuntimeError about integer parameters. ΔW cannot be added into a quantized
        # weight at all, so this refuses it by name instead.
        if not ref.is_floating_point():
            raise ValueError(
                f"SlotLoRALinear needs a floating-point base weight; got "
                f"{ref.dtype}. The slots inherit the base's dtype and the merge adds "
                "ΔW straight into `base.weight`, neither of which is meaningful for a "
                "quantized (int8/uint8/nf4) weight. Wrap the linear before quantizing "
                "it, or de-quantize first."
            )
        ranks = tuple(int(r) for r in slot_ranks)
        # sum(()) is 0, and taking it before SlotProj is built turned "you asked for no
        # slots" into SlotProj's "total_rank=0, in_features=16" -- a rank complaint --
        # while SlotOut's much clearer message was unreachable from here.
        if not ranks:
            raise ValueError(
                "SlotLoRALinear needs at least one slot; got an empty slot_ranks. With "
                "no slots there are no columns in B, so ΔW is identically zero and the "
                "adapter cannot learn anything."
            )
        # The children are built BEFORE the base is frozen, so a config error they
        # reject (a rank wider than the input, a gate whose slot count disagrees) leaves
        # the caller's model exactly as it was rather than half frozen by a constructor
        # that then raised.
        self.base = base
        self.slot_A = SlotProj(
            base.in_features,
            sum(ranks),
            scale,
            eps,
            iters,
            dtype=ref.dtype,
            device=ref.device,
        )
        self.slot_B = SlotOut(
            base.out_features, ranks, gate, dtype=ref.dtype, device=ref.device
        )
        for param in self.base.parameters():
            param.requires_grad_(False)

    def arm_diag(self) -> None:
        """Arm the orthogonality/interference diagnostics on BOTH halves of this layer.

        Exposed here, rather than left to a consumer reaching into ``slot_A`` and
        ``slot_B``, because the two halves are only meaningful TOGETHER and only from
        the SAME layer: ``⟨ΔW_s, ΔW_t⟩_F`` pairs this layer's ``Ā_tĀ_sᵀ`` block with
        this layer's ``B_tᵀB_s``. A consumer that walks the model for "the first
        SlotProj" and "the first SlotOut" separately pairs them by module registration
        order, and in a 7B model full of same-shaped projections a mismatched pair has
        the right shapes and yields a WRONG NUMBER rather than an error. One call on one
        layer removes that failure mode entirely.

        Both halves clear their previous readings; see :meth:`SlotProj.arm_diag` for why
        arming must never leave a stale one behind.
        """
        self.slot_A.arm_diag()
        self.slot_B.arm_diag()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``base(x) + Σ_k B_k (s · Ā_k x)``.

        The routing is NOT an argument: it reaches :class:`SlotOut` through the shared
        gate the model installed for this micro-batch, because this layer is called by
        code (a HuggingFace attention block) that knows nothing about slots.

        Args:
            x: Input of shape ``(B, ..., d_in)``, where ``B`` is the micro-batch the
                installed routing describes.

        Returns:
            The adapted output, of shape ``(B, ..., d_out)``.
        """
        return self.base(x) + self.slot_B(self.slot_A(x))

    def get_extra_state(self) -> dict:
        """Record the scale the checkpoint was trained with.

        ``scale`` multiplies every slot's contribution to ``ΔW``, so a converter that
        merges the slots back into the base weights has to use the SAME value training
        used. Nothing else in the checkpoint pins it: ``Ā`` is invariant to the scale of
        ``Z`` (see :func:`orthogonalize`), so the only surviving trace of ``s`` would be
        in the B side, and it is not separable there. A converter left to re-derive
        ``s`` from CLI flags that disagree with the training config produces a ``ΔW``
        off by a constant factor: the merge succeeds, the model runs, and the adapter is
        simply the wrong size, with no error anywhere. Writing it down is what turns
        that into something the converter can VERIFY.

        IT LIVES HERE, ON THE WRAPPER, AND NOT ON :class:`SlotProj` THAT OWNS ``scale``.
        That is forced by torch, not chosen for tidiness. ``SlotProj`` is individually
        FSDP-wrapped (the per-leaf policy at
        ``rlinf/hybrid_engines/fsdp/utils.py:303-311`` fires on it, deliberately -- it
        is what all-gathers the FULL ``Z`` for the module's own forward, which
        :func:`orthogonalize` requires), and
        ``torch.distributed.checkpoint.state_dict._get_fqns`` walks a state-dict key
        component by component with a per-wrapper branch for each: its ``else`` branch
        special-cases ``_extra_state`` and stops, but its FSDP branch (torch 2.6,
        ``state_dict.py:205-212``) does NOT, and ends on an unconditional
        ``getattr(curr_obj, "_extra_state")``. So a module that is both individually
        FSDP-wrapped AND carries extra state cannot be reached by
        ``get_model_state_dict`` at all -- it raises ``AttributeError: 'SlotProj' object
        has no attribute '_extra_state'`` on the FIRST weight sync, which is exactly how
        a 2-GPU smoke test died. ``SlotLoRALinear`` has child modules and no ``.weight``,
        so the per-leaf predicate does not fire on it: it is absorbed into the enclosing
        transformer layer's FSDP unit, the walk reaches it through the ``else`` branch,
        and the key is handled correctly. ``tests/unit_tests/test_slot_lora_fsdp.py``
        holds that shape in place under a real ``FullyShardedDataParallel``; moving this
        pair back down one level re-breaks every sync.

        Carried as extra state rather than a buffer on purpose: a buffer would be
        sharded and synchronized by FSDP, would drag tensor semantics into a
        configuration constant, and would break direct assignment
        (``layer.slot_A.scale = 2.5``). The repo's weight-sync paths already skip
        ``_extra_state`` keys (``bucket_syncer._bucket_key``,
        ``fsdp_model_manager.divide_model_to_bucket``), so this reaches checkpoints
        without reaching the rollout weight transfer.

        ``eps`` and ``total_rank`` are deliberately NOT recorded. ``total_rank`` is
        already ``slot_A.weight.shape[0]``, which the state dict carries: a mismatch
        raises a shape error on load, and duplicating it would create a second source of
        truth that can disagree with the tensor. ``eps`` only floors ``‖Z Zᵀ‖_F``
        (production sits ~7 orders of magnitude above it), so it cannot silently rescale
        a merged ``ΔW`` -- the one failure this exists to close -- and checking it would
        manufacture load failures for a knob that carries no risk.

        Returns:
            ``{"scale": float}``, read live off ``slot_A``. A dict rather than a bare
            float so more fields can be added later without breaking readers.
        """
        return {"scale": self.slot_A.scale}

    def set_extra_state(self, state: dict) -> None:
        """Verify the checkpoint's scale against this layer's, never reconcile.

        A disagreement here is a disagreement between the config that trained the
        checkpoint and the config loading it, and only a human knows which one is right.
        Overwriting from the checkpoint would silently make the run stop matching its
        own config; ignoring the checkpoint would make the recorded value decorative and
        reinstate the failure above. So this raises, naming both values, and leaves the
        choice where it belongs.

        Absence is a different case and is NOT an error here: a state dict with no
        ``_extra_state`` makes no claim about ``s``, so there is nothing to contradict
        and the layer keeps its configured value. That case never reaches this method --
        ``nn.Module._load_from_state_dict`` reports the missing key instead, and only
        when ``strict=True``, which is the right place for it. A checkpoint written
        before this moved up carries the record under ``slot_A._extra_state``, so it
        arrives as one missing key plus one unexpected key per adapted layer, and the
        converter's key diff refuses it by name rather than merging a scale nobody
        checked.

        Args:
            state: The object returned by :meth:`get_extra_state` at save time.

        Raises:
            ValueError: if ``state`` is not a mapping carrying ``"scale"``, or if the
                scale it carries differs from this layer's.
        """
        if not isinstance(state, dict) or "scale" not in state:
            raise ValueError(
                f"SlotLoRALinear extra state must be a dict carrying 'scale'; got "
                f"{state!r}. The checkpoint was not written by this class."
            )
        ckpt_scale = float(state["scale"])
        if ckpt_scale != self.slot_A.scale:
            raise ValueError(
                f"SlotLoRALinear scale mismatch: the checkpoint was trained with "
                f"scale={ckpt_scale!r} but this module is configured with "
                f"scale={self.slot_A.scale!r}. Every merged delta-W would come out off "
                "by that ratio and nothing downstream would report it. Fix the "
                "config to match the checkpoint (or vice versa); this is not "
                "reconciled here."
            )

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        """``s · B Ā`` -- exactly what the merge adds into the base weight.

        THE ONE INVARIANT: this must equal what the forward computed, or every converted
        checkpoint is silently wrong -- it still loads, still runs, and is simply a
        different model than the one that was trained. So ``Ā`` is recomputed from the
        stored ``Z`` through :meth:`SlotProj.orth_weight`, the same call
        :meth:`SlotProj.forward` makes, rather than through a "more accurate" fp32
        re-derivation of ``Ā`` from the same ``Z``. Those two are NOT the same matrix:
        ``orth_weight`` returns in the PARAMETER's dtype, so in production the forward
        consumes a bf16 ``Ā`` whose entries sit at the bf16 rounding floor, 1.66e-3
        relative Frobenius away from the fp32 one (measured at R=256, 3 seeds: 1.661e-3
        to 1.670e-3 at d_in=1024 and 1.662e-3 to 1.664e-3 at d_in=4096, i.e. the same
        floor at both widths), and that difference passes 1:1 into ``ΔW``.

        THE FP32 ``Ā`` IS THE MORE ACCURATE MATRIX AND THE WRONG ONE TO MERGE. Measured
        against the adapter output the TRAINING forward actually produced, this route is
        closer at every width and seed: 3.70e-3 relative versus 4.05e-3 for the fp32
        re-derivation (R=256, bf16; 3.69-3.71e-3 vs 4.04-4.07e-3 across 3 seeds at both
        widths). Neither is zero, because the training forward rounds ``h`` to bf16
        between the two matmuls and this one does not; the point is that reproducing the
        trained model beats improving on it. Downstream of the merge that margin is
        swallowed anyway -- ``ΔW`` is rounded into the bf16 base weight (1.6e-3 to 1.8e-3
        relative on ``W`` by itself), leaving the two routes indistinguishable at the
        merged model's OUTPUT (3.38e-3 vs 3.36e-3 at d_in=1024, 3.48e-3 vs 3.51e-3 at
        4096) -- so this choice buys correctness of definition, not accuracy.

        PRECISION IS A FLOOR, NOT A CAST. The product is formed no NARROWER than fp32,
        which is what matters in production (bf16 ``B @ Ā`` at R=256 is not a merge, it
        is a second rounding), but a parameter that is already WIDER keeps its own dtype
        rather than being thrown away -- the same convention :func:`orthogonalize` uses,
        which is what lets the fp64 tests assert this identity to 1e-12 instead of to
        fp32 slop. The whole thing runs under ``_pinned_precision``: autocast intercepts
        per op, so an ambient bf16 region would demote the ``B @ Ā`` matmul even though
        both operands were widened by hand, and TF32 (process-global, and turned on
        elsewhere in this repo) would leave that matmul with 10 mantissa bits.

        THE CALLER OWNS THE CAST. This returns fp32 or wider while ``base.weight`` is
        bf16 in production, so the merge is ``W.data.add_(layer.delta_weight().to(
        W.dtype))`` and the ``.to`` is part of the contract, not a tidiness. Measured:
        ``W.data.add_(layer.delta_weight())`` does NOT raise -- ``add_`` downcasts the
        operand to the destination dtype and lands on the same answer -- so a converter
        that omits it is lucky rather than correct, and stays lucky only for as long as
        the two dtypes happen to relate that way.

        IT DIVERGES FROM THE FORWARD IN ONE CONFIGURATION. Everything above assumes the
        forward orthogonalized the same ``Z`` this reads, which holds whenever the
        parameter dtype IS the model dtype -- true for this repo's shipped configs
        (``examples/embodiment/config/training_backend/fsdp.yaml:23`` leaves
        ``param_dtype: null``, so parameters stay at ``model.precision``). It stops
        holding under ``model.precision: fp32`` plus ``mixed_precision.param_dtype:
        bf16`` -- the CPU-offload arrangement
        ``examples/sft/config/libero_sft_dreamzero.yaml:48`` describes -- and equally
        under ``fsdp_config.amp_autocast.enabled: True``: the training forward then
        orthogonalizes a bf16 ``Z`` while the checkpoint hands the converter the fp32
        master copy, and this method silently becomes the fp32 rederivation its own
        argument above rejects. The result is still a valid merge, ~4e-3 relative rather
        than ~3.7e-3 away from the trained adapter, but it is no longer the SAME route
        the forward took. Nothing detects it; if such a config is ever used, the merge
        has to cast ``Z`` down before orthogonalizing.

        WHAT IT DOES NOT COVER IS THE BIAS. ``ΔW`` is weight-only; the bias lives on
        ``base`` and the merge carries it over unchanged, exactly once.

        IT NEEDS THE WHOLE ``Z``. ``Ā = (Z Zᵀ)^(-1/2) Z`` is not computable from a slice
        of ``Z``, so this must run on an UNSHARDED module: the converter's plain model,
        or inside ``summon_full_params``. Under FSDP with ``use_orig_params=False`` the
        slot parameters carry sharded views outside the module's own forward, and
        orthogonalizing a fragment of ``Z`` is a different matrix, not an approximation
        of the right one.

        Returns:
            ``(d_out, d_in)``, detached, in fp32 or wider. Exactly zero at
            initialization.
        """
        compute_dtype = _compute_dtype(self.slot_A.weight.dtype)
        with _pinned_precision(self.slot_A.weight.device.type):
            # The scale is folded into Ā in the PARAMETER's dtype, before widening,
            # because that is where SlotProj.forward folds it (``orth_weight() *
            # self.scale``). It is not cosmetic at every width: with the match_mt4
            # policy s = sqrt(d_in)/128, which at d_in 1024 and 4096 is 0.25 and 0.5 --
            # exact powers of two, so bf16 rounds nothing and the two placements are
            # BITWISE identical. At any other width it rounds: measured relative error
            # against the training forward's adapter output, folding it in first gives
            # 3.78e-3 vs 4.13e-3 at d_in=2048 and 3.70e-3 vs 4.07e-3 at d_in=11008
            # (llama-7b's MLP width), the same margin as the Ā route above and for the
            # same reason.
            # collect=False: this is not a forward, and an arm_diag() waiting for one
            # must survive a merge that happens to run in between. Consuming it here
            # would pair a Gram taken at THIS moment with a cross block taken after the
            # next optimizer step, and would write `_diag` from a non-forward, which is
            # precisely what SlotProj.arm_diag promises cannot happen.
            a = (self.slot_A.orth_weight(collect=False) * self.slot_A.scale).to(
                compute_dtype
            )
            return self.slot_B.weight.to(compute_dtype) @ a
