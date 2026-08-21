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

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ``_pinned_precision`` is imported across modules WITHIN this package on purpose. The
# diagnostic Gram below must be formed under the same "no autocast, no TF32" guard that
# :func:`orthogonalize` computes under, and a second copy of that guard living here
# could drift from the one it has to agree with. It stays underscored because it is
# package-internal precision plumbing, not something the rest of the repo should reach
# for -- and promoting it would mean editing ``orth.py`` to widen an API that has
# exactly one other caller.
from rlinf.models.slot_lora.orth import _pinned_precision, orthogonalize


def _check_gate_ids(ids: Optional[torch.Tensor]) -> None:
    """Validate a routing tensor against the contract in :class:`SlotGate`.

    Checked once per installation (i.e. once per micro-batch), never in the read
    path, which runs 200-400 times per forward.

    Args:
        ids: The candidate routing, or ``None`` for ungated.

    Raises:
        ValueError: if ``ids`` is neither ``None`` nor a 1-D ``torch.long`` tensor.
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

    __slots__ = ("_ids", "strict")

    def __init__(self, strict: bool = True) -> None:
        """Create the holder for one model.

        Args:
            strict: When true, reading through :meth:`current` with no routing
                installed raises instead of returning ``None``. Turn it off only for
                runs that genuinely have no routing.
        """
        self._ids: Optional[torch.Tensor] = None
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
                :meth:`ungated` to actually run ungated.

        Yields:
            This gate, for convenience.

        Raises:
            ValueError: if ``ids`` violates the contract (see :class:`SlotGate`). The
                previously installed routing is left untouched in that case.
        """
        _check_gate_ids(ids)
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
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """Build the Z parameter for one LoRA'd linear.

        Args:
            in_features: ``d_in`` of the linear this projects the input of.
            total_rank: ``R``, the summed rank of all slots. Must be
                ``<= in_features`` or no row-orthonormal Ā exists.
            scale: LoRA scaling applied to the projection output. See the note on
                the attribute below.
            eps: Passed through to :func:`orthogonalize` as the floor on the
                Frobenius norm of ``Z Zᵀ``.
            dtype: Parameter dtype; ``None`` uses the torch default.
            device: Parameter device; ``None`` uses the torch default.

        Raises:
            ValueError: if ``0 < total_rank <= in_features`` does not hold.
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
        # It IS persisted, via get_extra_state/set_extra_state -- see there for why
        # the checkpoint has to carry it.
        self.scale = float(scale)
        self.eps = float(eps)
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

    def orth_weight(self) -> torch.Tensor:
        """Ā = (Z Zᵀ)^(-1/2) Z, differentiable w.r.t. Z, in the parameter's dtype.

        Returns:
            The row-orthonormal ``(R, d_in)`` matrix. Also refreshes ``self._diag``
            and disarms ``self._collect_diag`` when armed.
        """
        a = orthogonalize(self.weight, eps=self.eps)
        if self._collect_diag:
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
                a32 = orthogonalize(self.weight.detach().float(), eps=self.eps)
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

    def get_extra_state(self) -> dict:
        """Record the scale the checkpoint was trained with.

        ``scale`` multiplies every slot's contribution to ``ΔW``, so a converter
        that merges the slots back into the base weights has to use the SAME
        value training used. Nothing else in the checkpoint pins it: ``Ā`` is
        invariant to the scale of ``Z`` (see :func:`orthogonalize`), so the only
        surviving trace of ``s`` would be in the B side, which this module does
        not own. A converter left to re-derive ``s`` from CLI flags that
        disagree with the training config produces a ``ΔW`` off by a constant
        factor: the merge succeeds, the model runs, and the adapter is simply the
        wrong size, with no error anywhere. Writing it down is what turns that
        into something the converter can VERIFY.

        Carried as extra state rather than a buffer on purpose: a buffer would be
        sharded and synchronized by FSDP, would drag tensor semantics into a
        configuration constant, and would break direct assignment
        (``p.scale = 2.5``). The repo's weight-sync paths already skip
        ``_extra_state`` keys (``bucket_syncer._bucket_key``,
        ``fsdp_model_manager.divide_model_to_bucket``), so this reaches
        checkpoints without reaching the rollout weight transfer.

        ``eps`` and ``total_rank`` are deliberately NOT recorded. ``total_rank``
        is already ``weight.shape[0]``, which the state dict carries: a mismatch
        raises a shape error on load, and duplicating it would create a second
        source of truth that can disagree with the tensor. ``eps`` only floors
        ``‖Z Zᵀ‖_F`` (production sits ~7 orders of magnitude above it), so it
        cannot silently rescale a merged ``ΔW`` -- the one failure this exists to
        close -- and checking it would manufacture load failures for a knob that
        carries no risk.

        Returns:
            ``{"scale": float}``. A dict rather than a bare float so more fields
            can be added later without breaking readers.
        """
        return {"scale": self.scale}

    def set_extra_state(self, state: dict) -> None:
        """Verify the checkpoint's scale against this module's, never reconcile.

        A disagreement here is a disagreement between the config that trained the
        checkpoint and the config loading it, and only a human knows which one is
        right. Overwriting from the checkpoint would silently make the run stop
        matching its own config; ignoring the checkpoint would make the recorded
        value decorative and reinstate the failure above. So this raises, naming
        both values, and leaves the choice where it belongs.

        Absence is a different case and is NOT an error here: a state dict with
        no ``_extra_state`` predates scale persistence and therefore makes no
        claim about ``s``, so there is nothing to contradict and the module keeps
        its configured value. That case never reaches this method --
        ``nn.Module._load_from_state_dict`` reports the missing key instead, and
        only when ``strict=True``, which is the right place for it.

        Args:
            state: The object returned by :meth:`get_extra_state` at save time.

        Raises:
            ValueError: if ``state`` is not a mapping carrying ``"scale"``, or if
                the scale it carries differs from this module's.
        """
        if not isinstance(state, dict) or "scale" not in state:
            raise ValueError(
                f"SlotProj extra state must be a dict carrying 'scale'; got "
                f"{state!r}. The checkpoint was not written by this class."
            )
        ckpt_scale = float(state["scale"])
        if ckpt_scale != self.scale:
            raise ValueError(
                f"SlotProj scale mismatch: the checkpoint was trained with "
                f"scale={ckpt_scale!r} but this module is configured with "
                f"scale={self.scale!r}. Every merged delta-W would come out off "
                "by that ratio and nothing downstream would report it. Fix the "
                "config to match the checkpoint (or vice versa); this is not "
                "reconciled here."
            )
