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
from contextvars import ContextVar
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.slot_lora.orth import orthogonalize

# Per-sample slot ownership for the CURRENT student forward: LongTensor[B] holding the
# slot index each sample belongs to, or -1 for "no slot owns this sample".
# A ContextVar rather than a plain global because the actor worker is an asyncio actor.
_SLOT_GATE: ContextVar[Optional[torch.Tensor]] = ContextVar(
    "rlinf_slot_gate", default=None
)


def get_slot_gate() -> Optional[torch.Tensor]:
    """The per-sample slot routing in effect for the current student forward.

    Returns:
        The ``LongTensor[B]`` most recently installed by :func:`slot_gate`, or
        ``None`` when no forward has scoped one (the ungated default: every slot
        sees every sample).
    """
    return _SLOT_GATE.get()


@contextmanager
def slot_gate(gate_ids: Optional[torch.Tensor]) -> Iterator[None]:
    """Scope the per-sample slot routing to one student forward.

    The routing has to reach hundreds of LoRA'd linears that the caller never
    touches directly, so it travels out of band rather than through the module
    signatures. Scoping it to a context manager is what keeps that safe: the
    previous value is restored on the way out, including on an exception, so a
    crashed forward cannot leak its routing into the next one.

    Args:
        gate_ids: ``LongTensor[B]`` giving the slot index that owns each sample
            in the batch, ``-1`` for samples no slot owns, or ``None`` for
            ungated.

    Yields:
        Nothing; the routing is read through :func:`get_slot_gate`.
    """
    token = _SLOT_GATE.set(gate_ids)
    try:
        yield
    finally:
        _SLOT_GATE.reset(token)


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
        """
        super().__init__()
        self.in_features = int(in_features)
        self.total_rank = int(total_rank)
        # A plain float, so it stays OUT of the state dict: it is a configuration
        # constant, not learned state. Checkpoint conversion that merges slots into
        # the base weights must therefore reconstruct the SAME value from config.
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
            with torch.no_grad():
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
