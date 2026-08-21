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

import contextlib

import torch

# fp32 is the FLOOR precision for the Gram matrix and its inverse square root:
# bf16/fp16 have far too few mantissa bits to keep the iteration (or an eigh)
# stable, so anything narrower than fp32 is upcast to it. Inputs that are
# already at or above fp32 (fp32, float64) are left in their own dtype instead
# of being downcast, so a float64 caller keeps float64-level accuracy.
_MIN_COMPUTE_DTYPE = torch.float32

# Newton-Schulz iteration count. Measured (fp32, Gaussian Z, R=256, d_in=1024 --
# the production shape with the smallest LoRA'd d_in): 8 iterations give
# orth_error 6.4e-4..8.2e-4 across seeds and are still PRE-ASYMPTOTIC (10
# iterations improve that ~20x), 10-25 iterations all sit at 3.0e-5..3.5e-5,
# i.e. the fp32 rounding floor for this shape. 12 is the converged value plus
# two iterations of margin for a Z that drifts to worse conditioning during
# training (at condition number 1e1: 8 -> 1.3, 10 -> 1.1e-2, 12 -> 6.5e-5).
# The margin is cheap: one iteration is two R x R matmuls, against the two
# R x R x d_in matmuls that bracket it.
_DEFAULT_NS_ITERS = 12


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """The dtype to compute in, applying the fp32 floor (see _MIN_COMPUTE_DTYPE)."""
    if dtype.itemsize >= _MIN_COMPUTE_DTYPE.itemsize:
        return dtype
    return _MIN_COMPUTE_DTYPE


def _no_autocast(device_type: str):
    """Context that pins the working dtype, or a no-op where autocast cannot run.

    ``autocast`` intercepts per op, so casting a tensor to fp32 by hand does NOT
    keep it there: ``matmul`` is on the bf16 autocast list and demotes the
    operands again. Every ambient autocast must therefore be disabled around the
    whole computation, not worked around inside it.
    """
    if torch.amp.is_autocast_available(device_type):
        return torch.autocast(device_type=device_type, enabled=False)
    return contextlib.nullcontext()


def _inv_sqrt_newton_schulz(gram: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """M^(-1/2) by the coupled Newton-Schulz iteration -- matmuls only, no sync.

    ``gram`` is normalized by its Frobenius norm first, which both puts the
    spectrum inside the iteration's convergence region and makes the whole
    function exactly scale invariant: Z -> cZ sends M -> c^2 M, and the c
    cancels between the normalization and the final rescaling.
    """
    norm = gram.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    y = gram / norm
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    x = eye
    for i in range(iters):
        t = 0.5 * (3.0 * eye - x @ y)
        x = t @ x
        # y is only read through t, so its final update would be dead work.
        if i + 1 < iters:
            y = y @ t
    return x / norm.sqrt()


def _inv_sqrt_eigh(gram: torch.Tensor, eps: float) -> torch.Tensor:
    """M^(-1/2) by eigendecomposition -- the exact reference path, not the default.

    It is a hard device sync, and its backward is NaN on a degenerate spectrum.
    Uses the same Frobenius normalization as the Newton-Schulz path, so ``eps``
    floors a spectrum that has already been made scale free rather than acting
    as an absolute (scale dependent) eigenvalue threshold.
    """
    norm = gram.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    evals, evecs = torch.linalg.eigh(gram / norm)
    inv_sqrt = (evecs * evals.clamp_min(eps).rsqrt().unsqueeze(-2)) @ evecs.transpose(
        -2, -1
    )
    return inv_sqrt / norm.sqrt()


def orthogonalize(
    z: torch.Tensor,
    method: str = "ns",
    iters: int = _DEFAULT_NS_ITERS,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Loewdin (symmetric) orthogonalization:  A = (Z Zᵀ)^(-1/2) Z.

    The rows of the result are orthonormal and span exactly the same subspace as
    the rows of ``z``. Among all row-orthonormal matrices this is the one closest
    to ``z`` in Frobenius norm, so a gradient step taken on ``z`` is preserved to
    first order.

    Differentiable end to end: this is how the orthogonality constraint enters the
    computation graph, instead of being a post-``optimizer.step()`` projection.

    This runs inside the forward pass of every LoRA'd linear layer, so the
    default backend is the coupled Newton-Schulz iteration (``method="ns"``):
    matmuls only, no device sync, and its gradient is finite everywhere.
    ``method="eigh"`` is exact but measured ~6x slower at R=256/d_in=4096 and
    blocks the launch queue for the full kernel time (a per-layer host sync
    destroys FSDP's all-gather/compute overlap); its backward also contains
    ``1/(lambda_i - lambda_j)`` terms and is NaN whenever the spectrum is
    degenerate -- notably when the rows of ``z`` are ALREADY orthogonal, where
    the forward looks perfect. That is why ``z`` must be initialized randomly
    (Gaussian), never with ``nn.init.orthogonal_``.

    The Gram matrix and its inverse square root are computed with fp32 as a
    FLOOR: dtypes narrower than fp32 (bf16, fp16) are upcast to fp32, while
    dtypes already at or above fp32 are computed in their own dtype. The whole
    computation runs under ``autocast(enabled=False)``, because autocast
    intercepts per op and would otherwise demote the explicitly upcast tensors
    back to bf16 (measured WITHOUT this guard: orth_error 5.2e-2 inside an
    ambient bf16 autocast versus 1.3e-5 outside it). NOTE that TF32 is a
    second, independent way to lose the fp32 guarantee -- it is a global
    backend setting (``torch.backends.cuda.matmul.allow_tf32``, implied by
    ``set_float32_matmul_precision("high")``) that this function deliberately
    does not override, and it leaves the fp32 path with 10 mantissa bits.

    ``A`` is invariant to the scale of ``Z`` (``orthogonalize(cZ) ==
    orthogonalize(Z)``). With the Newton-Schulz backend this is exact by
    construction rather than approximate, because the Frobenius normalization
    cancels ``c`` (verified down to c=1e-4). Scale invariance is why the ``Z``
    parameter group MUST use ``weight_decay=0``: decay would shrink ``Z`` toward
    zero, worsening the conditioning of ``Z Zᵀ``, with no effect on ``A``.

    Args:
        z: Slot matrix of shape ``(R, d_in)``, or ``(..., R, d_in)`` for a
            batch of them. Requires ``R <= d_in``; with more rows than columns
            ``Z Zᵀ`` is rank deficient and no row-orthonormal result exists.
        method: ``"ns"`` (default) for the coupled Newton-Schulz iteration, or
            ``"eigh"`` for the eigendecomposition reference path.
        iters: Newton-Schulz iteration count; ignored by ``method="eigh"``. The
            default is converged at the production shape (see
            ``_DEFAULT_NS_ITERS``); lowering it degrades orthogonality silently.
        eps: Floor on the Frobenius norm used to normalize ``Z Zᵀ``, purely a
            division-by-zero guard for an all-zero ``z``. It is NOT an absolute
            eigenvalue clamp, which is what the previous implementation used and
            which silently broke scale invariance whenever ``Z`` was small
            enough to push an eigenvalue under it (measured at c=1e-4: the
            result moved by 9.0e-3 and orth_error rose to 0.447).

    Returns:
        A tensor of the same shape and dtype as ``z`` whose last two dimensions
        form a row-orthonormal matrix. Because the result is cast back to the
        INPUT dtype, a bf16 ``z`` carries a bf16 rounding floor on how orthonormal
        it can be: at R=256 the achievable ``orth_error`` is about 1e-2 (measured
        0.0189 at d_in=1024, 0.0094 at d_in=4096, matching ``method="eigh"`` on
        the same inputs to within 1%). That is rounding, not failure -- a
        monitoring threshold set at 1e-6 would reject a perfectly correct run.

    Raises:
        ValueError: if ``z`` has fewer than 2 dimensions, if it has more rows
            than columns, or if ``method`` is not ``"ns"`` or ``"eigh"``.
    """
    if method not in ("ns", "eigh"):
        raise ValueError(f"unknown method {method!r}; expected 'ns' or 'eigh'")
    if z.ndim < 2:
        raise ValueError(f"orthogonalize needs a matrix, got shape {tuple(z.shape)}")
    if z.shape[-2] > z.shape[-1]:
        raise ValueError(
            f"orthogonalize needs rows <= cols (got {tuple(z.shape[-2:])}); "
            "Z Zᵀ would be rank-deficient and the result not row-orthonormal."
        )

    dtype = z.dtype
    with _no_autocast(z.device.type):
        zc = z.to(_compute_dtype(dtype))
        gram = zc @ zc.transpose(-2, -1)
        if method == "ns":
            inv_sqrt = _inv_sqrt_newton_schulz(gram, iters, eps)
        else:
            inv_sqrt = _inv_sqrt_eigh(gram, eps)
        return (inv_sqrt @ zc).to(dtype)


@torch.no_grad()
def orth_error(a: torch.Tensor) -> torch.Tensor:
    """‖A Aᵀ − I‖_F -- the direct correctness check on :func:`orthogonalize`.

    Diagnostics only: it never builds an autograd graph, and it is computed
    under the same fp32 floor and the same ``autocast(enabled=False)`` guard as
    :func:`orthogonalize`, so the number does not depend on the ambient
    precision context of the caller.

    Args:
        a: Matrix of shape ``(R, d_in)`` or a batch ``(..., R, d_in)``.

    Returns:
        The Frobenius norm reduced over the LAST TWO dimensions only, so a
        batched input gives one error PER matrix (shape ``(...,)``) rather than
        a single total that would drift with the batch size. An unbatched input
        gives a 0-dim tensor. See :func:`orthogonalize` for the bf16 floor: at
        R=256 in bf16, ~1e-2 is the expected value, not a failure.
    """
    with _no_autocast(a.device.type):
        ac = a.to(_compute_dtype(a.dtype))
        gram = ac @ ac.transpose(-2, -1)
        eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
        return (gram - eye).norm(dim=(-2, -1))
