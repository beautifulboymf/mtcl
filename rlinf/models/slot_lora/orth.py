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
from torch.utils.checkpoint import checkpoint

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
# training (at cond(Z) = 10 -- which is cond = 100 for the Gram matrix Z Zᵀ that
# the iteration actually sees, since squaring Z squares its condition number:
# 8 -> 1.3, 10 -> 1.1e-2, 12 -> 6.5e-5). The margin is cheap: one iteration is
# two R x R matmuls, against the two R x R x d_in matmuls that bracket it.
# The margin is also FINITE, and adding iterations is not a way to buy more of
# it -- see the conditioning envelope in ``orthogonalize``'s docstring.
_DEFAULT_NS_ITERS = 12


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """The dtype to compute in, applying the fp32 floor (see _MIN_COMPUTE_DTYPE)."""
    if dtype.itemsize >= _MIN_COMPUTE_DTYPE.itemsize:
        return dtype
    return _MIN_COMPUTE_DTYPE


@contextlib.contextmanager
def _pinned_precision(device_type: str):
    """Context that pins the working precision: no autocast, and no TF32.

    Two independent mechanisms can silently demote this computation below fp32,
    and both are ambient state owned by the caller rather than by this function.

    ``autocast`` intercepts per op, so casting a tensor to fp32 by hand does NOT
    keep it there: ``matmul`` is on the bf16 autocast list and demotes the
    operands again. Every ambient autocast must therefore be disabled around the
    whole computation, not worked around inside it.

    TF32 leaves an fp32 matmul with 10 mantissa bits, and it is a PROCESS-GLOBAL
    backend flag that other parts of this repo turn on for their own reasons
    (``set_float32_matmul_precision("high")`` in the FSDP IQL policy worker,
    ``allow_tf32 = True`` in the OpenSora world-model env), so this function
    cannot assume it is off -- it saves, disables and restores it, the same
    save/restore shape ``torch.backends.cudnn.flags`` uses. The forward pass this
    runs in is single threaded, so the window in which the flag is globally off
    is not observable by anything else. What is saved is the float32 matmul
    PRECISION, not the ``allow_tf32`` bool: the bool cannot represent "medium",
    and restoring it would silently upgrade a "medium" caller to "high". The
    whole thing is skipped when CUDA is unavailable, where the flags govern
    nothing.
    """
    autocast_guard = (
        torch.autocast(device_type=device_type, enabled=False)
        if torch.amp.is_autocast_available(device_type)
        else contextlib.nullcontext()
    )
    pin_tf32 = torch.cuda.is_available()
    if pin_tf32:
        prev_matmul_precision = torch.get_float32_matmul_precision()
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        with autocast_guard:
            yield
    finally:
        if pin_tf32:
            torch.set_float32_matmul_precision(prev_matmul_precision)
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32


def _ns_loop(y: torch.Tensor, eye: torch.Tensor, iters: int) -> torch.Tensor:
    """The coupled Newton-Schulz recurrence, split out so it can be checkpointed."""
    x = eye
    for i in range(iters):
        t = 0.5 * (3.0 * eye - x @ y)
        x = t @ x
        # y is only read through t, so its final update would be dead work.
        if i + 1 < iters:
            y = y @ t
    return x


def _inv_sqrt_newton_schulz(gram: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """M^(-1/2) by the coupled Newton-Schulz iteration -- matmuls only, no sync.

    ``gram`` is normalized by its Frobenius norm first, which both puts the
    spectrum inside the iteration's convergence region and makes the whole
    function exactly scale invariant: Z -> cZ sends M -> c^2 M, and the c
    cancels between the normalization and the final rescaling.

    The recurrence is gradient-CHECKPOINTED, because it trades a lot of live
    activation for very little compute. It keeps three R x R intermediates alive
    per step and autograd would otherwise save all ``iters`` of them: measured
    saved-tensor bytes for ONE call with a bf16 Z on A100, 12.32 MB in 44 unique
    tensors at d_in=1024 and 18.61 MB at d_in=4096, against 3.41 MB in 13 for the
    ``eigh`` path this replaced. Multiplied by the 200-400 LoRA'd linears of a 7B
    model that is +1.7 to +3.5 GB of live activation per micro-batch, which this
    project has already lost runs to. Checkpointing hands all of it back -- 3.41
    MB in 11 tensors at d_in=1024 and 9.70 MB at d_in=4096, i.e. the inputs and
    nothing else, level with the eigh path -- for the cost of recomputing
    ``iters`` x 2 matmuls of R x R, about 0.1 ms. Forward and gradient are
    bitwise unchanged by it (verified against an uncheckpointed replica of the
    same math, bf16 and fp32, at both widths). It is skipped when there is no
    graph to save, where it would be pure overhead.
    """
    norm = gram.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    y = gram / norm
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    if torch.is_grad_enabled() and y.requires_grad:
        x = checkpoint(_ns_loop, y, eye, iters, use_reentrant=False)
    else:
        x = _ns_loop(y, eye, iters)
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
    ambient bf16 autocast versus 1.3e-5 outside it). TF32 is a second,
    independent way to lose the fp32 guarantee -- a process-global backend
    setting (``torch.backends.cuda.matmul.allow_tf32``, implied by
    ``set_float32_matmul_precision("high")``) that leaves an fp32 matmul with 10
    mantissa bits -- so it is saved, disabled and restored around the
    computation as well. Measured on A100 with TF32 left ON, the achievable
    orth_error stops being a precision floor and becomes noise: bf16 0.01888 ->
    0.02922 at d_in=1024 and 0.00949 -> 0.02475 at d_in=4096, fp32 8.8e-6 ->
    0.0225. That is the difference that matters operationally, because it puts a
    healthy run (0.025-0.029) on top of a genuinely collapsed one (0.054 at
    cond(Z)=30) and destroys the only sentinel the training loop has.

    CONDITIONING ENVELOPE. ``iters`` is validated at GOOD conditioning, and past
    the envelope the Newton-Schulz error is a CLIFF, not a slope. Measured fp32
    at the production shape (R=256, d_in=1024, log-spaced singular values):
    cond(Z) 10 -> 6.7e-5, 15 -> 1.1e-4, 20 -> 1.07e-3, 30 -> 5.1e-2, 50 -> 0.637.
    So the validated envelope is roughly cond(Z) <~ 20 at R=256 with iters=12.
    Gaussian init gives cond(Z) ~ 3, so there is about one decade of drift
    headroom -- and its far edge is a step, not a slope, so nothing gradual
    announces the crossing.

    A bf16 reading of :func:`orth_error` is BLIND across that entire
    degradation: over the same sweep it stays pinned at 0.019 (measured 0.01900,
    0.01891, 0.01904 at cond(Z) 10, 15, 20) while the fp32 error grows 16x,
    because the bf16 rounding floor (~1e-2 at R=256) swamps the signal until the
    cliff has already been crossed. Monitoring that wants early warning must
    therefore read ``orth_error`` on an fp32 ``A`` -- orthogonalize an fp32 copy
    of ``Z`` for the diagnostic -- not on the bf16 ``A`` the model consumes.

    ``A`` is invariant to the scale of ``Z`` (``orthogonalize(cZ) ==
    orthogonalize(Z)``). With the Newton-Schulz backend the Frobenius
    normalization cancels ``c`` algebraically, so this is exact rather than
    approximate -- but only while both ends of that cancellation are
    representable. Measured exact over ``‖Z Zᵀ‖_F`` in [2.3e-7, 2.3e17]: below
    that the norm falls under ``eps``, the clamp freezes the normalization, and
    ``A`` shrinks with ``c`` instead of staying invariant (orth_error saturates
    at sqrt(R)); above it the fp32 SUM OF SQUARES inside ``gram.norm()``
    overflows -- ``‖Z Zᵀ‖_F`` = 2.3e17 is finite, 2.3e19 is inf. Production sits
    at ``‖Z Zᵀ‖_F`` ~ 17, so that is 8 and 16 orders of magnitude of headroom
    respectively: this is a documented limit, not a live risk. Scale invariance
    is why the ``Z`` parameter group MUST use ``weight_decay=0``: decay would
    shrink ``Z`` toward zero, worsening the conditioning of ``Z Zᵀ``, with no
    effect on ``A``.

    Args:
        z: Slot matrix of shape ``(R, d_in)``, or ``(..., R, d_in)`` for a
            batch of them. Requires ``R <= d_in``; with more rows than columns
            ``Z Zᵀ`` is rank deficient and no row-orthonormal result exists.
        method: ``"ns"`` (default) for the coupled Newton-Schulz iteration, or
            ``"eigh"`` for the eigendecomposition reference path.
        iters: Newton-Schulz iteration count; ignored by ``method="eigh"``. The
            default is converged at the production shape (see
            ``_DEFAULT_NS_ITERS``) and is bounded on BOTH sides. Lowering it
            degrades orthogonality silently. Raising it is not a free way to buy
            conditioning headroom either: the iteration converges only on the
            range of ``Z``, and on any direction where the Gram is singular the
            recurrence multiplies by 3/2 every step (measured: ‖X‖ grows at
            exactly 1.5x per iteration). ``R <= d_in`` is NECESSARY BUT NOT
            SUFFICIENT for a full-rank Gram -- duplicated rows satisfy it -- and
            on such a ``Z`` (R=256 with rank 128) no iteration count helps: 12,
            20 and 30 all give orth_error 11.31, which is exactly the
            sqrt(R - rank) floor of a rank-deficient result, and by 50 the 1.5^n
            growth has overtaken the fp32 rounding floor and the result is NaN.
            Near-singular behaves the same way, just sooner: at cond(Z)=1e4,
            iters=30 is already NaN.
        eps: Floor on the Frobenius norm used to normalize ``Z Zᵀ``. It acts as
            a division-by-zero guard for an all-zero ``z`` and is inert for
            everything else, but that is a statement about the operating range,
            not an identity: it is inert only while ``‖Z Zᵀ‖_F > eps``, which
            production clears by 7 orders of magnitude (``‖Z Zᵀ‖_F`` ~ 17). It
            is NOT an absolute eigenvalue clamp on ``Z Zᵀ``, which is what the
            previous implementation used and which silently broke scale
            invariance whenever ``Z`` was small enough to push an eigenvalue
            under it (measured at c=1e-4: the result moved by 9.0e-3 and
            orth_error rose to 0.447). ``method="eigh"`` does still use ``eps``
            as an eigenvalue clamp, but on the Frobenius-NORMALIZED spectrum, so
            it is a relative floor there rather than a scale-dependent one.

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
    with _pinned_precision(z.device.type):
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
    under the same fp32 floor and the same precision guard -- no autocast, no
    TF32 -- as :func:`orthogonalize`, so the number does not depend on the
    ambient precision context of the caller.

    What the number cannot do is give early warning, if it is read off the bf16
    ``A`` the model consumes: the bf16 rounding floor swamps the signal, and
    across a full slide from cond(Z)=10 to 20 the bf16 reading stays pinned at
    0.019 while the fp32 error grows 16x. Read it on an fp32 ``A``. See the
    conditioning envelope in :func:`orthogonalize`.

    Args:
        a: Matrix of shape ``(R, d_in)`` or a batch ``(..., R, d_in)``.

    Returns:
        The Frobenius norm reduced over the LAST TWO dimensions only, so a
        batched input gives one error PER matrix (shape ``(...,)``) rather than
        a single total that would drift with the batch size. An unbatched input
        gives a 0-dim tensor. See :func:`orthogonalize` for the bf16 floor: at
        R=256 in bf16, ~1e-2 is the expected value, not a failure.

    Raises:
        ValueError: if ``a`` has fewer than 2 dimensions.
    """
    if a.ndim < 2:
        raise ValueError(f"orth_error needs a matrix, got shape {tuple(a.shape)}")

    with _pinned_precision(a.device.type):
        ac = a.to(_compute_dtype(a.dtype))
        gram = ac @ ac.transpose(-2, -1)
        eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
        return (gram - eye).norm(dim=(-2, -1))
