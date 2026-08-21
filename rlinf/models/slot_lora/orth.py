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

import torch

# fp32 is the FLOOR precision for the eigendecomposition below: bf16/fp16 have far
# too few mantissa bits for a stable eigh, so anything narrower than fp32 is
# upcast to it. Inputs that are already at or above fp32 (fp32, float64) are left
# in their own dtype instead of being downcast, so a float64 caller keeps
# float64-level accuracy.
_MIN_COMPUTE_DTYPE = torch.float32


def orthogonalize(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Loewdin (symmetric) orthogonalization:  A = (Z Zᵀ)^(-1/2) Z.

    The rows of the result are orthonormal and span exactly the same subspace as
    the rows of ``z``. Among all row-orthonormal matrices this is the one closest
    to ``z`` in Frobenius norm, so a gradient step taken on ``z`` is preserved to
    first order.

    Differentiable end to end: this is how the orthogonality constraint enters the
    computation graph, instead of being a post-``optimizer.step()`` projection.

    The Gram matrix and its inverse square root are computed with fp32 as a FLOOR:
    dtypes narrower than fp32 (bf16, fp16) are upcast to fp32 for the
    eigendecomposition -- bf16 has ~3 decimal digits, far too few for a stable
    eigh -- while dtypes already at or above fp32 (fp32, float64) are computed in
    their own dtype. The result is cast back to the input dtype.

    Note A is invariant to the scale of Z (orthogonalize(cZ) == orthogonalize(Z)),
    which is why the Z parameter group MUST use weight_decay=0: decay would shrink
    Z toward zero, worsening the conditioning of Z Zᵀ, with no effect on A.
    """
    dtype = z.dtype
    compute_dtype = (
        dtype if dtype.itemsize >= _MIN_COMPUTE_DTYPE.itemsize else _MIN_COMPUTE_DTYPE
    )
    zc = z.to(compute_dtype)
    gram = zc @ zc.transpose(-2, -1)
    evals, evecs = torch.linalg.eigh(gram)
    inv_sqrt = (evecs * evals.clamp_min(eps).rsqrt().unsqueeze(-2)) @ evecs.transpose(
        -2, -1
    )
    return (inv_sqrt @ zc).to(dtype)


def orth_error(a: torch.Tensor) -> torch.Tensor:
    """‖A Aᵀ − I‖_F -- the direct correctness check on orthogonalize()."""
    dtype = a.dtype
    compute_dtype = (
        dtype if dtype.itemsize >= _MIN_COMPUTE_DTYPE.itemsize else _MIN_COMPUTE_DTYPE
    )
    ac = a.to(compute_dtype)
    gram = ac @ ac.transpose(-2, -1)
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    return (gram - eye).norm()
