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

import pytest
import torch

from rlinf.models.slot_lora.orth import orth_error, orthogonalize


def _z(rows=16, cols=64, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=g, dtype=dtype)


def _orthonormal_rows(rows=16, cols=64, seed=0):
    """Z whose Gram matrix is exactly I -- the degenerate spectrum for eigh."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(cols, rows, generator=g, dtype=torch.float64))
    return q.transpose(-2, -1).contiguous()


class TestOrthogonalize:
    def test_rows_are_orthonormal(self):
        a = orthogonalize(_z())
        eye = torch.eye(a.shape[0], dtype=a.dtype)
        assert torch.allclose(a @ a.T, eye, atol=1e-10)

    def test_scale_invariant(self):
        z = _z()
        assert torch.allclose(orthogonalize(3.7 * z), orthogonalize(z), atol=1e-10)

    def test_scale_invariance_holds_at_small_scale(self):
        # An ABSOLUTE eigenvalue clamp breaks here: at c=1e-4 the smallest
        # eigenvalue of (cZ)(cZ)^T is 7.4e-7, under the old eps=1e-6 floor,
        # which moved the result by 9.0e-3 and left orth_error at 0.447.
        z = _z(rows=64, cols=256, dtype=torch.float32)
        a = orthogonalize(z)
        a_scaled = orthogonalize(1e-4 * z)
        assert torch.allclose(a_scaled, a, atol=1e-5)
        assert orth_error(a_scaled).item() < 1e-4

    def test_preserves_row_space(self):
        # projecting Z onto span(A rows) must return Z exactly
        z = _z()
        a = orthogonalize(z)
        assert torch.allclose(z @ a.T @ a, z, atol=1e-8)

    def test_is_differentiable(self):
        z = _z().requires_grad_(True)
        orthogonalize(z).sum().backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_gradcheck_matches_numerical(self):
        # Stronger than test_is_differentiable: an implementation that wrongly
        # detached the Gram matrix would still produce a finite gradient.
        z = _z(rows=6, cols=16).requires_grad_(True)
        assert torch.autograd.gradcheck(orthogonalize, (z,), atol=1e-5)

    def test_degenerate_spectrum_gradient_is_finite(self):
        # Gram == I: every eigenvalue is equal, so eigh's 1/(lambda_i - lambda_j)
        # backward is NaN here even though its forward is exact.
        z = _orthonormal_rows().requires_grad_(True)
        orthogonalize(z).sum().backward()
        assert torch.isfinite(z.grad).all()

    def test_eigh_backend_still_available(self):
        z = _z()
        a_ns = orthogonalize(z, method="ns")
        a_eigh = orthogonalize(z, method="eigh")
        assert torch.allclose(a_ns, a_eigh, atol=1e-10)

    def test_raises_when_rows_exceed_cols(self):
        with pytest.raises(ValueError, match="rows <= cols"):
            orthogonalize(_z(rows=64, cols=32))

    def test_ns_converges_at_production_rank(self):
        # Production shape: R=256 slots, smallest LoRA'd d_in ~1024.
        a = orthogonalize(_z(rows=256, cols=1024, dtype=torch.float32))
        assert orth_error(a).item() < 1e-3

    def test_fp32_floor_survives_autocast(self):
        # autocast intercepts matmul per-op, so an explicit .to(float32) is not
        # enough on its own -- the computation must disable autocast.
        z = _z(rows=64, cols=256, dtype=torch.float32)
        a_outside = orthogonalize(z)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            a_inside = orthogonalize(z)
        err_outside = orth_error(a_outside).item()
        err_inside = orth_error(a_inside).item()
        assert err_inside < 10.0 * err_outside

    def test_preserves_input_dtype(self):
        a = orthogonalize(_z().to(torch.bfloat16))
        assert a.dtype == torch.bfloat16

    def test_bf16_orth_error_floor_at_production_rank(self):
        # The result is cast back to the input dtype, so at R=256 the bf16
        # rounding floor is ~1e-2. That is rounding, not a broken run: any
        # monitoring threshold set at 1e-6 would reject a correct model.
        z = _z(rows=256, cols=1024, dtype=torch.float32).to(torch.bfloat16)
        err = orth_error(orthogonalize(z)).item()
        assert 1e-3 < err < 0.05

    def test_batched_matches_looped(self):
        g = torch.Generator().manual_seed(3)
        z = torch.randn(4, 8, 32, generator=g, dtype=torch.float64)
        batched = orthogonalize(z)
        looped = torch.stack([orthogonalize(z[i]) for i in range(z.shape[0])])
        assert torch.allclose(batched, looped, atol=1e-12)


class TestOrthError:
    def test_orth_error_is_zero_after_orthogonalization(self):
        assert orth_error(orthogonalize(_z())).item() < 1e-10

    def test_orth_error_is_positive_for_raw_matrix(self):
        assert orth_error(_z()).item() > 1.0

    def test_orth_error_is_per_matrix_for_batched_input(self):
        g = torch.Generator().manual_seed(5)
        z = torch.randn(3, 8, 32, generator=g, dtype=torch.float64)
        per_matrix = orth_error(z)
        assert per_matrix.shape == (3,)
        looped = torch.stack([orth_error(z[i]) for i in range(z.shape[0])])
        assert torch.allclose(per_matrix, looped)
        # unbatched input still reduces to a 0-dim tensor
        assert orth_error(z[0]).shape == ()

    def test_orth_error_builds_no_graph(self):
        z = _z().requires_grad_(True)
        assert orth_error(z).requires_grad is False
