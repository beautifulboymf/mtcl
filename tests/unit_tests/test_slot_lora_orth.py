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

from rlinf.models.slot_lora.orth import orth_error, orthogonalize


def _z(rows=16, cols=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=g, dtype=torch.float64)


class TestOrthogonalize:
    def test_rows_are_orthonormal(self):
        a = orthogonalize(_z())
        eye = torch.eye(a.shape[0], dtype=a.dtype)
        assert torch.allclose(a @ a.T, eye, atol=1e-10)

    def test_scale_invariant(self):
        z = _z()
        assert torch.allclose(orthogonalize(3.7 * z), orthogonalize(z), atol=1e-10)

    def test_row_blocks_are_mutually_orthogonal(self):
        a = orthogonalize(_z(rows=16))
        cross = a[:6] @ a[6:].T
        assert cross.abs().max().item() < 1e-10

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

    def test_preserves_input_dtype(self):
        a = orthogonalize(_z().to(torch.bfloat16))
        assert a.dtype == torch.bfloat16

    def test_orth_error_is_zero_after_orthogonalization(self):
        assert orth_error(orthogonalize(_z())).item() < 1e-10

    def test_orth_error_is_positive_for_raw_matrix(self):
        assert orth_error(_z()).item() > 1.0
