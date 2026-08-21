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

from rlinf.models.slot_lora.modules import SlotProj, get_slot_gate, slot_gate


class TestSlotGate:
    def test_default_is_none(self):
        assert get_slot_gate() is None

    def test_context_sets_and_restores(self):
        ids = torch.tensor([0, 1])
        with slot_gate(ids):
            assert get_slot_gate() is ids
        assert get_slot_gate() is None

    def test_restores_on_exception(self):
        try:
            with slot_gate(torch.tensor([0])):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert get_slot_gate() is None

    def test_nested_contexts_restore_in_order(self):
        outer, inner = torch.tensor([0]), torch.tensor([1])
        with slot_gate(outer):
            with slot_gate(inner):
                assert get_slot_gate() is inner
            assert get_slot_gate() is outer
        assert get_slot_gate() is None


class TestSlotProj:
    def _proj(self, in_features=32, total_rank=8, scale=1.0):
        torch.manual_seed(0)
        return SlotProj(in_features, total_rank, scale, dtype=torch.float64)

    def test_is_a_leaf_module_so_fsdp_wraps_it_alone(self):
        # rlinf/hybrid_engines/fsdp/utils.py:306 wraps modules that have no children,
        # a .weight, and weight.requires_grad. That is what gives this module its own
        # FSDP flat param (uniform requires_grad, required with use_orig_params=False)
        # AND materializes the full Z inside its own forward, which orthogonalize needs.
        p = self._proj()
        assert list(p.named_children()) == []
        assert p.weight is not None
        assert p.weight.requires_grad

    def test_weight_shape(self):
        assert self._proj(in_features=32, total_rank=8).weight.shape == (8, 32)

    def test_orth_weight_rows_are_orthonormal(self):
        a = self._proj().orth_weight()
        assert torch.allclose(a @ a.T, torch.eye(8, dtype=a.dtype), atol=1e-10)

    def test_forward_applies_scale(self):
        p = self._proj(scale=1.0)
        x = torch.randn(4, 32, dtype=torch.float64)
        base = p(x)
        p.scale = 2.5
        assert torch.allclose(p(x), 2.5 * base, atol=1e-12)

    def test_forward_shape(self):
        out = self._proj(in_features=32, total_rank=8)(
            torch.randn(4, 7, 32, dtype=torch.float64)
        )
        assert out.shape == (4, 7, 8)

    def test_gradient_reaches_z(self):
        p = self._proj()
        p(torch.randn(4, 32, dtype=torch.float64)).sum().backward()
        assert p.weight.grad is not None
        assert torch.isfinite(p.weight.grad).all()

    def test_init_is_gaussian_not_orthogonal(self):
        # Z MUST NOT be initialized to an orthogonal matrix. orthogonalize's eigh
        # backend has a NaN gradient on exactly-degenerate spectra (Gram == I), and
        # while the default NS backend is finite there, an orthogonal init also wastes
        # the well-conditioned starting point the Newton-Schulz iteration count was
        # validated against (Gaussian init gives cond(Z) ~ 3).
        p = self._proj(in_features=256, total_rank=64)
        gram = p.weight @ p.weight.T
        off_diag = gram - torch.diag(torch.diag(gram))
        assert off_diag.abs().max().item() > 1e-6

    def test_diag_is_off_by_default(self):
        p = self._proj()
        p(torch.randn(4, 32, dtype=torch.float64))
        assert p._diag is None

    def test_diag_is_collected_once_when_armed(self):
        p = self._proj()
        p._collect_diag = True
        p(torch.randn(4, 32, dtype=torch.float64))
        assert p._diag is not None
        assert p._collect_diag is False
        assert "gram" in p._diag and "scale" in p._diag
        assert p._diag["gram"].shape == (8, 8)

    def test_diag_gram_is_computed_in_fp32_not_the_forward_dtype(self):
        # The monitor must see the fp32 orthogonality error, not the bf16 one the model
        # consumes. Measured: as cond(Z) goes 10 -> 20 the fp32 error degrades 16x
        # (6.8e-5 -> 1.07e-3) while a bf16 reading moves only 0.01887 -> 0.01894. The
        # rounding floor hides the entire slide; failure is a cliff, not a slope.
        torch.manual_seed(0)
        p = SlotProj(64, 16, 1.0, dtype=torch.bfloat16)
        p._collect_diag = True
        p(torch.randn(4, 64, dtype=torch.bfloat16))
        gram = p._diag["gram"]
        assert gram.dtype == torch.float32
        eye = torch.eye(16, dtype=torch.float32)
        # fp32 recomputation must be far below the bf16 rounding floor (~1e-2 at R=256,
        # smaller here but still orders of magnitude above what fp32 achieves)
        assert (gram - eye).norm().item() < 1e-4

    def test_diag_does_not_build_a_graph(self):
        p = self._proj()
        p._collect_diag = True
        p(torch.randn(4, 32, dtype=torch.float64))
        assert p._diag["gram"].requires_grad is False
