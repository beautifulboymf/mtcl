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

import io

import pytest
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


class TestSlotProjExtraState:
    """scale is config, not a tensor -- but the checkpoint still has to record it.

    Nothing else in the checkpoint pins ``s``: Ā is invariant to the scale of Z, so
    a converter that re-derives ``s`` from CLI flags disagreeing with the training
    config emits a delta-W off by a constant factor, with no error anywhere.
    """

    def _proj(self, scale):
        torch.manual_seed(0)
        return SlotProj(32, 8, scale, dtype=torch.float64)

    def test_state_dict_carries_the_scale(self):
        sd = self._proj(2.5).state_dict()
        assert sd["_extra_state"] == {"scale": 2.5}

    def test_state_dict_reads_the_live_attribute(self):
        # scale stays a plain assignable float (that is why it is extra state and
        # not a buffer), so the recorded value must follow a direct assignment.
        p = self._proj(1.0)
        p.scale = 3.0
        assert p.state_dict()["_extra_state"] == {"scale": 3.0}

    def test_round_trips_through_load_state_dict(self):
        src = self._proj(2.5)
        src.weight.data.add_(0.1)
        dst = SlotProj(32, 8, 2.5, dtype=torch.float64)
        dst.load_state_dict(src.state_dict())
        assert dst.scale == 2.5
        assert torch.equal(dst.weight, src.weight)

    def test_round_trips_through_a_torch_save_file(self):
        # extra state has to survive real serialization, not just an in-memory dict.
        buf = io.BytesIO()
        torch.save(self._proj(2.5).state_dict(), buf)
        buf.seek(0)
        dst = SlotProj(32, 8, 2.5, dtype=torch.float64)
        dst.load_state_dict(torch.load(buf, weights_only=False))
        assert dst.scale == 2.5

    def test_mismatched_scale_raises_naming_both_values(self):
        sd = self._proj(2.5).state_dict()
        dst = SlotProj(32, 8, 1.0, dtype=torch.float64)
        with pytest.raises(ValueError, match=r"scale=2\.5.*scale=1\.0"):
            dst.load_state_dict(sd)
        # and it must NOT have been reconciled in either direction
        assert dst.scale == 1.0
        assert sd["_extra_state"] == {"scale": 2.5}

    def test_malformed_extra_state_raises(self):
        p = self._proj(1.0)
        for bad in (2.5, {}, {"eps": 1e-6}, None):
            with pytest.raises(ValueError, match="must be a dict carrying 'scale'"):
                p.set_extra_state(bad)

    def test_pre_change_checkpoint_loads_and_keeps_the_configured_scale(self):
        # A state dict written before scale was persisted makes NO claim about the
        # scale, so there is nothing to contradict: the weight loads and the module
        # keeps the scale it was constructed with (the pre-fix status quo).
        legacy = {"weight": torch.zeros(8, 32, dtype=torch.float64)}
        p = self._proj(2.5)
        missing, unexpected = p.load_state_dict(legacy, strict=False)
        assert p.scale == 2.5
        assert torch.equal(p.weight, legacy["weight"])
        assert missing == ["_extra_state"] and unexpected == []

    def test_pre_change_checkpoint_is_a_missing_key_under_strict(self):
        # Deliberately left as PyTorch's default rather than silently tolerated:
        # "this state dict does not record its scale" is exactly what a converter
        # that must VERIFY needs to hear, and strict= is the caller's knob for it.
        legacy = {"weight": torch.zeros(8, 32, dtype=torch.float64)}
        with pytest.raises(RuntimeError, match="_extra_state"):
            self._proj(2.5).load_state_dict(legacy, strict=True)

    def test_extra_state_does_not_break_the_fsdp_leaf_predicate(self):
        # get/set_extra_state must not add children or parameters: the leaf-wrap
        # test at rlinf/hybrid_engines/fsdp/utils.py:306 still has to fire.
        p = self._proj(1.0)
        assert list(p.named_children()) == []
        assert len(list(p.parameters())) == 1
        assert p.weight.requires_grad
