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
import threading

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from rlinf.models.slot_lora.modules import (
    SlotGate,
    SlotLoRALinear,
    SlotOut,
    SlotProj,
)


def _ids(*values):
    return torch.tensor(list(values) or [0, 1, 0, -1], dtype=torch.long)


def _in_a_fresh_thread(fn):
    """Run ``fn()`` on a brand-new thread, returning its value or re-raising its error.

    A fresh thread starts with an EMPTY contextvars context, which is precisely the
    property that broke the ContextVar-based gate this replaced.
    """
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - forwarded to the caller below
            box["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


class _GateProbe(nn.Module):
    """Stand-in for the not-yet-written ``SlotOut``: records the gate each forward saw.

    ``SlotOut`` is the module that will actually consume the gate and it does not exist
    yet, so the recomputation tests need something with the same shape: a module that
    holds a reference to the shared gate, reads it INSIDE ``forward``, and owns a
    parameter so there is a real backward to drive the recomputation with.
    """

    def __init__(self, gate: SlotGate):
        super().__init__()
        self.gate = gate
        self.weight = nn.Parameter(torch.ones(3, dtype=torch.float64))
        self.seen = []

    def forward(self, x):
        self.seen.append(self.gate.current())
        return x * self.weight


class TestSlotGate:
    def test_unset_strict_gate_raises_rather_than_running_ungated(self):
        with pytest.raises(RuntimeError, match="no slot routing"):
            SlotGate().current()

    def test_unset_permissive_gate_reads_none(self):
        assert SlotGate(strict=False).current() is None

    def test_scoped_sets_and_restores(self):
        gate, ids = SlotGate(num_slots=2), _ids()
        with gate.scoped(ids):
            assert gate.current() is ids
        assert gate.current_unchecked() is None

    def test_restores_on_exception(self):
        gate = SlotGate(num_slots=2)
        with pytest.raises(RuntimeError, match="boom"):
            with gate.scoped(_ids()):
                raise RuntimeError("boom")
        assert gate.current_unchecked() is None

    def test_nested_scopes_restore_in_order(self):
        gate, outer, inner = SlotGate(num_slots=2), _ids(0), _ids(1)
        with gate.scoped(outer):
            with gate.scoped(inner):
                assert gate.current() is inner
            assert gate.current() is outer
        assert gate.current_unchecked() is None

    def test_one_holder_is_shared_by_every_gated_module(self):
        # Installing a routing must be O(1), not a walk over the 200-400 gated linears
        # of a 7B model: ONE holder, many references to it.
        gate = SlotGate(num_slots=2)
        probes = [_GateProbe(gate) for _ in range(4)]
        ids = _ids()
        with gate.scoped(ids):
            assert all(p.gate.current() is ids for p in probes)
        for p in probes:
            assert p.gate.current_unchecked() is None


class TestSlotGateStrictMode:
    """Strict mode is what closes the silent-degradation hole.

    An unset gate reaching a gated module means every slot receives gradient from
    every sample, i.e. the isolation mechanism is OFF, with no error, no NaN and a
    perfectly normal loss curve. Under strict that read raises instead.
    """

    def test_strict_raises_after_the_scope_exits(self):
        gate = SlotGate(num_slots=2)
        with gate.scoped(_ids()):
            pass
        with pytest.raises(RuntimeError, match="no slot routing"):
            gate.current()

    def test_strict_raises_when_the_scope_carried_none(self):
        # The routing helper has several early-return paths that yield None. Under
        # strict, "the router produced nothing" is a BUG, not "quietly run ungated".
        gate = SlotGate(num_slots=2)
        with gate.scoped(None):
            with pytest.raises(RuntimeError, match="no slot routing"):
                gate.current()

    def test_the_strict_message_names_the_way_out(self):
        with pytest.raises(RuntimeError, match=r"ungated\(\)"):
            SlotGate().current()

    def test_current_unchecked_never_raises(self):
        assert SlotGate().current_unchecked() is None

    def test_ungated_window_opts_out_and_restores_both_fields(self):
        gate, ids = SlotGate(num_slots=2), _ids()
        with gate.scoped(ids):
            with gate.ungated():
                assert gate.current() is None
                assert gate.strict is False
            assert gate.current() is ids
            assert gate.strict is True

    def test_ungated_window_restores_on_exception(self):
        gate = SlotGate(num_slots=2)
        with pytest.raises(RuntimeError, match="boom"):
            with gate.ungated():
                raise RuntimeError("boom")
        assert gate.strict is True

    def test_a_permissive_gate_is_a_first_class_mode(self):
        # Single-suite distillation legitimately has no routing at all; it must not be
        # forced through strict mode.
        gate = SlotGate(strict=False)
        probe = _GateProbe(gate)
        out = probe(torch.ones(2, 3, dtype=torch.float64))
        assert probe.seen == [None]
        assert out.shape == (2, 3)


class TestSlotGateIdsContract:
    def test_rejects_a_non_tensor(self):
        with pytest.raises(ValueError, match="LongTensor"):
            with SlotGate().scoped([0, 1]):
                pass

    def test_rejects_a_float_tensor(self):
        with pytest.raises(ValueError, match="torch.long"):
            with SlotGate().scoped(torch.tensor([0.0, 1.0])):
                pass

    def test_rejects_a_bool_tensor(self):
        with pytest.raises(ValueError, match="torch.long"):
            with SlotGate().scoped(torch.tensor([True, False])):
                pass

    def test_rejects_a_non_1d_tensor(self):
        with pytest.raises(ValueError, match="1-D"):
            with SlotGate().scoped(torch.zeros(2, 2, dtype=torch.long)):
                pass

    def test_accepts_none_as_the_ungated_value(self):
        gate = SlotGate(strict=False)
        with gate.scoped(None):
            assert gate.current() is None

    def test_accepts_minus_one_for_unrouted_samples(self):
        gate, ids = SlotGate(num_slots=2), _ids(-1, -1)
        with gate.scoped(ids):
            assert torch.equal(gate.current(), ids)

    def test_a_bad_gate_does_not_clobber_the_live_one(self):
        gate, good = SlotGate(num_slots=2), _ids()
        with gate.scoped(good):
            with pytest.raises(ValueError):
                with gate.scoped(torch.tensor([0.0])):
                    pass
            assert gate.current() is good


class TestSlotGateSurvivesThreadsAndCheckpointing:
    """The gate must not live in a ContextVar. Measured against that implementation:

    a fresh ``threading.Thread`` reads the ContextVar's DEFAULT (``None``), not the
    value the caller installed. The autograd engine runs backward nodes for CUDA
    tensors on per-device worker threads, and this repo turns gradient checkpointing on
    (``fsdp_model_manager.py:253``), which RE-EXECUTES the wrapped forward during
    backward -- on that other thread. The recomputed forward would then see no gate,
    every slot would take gradient from every sample, and nothing would report it.
    A shared mutable holder is immune to both thread boundaries and recomputation.
    """

    def test_the_gate_is_visible_from_a_fresh_thread(self):
        gate, ids = SlotGate(num_slots=2), _ids()
        with gate.scoped(ids):
            assert _in_a_fresh_thread(gate.current) is ids

    def test_a_fresh_thread_can_install_a_routing_the_main_thread_sees(self):
        gate, ids = SlotGate(num_slots=2), _ids()

        def install_and_read():
            with gate.scoped(ids):
                return gate.current()

        assert _in_a_fresh_thread(install_and_read) is ids
        assert gate.current_unchecked() is None

    def test_checkpoint_recomputation_sees_the_gate(self):
        gate, ids = SlotGate(num_slots=2), _ids(0, 1)
        probe = _GateProbe(gate)
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        with gate.scoped(ids):
            y = checkpoint(probe, x, use_reentrant=False)
            assert len(probe.seen) == 1
            y.sum().backward()
        # the wrapped forward ran twice: once for real, once recomputed in backward
        assert len(probe.seen) == 2
        assert all(seen is ids for seen in probe.seen)

    def test_checkpoint_recomputation_on_another_thread_sees_the_gate(self):
        # The production shape, reproduced on CPU: the forward is recomputed by a
        # thread that never entered the scope. The ContextVar version returns None here.
        gate, ids = SlotGate(num_slots=2), _ids(0, 1)
        probe = _GateProbe(gate)
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        with gate.scoped(ids):
            y = checkpoint(probe, x, use_reentrant=False)
            _in_a_fresh_thread(lambda: y.sum().backward())
        assert len(probe.seen) == 2
        assert all(seen is ids for seen in probe.seen)
        assert probe.weight.grad is not None

    def test_a_strict_gate_raises_when_recomputation_escapes_the_scope(self):
        # Backward AFTER the scope closed: the recomputed forward genuinely has no
        # routing. Strict mode makes that loud instead of silently ungated.
        gate, ids = SlotGate(num_slots=2), _ids(0, 1)
        probe = _GateProbe(gate)
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        with gate.scoped(ids):
            y = checkpoint(probe, x, use_reentrant=False)
        with pytest.raises(RuntimeError, match="no slot routing"):
            y.sum().backward()


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

    def test_bf16_survives_orth_weight_and_forward(self):
        # bf16 is the production dtype. The fp32 FLOOR inside orthogonalize is an
        # internal compute detail: it must not leak out and silently upcast either the
        # returned basis or the activation, which would cost a d_in-sized fp32 tensor
        # per LoRA'd linear and change the dtype the rest of the layer sees.
        torch.manual_seed(0)
        p = SlotProj(64, 16, 1.0, dtype=torch.bfloat16)
        assert p.weight.dtype == torch.bfloat16
        assert p.orth_weight().dtype == torch.bfloat16
        assert p(torch.randn(4, 64, dtype=torch.bfloat16)).dtype == torch.bfloat16

    def test_diag_is_off_by_default(self):
        p = self._proj()
        p(torch.randn(4, 32, dtype=torch.float64))
        assert p._diag is None

    def test_diag_is_collected_once_when_armed(self):
        p = self._proj()
        p.arm_diag()
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
        p.arm_diag()
        p(torch.randn(4, 64, dtype=torch.bfloat16))
        gram = p._diag["gram"]
        assert gram.dtype == torch.float32
        eye = torch.eye(16, dtype=torch.float32)
        # fp32 recomputation must be far below the bf16 rounding floor (~1e-2 at R=256,
        # smaller here but still orders of magnitude above what fp32 achieves)
        assert (gram - eye).norm().item() < 1e-4

    def test_diag_gram_stays_fp32_inside_an_ambient_autocast(self):
        # orthogonalize pins autocast off internally, but the matmul that FORMS the
        # Gram is outside it -- and autocast intercepts per op, so an fp32 `a32` is
        # demoted again by the matmul itself. Measured on this exact module without
        # the guard: gram comes back bfloat16 with ||G - I||_F = 4.61e-3 against
        # 1.46e-6 outside autocast, ~3000x worse, in the sentinel whose entire job is
        # to resolve a 6.8e-5 -> 1.07e-3 slide. The test above cannot catch this
        # because it runs with no ambient autocast.
        torch.manual_seed(0)
        p = SlotProj(64, 16, 1.0, dtype=torch.bfloat16)
        p.arm_diag()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            p(torch.randn(4, 64, dtype=torch.bfloat16))
        gram = p._diag["gram"]
        assert gram.dtype == torch.float32
        eye = torch.eye(16, dtype=torch.float32)
        assert (gram - eye).norm().item() < 1e-4

    def test_diag_does_not_build_a_graph(self):
        p = self._proj()
        p.arm_diag()
        p(torch.randn(4, 32, dtype=torch.float64))
        assert p._diag["gram"].requires_grad is False


class TestSlotProjRankValidation:
    """R <= d_in is a CONSTRUCTION-time fact, so it has to fail at construction.

    Left to orthogonalize it first fires at the first forward, deep inside a 7B model,
    after 200-400 instances have been built and FSDP-wrapped -- one bad config costing
    a full model build plus a wrap before it says anything.
    """

    def test_rank_above_in_features_raises_naming_both_numbers(self):
        with pytest.raises(ValueError, match=r"total_rank=64.*in_features=32"):
            SlotProj(32, 64, 1.0)

    def test_rank_equal_to_in_features_is_allowed(self):
        assert SlotProj(32, 32, 1.0).weight.shape == (32, 32)

    def test_zero_rank_raises(self):
        with pytest.raises(ValueError, match="total_rank=0"):
            SlotProj(32, 0, 1.0)

    def test_negative_rank_raises(self):
        with pytest.raises(ValueError, match="total_rank=-1"):
            SlotProj(32, -1, 1.0)

    def test_it_raises_before_allocating_the_parameter(self):
        # i.e. it is a constructor check, not a first-forward check
        with pytest.raises(ValueError):
            SlotProj(8, 16, 1.0)


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


class TestSlotOut:
    RANKS = (3, 5)  # slot0 -> columns 0:3, slot1 -> columns 3:8

    def _out(self, out_features=6, strict=True):
        torch.manual_seed(1)
        gate = SlotGate(num_slots=len(self.RANKS), strict=strict)
        m = SlotOut(out_features, self.RANKS, gate, dtype=torch.float64)
        with torch.no_grad():
            m.weight.copy_(
                torch.randn_like(m.weight)
            )  # B must be non-zero to test gating
        return m, gate

    def _h(self, batch=4):
        return torch.randn(
            batch, sum(self.RANKS), dtype=torch.float64, requires_grad=True
        )

    def test_offsets_and_total_rank(self):
        m, _ = self._out()
        assert m.total_rank == 8
        assert m.offsets == (0, 3)

    def test_b_is_zero_initialized(self):
        assert (
            torch.count_nonzero(SlotOut(6, self.RANKS, SlotGate(strict=False)).weight)
            == 0
        )

    def test_is_a_leaf_module_so_fsdp_wraps_it_alone(self):
        m, _ = self._out()
        assert list(m.named_children()) == []
        assert len(list(m.parameters())) == 1
        assert m.weight.requires_grad

    def test_forward_value_is_the_full_slot_sum_regardless_of_routing(self):
        m, gate = self._out()
        h = self._h()
        with gate.ungated():
            ungated = m(h)
        with gate.scoped(torch.tensor([0, 1, 0, -1])):
            routed = m(h)
        assert torch.allclose(ungated, routed, atol=1e-12)

    def test_forward_equals_a_plain_linear(self):
        m, gate = self._out()
        h = self._h()
        with gate.ungated():
            out = m(h)
        assert torch.allclose(out, torch.nn.functional.linear(h, m.weight), atol=1e-12)

    def test_gradient_reaches_only_the_owning_slot_columns(self):
        m, gate = self._out()
        h = self._h()
        with gate.scoped(torch.tensor([0, 0, 0, 0])):
            m(h).sum().backward()
        assert m.weight.grad[:, 0:3].abs().sum() > 0
        assert torch.count_nonzero(m.weight.grad[:, 3:8]) == 0

    def test_gradient_to_h_is_blocked_outside_the_owning_block(self):
        m, gate = self._out()
        h = self._h()
        with gate.scoped(torch.tensor([1, 1, 1, 1])):
            m(h).sum().backward()
        assert torch.count_nonzero(h.grad[:, 0:3]) == 0
        assert h.grad[:, 3:8].abs().sum() > 0

    def test_unrouted_samples_produce_no_gradient_at_all(self):
        m, gate = self._out()
        h = self._h()
        with gate.scoped(torch.tensor([-1, -1, -1, -1])):
            m(h).sum().backward()
        assert torch.count_nonzero(m.weight.grad) == 0
        assert torch.count_nonzero(h.grad) == 0

    def test_mixed_batch_routes_each_sample_to_its_own_slot(self):
        m, gate = self._out()
        h = self._h()
        with gate.scoped(torch.tensor([0, 1, -1, -1])):
            m(h).sum().backward()
        assert torch.count_nonzero(h.grad[0, 3:8]) == 0
        assert h.grad[0, 0:3].abs().sum() > 0
        assert torch.count_nonzero(h.grad[1, 0:3]) == 0
        assert h.grad[1, 3:8].abs().sum() > 0
        assert torch.count_nonzero(h.grad[2]) == 0
        assert torch.count_nonzero(h.grad[3]) == 0

    def test_per_slot_gradient_equals_the_isolated_reference(self):
        # The gated gradient for slot k must equal what you would get by training slot k
        # alone on its own samples -- gating must not merely zero the others, it must
        # leave the owner's gradient numerically untouched.
        m, gate = self._out()
        h = self._h()
        with gate.scoped(torch.tensor([0, 0, 1, 1])):
            m(h).sum().backward()
        gated = m.weight.grad.clone()

        m2, gate2 = self._out()
        h2 = self._h()
        with gate2.ungated():
            m2(h2[:2]).sum().backward()
        assert torch.allclose(gated[:, 0:3], m2.weight.grad[:, 0:3], atol=1e-12)

    def test_works_with_a_sequence_dimension(self):
        m, gate = self._out()
        h = torch.randn(4, 7, 8, dtype=torch.float64, requires_grad=True)
        with gate.scoped(torch.tensor([0, 0, 1, 1])):
            out = m(h)
            assert out.shape == (4, 7, 6)
            out.sum().backward()
        assert torch.count_nonzero(h.grad[0, :, 3:8]) == 0
        # the owner half too: a mutant that keeps the forward value and HALVES the
        # owner's gradient passes every "some block is zero" assertion in this file.
        assert h.grad[0, :, 0:3].abs().sum() > 0
        assert m.weight.grad[:, 0:3].abs().sum() > 0
        assert m.weight.grad[:, 3:8].abs().sum() > 0

    def test_rejects_routing_of_the_wrong_length(self):
        # A ValueError, not an assert: under PYTHONOPTIMIZE=1 asserts are STRIPPED, and
        # measured with them stripped, a length-1 routing against a batch of 4 assigns
        # all four samples to slot 0 (slot0 grad abs-sum 63.27, slot1 exactly 0) while a
        # length-4 routing against a batch of 1 silently changes the output shape from
        # (1, 6) to (4, 6). Both are host-side shape comparisons, so raising costs the
        # same as asserting.
        m, gate = self._out()
        h = self._h(batch=4)
        with gate.scoped(torch.tensor([0, 1])):
            with pytest.raises(ValueError, match=r"routing has 2 entries.*4 samples"):
                m(h)

    def test_rejects_a_routing_longer_than_the_micro_batch(self):
        m, gate = self._out()
        with gate.scoped(torch.tensor([0, 1, 0, 1])):
            with pytest.raises(ValueError, match=r"routing has 4 entries.*1 samples"):
                m(self._h(batch=1))

    def test_strict_gate_raises_when_routing_was_never_installed(self):
        m, _ = self._out(strict=True)
        with pytest.raises(RuntimeError, match="no slot routing"):
            m(self._h())

    def test_survives_checkpoint_recomputation(self):
        # Gradient checkpointing re-runs this forward during backward, on the autograd
        # engine's worker thread. The gate must still be visible there, or gating is
        # silently off and every slot gets gradient from every sample.
        m, gate = self._out()
        h = self._h()
        with gate.scoped(torch.tensor([0, 0, 0, 0])):
            out = checkpoint(m, h, use_reentrant=False)
            out.sum().backward()
        assert torch.count_nonzero(m.weight.grad[:, 3:8]) == 0
        # and the owner still learned: "everything is zero" would pass the line above.
        assert m.weight.grad[:, 0:3].abs().sum() > 0
        assert h.grad[:, 0:3].abs().sum() > 0


class TestSlotOutRankValidation:
    """Same reasoning as SlotProj's: a bad rank list is a config error, so it has to
    cost a construction rather than a full 7B model build plus an FSDP wrap."""

    def test_rejects_an_empty_rank_list(self):
        with pytest.raises(ValueError, match="at least one slot"):
            SlotOut(6, (), SlotGate(strict=False))

    def test_rejects_a_non_positive_rank(self):
        for bad in ((3, 0), (0,), (3, -1)):
            with pytest.raises(ValueError, match="strictly positive"):
                SlotOut(6, bad, SlotGate(strict=False))

    def test_rejects_a_gate_whose_slot_count_disagrees(self):
        # The gate's range check is only as good as its count: a gate saying 4 against
        # a 3-slot B accepts an id of 3, which matches no slot here and trains that
        # sample nothing -- the very failure num_slots exists to close.
        with pytest.raises(ValueError, match=r"2 slots \(3, 5\).*num_slots=4"):
            SlotOut(6, (3, 5), SlotGate(num_slots=4))

    def test_accepts_a_gate_whose_slot_count_agrees(self):
        assert SlotOut(6, (3, 5), SlotGate(num_slots=2)).total_rank == 8

    def test_accepts_a_countless_gate(self):
        # A gate with no count cannot install a routing at all, so there is nothing to
        # disagree with; single-suite runs must stay buildable.
        assert SlotOut(6, (3, 5), SlotGate(strict=False)).total_rank == 8


class TestSlotOutDiag:
    """The B-side halves of ⟨ΔW_s, ΔW_t⟩_F = tr(B_sᵀB_t · Ā_tĀ_sᵀ).

    Only the small (R_t, R_s) blocks are ever materialized -- ΔW itself is d_out x d_in
    and forming it to measure it would defeat the point of measuring it.
    """

    RANKS = (3, 5)

    def _out(self, out_features=6, dtype=torch.float64):
        torch.manual_seed(2)
        m = SlotOut(
            out_features,
            self.RANKS,
            SlotGate(num_slots=len(self.RANKS), strict=False),
            dtype=dtype,
        )
        with torch.no_grad():
            m.weight.copy_(torch.randn_like(m.weight))
        return m

    def _h(self, dtype=torch.float64):
        return torch.randn(4, sum(self.RANKS), dtype=dtype)

    def test_diag_is_off_by_default(self):
        m = self._out()
        with m.gate.ungated():
            m(self._h())
        assert m._diag is None

    def test_diag_is_collected_once_when_armed(self):
        m = self._out()
        m.arm_diag()
        with m.gate.ungated():
            m(self._h())
        assert m._diag is not None
        assert m._collect_diag is False
        assert m._diag["b_norms"].shape == (2,)
        assert set(m._diag["cross"]) == {(0, 1)}
        # B_tᵀ B_s pairs elementwise with Ā_t Ā_sᵀ, so it is (R_t, R_s).
        assert m._diag["cross"][(0, 1)].shape == (5, 3)

    def test_diag_values_match_the_b_blocks(self):
        m = self._out()
        m.arm_diag()
        with m.gate.ungated():
            m(self._h())
        b = m.weight.detach().float()
        b0, b1 = b[:, 0:3], b[:, 3:8]
        assert torch.allclose(
            m._diag["b_norms"], torch.stack([b0.norm(), b1.norm()]), atol=1e-6
        )
        assert torch.allclose(m._diag["cross"][(0, 1)], b1.T @ b0, atol=1e-6)

    def test_diag_is_collected_on_the_gated_path_too(self):
        # Every other case here goes through gate.ungated(), so moving the _collect_diag
        # branch inside the `ids is None` fast path would break none of them -- and would
        # silently stop collecting for every real training step, which is gated.
        m = self._out()
        m.arm_diag()
        with m.gate.scoped(torch.tensor([0, 1, 0, -1])):
            m(self._h())
        assert m._diag is not None
        assert m._collect_diag is False
        assert set(m._diag["cross"]) == {(0, 1)}
        assert m._diag["b_norms"].shape == (2,)

    def test_diag_stays_fp32_inside_an_ambient_autocast(self):
        # Same trap as SlotProj's Gram: autocast intercepts per op, so a bare matmul
        # forming BᵀB is demoted right back down under an ambient bf16 autocast.
        m = self._out(dtype=torch.bfloat16)
        m.arm_diag()
        with torch.autocast("cpu", dtype=torch.bfloat16), m.gate.ungated():
            m(self._h(dtype=torch.bfloat16))
        assert m._diag["cross"][(0, 1)].dtype == torch.float32
        assert m._diag["b_norms"].dtype == torch.float32

    def test_diag_does_not_build_a_graph(self):
        m = self._out()
        m.arm_diag()
        with m.gate.ungated():
            m(self._h())
        assert m._diag["b_norms"].requires_grad is False
        assert m._diag["cross"][(0, 1)].requires_grad is False


class TestSlotGateSlotCount:
    """Out-of-range slot ids: the failure mode strict mode does NOT catch.

    Measured before the fix: a 2-slot model handed ``ids = [0, 1, 7, 99]`` raised
    nothing, and samples 2 and 3 received zero gradient everywhere -- byte for byte the
    behaviour of the deliberate ``-1`` "no slot owns this sample". A router off-by-one,
    or a task-id-to-slot map missing an entry, therefore trains NOTHING for a whole
    suite with no error, no NaN and a normal-looking loss curve. Strict mode catches
    "no routing installed"; only the slot count catches "wrong routing installed".
    """

    def test_out_of_range_ids_are_rejected_at_install_time(self):
        with pytest.raises(ValueError, match=r"must be -1 .*or in \[0, 2\)"):
            with SlotGate(num_slots=2).scoped(_ids(0, 1, 7, 99)):
                pass

    def test_the_message_names_the_offending_values_and_positions(self):
        with pytest.raises(ValueError, match=r"\[7, 99\].*positions \[2, 3\]"):
            with SlotGate(num_slots=2).scoped(_ids(0, 1, 7, 99)):
                pass

    def test_the_off_by_one_id_equal_to_the_slot_count_is_rejected(self):
        # The likeliest bug of all: a 0-based map read as 1-based, or one slot too few.
        with pytest.raises(ValueError, match=r"\[0, 2\)"):
            with SlotGate(num_slots=2).scoped(_ids(0, 2)):
                pass

    def test_ids_below_minus_one_are_rejected(self):
        with pytest.raises(ValueError, match=r"\[-2\]"):
            with SlotGate(num_slots=2).scoped(_ids(0, -2)):
                pass

    def test_minus_one_remains_valid(self):
        gate, ids = SlotGate(num_slots=2), _ids(-1, -1)
        with gate.scoped(ids):
            assert torch.equal(gate.current(), ids)

    def test_every_in_range_id_is_accepted(self):
        gate, ids = SlotGate(num_slots=4), _ids(0, 3, -1, 2)
        with gate.scoped(ids):
            assert torch.equal(gate.current(), ids)

    def test_a_rejected_routing_does_not_clobber_the_live_one(self):
        gate, good = SlotGate(num_slots=2), _ids()
        with gate.scoped(good):
            with pytest.raises(ValueError):
                with gate.scoped(_ids(0, 5)):
                    pass
            assert gate.current() is good

    def test_the_contract_checks_run_before_the_range_check(self):
        # A float tensor has no meaningful range; it must be reported as a dtype error.
        with pytest.raises(ValueError, match="torch.long"):
            with SlotGate(num_slots=2).scoped(torch.tensor([0.0, 7.0])):
                pass

    def test_a_gate_with_no_slot_count_refuses_to_install_a_routing(self):
        # An unverifiable routing on a gate that cannot check it is exactly the hole
        # this closes, so installing one is an error rather than a reduced check.
        with pytest.raises(ValueError, match="num_slots"):
            with SlotGate().scoped(_ids(0, 1)):
                pass

    def test_a_gate_with_no_slot_count_is_still_a_first_class_ungated_gate(self):
        # Single-suite runs never install a routing; they must not be forced to invent
        # a slot count.
        gate = SlotGate(strict=False)
        with gate.scoped(None):
            assert gate.current() is None
        with gate.ungated():
            assert gate.current() is None

    def test_the_slot_count_is_validated_at_construction(self):
        for bad in (0, -1, 2.5, "2"):
            with pytest.raises(ValueError, match="num_slots"):
                SlotGate(num_slots=bad)

    def test_a_bool_slot_count_is_rejected_as_a_misplaced_strict_flag(self):
        # SlotGate(True) reads as the strict flag but binds to num_slots, where True
        # would quietly become a ONE-slot gate that rejects every id except 0 and -1.
        with pytest.raises(ValueError, match="strict"):
            SlotGate(True)

    def test_the_slot_count_is_readable(self):
        assert SlotGate(num_slots=4).num_slots == 4
        assert SlotGate().num_slots is None

    def test_end_to_end_a_two_slot_model_rejects_slot_seven(self):
        # The measured bug, whole: before the fix this ran to completion and trained
        # nothing for samples 2 and 3.
        gate = SlotGate(num_slots=2)
        m = SlotOut(6, (3, 5), gate, dtype=torch.float64)
        with pytest.raises(ValueError, match=r"\[0, 2\)"):
            with gate.scoped(_ids(0, 1, 7, 99)):
                m(torch.randn(4, 8, dtype=torch.float64)).sum().backward()


class TestSlotOutInputWidth:
    """``h`` must span exactly the columns the slots do.

    The gated path slices ``h[..., start:stop]`` per slot, so it always reads the FIRST
    ``total_rank`` columns: an ``h`` of width 12 against ``total_rank=8`` used to run
    fine and silently drop 4 columns, while the ungated fast path raised
    ``RuntimeError: mat1 and mat2 shapes cannot be multiplied``. A SlotProj.total_rank
    that disagrees with sum(SlotOut.slot_ranks) would therefore train to completion and
    only blow up at eval, by which time the checkpoint's delta-W is already wrong.
    """

    RANKS = (3, 5)

    def _out(self):
        torch.manual_seed(1)
        gate = SlotGate(num_slots=len(self.RANKS))
        return SlotOut(6, self.RANKS, gate, dtype=torch.float64), gate

    def test_too_wide_h_is_rejected_on_the_gated_path(self):
        m, gate = self._out()
        with gate.scoped(_ids(0, 1)):
            with pytest.raises(ValueError, match=r"width 12.*8 columns"):
                m(torch.randn(2, 12, dtype=torch.float64))

    def test_too_wide_h_is_rejected_on_the_ungated_path(self):
        m, gate = self._out()
        with gate.ungated():
            with pytest.raises(ValueError, match=r"width 12.*8 columns"):
                m(torch.randn(2, 12, dtype=torch.float64))

    def test_too_narrow_h_is_rejected_on_the_gated_path(self):
        m, gate = self._out()
        with gate.scoped(_ids(0, 1)):
            with pytest.raises(ValueError, match=r"width 5.*8 columns"):
                m(torch.randn(2, 5, dtype=torch.float64))

    def test_too_narrow_h_is_rejected_on_the_ungated_path(self):
        m, gate = self._out()
        with gate.ungated():
            with pytest.raises(ValueError, match=r"width 5.*8 columns"):
                m(torch.randn(2, 5, dtype=torch.float64))

    def test_a_sequence_dimension_does_not_confuse_the_width_check(self):
        m, gate = self._out()
        with gate.scoped(_ids(0, 1)):
            assert m(torch.randn(2, 7, 8, dtype=torch.float64)).shape == (2, 7, 6)


class TestSlotOutSlotLayouts:
    """Equal per-slot ranks, K = 1 and K >= 3.

    Every other SlotOut test uses (3, 5). Unequal ranks make a transposed block a shape
    error, which accidentally protects the diagnostic pairing from a silent slicing bug,
    and a two-slot layout cannot catch an offset bug that only shows up from slot 2 on.
    """

    def _out(self, ranks, out_features=6, strict=True):
        torch.manual_seed(7)
        gate = SlotGate(num_slots=len(ranks), strict=strict)
        m = SlotOut(out_features, ranks, gate, dtype=torch.float64)
        with torch.no_grad():
            m.weight.copy_(torch.randn_like(m.weight))
        return m, gate

    def test_equal_ranks_lay_out_contiguous_blocks(self):
        m, _ = self._out((4, 4, 4))
        assert m.total_rank == 12
        assert m.offsets == (0, 4, 8)

    def test_equal_ranks_isolate_every_slot(self):
        m, gate = self._out((4, 4, 4))
        h = torch.randn(3, 12, dtype=torch.float64, requires_grad=True)
        with gate.scoped(torch.tensor([0, 1, 2])):
            m(h).sum().backward()
        blocks = [slice(0, 4), slice(4, 8), slice(8, 12)]
        for sample, owned in enumerate(blocks):
            assert h.grad[sample, owned].abs().sum() > 0
            for other in blocks:
                if other != owned:
                    assert torch.count_nonzero(h.grad[sample, other]) == 0
        for owned in blocks:
            assert m.weight.grad[:, owned].abs().sum() > 0

    def test_equal_ranks_leave_the_owner_gradient_numerically_untouched(self):
        m, gate = self._out((4, 4, 4))
        h = torch.randn(3, 12, dtype=torch.float64, requires_grad=True)
        with gate.scoped(torch.tensor([1, 1, 2])):
            m(h).sum().backward()
        gated = m.weight.grad.clone()

        reference, ref_gate = self._out((4, 4, 4))
        with ref_gate.ungated():
            reference(h.detach()[:2]).sum().backward()
        assert torch.allclose(gated[:, 4:8], reference.weight.grad[:, 4:8], atol=1e-12)

    def test_three_slots_cross_the_expected_pairs(self):
        m, _ = self._out((4, 4, 4), strict=False)
        m.arm_diag()
        with m.gate.ungated():
            m(torch.randn(2, 12, dtype=torch.float64))
        assert set(m._diag["cross"]) == {(0, 1), (0, 2), (1, 2)}
        assert m._diag["cross"][(0, 2)].shape == (4, 4)
        assert m._diag["b_norms"].shape == (3,)

    def test_a_single_slot_takes_gradient_from_every_routed_sample(self):
        m, gate = self._out((4,))
        h = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        with gate.scoped(torch.tensor([0, 0, 0])):
            m(h).sum().backward()
        assert m.offsets == (0,)
        assert m.weight.grad.abs().sum() > 0
        assert h.grad.abs().sum() > 0

    def test_a_single_slot_still_honours_minus_one(self):
        m, gate = self._out((4,))
        h = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        with gate.scoped(torch.tensor([0, -1, -1])):
            m(h).sum().backward()
        assert h.grad[0].abs().sum() > 0
        assert torch.count_nonzero(h.grad[1:]) == 0

    def test_a_single_slot_has_no_cross_terms(self):
        m, _ = self._out((4,), strict=False)
        m.arm_diag()
        with m.gate.ungated():
            m(torch.randn(2, 4, dtype=torch.float64))
        assert m._diag["cross"] == {}
        assert m._diag["b_norms"].shape == (1,)


class TestSlotOutBf16:
    """bf16 is the production dtype; every other functional test here is fp64."""

    RANKS = (3, 5)

    def _out(self):
        torch.manual_seed(11)
        gate = SlotGate(num_slots=len(self.RANKS))
        m = SlotOut(6, self.RANKS, gate, dtype=torch.bfloat16)
        with torch.no_grad():
            m.weight.copy_(torch.randn_like(m.weight))
        return m, gate

    def test_bf16_forward_keeps_its_dtype(self):
        m, gate = self._out()
        with gate.scoped(_ids(0, 1, 0, -1)):
            out = m(torch.randn(4, 8, dtype=torch.bfloat16))
        assert out.dtype == torch.bfloat16
        assert out.shape == (4, 6)

    def test_bf16_backward_routes_gradient_to_the_owner_only(self):
        m, gate = self._out()
        h = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
        with gate.scoped(torch.tensor([0, 0, 0, 0])):
            m(h).sum().backward()
        assert m.weight.grad.dtype == torch.bfloat16
        assert torch.isfinite(m.weight.grad).all()
        assert m.weight.grad[:, 0:3].abs().sum() > 0
        assert torch.count_nonzero(m.weight.grad[:, 3:8]) == 0
        assert h.grad[:, 0:3].abs().sum() > 0
        assert torch.count_nonzero(h.grad[:, 3:8]) == 0

    def test_the_two_paths_agree_only_to_the_bf16_rounding_floor(self):
        # The routing has no train/inference mismatch, but the ARITHMETIC does: the
        # gated path accumulates K per-slot terms where the fast path does one matmul.
        # Measured max|delta| 8.882e-16 in fp64 and 9.375e-2 in bf16 against max|out|
        # 2.05e1 (relative ~4.6e-3). Never compare the two for bitwise equality.
        m, gate = self._out()
        h = torch.randn(4, 8, dtype=torch.bfloat16)
        with gate.ungated():
            fast = m(h).float()
        with gate.scoped(_ids(0, 1, 0, -1)):
            gated = m(h).float()
        assert (gated - fast).abs().max() <= 2e-2 * fast.abs().max()


class TestSlotOutGradientAccumulation:
    """Each micro-batch installs its own routing; the accumulated grad must respect all
    of them. With gradient accumulation the global batch is split, so this is the shape
    every real training step takes."""

    RANKS = (3, 5)

    def _out(self):
        torch.manual_seed(13)
        gate = SlotGate(num_slots=len(self.RANKS))
        m = SlotOut(6, self.RANKS, gate, dtype=torch.float64)
        with torch.no_grad():
            m.weight.copy_(torch.randn_like(m.weight))
        return m, gate

    def test_two_micro_batches_land_in_their_own_column_blocks(self):
        m, gate = self._out()
        h1 = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
        h2 = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)

        with gate.scoped(torch.tensor([0, 0, 0, 0])):
            m(h1).sum().backward()
        after_first = m.weight.grad.clone()
        assert after_first[:, 0:3].abs().sum() > 0
        assert torch.count_nonzero(after_first[:, 3:8]) == 0

        with gate.scoped(torch.tensor([1, 1])):
            m(h2).sum().backward()
        # slot 0's accumulated gradient must be exactly what micro-batch 1 left there
        assert torch.equal(m.weight.grad[:, 0:3], after_first[:, 0:3])
        assert m.weight.grad[:, 3:8].abs().sum() > 0

    def test_accumulation_adds_rather_than_replaces_within_one_slot(self):
        m, gate = self._out()
        h1 = torch.randn(2, 8, dtype=torch.float64)
        h2 = torch.randn(2, 8, dtype=torch.float64)
        with gate.scoped(torch.tensor([0, 0])):
            m(h1).sum().backward()
            first = m.weight.grad.clone()
            m(h2).sum().backward()
        both = m.weight.grad.clone()

        m2, gate2 = self._out()
        with gate2.scoped(torch.tensor([0, 0])):
            m2(h2).sum().backward()
        assert torch.allclose(both - first, m2.weight.grad, atol=1e-12)


class TestDiagStaleness:
    """``_diag`` must never read as fresh when no forward refreshed it.

    Measured before the fix: arm, forward, record; then mutate the weight and re-arm
    WITHOUT running a forward -- ``_diag`` still held the previous values and no reader
    could tell. A module that is armed but never REACHED by a forward (a frozen layer,
    a rollout-only step, a branch this batch did not take) would have last step's
    numbers plotted as this step's, in plots whose entire job is to answer "is this slot
    dead?".
    """

    def _proj(self):
        torch.manual_seed(17)
        return SlotProj(32, 8, 1.0, dtype=torch.float64)

    def _out(self):
        torch.manual_seed(19)
        m = SlotOut(6, (3, 5), SlotGate(num_slots=2, strict=False), dtype=torch.float64)
        with torch.no_grad():
            m.weight.copy_(torch.randn_like(m.weight))
        return m

    def test_slot_proj_arm_diag_sets_the_flag(self):
        p = self._proj()
        p.arm_diag()
        assert p._collect_diag is True

    def test_slot_out_arm_diag_sets_the_flag(self):
        m = self._out()
        m.arm_diag()
        assert m._collect_diag is True

    def test_slot_proj_re_arming_without_a_forward_reads_absent_not_stale(self):
        p = self._proj()
        p.arm_diag()
        p(torch.randn(4, 32, dtype=torch.float64))
        assert p._diag is not None
        with torch.no_grad():
            p.weight.mul_(3.0)
        p.arm_diag()
        assert p._diag is None

    def test_slot_out_re_arming_without_a_forward_reads_absent_not_stale(self):
        m = self._out()
        m.arm_diag()
        with m.gate.ungated():
            m(torch.randn(4, 8, dtype=torch.float64))
        stale = m._diag["b_norms"].clone()
        with torch.no_grad():
            m.weight.mul_(3.0)
        m.arm_diag()
        assert m._diag is None
        # and the next forward reports the CURRENT weight, not the recorded one
        with m.gate.ungated():
            m(torch.randn(4, 8, dtype=torch.float64))
        assert torch.allclose(m._diag["b_norms"], 3.0 * stale, atol=1e-5)

    def test_arm_diag_is_idempotent_before_a_forward(self):
        m = self._out()
        m.arm_diag()
        m.arm_diag()
        assert m._collect_diag is True and m._diag is None


class TestCrossGramPairingContract:
    """How a consumer turns the two recorded halves into a cross-slot interference.

    ``<dW_s, dW_t>_F = scale**2 * sum( (B_t^T B_s) elementwise (A_t A_s^T) )``. The B
    half comes from ``SlotOut._diag["cross"][(s, t)]`` and is (R_t, R_s); the A half is
    ``SlotProj._diag["gram"][t_slice, s_slice]``, in THAT order. With EQUAL per-slot
    ranks the transposed slicing has the same shape, so the wrong one produces a wrong
    number instead of a shape error -- which is why these use (4, 4).
    """

    RANKS = (4, 4)
    IN_FEATURES = 16
    SCALE = 2.5

    def _out_with_diag(self):
        torch.manual_seed(23)
        m = SlotOut(
            6,
            self.RANKS,
            SlotGate(num_slots=len(self.RANKS), strict=False),
            dtype=torch.float64,
        )
        with torch.no_grad():
            m.weight.copy_(torch.randn_like(m.weight))
        m.arm_diag()
        with m.gate.ungated():
            m(torch.randn(2, sum(self.RANKS), dtype=torch.float64))
        return m

    def test_the_documented_pairing_reproduces_the_frobenius_inner_product(self):
        # A deliberately NON-orthogonal A: with a real orthonormal A-bar every slicing
        # of the Gram is ~0, so an orthonormal basis cannot tell the right pairing from
        # the wrong one.
        m = self._out_with_diag()
        torch.manual_seed(29)
        a = torch.randn(sum(self.RANKS), self.IN_FEATURES, dtype=torch.float64)
        gram = a @ a.transpose(-2, -1)
        b = m.weight.detach()
        s_slice, t_slice = slice(0, 4), slice(4, 8)

        dw_s = self.SCALE * b[:, s_slice] @ a[s_slice]
        dw_t = self.SCALE * b[:, t_slice] @ a[t_slice]
        truth = (dw_s * dw_t).sum()

        cross = m._diag["cross"][(0, 1)].double()
        predicted = (cross * gram[t_slice, s_slice]).sum() * self.SCALE**2
        assert torch.allclose(predicted, truth, rtol=1e-5)

    def test_the_transposed_slicing_is_a_silent_wrong_answer(self):
        # Same shape, different number: nothing but this test says which one is right.
        m = self._out_with_diag()
        torch.manual_seed(29)
        a = torch.randn(sum(self.RANKS), self.IN_FEATURES, dtype=torch.float64)
        gram = a @ a.transpose(-2, -1)
        b = m.weight.detach()
        s_slice, t_slice = slice(0, 4), slice(4, 8)
        truth = (
            (self.SCALE * b[:, s_slice] @ a[s_slice])
            * (self.SCALE * b[:, t_slice] @ a[t_slice])
        ).sum()

        cross = m._diag["cross"][(0, 1)].double()
        wrong = (cross * gram[s_slice, t_slice]).sum() * self.SCALE**2
        assert wrong.shape == ()  # no shape error to save you
        assert (wrong - truth).abs() > 1e-3 * truth.abs()

    def test_end_to_end_the_orthonormal_basis_makes_the_cross_term_vanish(self):
        # The real pipeline: gram from SlotProj, scale from SlotProj's own record, cross
        # from SlotOut. A-bar is row-orthonormal, so the interference must be ~0 -- and
        # the B half is far from zero, so the vanishing comes from the Gram.
        m = self._out_with_diag()
        torch.manual_seed(31)
        p = SlotProj(self.IN_FEATURES, sum(self.RANKS), self.SCALE, dtype=torch.float64)
        p.arm_diag()
        p(torch.randn(2, self.IN_FEATURES, dtype=torch.float64))
        gram, scale = p._diag["gram"].double(), p._diag["scale"]
        assert scale == self.SCALE

        a = p.orth_weight().detach()
        b = m.weight.detach()
        s_slice, t_slice = slice(0, 4), slice(4, 8)
        dw_s = scale * b[:, s_slice] @ a[s_slice]
        dw_t = scale * b[:, t_slice] @ a[t_slice]
        truth = (dw_s * dw_t).sum()

        cross = m._diag["cross"][(0, 1)].double()
        predicted = (cross * gram[t_slice, s_slice]).sum() * scale**2

        floor = 1e-4 * dw_s.norm() * dw_t.norm()
        assert cross.abs().max() > 0.1  # the B half is NOT trivially zero
        assert truth.abs() < floor
        assert (predicted - truth).abs() < floor


def _fsdp_lora_leaf(module: nn.Module) -> bool:
    """The LoRA leaf-wrap predicate from ``rlinf/hybrid_engines/fsdp/utils.py:305-311``.

    Copied verbatim so the layout assumptions of this package are pinned by a test
    rather than by a comment. A module it accepts gets its OWN FSDP flat parameter.
    """
    return bool(
        len(list(module.named_children())) == 0
        and getattr(module, "weight", None) is not None
        and module.weight.requires_grad
        and getattr(module, "_to_lora", True) is True
    )


class TestSlotLoRALinear:
    """The drop-in replacement for one nn.Linear: frozen base + K orthogonal slots."""

    RANKS = (3, 5)  # slot0 -> columns 0:3, slot1 -> columns 3:8

    def _layer(
        self,
        in_features=16,
        out_features=6,
        # NOT 1.0: at unit scale a delta_weight() that dropped the scale entirely would
        # satisfy every merge test in this class.
        scale=0.7,
        bias=True,
        dtype=torch.float64,
        eps=1e-6,
        ranks=None,
        num_slots=None,
    ):
        ranks = self.RANKS if ranks is None else ranks
        torch.manual_seed(2)
        base = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        gate = SlotGate(num_slots=len(ranks) if num_slots is None else num_slots)
        return SlotLoRALinear(base, ranks, scale, gate, eps=eps), gate

    def _trained(self, **kwargs):
        """A layer whose B is non-zero. At init ΔW == 0 and every merge test is vacuous."""
        layer, gate = self._layer(**kwargs)
        with torch.no_grad():
            layer.slot_B.weight.copy_(torch.randn_like(layer.slot_B.weight))
        return layer, gate

    def _x(self, batch=4, in_features=16, dtype=torch.float64, requires_grad=False):
        return torch.randn(batch, in_features, dtype=dtype, requires_grad=requires_grad)

    # ---- structure -----------------------------------------------------------------

    def test_holds_the_base_and_the_two_slot_sides(self):
        layer, gate = self._layer()
        assert [name for name, _ in layer.named_children()] == [
            "base",
            "slot_A",
            "slot_B",
        ]
        assert isinstance(layer.slot_A, SlotProj)
        assert isinstance(layer.slot_B, SlotOut)
        assert layer.slot_A.weight.shape == (8, 16)  # (sum(ranks), d_in)
        assert layer.slot_B.weight.shape == (6, 8)  # (d_out, sum(ranks))

    def test_the_slots_share_the_gate_that_was_passed_in(self):
        # One gate object per MODEL: installing a routing must be one attribute write,
        # not a walk over the 200-400 gated linears of a 7B student.
        layer, gate = self._layer()
        assert layer.slot_B.gate is gate

    def test_slot_parameters_inherit_the_base_dtype_and_device(self):
        layer, _ = self._layer(dtype=torch.float64)
        ref = layer.base.weight
        for p in (layer.slot_A.weight, layer.slot_B.weight):
            assert p.dtype == ref.dtype
            assert p.device == ref.device

    def test_scale_and_eps_are_stored_on_the_projection(self):
        layer, _ = self._layer(scale=0.7, eps=1e-4)
        assert layer.slot_A.scale == 0.7
        assert layer.slot_A.eps == 1e-4

    def test_match_mt4_style_scale_is_stored_verbatim(self):
        # The scale POLICY (sqrt(d_in)/ref_rank, reproducing the row norm of PEFT's
        # gaussian-initialized A) lives in the injection pass. This class only has to
        # take the number it is handed and apply it -- see the delta_weight test below.
        layer, _ = self._layer(in_features=16, scale=(16**0.5) / 128)
        assert abs(layer.slot_A.scale - (16**0.5) / 128) < 1e-12

    def test_it_is_not_an_fsdp_leaf_but_its_children_are(self):
        # SlotLoRALinear HAS children, so the leaf predicate skips it and FSDP wraps
        # slot_A and slot_B individually; the frozen base is left to fold into the
        # enclosing transformer layer's flat param, which is uniformly frozen.
        layer, _ = self._layer()
        assert _fsdp_lora_leaf(layer) is False
        assert _fsdp_lora_leaf(layer.slot_A) is True
        assert _fsdp_lora_leaf(layer.slot_B) is True
        assert _fsdp_lora_leaf(layer.base) is False  # frozen -> not its own flat param

    # ---- freezing and trainability -------------------------------------------------

    def test_base_parameters_are_frozen(self):
        layer, _ = self._layer()
        assert layer.base.weight.requires_grad is False
        assert layer.base.bias.requires_grad is False

    def test_only_the_two_slot_parameters_are_trainable(self):
        layer, _ = self._layer()
        trainable = sorted(n for n, p in layer.named_parameters() if p.requires_grad)
        assert trainable == ["slot_A.weight", "slot_B.weight"]

    def test_a_rejected_construction_leaves_the_caller_base_untouched(self):
        # Validation happens BEFORE the base is frozen, so a config error caught here
        # does not leave the caller holding a half-frozen model.
        base = nn.Linear(16, 6, dtype=torch.float64)
        with pytest.raises(ValueError):
            SlotLoRALinear(base, self.RANKS, 1.0, SlotGate(num_slots=4))
        assert base.weight.requires_grad is True

    # ---- forward -------------------------------------------------------------------

    def test_output_equals_the_base_exactly_at_init(self):
        # B is zero-initialized, so ΔW == 0 and the student starts as its base model.
        layer, gate = self._layer()
        x = self._x()
        with gate.scoped(_ids(0, 1, 0, -1)):
            out = layer(x)
        assert torch.equal(out, layer.base(x))

    def test_forward_matches_base_plus_delta_weight(self):
        # THE property the whole merge path rests on. fp64, tight.
        layer, gate = self._trained()
        x = self._x()
        with gate.scoped(_ids(0, 1, 0, -1)):
            out = layer(x)
        expected = layer.base(x) + x @ layer.delta_weight().to(x.dtype).T
        assert torch.allclose(out, expected, atol=1e-12)

    def test_forward_carries_a_sequence_dimension_through(self):
        layer, gate = self._trained()
        x = torch.randn(2, 3, 16, dtype=torch.float64)
        with gate.scoped(_ids(0, 1)):  # one id per SAMPLE, not per token
            out = layer(x)
        assert out.shape == (2, 3, 6)
        expected = layer.base(x) + x @ layer.delta_weight().to(x.dtype).T
        assert torch.allclose(out, expected, atol=1e-12)

    def test_forward_works_without_a_bias(self):
        layer, gate = self._trained(bias=False)
        assert layer.base.bias is None
        assert sorted(n for n, p in layer.named_parameters() if p.requires_grad) == [
            "slot_A.weight",
            "slot_B.weight",
        ]
        x = self._x()
        with gate.scoped(_ids(0, 1, 0, -1)):
            out = layer(x)
        expected = layer.base(x) + x @ layer.delta_weight().to(x.dtype).T
        assert torch.allclose(out, expected, atol=1e-12)

    # ---- gating --------------------------------------------------------------------

    def test_gradient_reaches_only_the_owning_slot_columns(self):
        # The routing arrives through the SHARED gate holder, never as a call argument:
        # nothing in this test passes ids to the layer.
        layer, gate = self._trained()
        x = self._x(requires_grad=True)
        with gate.scoped(torch.tensor([0, 0, 0, 0])):
            layer(x).sum().backward()
        assert layer.slot_B.weight.grad[:, 0:3].abs().sum() > 0
        assert torch.count_nonzero(layer.slot_B.weight.grad[:, 3:8]) == 0
        assert layer.base.weight.grad is None  # frozen

    def test_an_entirely_unrouted_batch_trains_nothing(self):
        layer, gate = self._trained()
        x = self._x(requires_grad=True)
        with gate.scoped(torch.tensor([-1, -1, -1, -1])):
            layer(x).sum().backward()
        assert torch.count_nonzero(layer.slot_B.weight.grad) == 0
        assert torch.count_nonzero(layer.slot_A.weight.grad) == 0

    def test_a_strict_gate_with_no_routing_raises_instead_of_running_ungated(self):
        layer, _ = self._trained()
        with pytest.raises(RuntimeError, match="no slot routing"):
            layer(self._x())

    def test_survives_checkpoint_recomputation(self):
        # Gradient checkpointing re-runs this forward during backward on the autograd
        # engine's worker thread; the gate has to still be visible there.
        layer, gate = self._trained()
        x = self._x(requires_grad=True)
        with gate.scoped(torch.tensor([0, 0, 0, 0])):
            out = checkpoint(layer, x, use_reentrant=False)
            out.sum().backward()
        assert torch.count_nonzero(layer.slot_B.weight.grad[:, 3:8]) == 0
        assert layer.slot_B.weight.grad[:, 0:3].abs().sum() > 0  # not "all zero"
        assert x.grad.abs().sum() > 0

    # ---- delta_weight --------------------------------------------------------------

    def test_delta_weight_is_exactly_zero_at_init(self):
        layer, _ = self._layer()
        assert layer.delta_weight().abs().max().item() == 0.0

    def test_delta_weight_has_the_shape_of_the_base_weight(self):
        layer, _ = self._trained()
        assert layer.delta_weight().shape == layer.base.weight.shape

    def test_delta_weight_scales_linearly_with_the_scale(self):
        # Ā is invariant to the scale of Z, so the ONLY place s enters ΔW is this factor.
        one, _ = self._trained(scale=1.0)
        many, _ = self._trained(scale=2.5)
        assert torch.equal(one.slot_B.weight, many.slot_B.weight)  # same seed
        assert torch.allclose(many.delta_weight(), 2.5 * one.delta_weight(), atol=1e-12)

    def test_delta_weight_builds_no_graph(self):
        layer, _ = self._trained()
        delta = layer.delta_weight()
        assert delta.requires_grad is False
        assert delta.grad_fn is None
        assert layer.slot_A.weight.grad is None
        assert layer.slot_B.weight.grad is None

    def test_delta_weight_is_never_narrower_than_fp32(self):
        # fp32 is a FLOOR, the same convention orthogonalize uses: a bf16 adapter is
        # widened, a fp64 one is NOT thrown away.
        bf16, _ = self._trained(dtype=torch.bfloat16)
        assert bf16.delta_weight().dtype == torch.float32
        fp32, _ = self._trained(dtype=torch.float32)
        assert fp32.delta_weight().dtype == torch.float32
        fp64, _ = self._trained(dtype=torch.float64)
        assert fp64.delta_weight().dtype == torch.float64

    def test_delta_weight_keeps_its_precision_inside_an_ambient_autocast(self):
        # autocast intercepts per op, so the B @ Ā matmul would be demoted to bf16 even
        # though both operands were widened by hand -- silently halving the precision of
        # every merged checkpoint produced from inside an autocast region.
        layer, _ = self._trained(dtype=torch.float32)
        outside = layer.delta_weight()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            inside = layer.delta_weight()
        assert inside.dtype == torch.float32
        assert torch.equal(inside, outside)

    # ---- the merge the converter actually performs ----------------------------------

    def _merged(self, layer, bias=None):
        merged = nn.Linear(
            layer.base.in_features,
            layer.base.out_features,
            bias=bias is not None,
            dtype=layer.base.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(layer.base.weight + layer.delta_weight())
            if bias is not None:
                merged.bias.copy_(bias)
        return merged

    def test_a_merged_plain_linear_reproduces_the_gated_forward(self):
        layer, gate = self._trained()
        x = self._x()
        with gate.scoped(_ids(0, 1, 0, -1)):
            trained = layer(x)
        merged = self._merged(layer, bias=layer.base.bias)
        assert torch.allclose(merged(x), trained, atol=1e-12)

    def test_the_merge_test_would_catch_a_doubled_bias(self):
        # delta_weight() is WEIGHT-only; the bias belongs to the base and must be
        # carried over ONCE. A converter that adds it into both halves passes every
        # weight-space check and produces a model that is wrong by exactly one bias.
        layer, gate = self._trained()
        x = self._x()
        with gate.scoped(_ids(0, 1, 0, -1)):
            trained = layer(x)
        doubled = self._merged(layer, bias=layer.base.bias * 2)
        assert not torch.allclose(doubled(x), trained, atol=1e-12)
        assert torch.allclose(doubled(x) - trained, layer.base.bias.expand_as(trained))

    def test_the_merge_test_would_catch_a_dropped_bias(self):
        layer, gate = self._trained()
        x = self._x()
        with gate.scoped(_ids(0, 1, 0, -1)):
            trained = layer(x)
        assert not torch.allclose(self._merged(layer)(x), trained, atol=1e-12)

    def test_the_checkpoint_carries_everything_the_merge_needs(self):
        # The converter rebuilds this structure, loads the state dict with
        # missing == unexpected == 0, and merges. Z is what is stored -- never Ā.
        layer, _ = self._trained(scale=0.7)
        state = layer.state_dict()
        assert set(state) == {
            "base.weight",
            "base.bias",
            "slot_A.weight",
            "slot_A._extra_state",
            "slot_B.weight",
        }
        fresh, _ = self._layer(scale=0.7)
        report = fresh.load_state_dict(state, strict=True)
        assert report.missing_keys == [] and report.unexpected_keys == []
        assert torch.equal(fresh.delta_weight(), layer.delta_weight())

    # ---- diagnostics ---------------------------------------------------------------

    def test_arm_diag_arms_both_halves_of_the_same_layer(self):
        # The two halves of ⟨ΔW_s, ΔW_t⟩_F must come from ONE layer. Walking a model for
        # "the first SlotProj" and "the first SlotOut" separately pairs them by module
        # registration order, and a mismatched pair has the right SHAPES in a model whose
        # layers are all d_in x d_out -- a wrong number, not an error.
        layer, gate = self._trained()
        layer.arm_diag()
        assert layer.slot_A._collect_diag is True
        assert layer.slot_B._collect_diag is True
        assert layer.slot_A._diag is None and layer.slot_B._diag is None
        with gate.scoped(_ids(0, 1, 0, -1)):
            layer(self._x())
        assert layer.slot_A._diag is not None and layer.slot_B._diag is not None
        assert layer.slot_A._collect_diag is False
        assert layer.slot_B._collect_diag is False

    def test_arm_diag_arms_only_the_layer_it_was_called_on(self):
        armed, gate = self._trained()
        other, _ = self._trained()
        other.slot_B.gate = gate
        armed.arm_diag()
        with gate.scoped(_ids(0, 1, 0, -1)):
            x = self._x()
            armed(x)
            other(x)
        assert armed.slot_A._diag is not None
        assert other.slot_A._diag is None and other.slot_B._diag is None

    def test_re_arming_without_a_forward_reads_absent_not_stale(self):
        layer, gate = self._trained()
        layer.arm_diag()
        with gate.scoped(_ids(0, 1, 0, -1)):
            layer(self._x())
        layer.arm_diag()
        assert layer.slot_A._diag is None and layer.slot_B._diag is None

    # ---- construction-time validation ----------------------------------------------

    def test_a_rank_wider_than_the_input_is_rejected(self):
        with pytest.raises(ValueError, match="total_rank <= in_features"):
            self._layer(in_features=4)  # sum(RANKS) == 8 > 4

    def test_a_gate_whose_slot_count_disagrees_is_rejected(self):
        with pytest.raises(ValueError, match="num_slots=4"):
            self._layer(num_slots=4)

    def test_a_base_that_carries_its_own_adapter_is_rejected(self):
        # A PEFT lora.Linear IS an nn.Linear subclass, and wrapping one would put a
        # second adapter inside `base` that delta_weight() knows nothing about: the
        # merged checkpoint would silently drop it.
        base = nn.Linear(16, 6, dtype=torch.float64)
        base.lora_A = nn.Linear(16, 4, bias=False, dtype=torch.float64)
        with pytest.raises(ValueError, match="child module"):
            SlotLoRALinear(base, self.RANKS, 1.0, SlotGate(num_slots=2))

    def test_a_non_linear_base_is_rejected(self):
        with pytest.raises(ValueError, match="nn.Linear"):
            SlotLoRALinear(
                nn.Conv1d(16, 6, 1, dtype=torch.float64),
                self.RANKS,
                1.0,
                SlotGate(num_slots=2),
            )


class TestSlotLoRALinearBf16:
    """bf16 is the production dtype; the rest of the class is exercised in fp64."""

    RANKS = (3, 5)

    def _layer(self, in_features=16, out_features=6, scale=0.7):
        torch.manual_seed(3)
        base = nn.Linear(in_features, out_features, dtype=torch.bfloat16)
        gate = SlotGate(num_slots=len(self.RANKS))
        layer = SlotLoRALinear(base, self.RANKS, scale, gate)
        with torch.no_grad():
            layer.slot_B.weight.copy_(torch.randn_like(layer.slot_B.weight) * 0.1)
        return layer, gate

    def test_bf16_base_gives_bf16_slots_and_a_bf16_output(self):
        layer, gate = self._layer()
        assert layer.slot_A.weight.dtype == torch.bfloat16
        assert layer.slot_B.weight.dtype == torch.bfloat16
        with gate.scoped(_ids(0, 1, 0, -1)):
            out = layer(torch.randn(4, 16, dtype=torch.bfloat16))
        assert out.dtype == torch.bfloat16
        assert out.shape == (4, 6)

    def test_bf16_backward_routes_gradient_to_the_owner_only(self):
        layer, gate = self._layer()
        x = torch.randn(4, 16, dtype=torch.bfloat16, requires_grad=True)
        with gate.scoped(torch.tensor([0, 0, 0, 0])):
            layer(x).sum().backward()
        assert layer.slot_B.weight.grad.dtype == torch.bfloat16
        assert torch.isfinite(layer.slot_B.weight.grad).all()
        assert layer.slot_B.weight.grad[:, 0:3].abs().sum() > 0
        assert torch.count_nonzero(layer.slot_B.weight.grad[:, 3:8]) == 0

    def test_the_merged_bf16_weight_agrees_to_the_bf16_rounding_floor(self):
        # The training forward computes Ā, h and the per-slot matmuls in bf16; the merge
        # forms ΔW in fp32 and rounds ONCE into the base weight. The two therefore agree
        # only to bf16 rounding, never bitwise: measured max|Δ| 3.906e-3 against max|out|
        # 2.281 here (relative 1.7e-3), and 3.4e-3 relative Frobenius at the production
        # shape (R=256, d_in=4096) -- see the delta_weight docstring.
        layer, gate = self._layer()
        x = torch.randn(4, 16, dtype=torch.bfloat16)
        with gate.scoped(_ids(0, 1, 0, -1)):
            trained = layer(x).float()
        merged = nn.Linear(16, 6, dtype=torch.bfloat16)
        with torch.no_grad():
            merged.weight.copy_(
                layer.base.weight + layer.delta_weight().to(torch.bfloat16)
            )
            merged.bias.copy_(layer.base.bias)
        assert (merged(x).float() - trained).abs().max() <= 2e-2 * trained.abs().max()
