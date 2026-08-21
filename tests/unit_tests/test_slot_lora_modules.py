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

from rlinf.models.slot_lora.modules import SlotGate, SlotOut, SlotProj


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
        gate, ids = SlotGate(), _ids()
        with gate.scoped(ids):
            assert gate.current() is ids
        assert gate.current_unchecked() is None

    def test_restores_on_exception(self):
        gate = SlotGate()
        with pytest.raises(RuntimeError, match="boom"):
            with gate.scoped(_ids()):
                raise RuntimeError("boom")
        assert gate.current_unchecked() is None

    def test_nested_scopes_restore_in_order(self):
        gate, outer, inner = SlotGate(), _ids(0), _ids(1)
        with gate.scoped(outer):
            with gate.scoped(inner):
                assert gate.current() is inner
            assert gate.current() is outer
        assert gate.current_unchecked() is None

    def test_one_holder_is_shared_by_every_gated_module(self):
        # Installing a routing must be O(1), not a walk over the 200-400 gated linears
        # of a 7B model: ONE holder, many references to it.
        gate = SlotGate()
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
        gate = SlotGate()
        with gate.scoped(_ids()):
            pass
        with pytest.raises(RuntimeError, match="no slot routing"):
            gate.current()

    def test_strict_raises_when_the_scope_carried_none(self):
        # The routing helper has several early-return paths that yield None. Under
        # strict, "the router produced nothing" is a BUG, not "quietly run ungated".
        gate = SlotGate()
        with gate.scoped(None):
            with pytest.raises(RuntimeError, match="no slot routing"):
                gate.current()

    def test_the_strict_message_names_the_way_out(self):
        with pytest.raises(RuntimeError, match=r"ungated\(\)"):
            SlotGate().current()

    def test_current_unchecked_never_raises(self):
        assert SlotGate().current_unchecked() is None

    def test_ungated_window_opts_out_and_restores_both_fields(self):
        gate, ids = SlotGate(), _ids()
        with gate.scoped(ids):
            with gate.ungated():
                assert gate.current() is None
                assert gate.strict is False
            assert gate.current() is ids
            assert gate.strict is True

    def test_ungated_window_restores_on_exception(self):
        gate = SlotGate()
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
        gate, ids = SlotGate(), _ids(-1, -1)
        with gate.scoped(ids):
            assert torch.equal(gate.current(), ids)

    def test_a_bad_gate_does_not_clobber_the_live_one(self):
        gate, good = SlotGate(), _ids()
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
        gate, ids = SlotGate(), _ids()
        with gate.scoped(ids):
            assert _in_a_fresh_thread(gate.current) is ids

    def test_a_fresh_thread_can_install_a_routing_the_main_thread_sees(self):
        gate, ids = SlotGate(), _ids()

        def install_and_read():
            with gate.scoped(ids):
                return gate.current()

        assert _in_a_fresh_thread(install_and_read) is ids
        assert gate.current_unchecked() is None

    def test_checkpoint_recomputation_sees_the_gate(self):
        gate, ids = SlotGate(), _ids(0, 1)
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
        gate, ids = SlotGate(), _ids(0, 1)
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
        gate, ids = SlotGate(), _ids(0, 1)
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
        p._collect_diag = True
        with torch.autocast("cpu", dtype=torch.bfloat16):
            p(torch.randn(4, 64, dtype=torch.bfloat16))
        gram = p._diag["gram"]
        assert gram.dtype == torch.float32
        eye = torch.eye(16, dtype=torch.float32)
        assert (gram - eye).norm().item() < 1e-4

    def test_diag_does_not_build_a_graph(self):
        p = self._proj()
        p._collect_diag = True
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
        gate = SlotGate(strict=strict)
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

    def test_rejects_routing_of_the_wrong_length(self):
        m, gate = self._out()
        h = self._h(batch=4)
        with gate.scoped(torch.tensor([0, 1])):
            try:
                m(h)
            except AssertionError:
                return
        raise AssertionError("expected an assertion on mismatched routing length")

    def test_strict_gate_raises_when_routing_was_never_installed(self):
        m, _ = self._out(strict=True)
        try:
            m(self._h())
        except RuntimeError:
            return
        raise AssertionError("expected a strict-gate RuntimeError")

    def test_survives_checkpoint_recomputation(self):
        # Gradient checkpointing re-runs this forward during backward, on the autograd
        # engine's worker thread. The gate must still be visible there, or gating is
        # silently off and every slot gets gradient from every sample.
        from torch.utils.checkpoint import checkpoint

        m, gate = self._out()
        h = self._h()
        with gate.scoped(torch.tensor([0, 0, 0, 0])):
            out = checkpoint(m, h, use_reentrant=False)
            out.sum().backward()
        assert torch.count_nonzero(m.weight.grad[:, 3:8]) == 0


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


class TestSlotOutDiag:
    """The B-side halves of ⟨ΔW_s, ΔW_t⟩_F = tr(B_sᵀB_t · Ā_tĀ_sᵀ).

    Only the small (R_t, R_s) blocks are ever materialized -- ΔW itself is d_out x d_in
    and forming it to measure it would defeat the point of measuring it.
    """

    RANKS = (3, 5)

    def _out(self, out_features=6, dtype=torch.float64):
        torch.manual_seed(2)
        m = SlotOut(out_features, self.RANKS, SlotGate(strict=False), dtype=dtype)
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
        m._collect_diag = True
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
        m._collect_diag = True
        with m.gate.ungated():
            m(self._h())
        b = m.weight.detach().float()
        b0, b1 = b[:, 0:3], b[:, 3:8]
        assert torch.allclose(
            m._diag["b_norms"], torch.stack([b0.norm(), b1.norm()]), atol=1e-6
        )
        assert torch.allclose(m._diag["cross"][(0, 1)], b1.T @ b0, atol=1e-6)

    def test_diag_stays_fp32_inside_an_ambient_autocast(self):
        # Same trap as SlotProj's Gram: autocast intercepts per op, so a bare matmul
        # forming BᵀB is demoted right back down under an ambient bf16 autocast.
        m = self._out(dtype=torch.bfloat16)
        m._collect_diag = True
        with torch.autocast("cpu", dtype=torch.bfloat16), m.gate.ungated():
            m(self._h(dtype=torch.bfloat16))
        assert m._diag["cross"][(0, 1)].dtype == torch.float32
        assert m._diag["b_norms"].dtype == torch.float32

    def test_diag_does_not_build_a_graph(self):
        m = self._out()
        m._collect_diag = True
        with m.gate.ungated():
            m(self._h())
        assert m._diag["b_norms"].requires_grad is False
        assert m._diag["cross"][(0, 1)].requires_grad is False
