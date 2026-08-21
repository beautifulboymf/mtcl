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

import logging

import pytest
import torch
import torch.nn as nn

from rlinf.models.slot_lora.inject import (
    collect_slot_diag,
    enable_slot_diag,
    inject_slot_lora,
)
from rlinf.models.slot_lora.modules import (
    SlotGate,
    SlotLoRALinear,
    SlotOut,
    SlotProj,
)
from rlinf.models.slot_lora.orth import (
    _DEFAULT_NS_ITERS,
    orth_error,
    orthogonalize,
)

DIM = 16
HIDDEN = 24
TARGETS = ("q_proj", "o_proj")
RANKS = (2, 3)
# The four paths a Toy() must yield, in named_modules order.
PATHS = (
    "blocks.0.attn.q_proj",
    "blocks.0.attn.o_proj",
    "blocks.1.attn.q_proj",
    "blocks.1.attn.o_proj",
)


class Attn(nn.Module):
    """Two targeted linears with DIFFERENT in_features, so a scale that reads
    out_features by mistake cannot pass ``test_match_mt4_scale_uses_in_features``.
    """

    def __init__(self, dim=DIM, hidden=HIDDEN):
        super().__init__()
        self.q_proj = nn.Linear(dim, hidden, dtype=torch.float64)
        self.o_proj = nn.Linear(hidden, dim, dtype=torch.float64)

    def forward(self, x):
        return self.o_proj(self.q_proj(x))


class Block(nn.Module):
    def __init__(self, dim=DIM, hidden=HIDDEN):
        super().__init__()
        self.attn = Attn(dim, hidden)
        self.mlp = nn.Linear(dim, dim, dtype=torch.float64)  # NOT a target

    def forward(self, x):
        return self.mlp(self.attn(x))


class Toy(nn.Module):
    """Targets nested two levels down (``blocks.i.attn.*``), untargeted linears at
    both the top level and inside every block.
    """

    def __init__(self, n_blocks=2, dim=DIM, hidden=HIDDEN, value_head=False):
        super().__init__()
        self.embed = nn.Linear(dim, dim, dtype=torch.float64)  # NOT a target
        self.blocks = nn.ModuleList(Block(dim, hidden) for _ in range(n_blocks))
        if value_head:
            self.value_head = nn.Linear(dim, 1, dtype=torch.float64)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return x


class PatchEmbed(nn.Module):
    """A ``Conv2d`` whose ATTRIBUTE NAME is a target, next to one that is not.

    Shaped after the real ``vision_backbone.featurizer.patch_embed.proj`` -- an
    ``nn.Conv2d`` sitting under a name in the shared target list, which PEFT adapts and
    the slot path cannot.
    """

    def __init__(self, dim=DIM):
        super().__init__()
        self.q_proj = nn.Conv2d(3, dim, kernel_size=2, dtype=torch.float64)
        self.patch_conv_only = nn.Conv2d(3, dim, kernel_size=2, dtype=torch.float64)


class ConvToy(Toy):
    """A :class:`Toy` with the name-matched ``Conv2d`` bolted on."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patch_embed = PatchEmbed()


class FSDPLike(nn.Module):
    """The two attribute behaviours of ``FullyShardedDataParallel`` that the
    diagnostics have to survive.

    VERIFIED against a real ``FSDP(module, auto_wrap_policy=<the repo's per-leaf
    lora policy>, use_orig_params=False)`` on a gloo world of one, CPU, torch
    2.6.0:

    * ``.modules()`` still yields the real ``SlotProj``/``SlotOut``/
      ``SlotLoRALinear`` instances, because the wrapped module is registered as the
      child ``_fsdp_wrapped_module`` -- so ``isinstance`` finding is safe.
    * ``layer.slot_A`` is the WRAPPER, not the ``SlotProj``. Reading
      ``layer.slot_A._diag`` forwards through ``__getattr__`` and returns the real
      module's value, but ASSIGNING ``layer.slot_A._diag = None`` writes into the
      wrapper's own ``__dict__`` and shadows the real attribute from then on --
      measured: after one such write, every later read returned the shadow even
      though the real ``SlotProj`` had refreshed its ``_diag``.

    So a ``collect_slot_diag`` that clears through the attribute silently stops
    reporting after its first call under FSDP. That is what
    ``TestDiagnosticsUnderFSDPWrapping`` pins down.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self._fsdp_wrapped_module = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("_fsdp_wrapped_module"), name)

    def forward(self, *args, **kwargs):
        return self._fsdp_wrapped_module(*args, **kwargs)


def _x(batch=3, dim=DIM):
    return torch.randn(batch, dim, dtype=torch.float64)


def _randomize_b(model, seed=0):
    """Give every slot a non-zero B, so the diagnostics have something to report."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, SlotOut):
                module.weight.copy_(
                    torch.randn(
                        module.weight.shape, generator=generator, dtype=torch.float64
                    )
                )
    return model


def _armed_layer(model):
    """The one SlotLoRALinear whose two halves both hold a fresh reading."""
    layers = [
        m
        for m in model.modules()
        if isinstance(m, SlotLoRALinear)
        and _half(m, SlotProj)._diag is not None
        and _half(m, SlotOut)._diag is not None
    ]
    assert len(layers) == 1, f"expected exactly one armed layer, got {len(layers)}"
    return layers[0]


def _armed_layer_of(model):
    """The layer ``enable_slot_diag`` will arm (named before arming, not after)."""
    return next(m for m in model.modules() if isinstance(m, SlotLoRALinear))


def _half(layer, cls):
    """The real ``SlotProj``/``SlotOut`` inside ``layer``, past any FSDP wrapper."""
    return next(m for m in layer.modules() if isinstance(m, cls))


class TestInjectSlotLora:
    def test_replaces_every_target_module(self):
        model = Toy()
        result = inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        assert tuple(result.paths) == PATHS
        assert isinstance(model.blocks[0].attn.q_proj, SlotLoRALinear)
        assert isinstance(model.blocks[1].attn.o_proj, SlotLoRALinear)
        assert sum(isinstance(m, SlotLoRALinear) for m in model.modules()) == 4

    def test_leaves_non_target_modules_alone(self):
        model = Toy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        assert type(model.embed) is nn.Linear
        assert type(model.blocks[0].mlp) is nn.Linear
        assert type(model.blocks[1].mlp) is nn.Linear

    def test_matches_on_the_attribute_name_not_the_type_alone(self):
        """Every nn.Linear whose attribute name is not in target_modules survives,
        even though it is structurally identical to one that is replaced.
        """
        model = Toy()
        inject_slot_lora(model, RANKS, ("q_proj",), scale_mode="unit")
        assert isinstance(model.blocks[0].attn.q_proj, SlotLoRALinear)
        assert type(model.blocks[0].attn.o_proj) is nn.Linear

    def test_does_not_rewrap_the_frozen_base(self):
        model = Toy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        inner = model.blocks[0].attn.q_proj.base
        assert type(inner) is nn.Linear
        assert not isinstance(inner, SlotLoRALinear)
        assert sum(isinstance(m, SlotProj) for m in model.modules()) == 4

    def test_a_second_injection_does_not_double_wrap(self):
        """The base of an already-injected layer keeps the target's attribute name
        only one level down (``q_proj.base``), so a walk that descends into it would
        wrap the frozen base a second time and hide it from delta_weight().
        """
        model = Toy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        with pytest.raises(ValueError, match="matched no nn.Linear"):
            inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        assert sum(isinstance(m, SlotLoRALinear) for m in model.modules()) == 4
        assert type(model.blocks[0].attn.q_proj.base) is nn.Linear

    def test_output_is_unchanged_at_init(self):
        model, x = Toy(), _x()
        before = model(x)
        result = inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        with result.gate.ungated():
            after = model(x)
        assert torch.allclose(after, before, atol=1e-12)

    def test_output_is_unchanged_at_init_with_a_routing_installed(self):
        model, x = Toy(), _x(batch=4)
        before = model(x)
        result = inject_slot_lora(model, RANKS, TARGETS, scale_mode="match_mt4")
        with result.gate.scoped(torch.tensor([0, 1, -1, 0])):
            after = model(x)
        assert torch.allclose(after, before, atol=1e-12)

    def test_only_slot_params_are_trainable(self):
        model = Toy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        seen = 0
        for name, param in model.named_parameters():
            expected = ".slot_A." in name or ".slot_B." in name
            assert param.requires_grad is expected, name
            seen += int(expected)
        assert seen == 8  # 4 layers x (Z, B)

    def test_freezes_a_pre_existing_trainable_head_and_names_it(self, caplog):
        """Freezing it is right (FSDP needs uniform requires_grad in a flat param),
        but a caller that meant to keep training it must not have to infer that from
        a metric that stops moving.
        """
        model = Toy(value_head=True)
        with caplog.at_level(logging.INFO, logger="rlinf.models.slot_lora.inject"):
            inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        assert model.value_head.weight.requires_grad is False
        assert "value_head.weight" in caplog.text

    def test_the_freeze_report_is_a_single_capped_line(self, caplog):
        """Before injection a freshly loaded HF model has requires_grad=True on
        everything, so an uncapped per-name report is thousands of lines on a 7B
        model -- which is a way of not being read.
        """
        model = Toy(n_blocks=6, value_head=True)
        with caplog.at_level(logging.INFO, logger="rlinf.models.slot_lora.inject"):
            inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        assert len(caplog.records) == 1
        assert caplog.records[0].getMessage().count("\n") == 0
        assert "more)" in caplog.records[0].getMessage()

    def test_match_mt4_scale_uses_in_features(self):
        model = Toy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="match_mt4", ref_rank=128)
        attn = model.blocks[0].attn
        assert abs(attn.q_proj.slot_A.scale - (DIM**0.5) / 128) < 1e-12
        assert abs(attn.o_proj.slot_A.scale - (HIDDEN**0.5) / 128) < 1e-12

    def test_match_mt4_scale_honours_ref_rank(self):
        model = Toy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="match_mt4", ref_rank=32)
        assert abs(model.blocks[0].attn.q_proj.slot_A.scale - (DIM**0.5) / 32) < 1e-12

    def test_unit_scale_is_one(self):
        model = Toy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit", ref_rank=32)
        assert model.blocks[0].attn.q_proj.slot_A.scale == 1.0
        assert model.blocks[0].attn.o_proj.slot_A.scale == 1.0

    def test_rejects_unknown_scale_mode_without_touching_the_model(self):
        model = Toy()
        with pytest.raises(ValueError, match="scale_mode"):
            inject_slot_lora(model, RANKS, TARGETS, scale_mode="nope")
        assert type(model.blocks[0].attn.q_proj) is nn.Linear
        assert model.embed.weight.requires_grad is True

    def test_rejects_a_non_positive_ref_rank(self):
        with pytest.raises(ValueError, match="ref_rank"):
            inject_slot_lora(Toy(), RANKS, TARGETS, scale_mode="match_mt4", ref_rank=0)

    def test_rejects_a_target_list_that_matches_nothing(self):
        """A typo in target_modules would otherwise freeze the whole model and
        return an adapter with no slots at all -- a run that trains nothing.
        """
        model = Toy()
        with pytest.raises(ValueError, match="matched no nn.Linear"):
            inject_slot_lora(model, RANKS, ("qkv_proj",), scale_mode="unit")
        assert model.embed.weight.requires_grad is True

    def test_eps_reaches_the_projection(self):
        model = Toy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit", eps=1e-4)
        assert model.blocks[0].attn.q_proj.slot_A.eps == 1e-4

    def test_iters_reaches_the_projection(self):
        model = Toy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit", iters=16)
        assert model.blocks[0].attn.q_proj.slot_A.iters == 16

    def test_iters_defaults_to_the_module_constant(self):
        model = Toy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        assert model.blocks[0].attn.q_proj.slot_A.iters == _DEFAULT_NS_ITERS

    def test_iters_actually_changes_the_orthogonalization(self):
        """Not just stored: a starved iteration count must show up in ``Ā``.

        One Newton-Schulz step cannot converge, so ``orth_error`` at ``iters=1`` has to
        be orders of magnitude worse than at the default. Without this the knob could
        be threaded to the attribute and dropped on the way to ``orthogonalize``, and
        the documented remediation ("raise iters from 12 to 16") would still do nothing.
        """
        starved, healthy = Toy(), Toy()
        starved.load_state_dict(healthy.state_dict())
        inject_slot_lora(starved, RANKS, TARGETS, scale_mode="unit", iters=1)
        inject_slot_lora(healthy, RANKS, TARGETS, scale_mode="unit")
        starved_proj = starved.blocks[0].attn.q_proj.slot_A
        healthy_proj = healthy.blocks[0].attn.q_proj.slot_A
        # Same Z in both, so the only difference is the iteration count.
        with torch.no_grad():
            starved_proj.weight.copy_(healthy_proj.weight)
        assert orth_error(starved_proj.orth_weight(collect=False)) > 100 * orth_error(
            healthy_proj.orth_weight(collect=False)
        )


class TestNameMatchedButWrongType:
    """A child whose NAME is a target but whose TYPE is not an ``nn.Linear``.

    This is not hypothetical. Two modules of the real student --
    ``vision_backbone.featurizer.patch_embed.proj`` and
    ``vision_backbone.fused_featurizer.patch_embed.proj``, both ``Conv2d(3, 128, 14)``
    -- match ``"proj"`` in ``SLOT_LORA_TARGET_MODULES``, and PEFT adapts them (measured
    off the mt4 baseline's own adapter: 439 ``lora_A`` tensors, 437 of them 2-D and 2 of
    them 4-D). The slot path cannot, so it adapts 437 and leaves those two frozen. That
    divergence from the baseline is legitimate but it must be COUNTED and SAID, never
    inferred from a parameter count.
    """

    def test_a_name_matched_conv_is_reported_not_silently_dropped(self):
        model = ConvToy()
        result = inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        assert result.skipped == (("patch_embed.q_proj", "Conv2d"),)
        assert "patch_embed.q_proj" not in result.paths

    def test_the_skipped_module_is_left_alone_and_frozen(self):
        model = ConvToy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        assert type(model.patch_embed.q_proj) is nn.Conv2d
        assert model.patch_embed.q_proj.weight.requires_grad is False

    def test_the_linears_around_it_are_still_adapted(self):
        model = ConvToy()
        result = inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        assert tuple(result.paths) == PATHS

    def test_it_is_logged_once_at_warning_with_the_path_and_the_type(self, caplog):
        with caplog.at_level(logging.WARNING, logger="rlinf.models.slot_lora.inject"):
            inject_slot_lora(ConvToy(), RANKS, TARGETS, scale_mode="unit")
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert message.count("\n") == 0
        assert "patch_embed.q_proj" in message
        assert "Conv2d" in message

    def test_nothing_is_logged_at_warning_when_every_target_is_a_linear(self, caplog):
        with caplog.at_level(logging.WARNING, logger="rlinf.models.slot_lora.inject"):
            result = inject_slot_lora(Toy(), RANKS, TARGETS, scale_mode="unit")
        assert result.skipped == ()
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_a_target_list_matching_only_non_linears_still_raises(self):
        """The "matched no nn.Linear" guard must survive the new bookkeeping."""
        model = ConvToy()
        with pytest.raises(ValueError, match="matched no nn.Linear"):
            inject_slot_lora(model, RANKS, ("patch_conv_only",), scale_mode="unit")


class TestSharedGate:
    def test_one_gate_instance_reaches_every_layer(self):
        model = Toy()
        result = inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        gates = {id(m.gate) for m in model.modules() if isinstance(m, SlotOut)}
        assert gates == {id(result.gate)}

    def test_the_gate_knows_the_slot_count(self):
        result = inject_slot_lora(Toy(), RANKS, TARGETS, scale_mode="unit")
        assert isinstance(result.gate, SlotGate)
        assert result.gate.num_slots == len(RANKS)
        assert result.gate.strict is True

    def test_strict_gate_flag_is_honoured(self):
        result = inject_slot_lora(
            Toy(), RANKS, TARGETS, scale_mode="unit", strict_gate=False
        )
        assert result.gate.strict is False
        # A non-strict gate runs ungated with no scope at all.
        assert result.gate.current() is None

    def test_a_strict_gate_refuses_an_unrouted_forward(self):
        model = Toy()
        inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        with pytest.raises(RuntimeError, match="no slot routing installed"):
            model(_x())

    def test_one_scope_routes_every_layer(self):
        model = Toy()
        result = inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        _randomize_b(model)
        with result.gate.scoped(torch.tensor([0, 0, 0])):
            model(_x()).sum().backward()
        for layer in (m for m in model.modules() if isinstance(m, SlotLoRALinear)):
            grad = layer.slot_B.weight.grad
            assert grad is not None
            # slot 0 owns columns 0:2, and nothing else took gradient.
            assert grad[:, : RANKS[0]].abs().sum() > 0
            assert grad[:, RANKS[0] :].abs().sum() == 0


class TestReturnValue:
    def test_paths_are_dotted_and_resolvable(self):
        model = Toy()
        result = inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        by_name = dict(model.named_modules())
        for path in result.paths:
            assert isinstance(by_name[path], SlotLoRALinear)

    def test_result_is_not_a_bare_sequence(self):
        """``len(result)`` / ``" ".join(result)`` were how the paths list used to be
        consumed. Both must raise rather than quietly report the field count.
        """
        result = inject_slot_lora(Toy(), RANKS, TARGETS, scale_mode="unit")
        with pytest.raises(TypeError):
            len(result)
        with pytest.raises(TypeError):
            iter(result)


class TestSlotDiagnostics:
    def _model(self, scale_mode="unit", ranks=RANKS, seed=0):
        model = Toy()
        result = inject_slot_lora(model, ranks, TARGETS, scale_mode=scale_mode)
        _randomize_b(model, seed=seed)
        return model, result

    def _forward(self, model, result, batch=3):
        with result.gate.ungated():
            model(_x(batch))

    def test_enable_then_collect_returns_metrics(self):
        model, result = self._model()
        assert enable_slot_diag(model) is True
        self._forward(model, result)
        diag = collect_slot_diag(model)
        assert set(diag) == {
            "slot/orth_err",
            "slot/dw_norm_0",
            "slot/dw_norm_1",
            "slot/cos_0_1",
        }
        assert all(isinstance(v, float) for v in diag.values())
        assert diag["slot/orth_err"] < 1e-3  # the "healthy" threshold
        assert abs(diag["slot/cos_0_1"]) < 1e-3  # a real Ā really is orthogonal
        assert diag["slot/dw_norm_0"] > 0.0
        assert diag["slot/dw_norm_1"] > 0.0

    def test_orth_err_matches_an_independent_recomputation(self):
        model, result = self._model()
        enable_slot_diag(model)
        self._forward(model, result)
        layer = _armed_layer(model)
        proj = _half(layer, SlotProj)
        reference = orth_error(
            orthogonalize(proj.weight.detach().float(), eps=proj.eps)
        ).item()
        diag = collect_slot_diag(model)
        assert diag["slot/orth_err"] == pytest.approx(reference, rel=1e-5, abs=1e-9)

    def test_dw_norm_is_scale_times_b_norm(self):
        model, result = self._model(scale_mode="match_mt4")
        enable_slot_diag(model)
        self._forward(model, result)
        layer = _armed_layer(model)
        scale = _half(layer, SlotProj).scale
        out = _half(layer, SlotOut)
        diag = collect_slot_diag(model)
        for k, (start, rank) in enumerate(zip(out.offsets, out.slot_ranks)):
            expected = scale * out.weight[:, start : start + rank].float().norm().item()
            assert diag[f"slot/dw_norm_{k}"] == pytest.approx(expected, rel=1e-6)

    def test_three_slots_report_every_pair(self):
        model, result = self._model(ranks=(2, 3, 4))
        enable_slot_diag(model)
        self._forward(model, result)
        diag = collect_slot_diag(model)
        assert set(diag) == {
            "slot/orth_err",
            "slot/dw_norm_0",
            "slot/dw_norm_1",
            "slot/dw_norm_2",
            "slot/cos_0_1",
            "slot/cos_0_2",
            "slot/cos_1_2",
        }

    def test_collect_without_enable_returns_empty(self):
        model, result = self._model()
        self._forward(model, result)
        assert collect_slot_diag(model) == {}

    def test_collect_without_a_forward_returns_empty(self):
        model, _ = self._model()
        assert enable_slot_diag(model) is True
        assert collect_slot_diag(model) == {}

    def test_collect_clears_the_reading_so_it_can_never_go_stale(self):
        model, result = self._model()
        enable_slot_diag(model)
        self._forward(model, result)
        assert collect_slot_diag(model) != {}
        # A second read with no re-arm and no forward must report NOTHING rather
        # than last step's constant.
        assert collect_slot_diag(model) == {}

    def test_enable_returns_false_without_slots(self):
        assert enable_slot_diag(Toy()) is False
        assert collect_slot_diag(Toy()) == {}

    def test_exactly_one_layer_is_armed(self):
        model, result = self._model()
        enable_slot_diag(model)
        self._forward(model, result)
        armed = [m for m in model.modules() if isinstance(m, SlotProj)]
        assert sum(m._diag is not None for m in armed) == 1
        assert sum(m._collect_diag for m in armed) == 0  # the forward disarmed it

    def test_the_two_halves_come_from_the_same_layer(self):
        """The Gram and the B cross-products must be this layer's. Only the first
        layer may hold a reading; every other half must be empty.
        """
        model, result = self._model()
        enable_slot_diag(model)
        self._forward(model, result)
        layer = _armed_layer(model)
        assert layer is next(
            m for m in model.modules() if isinstance(m, SlotLoRALinear)
        )
        others = [
            m
            for m in model.modules()
            if isinstance(m, (SlotProj, SlotOut))
            and m not in (layer.slot_A, layer.slot_B)
        ]
        assert all(m._diag is None for m in others)

    def test_metrics_survive_a_re_arm(self):
        model, result = self._model()
        for _ in range(3):
            assert enable_slot_diag(model) is True
            self._forward(model, result)
            assert collect_slot_diag(model) != {}


class TestCosineAgainstMaterializedDeltaW:
    """The test that discriminates the two silent traps in the cosine.

    ``cross[(s, t)] = B_tᵀB_s`` has shape ``(r_t, r_s)`` and pairs with
    ``gram[t_slice, s_slice]``. With a genuinely orthonormal ``Ā`` both slicings
    give ~0 and neither the transpose nor the ``scale²`` mistake is visible, so
    this builds a ``Ā`` whose row blocks are each orthonormal ON THEIR OWN (which
    keeps ``‖ΔW_k‖_F = scale·‖B_k‖_F`` exact, the identity ``dw_norm_k`` relies
    on) but are NOT mutually orthogonal, so the off-diagonal Gram blocks are
    non-zero and asymmetric.

    ``SlotProj`` always orthogonalizes, so such an ``Ā`` cannot come out of a real
    ``Z``; only the Gram is substituted, and it is substituted into the very object
    ``SlotProj.orth_weight`` stashes. The B side stays entirely authentic.
    """

    @staticmethod
    def _blocks_orthonormal_but_not_mutually(ranks, d_in, seed):
        generator = torch.Generator().manual_seed(seed)
        blocks = []
        for rank in ranks:
            raw = torch.randn(d_in, rank, generator=generator, dtype=torch.float64)
            q, _ = torch.linalg.qr(raw)  # (d_in, rank), orthonormal COLUMNS
            blocks.append(q.transpose(0, 1))  # (rank, d_in), orthonormal ROWS
        return torch.cat(blocks, dim=0)

    def _run(self, ranks, seed):
        model = Toy()
        result = inject_slot_lora(model, ranks, TARGETS, scale_mode="match_mt4")
        _randomize_b(model, seed=seed)
        enable_slot_diag(model)
        with result.gate.ungated():
            model(_x())
        layer = _armed_layer(model)
        proj, out = _half(layer, SlotProj), _half(layer, SlotOut)

        a = self._blocks_orthonormal_but_not_mutually(
            ranks, layer.base.in_features, seed
        )
        # Substituted in fp32, exactly as SlotProj.orth_weight stashes it.
        proj._diag = {"gram": (a @ a.transpose(0, 1)).float(), "scale": proj.scale}

        scale = proj.scale
        b = out.weight.detach().double()
        deltas = [
            scale * b[:, s : s + r] @ a[s : s + r]
            for s, r in zip(out.offsets, out.slot_ranks)
        ]
        diag = collect_slot_diag(model)
        return diag, deltas, a, b, out

    @pytest.mark.parametrize("ranks", [(2, 3), (4, 4, 4), (2, 3, 4)])
    def test_cosine_equals_the_materialized_delta_w_cosine(self, ranks):
        diag, deltas, _, _, _ = self._run(ranks, seed=7)
        for s in range(len(ranks)):
            assert diag[f"slot/dw_norm_{s}"] == pytest.approx(
                deltas[s].norm().item(), rel=1e-5
            )
            for t in range(s + 1, len(ranks)):
                inner = (deltas[s] * deltas[t]).sum()
                expected = (inner / (deltas[s].norm() * deltas[t].norm())).item()
                assert diag[f"slot/cos_{s}_{t}"] == pytest.approx(expected, rel=1e-5)

    def test_the_transposed_gram_slice_would_give_a_different_answer(self):
        """Proves the test above can actually SEE the mistake it exists to catch.

        With EQUAL per-slot ranks ``gram[s_slice, t_slice]`` has the same shape as
        ``gram[t_slice, s_slice]``, so the wrong slicing raises nothing and simply
        reports a different number.
        """
        ranks = (4, 4, 4)
        diag, _, a, b, out = self._run(ranks, seed=7)
        gram = a @ a.transpose(0, 1)
        for s in range(len(ranks)):
            for t in range(s + 1, len(ranks)):
                s_slice = slice(out.offsets[s], out.offsets[s] + out.slot_ranks[s])
                t_slice = slice(out.offsets[t], out.offsets[t] + out.slot_ranks[t])
                cross = b[:, t_slice].transpose(0, 1) @ b[:, s_slice]  # (r_t, r_s)
                denom = b[:, s_slice].norm() * b[:, t_slice].norm()
                right = ((cross * gram[t_slice, s_slice]).sum() / denom).item()
                wrong = ((cross * gram[s_slice, t_slice]).sum() / denom).item()
                assert diag[f"slot/cos_{s}_{t}"] == pytest.approx(right, rel=1e-5)
                assert abs(wrong - right) > 1e-3 * max(abs(right), 1e-6)

    def test_unequal_ranks_turn_the_wrong_slice_into_a_shape_error(self):
        """The other half of the story: production uses 128/64/48/16, where the
        wrong slicing cannot even be evaluated. Equal ranks are the dangerous case.
        """
        ranks = (2, 3, 4)
        _, _, a, b, out = self._run(ranks, seed=7)
        gram = a @ a.transpose(0, 1)
        s_slice = slice(out.offsets[0], out.offsets[0] + out.slot_ranks[0])
        t_slice = slice(out.offsets[1], out.offsets[1] + out.slot_ranks[1])
        cross = b[:, t_slice].transpose(0, 1) @ b[:, s_slice]
        assert cross.shape == gram[t_slice, s_slice].shape
        with pytest.raises(RuntimeError):
            cross * gram[s_slice, t_slice]

    def test_the_scale_cancels_in_the_cosine_but_not_in_dw_norm(self):
        """``⟨ΔW_s, ΔW_t⟩ = scale²·Σ(...)`` and ``‖ΔW_k‖ = scale·‖B_k‖``, so a
        ``scale²`` applied to the cosine would rescale a scale-invariant number.
        """
        cosines, norms = [], []
        for ref_rank in (128, 32):
            model = Toy()
            result = inject_slot_lora(
                model, RANKS, TARGETS, scale_mode="match_mt4", ref_rank=ref_rank
            )
            _randomize_b(model, seed=3)
            layer = _armed_layer_of(model)
            # Force a non-trivial cosine: without it both readings are ~0 and the
            # comparison proves nothing.
            proj = _half(layer, SlotProj)
            a = self._blocks_orthonormal_but_not_mutually(
                RANKS, layer.base.in_features, seed=3
            )
            enable_slot_diag(model)
            with result.gate.ungated():
                model(_x())
            proj._diag = {"gram": (a @ a.transpose(0, 1)).float(), "scale": proj.scale}
            diag = collect_slot_diag(model)
            cosines.append(diag["slot/cos_0_1"])
            norms.append(diag["slot/dw_norm_0"])
        assert abs(cosines[0]) > 1e-3
        assert cosines[0] == pytest.approx(cosines[1], rel=1e-6)
        assert norms[1] == pytest.approx(norms[0] * 4.0, rel=1e-6)


class TestDiagnosticsUnderFSDPWrapping:
    """See :class:`FSDPLike` for what is being reproduced and how it was verified."""

    def _model(self):
        model = Toy()
        result = inject_slot_lora(model, RANKS, TARGETS, scale_mode="unit")
        _randomize_b(model)
        for layer in [m for m in model.modules() if isinstance(m, SlotLoRALinear)]:
            layer.slot_A = FSDPLike(layer.slot_A)
            layer.slot_B = FSDPLike(layer.slot_B)
        return model, result

    def test_modules_still_finds_the_real_instances(self):
        model, _ = self._model()
        assert sum(isinstance(m, SlotProj) for m in model.modules()) == 4
        assert sum(isinstance(m, SlotOut) for m in model.modules()) == 4
        assert sum(isinstance(m, SlotLoRALinear) for m in model.modules()) == 4

    def test_arm_forward_collect_works_through_the_wrapper(self):
        model, result = self._model()
        assert enable_slot_diag(model) is True
        with result.gate.ungated():
            model(_x())
        diag = collect_slot_diag(model)
        assert diag["slot/dw_norm_0"] > 0.0

    def test_collect_clears_the_real_module_not_the_wrapper(self):
        model, result = self._model()
        layer = _armed_layer_of(model)
        enable_slot_diag(model)
        with result.gate.ungated():
            model(_x())
        collect_slot_diag(model)
        assert _half(layer, SlotProj)._diag is None
        assert _half(layer, SlotOut)._diag is None
        assert "_diag" not in layer.slot_A.__dict__
        assert "_diag" not in layer.slot_B.__dict__

    def test_a_second_arm_collect_cycle_still_reports(self):
        """The regression: clearing through ``layer.slot_A._diag = None`` shadows the
        real attribute, and every later collect returns ``{}`` with nothing raised.
        """
        model, result = self._model()
        for _ in range(3):
            assert enable_slot_diag(model) is True
            with result.gate.ungated():
                model(_x())
            assert collect_slot_diag(model) != {}
