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

"""The slot-LoRI branch of ``rlinf.models.get_model``.

The real caller hands ``get_model`` a 7B VLA and an ``omegaconf`` config, and neither
half of that is needed to exercise the branch: what the branch does is read config,
choose a path, and inject. So the model here is a toy with two targeted linears
registered through the PUBLIC :func:`register_model` API (the same door a new model
type comes in by), and the config is a real ``DictConfig``, because that is what the
caller passes and its ``.get`` semantics on missing/None keys are part of what is
under test.

``get_peft_model`` is monkeypatched to a spy rather than left to run. That is not
avoidance: WHICH path was taken is exactly the thing these tests are about, and the
spy also captures the ``LoraConfig`` the PEFT path built, which is how
``test_peft_branch_reads_the_hoisted_target_list`` can prove the two arms of the
comparison target the same modules.
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import rlinf.models as models
from rlinf.config import SupportedModel
from rlinf.models import (
    SLOT_LORA_TARGET_MODULES,
    _assert_slot_leaves_fsdp_wrappable,
    find_slot_gate,
    get_model,
    register_model,
)
from rlinf.models.slot_lora import SlotGate, SlotLoRALinear, SlotOut, SlotProj
from rlinf.models.slot_lora.orth import _DEFAULT_NS_ITERS

DIM = 8
MODEL_TYPE = "slot_lora_unit_test_toy"
# Two slots with DIFFERENT ranks, and a slot_ranks mapping whose INSERTION order is the
# reverse of slot_order: a rank list built by iterating the mapping comes out (3, 2).
ORDER = ["libero_spatial", "libero_object"]
RANKS = OmegaConf.create({"libero_object": 3, "libero_spatial": 2})
EXPECTED_RANKS = (2, 3)


class Attn(nn.Module):
    """Two targeted linears (``q_proj``, ``k_proj``) and one untargeted (``mlp``)."""

    def __init__(self, dim=DIM):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.mlp = nn.Linear(dim, dim)  # NOT in SLOT_LORA_TARGET_MODULES

    def forward(self, x):
        return self.mlp(self.k_proj(self.q_proj(x)))


class Toy(nn.Module):
    def __init__(self, value_head=False):
        super().__init__()
        self.attn = Attn()
        if value_head:
            # Shaped like rlinf.models.embodiment.modules.ValueHead: an nn.Sequential
            # of childless nn.Linear leaves, which is what makes re-enabling it
            # FSDP-safe under the per-leaf wrap policy.
            self.value_head = nn.Sequential(
                nn.Linear(DIM, 4), nn.GELU(), nn.Linear(4, 1)
            )

    def forward(self, x):
        return self.attn(x)


@pytest.fixture
def toy_type():
    """Register the toy under a model type, and take it back out again.

    Registration is global (``_MODEL_REGISTRY`` and ``SupportedModel.models``), so it
    is undone on the way out; a leaked test-only model type would show up in the
    "supported models" list of every later failure message in the same session.
    """

    def build(cfg, torch_dtype):
        return Toy(value_head=bool(cfg.get("add_value_head", False)))

    # category != "embodied" keeps it out of EMBODIED_MODEL, which nothing here needs.
    register_model(MODEL_TYPE, build, category="unit_test", force=True)
    yield MODEL_TYPE
    models._MODEL_REGISTRY.pop(MODEL_TYPE, None)
    SupportedModel.models.pop(MODEL_TYPE, None)


@pytest.fixture
def peft_spy(monkeypatch):
    """Record every ``get_peft_model`` call instead of making one.

    ``get_model`` does ``from peft import ... get_peft_model`` INSIDE the function, so
    the attribute is looked up on the module at call time and patching it here is what
    the branch sees.
    """
    import peft

    calls = []

    def fake_get_peft_model(model, config, *args, **kwargs):
        calls.append((model, config))
        return model

    monkeypatch.setattr(peft, "get_peft_model", fake_get_peft_model)
    return calls


def make_cfg(slot_lora=None, **overrides):
    cfg = {
        "model_type": MODEL_TYPE,
        "precision": "fp32",
        "is_lora": True,
        "lora_rank": 8,
        "lora_path": None,
        # Keeps the test off the GPU: without it get_model would .to("cuda") and open a
        # CUDA context on a box whose cards belong to someone else.
        "load_to_device": False,
    }
    if slot_lora is not None:
        cfg["slot_lora"] = slot_lora
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def slot_cfg(**overrides):
    cfg = {"enabled": True, "slot_ranks": RANKS, "slot_order": ORDER}
    cfg.update(overrides)
    return cfg


def slot_layers(model):
    return [m for m in model.modules() if isinstance(m, SlotLoRALinear)]


class TestWhichBranchIsTaken:
    def test_slot_config_injects_slots_and_never_calls_peft(self, toy_type, peft_spy):
        model = get_model(make_cfg(slot_cfg()))

        assert peft_spy == []
        assert isinstance(model.attn.q_proj, SlotLoRALinear)
        assert isinstance(model.attn.k_proj, SlotLoRALinear)
        # The untargeted linear is untouched, so "adapt everything" cannot pass.
        assert isinstance(model.attn.mlp, nn.Linear)
        assert not isinstance(model.attn.mlp, SlotLoRALinear)

    def test_plain_lora_config_takes_the_peft_path(self, toy_type, peft_spy):
        model = get_model(make_cfg())

        assert len(peft_spy) == 1
        assert slot_layers(model) == []
        assert find_slot_gate(model) is None

    def test_slot_lora_disabled_takes_the_peft_path(self, toy_type, peft_spy):
        model = get_model(make_cfg(slot_cfg(enabled=False)))

        assert len(peft_spy) == 1
        assert slot_layers(model) == []

    def test_slot_lora_null_takes_the_peft_path(self, toy_type, peft_spy):
        """``slot_lora:`` with nothing under it is None, not an empty mapping."""
        cfg = make_cfg()
        cfg.slot_lora = None
        model = get_model(cfg)

        assert len(peft_spy) == 1
        assert slot_layers(model) == []

    def test_is_lora_false_injects_nothing(self, toy_type, peft_spy):
        """The slot path lives INSIDE `if cfg.is_lora:` on purpose: the per-leaf FSDP
        wrap policy the slots need is only registered when is_lora is set."""
        model = get_model(make_cfg(slot_cfg(), is_lora=False))

        assert peft_spy == []
        assert slot_layers(model) == []


class TestTargetModules:
    def test_peft_branch_reads_the_hoisted_target_list(self, toy_type, peft_spy):
        """The comparison is only a comparison if both arms adapt the same modules."""
        get_model(make_cfg())

        _, lora_config = peft_spy[0]
        assert set(lora_config.target_modules) == set(SLOT_LORA_TARGET_MODULES)
        assert len(SLOT_LORA_TARGET_MODULES) == len(set(SLOT_LORA_TARGET_MODULES))

    def test_slot_branch_adapts_exactly_the_same_names(self, toy_type, peft_spy):
        """Every slot-adapted path ends in a name from the shared constant."""
        model = get_model(make_cfg(slot_cfg()))

        adapted = [
            name
            for name, module in model.named_modules()
            if isinstance(module, SlotLoRALinear)
        ]
        assert adapted == ["attn.q_proj", "attn.k_proj"]
        assert all(name.split(".")[-1] in SLOT_LORA_TARGET_MODULES for name in adapted)

    def test_peft_cannot_mutate_the_shared_constant(self, toy_type, peft_spy):
        before = list(SLOT_LORA_TARGET_MODULES)
        get_model(make_cfg())
        assert SLOT_LORA_TARGET_MODULES == before


class TestRankOrder:
    def test_rank_list_follows_slot_order_not_mapping_order(self, toy_type, peft_spy):
        """RANKS is inserted object-then-spatial; ORDER is spatial-then-object."""
        assert list(RANKS.keys()) == list(reversed(ORDER)), "fixture no longer tests it"

        model = get_model(make_cfg(slot_cfg()))

        for layer in slot_layers(model):
            assert layer.slot_B.slot_ranks == EXPECTED_RANKS

    def test_reversing_slot_order_reverses_the_ranks(self, toy_type, peft_spy):
        """The pairing is by NAME, not a coincidence of two-element tuples."""
        model = get_model(make_cfg(slot_cfg(slot_order=list(reversed(ORDER)))))

        for layer in slot_layers(model):
            assert layer.slot_B.slot_ranks == tuple(reversed(EXPECTED_RANKS))

    def test_total_rank_is_the_sum(self, toy_type, peft_spy):
        model = get_model(make_cfg(slot_cfg()))

        assert model.attn.q_proj.slot_A.total_rank == sum(EXPECTED_RANKS)
        assert model.attn.q_proj.slot_A.weight.shape == (sum(EXPECTED_RANKS), DIM)


class TestGateIsReachable:
    def test_find_slot_gate_returns_the_gate_every_layer_holds(
        self, toy_type, peft_spy
    ):
        model = get_model(make_cfg(slot_cfg()))

        gate = find_slot_gate(model)
        assert isinstance(gate, SlotGate)
        assert gate.num_slots == len(ORDER)
        assert gate.strict is True
        for layer in slot_layers(model):
            assert layer.slot_B.gate is gate

    def test_the_stashed_handle_and_the_walk_agree(self, toy_type, peft_spy):
        model = get_model(make_cfg(slot_cfg()))

        walked = next(m for m in model.modules() if isinstance(m, SlotOut)).gate
        assert model._slot_gate is walked

    def test_find_slot_gate_falls_back_to_the_walk(self, toy_type, peft_spy):
        """A model that arrived without the stashed handle still answers."""
        model = get_model(make_cfg(slot_cfg()))
        gate = model._slot_gate
        del model._slot_gate

        assert find_slot_gate(model) is gate

    def test_the_gate_actually_routes_a_forward(self, toy_type, peft_spy):
        """End to end: the handle a caller reaches is the one the forward reads."""
        model = get_model(make_cfg(slot_cfg()))
        gate = find_slot_gate(model)
        x = torch.randn(2, DIM)

        with pytest.raises(RuntimeError, match="no slot routing installed"):
            model(x)
        with gate.scoped(torch.tensor([0, 1])):
            model(x)  # must not raise

    def test_slot_order_is_stashed_for_the_router(self, toy_type, peft_spy):
        model = get_model(make_cfg(slot_cfg()))

        assert model._slot_order == tuple(ORDER)

    def test_gate_and_order_stay_out_of_the_state_dict(self, toy_type, peft_spy):
        """Neither is a Parameter, a buffer or a Module, so neither is checkpointed."""
        model = get_model(make_cfg(slot_cfg()))

        assert not any(
            "_slot_gate" in k or "_slot_order" in k for k in model.state_dict()
        )


class TestConfigErrors:
    @pytest.mark.parametrize("key", ["slot_ranks", "slot_order"])
    def test_missing_required_key_raises(self, toy_type, peft_spy, key):
        cfg = slot_cfg()
        del cfg[key]

        with pytest.raises(ValueError, match=key):
            get_model(make_cfg(cfg))

    @pytest.mark.parametrize("key", ["slot_ranks", "slot_order"])
    def test_null_required_key_raises(self, toy_type, peft_spy, key):
        with pytest.raises(ValueError, match=key):
            get_model(make_cfg(slot_cfg(**{key: None})))

    def test_repeated_suite_in_slot_order_raises(self, toy_type, peft_spy):
        with pytest.raises(ValueError, match="repeats"):
            get_model(make_cfg(slot_cfg(slot_order=[ORDER[0], ORDER[1], ORDER[0]])))

    def test_suite_without_a_rank_raises(self, toy_type, peft_spy):
        with pytest.raises(ValueError, match="libero_goal"):
            get_model(make_cfg(slot_cfg(slot_order=ORDER + ["libero_goal"])))

    def test_rank_for_an_unlisted_suite_raises(self, toy_type, peft_spy):
        """The typo that would otherwise silently build a model with fewer slots."""
        ranks = OmegaConf.create({**RANKS, "libero_goal": 4})

        with pytest.raises(ValueError, match="libero_goal"):
            get_model(make_cfg(slot_cfg(slot_ranks=ranks)))

    def test_slot_ranks_as_a_bare_list_raises(self, toy_type, peft_spy):
        with pytest.raises(ValueError, match="mapping"):
            get_model(make_cfg(slot_cfg(slot_ranks=[2, 3])))

    def test_slot_order_as_a_bare_string_raises(self, toy_type, peft_spy):
        """Otherwise it iterates into one slot per CHARACTER."""
        with pytest.raises(ValueError, match="LIST of suite names"):
            get_model(make_cfg(slot_cfg(slot_order=ORDER[0])))

    def test_lora_path_with_slots_raises(self, toy_type, peft_spy):
        """Otherwise the adapter is silently never loaded."""
        with pytest.raises(ValueError, match="lora_path"):
            get_model(make_cfg(slot_cfg(), lora_path="/some/adapter"))

    def test_a_config_error_never_reaches_peft(self, toy_type, peft_spy):
        with pytest.raises(ValueError):
            get_model(make_cfg(slot_cfg(slot_order=[])))
        assert peft_spy == []


class TestScaleAndOptionalKeys:
    def test_defaults_are_match_mt4_over_ref_rank_128(self, toy_type, peft_spy):
        model = get_model(make_cfg(slot_cfg()))

        assert model.attn.q_proj.slot_A.scale == pytest.approx(DIM**0.5 / 128)
        assert model.attn.q_proj.slot_A.eps == pytest.approx(1e-6)

    def test_a_scale_mode_unit_is_honoured(self, toy_type, peft_spy):
        model = get_model(make_cfg(slot_cfg(a_scale_mode="unit")))

        assert model.attn.q_proj.slot_A.scale == 1.0

    def test_a_scale_ref_rank_is_honoured(self, toy_type, peft_spy):
        model = get_model(make_cfg(slot_cfg(a_scale_ref_rank=64)))

        assert model.attn.q_proj.slot_A.scale == pytest.approx(DIM**0.5 / 64)

    def test_orth_eps_is_honoured(self, toy_type, peft_spy):
        model = get_model(make_cfg(slot_cfg(orth_eps=1e-4)))

        assert model.attn.q_proj.slot_A.eps == pytest.approx(1e-4)

    def test_orth_iters_is_honoured(self, toy_type, peft_spy):
        # The design doc's remediation for a degrading slot/orth_err is "raise iters
        # from 12 to 16". Without a config path that instruction is unreachable.
        model = get_model(make_cfg(slot_cfg(orth_iters=16)))

        assert model.attn.q_proj.slot_A.iters == 16

    def test_orth_iters_defaults_to_twelve(self, toy_type, peft_spy):
        model = get_model(make_cfg(slot_cfg()))

        assert model.attn.q_proj.slot_A.iters == _DEFAULT_NS_ITERS
        assert _DEFAULT_NS_ITERS == 12

    def test_a_non_positive_orth_iters_raises(self, toy_type, peft_spy):
        # Zero iterations returns Z itself, unnormalized -- silently non-orthogonal.
        with pytest.raises(ValueError, match="iters=0"):
            get_model(make_cfg(slot_cfg(orth_iters=0)))

    def test_an_unknown_scale_mode_raises(self, toy_type, peft_spy):
        with pytest.raises(ValueError, match="scale_mode"):
            get_model(make_cfg(slot_cfg(a_scale_mode="nonsense")))


class TestFrozenAndTrainable:
    def test_only_slot_parameters_are_trainable(self, toy_type, peft_spy):
        model = get_model(make_cfg(slot_cfg()))

        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert trainable == {
            "attn.q_proj.slot_A.weight",
            "attn.q_proj.slot_B.weight",
            "attn.k_proj.slot_A.weight",
            "attn.k_proj.slot_B.weight",
        }

    def test_the_value_head_is_re_enabled(self, toy_type, peft_spy):
        """Mirrors the PEFT branch: injection freezes it, this puts it back."""
        model = get_model(make_cfg(slot_cfg(), add_value_head=True))

        head = {n for n, p in model.named_parameters() if p.requires_grad}
        assert {n for n in head if n.startswith("value_head.")} == {
            "value_head.0.weight",
            "value_head.0.bias",
            "value_head.2.weight",
            "value_head.2.bias",
        }
        assert not model.attn.q_proj.base.weight.requires_grad

    def test_injection_is_the_identity_at_init(self, toy_type, peft_spy):
        """B is zero, so switching the branch on cannot move a baseline."""
        reference = Toy()
        model = get_model(make_cfg(slot_cfg()))
        model.load_state_dict(
            {
                k.replace("q_proj.", "q_proj.base.").replace(
                    "k_proj.", "k_proj.base."
                ): v
                for k, v in reference.state_dict().items()
            },
            strict=False,
        )
        x = torch.randn(3, DIM)

        with find_slot_gate(model).scoped(torch.tensor([0, 1, -1])):
            assert torch.allclose(model(x), reference(x))


class TestToLoraAssertion:
    def test_passes_on_a_freshly_injected_model(self, toy_type, peft_spy):
        model = get_model(make_cfg(slot_cfg()))

        # Two SlotLoRALinear, each with a SlotProj and a SlotOut.
        assert _assert_slot_leaves_fsdp_wrappable(model) == 4

    @pytest.mark.parametrize("cls", [SlotProj, SlotOut])
    def test_fires_when_a_slot_leaf_is_tagged_not_to_lora(
        self, toy_type, peft_spy, cls
    ):
        """What tag_vlm_subtree(model, False) would do if it ever ran on this path."""
        model = get_model(make_cfg(slot_cfg()))
        victim = next(m for m in model.modules() if isinstance(m, cls))
        victim._to_lora = False

        with pytest.raises(AssertionError, match="_to_lora=False"):
            _assert_slot_leaves_fsdp_wrappable(model)

    def test_survives_tag_vlm_subtree_true(self, toy_type, peft_spy):
        """`True` is the value the FSDP predicate wants; only `False` is fatal."""
        model = get_model(make_cfg(slot_cfg()))
        models.tag_vlm_subtree(model, True)

        assert _assert_slot_leaves_fsdp_wrappable(model) == 4

    def test_counts_zero_on_a_model_with_no_slots(self, toy_type, peft_spy):
        assert _assert_slot_leaves_fsdp_wrappable(Toy()) == 0
