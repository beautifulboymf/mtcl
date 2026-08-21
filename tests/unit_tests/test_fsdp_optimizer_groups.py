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

"""Optimizer parameter grouping in ``FSDPModelManager.build_optimizer``.

The slot-LoRI student needs ``Z`` (``.slot_A.``) and ``B`` (``.slot_B.``) in their own
optimizer groups: ``Ā = (Z Zᵀ)^(-1/2) Z`` is invariant to the scale of ``Z``, so weight
decay on ``Z`` cannot change the function -- it only shrinks ``Z`` and degrades the
conditioning of ``Z Zᵀ``, which is the one thing that makes the orthogonalization stop
being orthogonal. Hence ``weight_decay == 0.0`` on the ``slot_A`` group, and a ``name``
on every group so the alternating-factor schedule can find them.
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    PARAM_GROUP_ACTOR,
    PARAM_GROUP_CRITIC,
    PARAM_GROUP_SLOT_A,
    PARAM_GROUP_SLOT_B,
    FSDPModelManager,
    build_slot_aware_param_groups,
)
from rlinf.utils.utils import warmup_optimizer_state

LR = 1e-4
VALUE_LR = 5e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 1e-2
ADAM_EPS = 1e-8


# --------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------


class _Leaf(nn.Module):
    """A one-parameter leaf, the shape both SlotProj and SlotOut have."""

    def __init__(self, *shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*shape))


class _SlotLinear(nn.Module):
    """The child layout of ``SlotLoRALinear``: frozen ``base`` + ``slot_A`` + ``slot_B``."""

    def __init__(self, d_in=6, d_out=4, rank=2):
        super().__init__()
        self.base = nn.Linear(d_in, d_out)
        self.slot_A = _Leaf(rank, d_in)
        self.slot_B = _Leaf(d_out, rank)


class _PlainModel(nn.Module):
    """No slots anywhere: the pre-slot-LoRI world build_optimizer must still serve."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(6, 4)
        self.value_head = nn.Linear(4, 1)


class _SlotModel(nn.Module):
    """Slots, a plain trainable tensor and a value head all at once."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_SlotLinear(), _SlotLinear()])
        self.embed = nn.Linear(6, 6)
        self.value_head = nn.Linear(4, 1)


def _named_trainable(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _groups(model, **overrides):
    kwargs = {
        "lr": LR,
        "value_lr": VALUE_LR,
        "betas": BETAS,
        "weight_decay": WEIGHT_DECAY,
    }
    kwargs.update(overrides)
    return build_slot_aware_param_groups(_named_trainable(model), **kwargs)


def _by_name(groups):
    return {g["name"]: g for g in groups}


def _ids(params):
    return {id(p) for p in params}


def _make_manager(**optim_overrides):
    """A manager with just enough state for ``build_optimizer``.

    ``__init__`` builds a device mesh and a tokenizer, neither of which
    ``build_optimizer`` touches, and both of which need a live process group. The
    method under test reads exactly ``_cfg.optim``, ``_logger`` and
    ``store_requires_grad_param_name``, so those are what the test supplies.
    """
    optim = {
        "lr": LR,
        "value_lr": VALUE_LR,
        "adam_beta1": BETAS[0],
        "adam_beta2": BETAS[1],
        "adam_eps": ADAM_EPS,
        "weight_decay": WEIGHT_DECAY,
    }
    optim.update(optim_overrides)
    manager = FSDPModelManager.__new__(FSDPModelManager)
    manager._cfg = OmegaConf.create({"optim": optim})
    manager._logger = _NullLogger()
    manager.store_requires_grad_param_name = []
    return manager


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


# --------------------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------------------


def test_slot_model_yields_four_named_groups_in_order():
    model = _SlotModel()
    groups = _groups(model)

    assert [g["name"] for g in groups] == [
        PARAM_GROUP_ACTOR,
        PARAM_GROUP_CRITIC,
        PARAM_GROUP_SLOT_A,
        PARAM_GROUP_SLOT_B,
    ]

    named = _by_name(groups)
    expected_slot_a = _ids(
        p for n, p in model.named_parameters() if n.endswith(".slot_A.weight")
    )
    expected_slot_b = _ids(
        p for n, p in model.named_parameters() if n.endswith(".slot_B.weight")
    )
    assert len(expected_slot_a) == 2 and len(expected_slot_b) == 2
    assert _ids(named[PARAM_GROUP_SLOT_A]["params"]) == expected_slot_a
    assert _ids(named[PARAM_GROUP_SLOT_B]["params"]) == expected_slot_b

    # Everything else lands where it did before: value head -> critic, rest -> actor.
    assert _ids(named[PARAM_GROUP_CRITIC]["params"]) == _ids(
        model.value_head.parameters()
    )
    expected_actor = _ids(
        p
        for n, p in model.named_parameters()
        if "value_head" not in n and ".slot_A." not in n and ".slot_B." not in n
    )
    assert _ids(named[PARAM_GROUP_ACTOR]["params"]) == expected_actor

    # Partition, not overlap: every trainable parameter in exactly one group.
    all_ids = [id(p) for g in groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == _ids(model.parameters())


def test_slot_a_group_has_zero_weight_decay_and_others_keep_the_global():
    named = _by_name(_groups(_SlotModel()))

    assert named[PARAM_GROUP_SLOT_A]["weight_decay"] == 0.0
    for group_name in (PARAM_GROUP_ACTOR, PARAM_GROUP_CRITIC, PARAM_GROUP_SLOT_B):
        assert named[group_name]["weight_decay"] == WEIGHT_DECAY


def test_slot_a_weight_decay_stays_zero_for_any_global_weight_decay():
    for global_wd in (0.0, 1e-2, 0.5):
        named = _by_name(_groups(_SlotModel(), weight_decay=global_wd))
        assert named[PARAM_GROUP_SLOT_A]["weight_decay"] == 0.0
        assert named[PARAM_GROUP_SLOT_B]["weight_decay"] == global_wd


def test_learning_rates_per_group():
    named = _by_name(_groups(_SlotModel()))
    assert named[PARAM_GROUP_ACTOR]["lr"] == LR
    assert named[PARAM_GROUP_CRITIC]["lr"] == VALUE_LR
    assert named[PARAM_GROUP_SLOT_A]["lr"] == LR
    assert named[PARAM_GROUP_SLOT_B]["lr"] == LR
    assert all(g["betas"] == BETAS for g in named.values())


def test_no_slot_params_reproduces_the_legacy_two_groups():
    model = _PlainModel()
    groups = _groups(model)

    assert len(groups) == 2
    assert [g["name"] for g in groups] == [PARAM_GROUP_ACTOR, PARAM_GROUP_CRITIC]
    assert groups[0]["lr"] == LR
    assert groups[1]["lr"] == VALUE_LR
    assert groups[0]["betas"] == BETAS and groups[1]["betas"] == BETAS
    assert groups[0]["weight_decay"] == WEIGHT_DECAY
    assert groups[1]["weight_decay"] == WEIGHT_DECAY
    assert _ids(groups[0]["params"]) == _ids(model.backbone.parameters())
    assert _ids(groups[1]["params"]) == _ids(model.value_head.parameters())


def test_empty_groups_are_omitted():
    """Slot-only model: the injection freeze leaves nothing but slots trainable."""
    model = _SlotModel()
    for name, param in model.named_parameters():
        param.requires_grad_(".slot_A." in name or ".slot_B." in name)

    groups = _groups(model)
    assert [g["name"] for g in groups] == [PARAM_GROUP_SLOT_A, PARAM_GROUP_SLOT_B]
    assert all(len(g["params"]) == 2 for g in groups)


def test_no_trainable_params_yields_no_groups():
    model = _PlainModel()
    model.requires_grad_(False)
    assert _groups(model) == []


def test_frozen_params_are_never_grouped():
    model = _SlotModel()
    model.layers[0].slot_A.weight.requires_grad_(False)
    named = _by_name(_groups(model))
    assert _ids(named[PARAM_GROUP_SLOT_A]["params"]) == {
        id(model.layers[1].slot_A.weight)
    }


# --------------------------------------------------------------------------------------
# value_head precedence
# --------------------------------------------------------------------------------------


def test_value_head_wins_over_a_slot_match():
    """A value-head parameter that also matches ``.slot_A.`` stays in the critic group.

    Precedence is value_head-first so that the classification is byte-for-byte what it
    was before slots existed, and so that it agrees with the critic-warmup branch (which
    collects value-head parameters and freezes literally everything else). ``value_lr``
    is the more consequential knob than weight decay, and this collision cannot arise in
    practice: the injection pass freezes every non-slot parameter, so a value head under
    slot-LoRI is not trainable at all.
    """

    class _ValueHeadWithSlots(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(6, 4)
            self.value_head = _SlotLinear(4, 1, 2)

    model = _ValueHeadWithSlots()
    groups = _groups(model)

    assert [g["name"] for g in groups] == [PARAM_GROUP_ACTOR, PARAM_GROUP_CRITIC]
    critic = _by_name(groups)[PARAM_GROUP_CRITIC]
    assert critic["lr"] == VALUE_LR
    assert id(model.value_head.slot_A.weight) in _ids(critic["params"])
    assert id(model.value_head.slot_B.weight) in _ids(critic["params"])


def test_model_dot_value_head_prefix_still_routes_to_critic():
    class _Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.value_head = nn.Linear(4, 1)
            self.other = nn.Linear(6, 4)

    model = _Wrapped()
    named = _by_name(_groups(model))
    assert _ids(named[PARAM_GROUP_CRITIC]["params"]) == _ids(
        model.model.value_head.parameters()
    )


# --------------------------------------------------------------------------------------
# the groups have to survive AdamW
# --------------------------------------------------------------------------------------


def test_adamw_preserves_the_name_key():
    groups = _groups(_SlotModel())
    optimizer = torch.optim.AdamW(groups, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)
    assert [g["name"] for g in optimizer.param_groups] == [
        PARAM_GROUP_ACTOR,
        PARAM_GROUP_CRITIC,
        PARAM_GROUP_SLOT_A,
        PARAM_GROUP_SLOT_B,
    ]


def test_adamw_applies_the_per_group_weight_decay():
    """Zero gradients: AdamW's Adam term is zero, so only decoupled decay can move a
    parameter. ``slot_A`` must come back bit-identical while a decayed group shrinks."""
    model = _SlotModel()
    groups = _groups(model)
    optimizer = torch.optim.AdamW(groups, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)

    z = model.layers[0].slot_A.weight
    b = model.layers[0].slot_B.weight
    actor = model.embed.weight
    b.data.fill_(1.0)  # SlotOut is zero-initialised; decay of zero is invisible
    before = {p: p.detach().clone() for p in (z, b, actor)}
    for p in (z, b, actor):
        p.grad = torch.zeros_like(p)

    optimizer.step()

    assert torch.equal(z, before[z]), "weight decay leaked onto Z"
    expected = 1.0 - LR * WEIGHT_DECAY
    for p in (b, actor):
        assert not torch.equal(p, before[p])
        torch.testing.assert_close(p, before[p] * expected, rtol=0, atol=1e-9)


def test_adamw_defaults_still_carry_the_global_weight_decay():
    optimizer = torch.optim.AdamW(
        _groups(_SlotModel()), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY
    )
    assert optimizer.defaults["weight_decay"] == WEIGHT_DECAY
    assert optimizer.defaults["eps"] == ADAM_EPS


def test_warmup_optimizer_state_initialises_state_without_moving_parameters():
    model = _SlotModel()
    groups = _groups(model)
    optimizer = torch.optim.AdamW(groups, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)
    before = {id(p): p.detach().clone() for g in groups for p in g["params"]}

    warmup_optimizer_state(optimizer)

    assert [g["lr"] for g in optimizer.param_groups] == [LR, VALUE_LR, LR, LR]
    for group in optimizer.param_groups:
        for p in group["params"]:
            assert p in optimizer.state and "exp_avg" in optimizer.state[p]
            assert torch.equal(p, before[id(p)])
            assert p.grad is None


# --------------------------------------------------------------------------------------
# the real method
# --------------------------------------------------------------------------------------


def test_build_optimizer_on_a_slot_model():
    manager = _make_manager()
    model = _SlotModel()
    optimizer = manager.build_optimizer(model)

    assert isinstance(optimizer, torch.optim.AdamW)
    names = [g["name"] for g in optimizer.param_groups]
    assert names == [
        PARAM_GROUP_ACTOR,
        PARAM_GROUP_CRITIC,
        PARAM_GROUP_SLOT_A,
        PARAM_GROUP_SLOT_B,
    ]
    by_name = _by_name(optimizer.param_groups)
    assert by_name[PARAM_GROUP_SLOT_A]["weight_decay"] == 0.0
    assert by_name[PARAM_GROUP_SLOT_B]["weight_decay"] == WEIGHT_DECAY
    assert by_name[PARAM_GROUP_CRITIC]["lr"] == VALUE_LR
    # warmup_optimizer_state ran: state is populated for every parameter.
    assert all(
        p in optimizer.state for g in optimizer.param_groups for p in g["params"]
    )


def test_build_optimizer_on_a_plain_model_is_unchanged():
    manager = _make_manager()
    model = _PlainModel()
    optimizer = manager.build_optimizer(model)

    assert len(optimizer.param_groups) == 2
    assert [g["lr"] for g in optimizer.param_groups] == [LR, VALUE_LR]
    assert [g["weight_decay"] for g in optimizer.param_groups] == [
        WEIGHT_DECAY,
        WEIGHT_DECAY,
    ]
    assert [g["betas"] for g in optimizer.param_groups] == [BETAS, BETAS]
    assert [g["name"] for g in optimizer.param_groups] == [
        PARAM_GROUP_ACTOR,
        PARAM_GROUP_CRITIC,
    ]


def test_build_optimizer_default_weight_decay_is_one_percent():
    """The global default the slot_A group has to override."""
    manager = _make_manager()
    del manager._cfg.optim.weight_decay
    optimizer = manager.build_optimizer(_SlotModel())
    by_name = _by_name(optimizer.param_groups)
    assert by_name[PARAM_GROUP_ACTOR]["weight_decay"] == pytest.approx(1e-2)
    assert by_name[PARAM_GROUP_SLOT_A]["weight_decay"] == 0.0


def test_build_optimizer_critic_warmup_then_rebuild():
    """Warmup freezes everything but the value head; the rebuild restores the slots."""
    manager = _make_manager()
    model = _SlotModel()

    optimizer = manager.build_optimizer(model, enable_critic_warmup=True)
    assert [g["name"] for g in optimizer.param_groups] == [PARAM_GROUP_CRITIC]
    assert optimizer.param_groups[0]["lr"] == VALUE_LR
    assert not model.layers[0].slot_A.weight.requires_grad

    optimizer = manager.build_optimizer(model)
    assert [g["name"] for g in optimizer.param_groups] == [
        PARAM_GROUP_ACTOR,
        PARAM_GROUP_CRITIC,
        PARAM_GROUP_SLOT_A,
        PARAM_GROUP_SLOT_B,
    ]
    assert model.layers[0].slot_A.weight.requires_grad


# --------------------------------------------------------------------------------------
# the name convention itself
# --------------------------------------------------------------------------------------


def test_real_injection_pass_produces_matchable_names():
    """The contract between the two modules: the names ``inject_slot_lora`` creates are
    the names this grouping matches. Tested against the real injection rather than the
    hand-built stand-ins above, because a rename on either side would otherwise show up
    only as an empty slot group in a 7B run."""
    slot_lora = pytest.importorskip(
        "rlinf.models.slot_lora",
        reason="slot-LoRI package unavailable; the grouping contract cannot be checked",
    )

    class _Attn(nn.Module):
        def __init__(self, d=8):
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)

    class _Model(nn.Module):
        def __init__(self, d=8):
            super().__init__()
            self.layers = nn.ModuleList([_Attn(d) for _ in range(2)])

    model = _Model()
    slot_lora.inject_slot_lora(
        model, slot_ranks=(2, 2), target_modules=["q_proj", "v_proj"], scale_mode="unit"
    )

    optimizer = _make_manager().build_optimizer(model)
    by_name = _by_name(optimizer.param_groups)
    # The injection freeze leaves nothing but the slots trainable.
    assert set(by_name) == {PARAM_GROUP_SLOT_A, PARAM_GROUP_SLOT_B}
    assert len(by_name[PARAM_GROUP_SLOT_A]["params"]) == 4  # 2 layers x 2 targets
    assert len(by_name[PARAM_GROUP_SLOT_B]["params"]) == 4
    assert by_name[PARAM_GROUP_SLOT_A]["weight_decay"] == 0.0
    assert by_name[PARAM_GROUP_SLOT_B]["weight_decay"] == WEIGHT_DECAY

    expected_z = _ids(
        m.slot_A.weight
        for m in model.modules()
        if isinstance(m, slot_lora.SlotLoRALinear)
    )
    assert _ids(by_name[PARAM_GROUP_SLOT_A]["params"]) == expected_z


@pytest.mark.parametrize(
    "name, expected",
    [
        # The dots are part of the marker: only an attribute literally named slot_A
        # matches, not a parameter whose name merely starts or ends with the text.
        ("layers.0.q_proj.slot_A.weight", PARAM_GROUP_SLOT_A),
        ("layers.0.q_proj.slot_B.weight", PARAM_GROUP_SLOT_B),
        # An FSDP1 flat parameter keeps the wrapped leaf's path in its name.
        (
            "_fsdp_wrapped_module.layers.0.q_proj.slot_A._fsdp_wrapped_module._flat_param",
            PARAM_GROUP_SLOT_A,
        ),
        ("layers.0.q_proj.slot_A", PARAM_GROUP_ACTOR),  # no trailing dot
        ("slot_A.weight", PARAM_GROUP_ACTOR),  # no leading dot
        ("layers.0.slot_Away.weight", PARAM_GROUP_ACTOR),
        ("layers.0.base.weight", PARAM_GROUP_ACTOR),
    ],
)
def test_name_classification(name, expected):
    param = nn.Parameter(torch.ones(2, 2))
    groups = build_slot_aware_param_groups(
        [(name, param)],
        lr=LR,
        value_lr=VALUE_LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    assert len(groups) == 1
    assert groups[0]["name"] == expected


def test_group_names_survive_a_state_dict_round_trip():
    """Checkpoint resume rebuilds the optimizer and loads state into it. The extra
    ``name`` key has to come back, or the alternating schedule silently finds no group
    to retune after a resume."""
    model = _SlotModel()
    saved = torch.optim.AdamW(
        _groups(model), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY
    ).state_dict()

    restored = torch.optim.AdamW(
        _groups(model), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY
    )
    restored.load_state_dict(saved)

    by_name = _by_name(restored.param_groups)
    assert set(by_name) == {
        PARAM_GROUP_ACTOR,
        PARAM_GROUP_CRITIC,
        PARAM_GROUP_SLOT_A,
        PARAM_GROUP_SLOT_B,
    }
    assert by_name[PARAM_GROUP_SLOT_A]["weight_decay"] == 0.0
    assert by_name[PARAM_GROUP_CRITIC]["lr"] == VALUE_LR
