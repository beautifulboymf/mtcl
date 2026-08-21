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

"""The slot-LoRA checkpoint converter, on toy models.

The real converter turns a 7B RLinf FSDP checkpoint into a plain HuggingFace model
directory, and none of the 7B is needed to test what can go wrong: the failures this
file is about are the checkpoint not describing the structure the converter built, the
scale disagreeing with the one training used, and the merge arithmetic itself. All
three are visible on two linears.

EVERY NUMERICAL ASSERTION IS IN float64, and tightly. The production merge sits ~3.4e-3
relative away from the training-time forward, which is the bf16 rounding floor of Ā and
of the merged weight, not a merge error -- so a bf16 test cannot tell a correct merge
from one that is wrong by less than a percent. In float64 the identity
``merged(x) == slot_layer(x)`` is exact to summation order (~1e-15), which is what makes
a dropped bias, a doubled bias or a mis-scaled ΔW impossible to hide. One test
deliberately runs in bf16, and it asserts the ~1e-2 tolerance the production path is
entitled to and nothing tighter.
"""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from rlinf.models.slot_lora import SlotLoRALinear, SlotOut, SlotProj

# The converter is a SCRIPT, not an importable package member, so it is loaded by path.
# That is deliberate on the converter's side (it lives next to the other opd_distill
# scripts, where a user runs it), and it costs this file six lines.
_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "opd_distill"
    / "scripts"
    / "convert_oft_slot_ckpt.py"
)
_spec = importlib.util.spec_from_file_location("convert_oft_slot_ckpt", _SCRIPT)
convert = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = convert
_spec.loader.exec_module(convert)


DIM = 16
HIDDEN = 24
RANKS = (2, 3)
# q_proj and k_proj are both in the REAL SLOT_LORA_TARGET_MODULES, so these toys go
# through the same target list the 7B student does; `mlp` is not, and stays a plain
# nn.Linear whose weights the merge must leave alone.
TARGETS = None


class Attn(nn.Module):
    """Two targeted linears with different in_features, plus one untargeted."""

    def __init__(self, dtype):
        super().__init__()
        self.q_proj = nn.Linear(DIM, HIDDEN, dtype=dtype)
        self.k_proj = nn.Linear(HIDDEN, DIM, dtype=dtype)
        self.mlp = nn.Linear(DIM, DIM, dtype=dtype)  # NOT a slot target

    def forward(self, x):
        return self.mlp(self.k_proj(self.q_proj(x)))


class Toy(nn.Module):
    """A two-linear stand-in for the student."""

    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.attn = Attn(dtype)
        # Biases that are NOT the default init, and are large enough that dropping one
        # or applying it twice moves the output far outside every tolerance here.
        with torch.no_grad():
            for linear in (self.attn.q_proj, self.attn.k_proj, self.attn.mlp):
                linear.bias.copy_(
                    torch.arange(1, linear.out_features + 1, dtype=dtype) * 0.5
                )

    def forward(self, x):
        return self.attn(x)


def _train(model, seed=0):
    """Move the slots off their initialization, the way a real run would.

    ``B`` starts at exactly zero and ``Z`` at a fixed Gaussian draw, so a checkpoint
    that was never trained merges to a no-op and every assertion here would pass
    against a converter that did nothing at all. Both sides are moved: ``B`` because it
    is the whole of ``ΔW``, and ``Z`` because a converter that regenerated ``Ā`` from a
    fresh init instead of from the checkpoint's ``Z`` would otherwise still agree.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, SlotOut):
                module.weight.copy_(
                    torch.randn(
                        module.weight.shape,
                        generator=generator,
                        dtype=module.weight.dtype,
                    )
                )
            elif isinstance(module, SlotProj):
                module.weight.add_(
                    torch.randn(
                        module.weight.shape,
                        generator=generator,
                        dtype=module.weight.dtype,
                    )
                    * 0.5
                )
    return model


def _snapshot(model):
    """A detached, deep copy of a state dict -- a stand-in for ``full_weights.pt``."""
    return {
        key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in model.state_dict().items()
    }


def _trained_checkpoint(dtype=torch.float64, ranks=RANKS, seed=0, **inject_kwargs):
    """A trained slot model, its injection handle, and the checkpoint it would save."""
    torch.manual_seed(11)
    model = Toy(dtype=dtype)
    injection = convert.wrap_with_slots(
        model, ranks, target_modules=TARGETS, **inject_kwargs
    )
    _train(model, seed=seed)
    return model, injection, _snapshot(model)


def _fresh_base(dtype=torch.float64):
    """A base model whose weights DIFFER from the trained one's.

    The checkpoint carries ``base.weight`` too, so a converter that failed to load the
    base -- or that merged into the wrong tensor -- has to show up as a wrong output
    rather than as an accident of both models having been seeded the same way.
    """
    torch.manual_seed(99)
    return Toy(dtype=dtype)


def _inputs(dtype=torch.float64, batch=4):
    torch.manual_seed(5)
    return torch.randn(batch, DIM, dtype=dtype)


# --------------------------------------------------------------------------------
# --slot-ranks parsing
# --------------------------------------------------------------------------------


class TestParseSlotRanks:
    def test_parses_the_production_default(self):
        assert convert.parse_slot_ranks("128,64,48,16") == (128, 64, 48, 16)

    def test_tolerates_whitespace(self):
        assert convert.parse_slot_ranks(" 2 , 3 ") == (2, 3)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            convert.parse_slot_ranks("")

    @pytest.mark.parametrize("text", ["2,0,3", "2,-1"])
    def test_rejects_non_positive(self, text):
        with pytest.raises(ValueError, match="positive"):
            convert.parse_slot_ranks(text)

    def test_rejects_non_integer(self):
        with pytest.raises(ValueError, match="comma-separated"):
            convert.parse_slot_ranks("2,three")


# --------------------------------------------------------------------------------
# The round trip
# --------------------------------------------------------------------------------


class TestRoundTrip:
    def test_merged_model_reproduces_the_gated_forward(self):
        """The whole point: convert(train(x)) == train(x), to summation order.

        The reference is the GATED forward -- the one the training loop actually ran,
        with a routing installed -- because that is the model the checkpoint claims to
        be. It differs from the merged single-matmul path only in the order the slot
        contributions are summed (measured ~1e-15 in float64), never in value: the gate
        selects between two tensors holding the same bits.
        """
        trained, injection, ckpt = _trained_checkpoint()
        x = _inputs()
        ids = torch.tensor([0, 1, -1, 1])
        with injection.gate.scoped(ids):
            reference = trained(x)

        merged = convert.merge_slot_checkpoint(
            _fresh_base(), ckpt, RANKS, target_modules=TARGETS
        )
        assert torch.allclose(merged(x), reference, rtol=0.0, atol=1e-12)

    def test_merged_model_reproduces_the_ungated_forward(self):
        """The eval/rollout forward, which is the one the converted model replaces."""
        trained, injection, ckpt = _trained_checkpoint()
        x = _inputs()
        with injection.gate.ungated():
            reference = trained(x)

        merged = convert.merge_slot_checkpoint(
            _fresh_base(), ckpt, RANKS, target_modules=TARGETS
        )
        assert torch.allclose(merged(x), reference, rtol=0.0, atol=1e-12)

    def test_the_merge_is_not_a_no_op(self):
        """Guards every other assertion here against a converter that does nothing.

        ``B`` is zero-initialized, so an unloaded or unmerged model IS the base model,
        and a round-trip test whose checkpoint happened to carry a zero ``ΔW`` would
        pass against a converter that skipped the merge entirely. Compared against the
        checkpoint's own ``base.weight``, which is the exact tensor ``ΔW`` was added to.
        """
        _, _, ckpt = _trained_checkpoint()
        merged = convert.merge_slot_checkpoint(
            _fresh_base(), ckpt, RANKS, target_modules=TARGETS
        )
        for path in ("attn.q_proj", "attn.k_proj"):
            weight = merged.get_submodule(path).weight
            base = ckpt[f"{path}.base.weight"]
            assert (weight - base).abs().max() > 1e-3

    def test_scale_reaches_the_merged_weights(self):
        """``match_mt4`` and ``unit`` must NOT produce the same merged model.

        ``scale`` multiplies every slot's contribution, and it is the one factor that
        leaves no trace in a tensor shape. If the two modes merged identically, the
        scale would not be reaching ``ΔW`` at all and the checkpoint's persisted value
        would be decorative.
        """
        _, _, unit_ckpt = _trained_checkpoint(scale_mode="unit")
        unit_merged = convert.merge_slot_checkpoint(
            _fresh_base(), unit_ckpt, RANKS, scale_mode="unit", target_modules=TARGETS
        )
        _, _, mt4_ckpt = _trained_checkpoint(scale_mode="match_mt4", ref_rank=8)
        mt4_merged = convert.merge_slot_checkpoint(
            _fresh_base(),
            mt4_ckpt,
            RANKS,
            scale_mode="match_mt4",
            ref_rank=8,
            target_modules=TARGETS,
        )
        unit_w = unit_merged.attn.q_proj.weight
        mt4_w = mt4_merged.attn.q_proj.weight
        assert not torch.allclose(unit_w, mt4_w)


class TestBias:
    def test_bias_is_carried_across_exactly_once(self):
        """``ΔW`` is weight-only; the bias comes over from ``base``, unchanged.

        Asserted against the CHECKPOINT's bias rather than against the fresh model's,
        so both halves of the failure are covered: dropping it leaves the fresh base's
        bias in place (different values), and adding it twice doubles it.
        """
        _, _, ckpt = _trained_checkpoint()
        merged = convert.merge_slot_checkpoint(
            _fresh_base(), ckpt, RANKS, target_modules=TARGETS
        )
        for path in ("attn.q_proj", "attn.k_proj"):
            module = merged.get_submodule(path)
            assert torch.equal(module.bias, ckpt[f"{path}.base.bias"])

    def test_untargeted_module_is_untouched(self):
        """``mlp`` is not a slot target: its weights come straight from the ckpt."""
        _, _, ckpt = _trained_checkpoint()
        merged = convert.merge_slot_checkpoint(
            _fresh_base(), ckpt, RANKS, target_modules=TARGETS
        )
        assert torch.equal(merged.attn.mlp.weight, ckpt["attn.mlp.weight"])
        assert torch.equal(merged.attn.mlp.bias, ckpt["attn.mlp.bias"])


# --------------------------------------------------------------------------------
# The merged model is a PLAIN model
# --------------------------------------------------------------------------------


class TestMergedModelIsPlain:
    def test_no_slot_modules_survive(self):
        _, _, ckpt = _trained_checkpoint()
        merged = convert.merge_slot_checkpoint(
            _fresh_base(), ckpt, RANKS, target_modules=TARGETS
        )
        for module in merged.modules():
            assert not isinstance(module, (SlotLoRALinear, SlotProj, SlotOut))
        assert isinstance(merged.attn.q_proj, nn.Linear)
        assert isinstance(merged.attn.k_proj, nn.Linear)

    def test_no_slot_keys_survive(self):
        """The saved state dict must be the BASE model's, key for key.

        A leftover ``slot_`` / ``base.`` / ``_extra_state`` key is not cosmetic: the
        eval pipeline loads this directory with a plain ``from_pretrained``, where an
        unexpected key is at best a warning and at worst a silently unloaded weight.
        """
        _, _, ckpt = _trained_checkpoint()
        merged = convert.merge_slot_checkpoint(
            _fresh_base(), ckpt, RANKS, target_modules=TARGETS
        )
        keys = set(merged.state_dict())
        assert not [k for k in keys if "slot_" in k]
        assert not [k for k in keys if ".base." in k]
        assert not [k for k in keys if "_extra_state" in k]
        assert keys == set(Toy().state_dict())

    def test_reports_one_merge_per_injected_layer(self):
        model = _fresh_base()
        injection = convert.wrap_with_slots(model, RANKS, target_modules=TARGETS)
        assert convert.merge_slots(model) == len(injection.paths) == 2


# --------------------------------------------------------------------------------
# Loud rejections
# --------------------------------------------------------------------------------


class TestRankMismatch:
    def test_a_different_total_rank_is_rejected(self):
        """The case the shapes CAN catch: ``sum(slot_ranks)`` disagrees."""
        _, _, ckpt = _trained_checkpoint(ranks=(2, 3))
        with pytest.raises(convert.SlotCheckpointMismatch, match="total rank"):
            convert.merge_slot_checkpoint(
                _fresh_base(), ckpt, (2, 4), target_modules=TARGETS
            )

    def test_a_different_slot_count_with_a_different_sum_is_rejected(self):
        _, _, ckpt = _trained_checkpoint(ranks=(2, 3))
        with pytest.raises(convert.SlotCheckpointMismatch, match="total rank"):
            convert.merge_slot_checkpoint(
                _fresh_base(), ckpt, (2, 3, 1), target_modules=TARGETS
            )

    def test_a_permuted_rank_order_merges_identically(self):
        """Documented, not desired: the merge cannot see the slot PARTITION.

        ``ΔW = Σ_k B_k Ā_k = B Ā``, and neither factor's shape depends on where the
        slot boundaries fall -- only on ``sum(slot_ranks)``. So a ``--slot-ranks`` given
        in the wrong ``slot_order`` loads without complaint and merges to the SAME
        weights, because at merge time every slot is being summed anyway. This is a
        real hole in the "wrong ranks are rejected loudly" guarantee, and it is benign
        only for the merge: the same mistake in a TRAINING config routes one suite's
        gradient into another suite's slot. The production ranks (128, 64, 48, 16) all
        sum to 256 under every permutation, so no permutation of them is detectable
        here.
        """
        _, _, ckpt = _trained_checkpoint(ranks=(2, 3))
        forward = convert.merge_slot_checkpoint(
            _fresh_base(), ckpt, (2, 3), target_modules=TARGETS
        )
        backward = convert.merge_slot_checkpoint(
            _fresh_base(), copy.deepcopy(ckpt), (3, 2), target_modules=TARGETS
        )
        for a, b in zip(forward.state_dict().values(), backward.state_dict().values()):
            assert torch.equal(a, b)


class TestScaleMismatch:
    def test_a_scale_that_disagrees_with_the_checkpoint_is_rejected(self):
        """The failure the persisted scale exists to close, end to end.

        Nothing about a wrong ``ref_rank`` changes a shape or a key: without the
        checkpoint's own record of ``s`` the merge would succeed and every ΔW would be
        off by the ratio of the two scales, with no error anywhere.
        """
        _, _, ckpt = _trained_checkpoint(scale_mode="match_mt4", ref_rank=8)
        with pytest.raises(ValueError, match="scale mismatch"):
            convert.merge_slot_checkpoint(
                _fresh_base(),
                ckpt,
                RANKS,
                scale_mode="match_mt4",
                ref_rank=16,
                target_modules=TARGETS,
            )

    def test_a_scale_mode_that_disagrees_is_rejected(self):
        _, _, ckpt = _trained_checkpoint(scale_mode="unit")
        with pytest.raises(ValueError, match="scale mismatch"):
            convert.merge_slot_checkpoint(
                _fresh_base(),
                ckpt,
                RANKS,
                scale_mode="match_mt4",
                ref_rank=8,
                target_modules=TARGETS,
            )

    def test_the_matching_scale_is_accepted(self):
        _, _, ckpt = _trained_checkpoint(scale_mode="match_mt4", ref_rank=8)
        merged = convert.merge_slot_checkpoint(
            _fresh_base(),
            ckpt,
            RANKS,
            scale_mode="match_mt4",
            ref_rank=8,
            target_modules=TARGETS,
        )
        assert isinstance(merged.attn.q_proj, nn.Linear)


class TestKeyMismatch:
    def test_unexpected_keys_are_rejected(self):
        """A checkpoint from a DIFFERENT structure -- here, an un-injected model."""
        torch.manual_seed(11)
        plain = _snapshot(Toy())
        with pytest.raises(convert.SlotCheckpointMismatch, match="unexpected"):
            convert.merge_slot_checkpoint(
                _fresh_base(), plain, RANKS, target_modules=TARGETS
            )

    def test_missing_keys_are_rejected(self):
        _, _, ckpt = _trained_checkpoint()
        ckpt.pop("attn.q_proj.slot_B.weight")
        with pytest.raises(convert.SlotCheckpointMismatch, match="missing"):
            convert.merge_slot_checkpoint(
                _fresh_base(), ckpt, RANKS, target_modules=TARGETS
            )

    def test_a_missing_extra_state_is_rejected(self):
        """A checkpoint that recorded no scale is refused, not silently accepted.

        ``_extra_state`` is the checkpoint's only record of the LoRA scaling, and it is
        the whole basis of the "verified, not re-derived" guarantee: with it absent,
        ``set_extra_state`` is never called and the CLI's ``--slot-ref-rank`` goes
        unchecked. So it is a missing KEY like any other and the conversion stops.
        """
        _, _, ckpt = _trained_checkpoint()
        removed = [k for k in ckpt if k.endswith("_extra_state")]
        assert removed, "the checkpoint is supposed to carry the scale"
        for key in removed:
            ckpt.pop(key)
        with pytest.raises(convert.SlotCheckpointMismatch, match="missing"):
            convert.merge_slot_checkpoint(
                _fresh_base(), ckpt, RANKS, target_modules=TARGETS
            )


# --------------------------------------------------------------------------------
# dtype
# --------------------------------------------------------------------------------


class TestDtype:
    def test_merged_weight_keeps_the_base_dtype(self):
        """``delta_weight()`` returns fp32; the base weight must stay bf16.

        A merged model that came out fp32 would be twice the size on disk and a
        different dtype than the eval pipeline builds, so this is not only about the
        cast being explicit.
        """
        _, _, ckpt = _trained_checkpoint(dtype=torch.bfloat16)
        merged = convert.merge_slot_checkpoint(
            _fresh_base(dtype=torch.bfloat16), ckpt, RANKS, target_modules=TARGETS
        )
        assert merged.attn.q_proj.weight.dtype is torch.bfloat16
        assert merged.attn.q_proj.bias.dtype is torch.bfloat16

    def test_bf16_round_trip_within_the_rounding_floor(self):
        """The production tolerance: ~3.4e-3 relative, and it is NOT a merge error.

        ``Ā`` is orthogonalized in the parameter's dtype (bf16 here, as in production),
        ``ΔW`` is then rounded into a bf16 base weight, and the training forward rounds
        its intermediate activation where the merged single matmul does not. 1e-2
        relative is what this path is entitled to; the float64 tests above are what
        pins the arithmetic.
        """
        trained, injection, ckpt = _trained_checkpoint(dtype=torch.bfloat16)
        x = _inputs(dtype=torch.bfloat16)
        with injection.gate.ungated():
            reference = trained(x)
        merged = convert.merge_slot_checkpoint(
            _fresh_base(dtype=torch.bfloat16), ckpt, RANKS, target_modules=TARGETS
        )
        got = merged(x).float()
        want = reference.float()
        relative = (got - want).norm() / want.norm().clamp_min(1e-12)
        assert relative < 1e-2, relative.item()


# --------------------------------------------------------------------------------
# Shared surface with the single-LoRA converter
# --------------------------------------------------------------------------------


class TestSharedWithBaselineConverter:
    def test_base_model_construction_is_not_duplicated(self):
        """One source of truth for "how RLinf builds this model".

        The two converters have to build the SAME base model; a second copy of that
        construction is a copy that can be fixed in one place and left wrong in the
        other, and the symptom would be a converted model that quietly differs from the
        trained one. So the slot converter imports the baseline's, and this asserts the
        identity rather than the behaviour.
        """
        baseline = convert.baseline_converter()
        assert convert.build_base_model is baseline.build_base_model
        assert convert.AUX_FILES is baseline.AUX_FILES

    def test_target_modules_match_the_training_path(self):
        from rlinf.models import SLOT_LORA_TARGET_MODULES

        assert convert.default_target_modules() == list(SLOT_LORA_TARGET_MODULES)


class TestCheckpointFileReadPath:
    def test_the_scale_survives_torch_save_and_a_weights_only_mmap_load(self, tmp_path):
        """``main()`` reads the checkpoint with ``mmap=True, weights_only=True``.

        Both flags are restrictions -- one maps storages instead of reading them, the
        other refuses to unpickle arbitrary objects -- and the scale does not travel as
        a tensor: it is a ``{"scale": float}`` dict under an ``_extra_state`` key. If
        either flag dropped or rejected it the whole verification would fall back to
        "the checkpoint has no scale", so this pins the real read path rather than the
        in-memory state dict the other tests hand around.
        """
        _, _, ckpt = _trained_checkpoint()
        path = tmp_path / "full_weights.pt"
        torch.save(ckpt, path)
        loaded = torch.load(path, map_location="cpu", mmap=True, weights_only=True)

        extra = {k: v for k, v in loaded.items() if k.endswith("_extra_state")}
        assert extra
        assert all(isinstance(v, dict) and "scale" in v for v in extra.values())

        merged = convert.merge_slot_checkpoint(
            _fresh_base(), loaded, RANKS, target_modules=TARGETS
        )
        assert isinstance(merged.attn.q_proj, nn.Linear)
