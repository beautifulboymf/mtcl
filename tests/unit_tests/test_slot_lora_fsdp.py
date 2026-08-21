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

"""The slot modules under a REAL ``FullyShardedDataParallel``, not a described one.

Every other slot-LoRI test runs on plain CPU modules. That left the load-bearing FSDP
chain -- the per-leaf wrap policy fires on ``SlotProj``/``SlotOut``, the full ``Z`` is
therefore all-gathered inside their own forward, and the resulting module tree still
survives ``get_model_state_dict`` -- asserted only by prose in docstrings. A 2-GPU smoke
test then died at the first ``sync_model_to_rollout`` on exactly the part no test
covered: ``torch.distributed.checkpoint.state_dict._get_fqns`` walks a state-dict key
component by component, and its FSDP branch (unlike its ``else`` branch) has no
``_extra_state`` special case, so it ends on an unconditional
``getattr(curr_obj, "_extra_state")`` and raises ``AttributeError`` for any module that
is BOTH individually FSDP-wrapped AND carries extra state.

So this file builds the real thing: a toy transformer, the repo's own
``get_fsdp_wrap_policy(..., is_lora=True)``, ``use_orig_params=False``, one process, one
GPU, a few hundred megabytes, seconds. It asserts the wrap SHAPE the orthogonalization
depends on, and then exercises both state-dict paths the training loop uses -- the
sharded one (``full_state_dict=False``, weight sync to the rollout engine) and the full
one (``full_state_dict=True``, checkpoint save) -- and round-trips the full one into a
fresh CPU-injected copy.

WHAT ONE PROCESS DOES NOT COVER: at world size 1 FSDP downgrades ``FULL_SHARD`` to
``NO_SHARD`` and warns that ``full_state_dict=False`` will return a full state dict
anyway, so nothing here exercises a real parameter SHARD. It does not have to. The
failure this pins is in ``_get_fqns``, which walks the MODULE TREE and never looks at a
tensor, so it reproduces identically at any world size -- verified: these tests raise
the smoke test's exact ``AttributeError: 'SlotProj' object has no attribute
'_extra_state'`` against the pre-fix code. Numerics under a genuine shard (the
all-gathered ``Z``, the merge) stay the multi-GPU smoke test's job.
"""

import os
import socket

import pytest
import torch
import torch.nn as nn

from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models.slot_lora import inject_slot_lora
from rlinf.models.slot_lora.modules import SlotLoRALinear, SlotOut, SlotProj

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="a real FullyShardedDataParallel needs a CUDA device and a process group",
)

HIDDEN = 64
LAYERS = 2
SLOT_RANKS = (8, 4)
TARGETS = ("q_proj", "v_proj")
REF_RANK = 16  # small so match_mt4's s = sqrt(d_in)/ref_rank stays O(1) here


class _Block(nn.Module):
    """Stands in for a transformer layer: the unit the layer-class policy wraps."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=True)

    def forward(self, x):
        return self.v_proj(self.q_proj(x))


class _Toy(nn.Module):
    """A model the repo's default wrap path recognizes.

    ``_no_split_modules`` is what ``get_fsdp_wrap_policy`` reads when the config carries
    no custom ``wrap_policy``, so naming ``_Block`` here reproduces production's nesting:
    an FSDP unit per layer, with the slot leaves as their OWN units inside it. That
    nesting is the point -- it is what puts an FSDP wrapper on the path ``_get_fqns``
    walks to reach a ``SlotLoRALinear``.
    """

    _no_split_modules = ["_Block"]

    def __init__(self, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.layers = nn.ModuleList([_Block(hidden) for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


@pytest.fixture(scope="module")
def process_group():
    """One world-size-1 process group for the whole file, torn down at the end.

    Module-scoped because initializing NCCL costs more than every test here put
    together, and leaving a group behind would leak into whatever test file runs next
    in the same process.
    """
    already = torch.distributed.is_initialized()
    if not already:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", _free_port())
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    try:
        yield
    finally:
        if not already and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _inject(model):
    """The one injection configuration both the FSDP model and its CPU twin use."""
    return inject_slot_lora(
        model,
        slot_ranks=SLOT_RANKS,
        target_modules=TARGETS,
        scale_mode="match_mt4",
        ref_rank=REF_RANK,
    )


def _build_sharded():
    """A slot-injected toy model wrapped exactly the way the FSDP strategy wraps one.

    Mirrors ``FSDPStrategy.wrap_model``
    (``rlinf/hybrid_engines/fsdp/strategy/fsdp.py``) in the parts that decide the module
    tree: the policy comes from ``get_fsdp_wrap_policy`` with ``is_lora=True``, and
    ``use_orig_params=False`` is this repo's default. Mixed precision and CPU offload are
    left off -- they change dtypes and placement, not which module becomes a unit.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    torch.manual_seed(0)
    model = _Toy().cuda()
    _inject(model)
    policy = get_fsdp_wrap_policy(
        module=model, config={}, is_lora=True, model_type="qwen2.5"
    )
    assert policy is not None, "the wrap policy must exist or this test proves nothing"
    return FSDP(
        module=model,
        auto_wrap_policy=policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=True,
        use_orig_params=False,
    )


def _state_dict(model, full_state_dict: bool):
    """The exact call ``FSDPStrategyBase.get_model_state_dict`` makes."""
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    return get_model_state_dict(
        model=model,
        options=StateDictOptions(cpu_offload=False, full_state_dict=full_state_dict),
    )


class TestSlotLoRAUnderRealFSDP:
    def test_slot_leaves_are_their_own_fsdp_units(self, process_group):
        """The wrap SHAPE the orthogonalization depends on, asserted against real FSDP.

        ``Ā = (Z Zᵀ)^(-1/2) Z`` is not computable from a shard of ``Z``. Its being
        computable at all rests on ``SlotProj`` owning an FSDP unit, so that the FULL
        ``Z`` is all-gathered for the duration of that module's own forward. If the
        per-leaf policy ever stopped firing, training would keep running and silently
        orthogonalize a fragment.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        sharded = _build_sharded()
        wrapped = {
            name: mod
            for name, mod in sharded.named_modules()
            if isinstance(mod, FSDP) and isinstance(mod.module, (SlotProj, SlotOut))
        }
        # 2 layers x 2 targets x 2 halves
        assert len(wrapped) == LAYERS * len(TARGETS) * 2, sorted(wrapped)
        for name, unit in wrapped.items():
            assert unit.module.weight.requires_grad, name
        # ...and the WRAPPER is not a unit: it has children and no `.weight`, so the
        # leaf predicate must not fire on it. That is what keeps its extra state off
        # the FSDP branch of `_get_fqns`.
        assert not [
            n
            for n, m in sharded.named_modules()
            if isinstance(m, FSDP) and isinstance(m.module, SlotLoRALinear)
        ]

    def test_sharded_state_dict_is_reachable(self, process_group):
        """``full_state_dict=False`` -- the ``sync_model_to_rollout`` path that broke.

        ``FSDPActorWorker.get_rollout_state_dict`` calls exactly this on every weight
        sync, so a raise here is a run that dies at its first sync, after the model is
        built and the first rollout is done.
        """
        sharded = _build_sharded()
        state = _state_dict(sharded, full_state_dict=False)
        assert state, "a sharded state dict must not come back empty"
        extra = [k for k in state if k.endswith("_extra_state")]
        assert extra, (
            "the scale is the checkpoint's ONLY record of s and it has to survive "
            "this path; an empty list here means the guarantee was deleted rather "
            "than moved"
        )

    def test_full_state_dict_round_trips_into_a_cpu_copy(self, process_group):
        """``full_state_dict=True`` -- the checkpoint-save path, and the merge contract.

        Zero missing and zero unexpected keys against a fresh CPU-injected model is what
        the converter demands before it merges; a key that does not line up merges
        ``ΔW = 0`` and yields the base model under a new name.
        """
        sharded = _build_sharded()
        state = _state_dict(sharded, full_state_dict=True)
        assert [k for k in state if k.endswith("_extra_state")]

        torch.manual_seed(1)  # different init: the load has to do the work
        cpu_copy = _Toy()
        _inject(cpu_copy)
        report = cpu_copy.load_state_dict(
            {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in state.items()},
            strict=True,
        )
        assert report.missing_keys == [] and report.unexpected_keys == []

    def test_the_full_state_dict_loads_back_into_a_sharded_model(self, process_group):
        """``set_model_state_dict`` -- the resume path, which walks the same FQNs.

        ``FSDPStrategyBase.load_model_with_state_dict`` is the mirror of the save above
        and goes through the same ``_get_fqns``, so the same torch limitation would
        break a RESUME rather than a sync -- later, and after a checkpoint that looked
        fine. The perturbation is what makes the assertion non-vacuous: both models are
        built from the same seed, so without it a no-op load would pass.
        """
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        source = _build_sharded()
        state = _state_dict(source, full_state_dict=True)
        state = {
            k: (v * 2 if k.endswith("slot_A.weight") else v) for k, v in state.items()
        }

        target = _build_sharded()
        set_model_state_dict(
            model=target,
            model_state_dict=state,
            options=StateDictOptions(cpu_offload=False, full_state_dict=True),
        )
        reloaded = _state_dict(target, full_state_dict=True)
        assert set(reloaded) == set(state)
        for key, want in state.items():
            got = reloaded[key]
            if torch.is_tensor(want):
                assert torch.equal(got.cpu(), want.cpu()), key
            else:
                assert got == want, key

    def test_a_wrong_scale_is_still_refused_after_the_round_trip(self, process_group):
        """The guarantee the extra state exists for, exercised through real FSDP.

        Nothing else in the checkpoint pins ``s`` (``Ā`` is invariant to the scale of
        ``Z``), so a converter that derives it from disagreeing CLI flags would produce
        a ``ΔW`` off by a constant factor with no error anywhere. Whatever module now
        carries the record, loading into a differently-scaled copy must RAISE.
        """
        sharded = _build_sharded()
        state = _state_dict(sharded, full_state_dict=True)

        wrong = _Toy()
        inject_slot_lora(
            wrong,
            slot_ranks=SLOT_RANKS,
            target_modules=TARGETS,
            scale_mode="match_mt4",
            ref_rank=REF_RANK * 2,  # halves every s
        )
        with pytest.raises(ValueError, match="scale mismatch"):
            wrong.load_state_dict(
                {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in state.items()},
                strict=True,
            )
