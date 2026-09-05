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
"""seqslot: the sequential per-suite rounds -- round table, masked-self anchor, frozen A.

Three mechanisms, each tested where it can actually fail:

* ``parse_slot_round_table`` -- the config errors that would otherwise surface ~25
  minutes into a run as a wrong-round training step that LOOKS fine.
* ``SlotGate.excluding`` / the exclusion branch of ``SlotOut.forward`` -- the anchor
  teacher's VALUE. ``student minus slot k`` must equal a model whose slot-k columns
  are zero, bit-for-bit restorable, and must refuse to coexist with a routed
  (training) forward.
* ``SlotProj(frozen_orth=True)`` -- the stored weight IS the row-orthonormal A-bar,
  ``orth_weight`` returns it without a Newton-Schulz pass, and the diagnostics still
  measure honestly.
"""

import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, "/home/fanruochen/CL/RLinf")

from rlinf.models.slot_lora.modules import (  # noqa: E402
    SlotGate,
    SlotLoRALinear,
    SlotOut,
    SlotProj,
    make_slot_ids,
)
from rlinf.workers.actor.fsdp_actor_worker import (  # noqa: E402
    parse_slot_round_table,
    slot_alt_schedule_for_step,
)

ROUNDS = [[0, "libero_object"], [6, "libero_10"], [12, "libero_spatial"], [18, "libero_goal"]]


# ---------------------------------------------------------------- round table
def test_none_and_empty_are_off():
    assert parse_slot_round_table(None) is None
    assert parse_slot_round_table([]) is None


def test_valid_table_sorts_ascending():
    got = parse_slot_round_table([[12, "c"], [0, "a"], [6, "b"]])
    assert got == [(0, "a"), (6, "b"), (12, "c")]


def test_round_lookup_through_the_shared_stage_selector():
    st = parse_slot_round_table(ROUNDS)
    picks = [slot_alt_schedule_for_step(st, None, s) for s in range(24)]
    assert picks[:6] == ["libero_object"] * 6
    assert picks[6:12] == ["libero_10"] * 6
    assert picks[12:18] == ["libero_spatial"] * 6
    assert picks[18:] == ["libero_goal"] * 6
    # beyond the last boundary the last round stays in force
    assert slot_alt_schedule_for_step(st, None, 999) == "libero_goal"


def test_unknown_step_keeps_the_default_not_stage0():
    st = parse_slot_round_table(ROUNDS)
    # a resumed run whose counter is not wired: keep whatever the caller passes as
    # the fallback (the round already in force), never snap to round 0
    assert slot_alt_schedule_for_step(st, "libero_spatial", None) == "libero_spatial"


@pytest.mark.parametrize(
    "bad",
    [
        42,  # not a sequence of pairs
        [[0, "a", "extra"]],  # not a pair
        [["0", "a"]],  # step not an int
        [[True, "a"]],  # bool masquerading as int
        [[-1, "a"]],  # negative step
        [[0, "a"], [0, "b"]],  # duplicated step
        [[3, "a"]],  # no stage at 0
        [[0, ""]],  # empty suite
        [[0, 7]],  # non-string suite
    ],
)
def test_malformed_tables_raise(bad):
    with pytest.raises(ValueError):
        parse_slot_round_table(bad)


# ---------------------------------------------------------------- excluding()
def _small_layer(k=3, d_in=16, d_out=8, ranks=(4, 4, 4), seed=0):
    torch.manual_seed(seed)
    gate = SlotGate(num_slots=k, strict=True)
    base = nn.Linear(d_in, d_out, bias=False)
    layer = SlotLoRALinear(base, tuple(ranks), 1.0, gate)
    # non-zero B everywhere, so an exclusion that removed nothing would be visible
    with torch.no_grad():
        layer.slot_B.weight.normal_()
    return layer, gate


def test_excluding_value_is_full_minus_slot_k():
    layer, gate = _small_layer()
    x = torch.randn(5, 16)
    for k in range(3):
        with gate.ungated():
            full = layer(x)
        with gate.excluding(k):
            got = layer(x)
        # reference: zero slot k's B columns and run the plain ungated forward
        ref_layer, ref_gate = _small_layer()
        with torch.no_grad():
            ref_layer.load_state_dict(layer.state_dict())
            out = ref_layer.slot_B
            s = out.offsets[k]
            ref_layer.slot_B.weight[:, s : s + out.slot_ranks[k]].zero_()
        with ref_gate.ungated():
            want = ref_layer(x)
        assert torch.allclose(got, want, atol=1e-5), f"slot {k}"
        # and it is NOT the full sum (B was random-nonzero)
        assert not torch.allclose(got, full, atol=1e-5)


def test_excluding_restores_state_even_on_exception():
    _, gate = _small_layer()
    ids = make_slot_ids([0, 1, 2], 3)
    with gate.scoped(ids):
        before = (gate._ids, gate.strict, gate.exclude_slot)
        with pytest.raises(RuntimeError):
            with gate.excluding(1):
                assert gate.exclude_slot == 1 and gate._ids is None
                raise RuntimeError("boom")
        assert (gate._ids, gate.strict, gate.exclude_slot) == before


@pytest.mark.parametrize("bad", [True, -1, 3, 2.0, "1", None])
def test_excluding_rejects_invalid_slot(bad):
    _, gate = _small_layer()
    with pytest.raises(ValueError):
        with gate.excluding(bad):
            pass


def test_excluding_on_countless_gate_raises():
    gate = SlotGate(num_slots=None, strict=False)
    with pytest.raises(ValueError):
        with gate.excluding(0):
            pass


def test_routed_forward_inside_excluding_raises():
    layer, gate = _small_layer()
    x = torch.randn(4, 16)
    ids = make_slot_ids([0, 1, 2, 0], 3)
    # simulate the leak: a training (routed) forward running inside the anchor context
    with gate.excluding(1):
        gate._ids = ids  # bypass scoped() on purpose; the forward must still refuse
        with pytest.raises(RuntimeError, match="exclude_slot"):
            layer(x)


def test_excluding_gradient_free_teacher_and_gated_student_coexist():
    """The seqslot step in miniature: gated student fwd, no-grad excluded fwd, backward."""
    layer, gate = _small_layer()
    x = torch.randn(6, 16)
    ids = make_slot_ids([1] * 6, 3)  # seqslot: every sample owns the round's slot
    with gate.scoped(ids):
        student = layer(x)
        with torch.no_grad(), gate.excluding(1):
            teacher = layer(x)
        # after the anchor forward the routing must be back for the backward
        assert gate.current() is not None
        loss = ((student - teacher.detach()) ** 2).mean()
        loss.backward()
    g = layer.slot_B.weight.grad
    assert g is not None
    out = layer.slot_B
    s1 = out.offsets[1]
    inside = g[:, s1 : s1 + out.slot_ranks[1]]
    outside = torch.cat(
        [g[:, : s1], g[:, s1 + out.slot_ranks[1] :]], dim=1
    )
    assert inside.abs().sum() > 0  # the round's slot learns
    assert outside.abs().sum() == 0  # every other slot got an exact zero


# ---------------------------------------------------------------- frozen_orth
def test_frozen_orth_init_is_row_orthonormal():
    p = SlotProj(32, 12, 1.0, frozen_orth=True)
    gram = p.weight.detach().float() @ p.weight.detach().float().T
    assert torch.allclose(gram, torch.eye(12), atol=1e-4)


def test_frozen_orth_orth_weight_returns_the_stored_weight():
    p = SlotProj(32, 12, 1.0, frozen_orth=True)
    assert p.orth_weight() is p.weight  # no Newton-Schulz, no copy


def test_unfrozen_default_is_gaussian_not_orthonormal():
    p = SlotProj(32, 12, 1.0)  # default: existing behavior untouched
    gram = p.weight.detach().float() @ p.weight.detach().float().T
    assert not torch.allclose(gram, torch.eye(12), atol=1e-3)
    assert p.orth_weight() is not p.weight


def test_frozen_orth_forward_matches_polar_of_itself():
    # NS on an already-orthonormal Z is (numerically) the identity, so the frozen
    # forward and a hypothetical polar forward agree -- which is what makes the
    # CONVERTER's --slot-frozen-orth flag a correctness knob rather than a big one.
    p = SlotProj(32, 12, 1.0, frozen_orth=True)
    from rlinf.models.slot_lora.orth import orthogonalize

    a = p.weight.detach().float()
    assert torch.allclose(orthogonalize(a, iters=p.iters, eps=p.eps), a, atol=1e-5)


def test_frozen_orth_diag_still_collects():
    p = SlotProj(32, 12, 1.0, frozen_orth=True)
    p.arm_diag()
    p.orth_weight()
    assert p._diag is not None and "gram" in p._diag


def test_frozen_orth_threads_through_slotloralinear():
    gate = SlotGate(num_slots=2, strict=False)
    layer = SlotLoRALinear(
        nn.Linear(16, 8, bias=False), (4, 4), 1.0, gate, frozen_orth=True
    )
    assert layer.slot_A.frozen_orth is True
    assert layer.slot_A.orth_weight() is layer.slot_A.weight


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
