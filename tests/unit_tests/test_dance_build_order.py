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
"""dance_build_order: the two contracts that make the DanceOPD transfer real.

If either fails silently the run LOOKS fine: a mixed-suite update quietly re-creates the
gradient interference the mechanism exists to remove, and a broken thinning quietly trains
dense-correlated again.
"""

import sys

import pytest
import torch

sys.path.insert(0, "/home/fanruochen/CL/RLinf")

from rlinf.workers.actor.fsdp_actor_worker import dance_build_order  # noqa: E402


def _ids(n_per_suite, num_suites=4, unrouted=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.cat(
        [torch.full((n_per_suite,), k) for k in range(num_suites)]
        + [torch.full((unrouted,), -1)]
    )
    return ids[torch.randperm(ids.numel(), generator=g)]


def _suite_of(order, ids, bspr):
    return [ids[order[i * bspr : (i + 1) * bspr]].unique().tolist() for i in range(order.numel() // bspr)]


def test_every_update_is_single_suite():
    ids = _ids(256)
    order, sched = dance_build_order(ids, 32, 0.25, 4, seed=7)
    per_update = _suite_of(order, ids, 32)
    assert all(len(u) == 1 for u in per_update), per_update
    assert [u[0] for u in per_update] == sched


def test_rotation_covers_all_suites_equally():
    ids = _ids(256)
    _, sched = dance_build_order(ids, 32, 0.25, 4, seed=7)
    assert sched[:4] == [0, 1, 2, 3]
    counts = {k: sched.count(k) for k in range(4)}
    assert len(set(counts.values())) == 1  # equal updates per suite


def test_keep_frac_thins_each_suite():
    ids = _ids(400)
    order, _ = dance_build_order(ids, 25, 0.25, 4, seed=1)
    # kept unique samples per suite ~ 100*0.25
    for k in range(4):
        uniq = set(order[ids[order] == k].tolist())
        assert 90 <= len(uniq) <= 110 or len(uniq) == 100 or abs(len(uniq) - 100) <= 10


def test_unrouted_samples_never_appear():
    ids = _ids(64, unrouted=64)
    order, _ = dance_build_order(ids, 16, 1.0, 4, seed=3)
    assert (ids[order] == -1).sum().item() == 0


def test_length_is_multiple_of_batch():
    ids = _ids(100)
    order, sched = dance_build_order(ids, 24, 0.5, 4, seed=5)
    assert order.numel() % 24 == 0
    assert order.numel() // 24 == len(sched)


def test_imbalanced_ranks_wrap_around_not_borrow():
    # suite 0 has only 8 samples but must fill 32-sample updates: wraparound reuses
    # its OWN pool; no foreign suite leaks in.
    ids = torch.cat([torch.zeros(8), torch.ones(96), torch.full((96,), 2.0), torch.full((96,), 3.0)]).long()
    order, sched = dance_build_order(ids, 32, 1.0, 4, seed=2)
    per_update = _suite_of(order, ids, 32)
    for suites, k in zip(per_update, sched):
        assert suites == [k]


def test_missing_suite_is_skipped_not_fabricated():
    ids = torch.cat([torch.zeros(64), torch.ones(64)]).long()  # suites 2,3 absent
    order, sched = dance_build_order(ids, 16, 1.0, 4, seed=4)
    assert set(sched) == {0, 1}


def test_deterministic_given_seed():
    ids = _ids(128)
    o1, s1 = dance_build_order(ids, 16, 0.5, 4, seed=11)
    o2, s2 = dance_build_order(ids, 16, 0.5, 4, seed=11)
    assert torch.equal(o1, o2) and s1 == s2
    o3, _ = dance_build_order(ids, 16, 0.5, 4, seed=12)
    assert not torch.equal(o1, o3)


@pytest.mark.parametrize("bad", [
    dict(batch=0), dict(frac=0.0), dict(frac=1.5),
])
def test_bad_args_raise(bad):
    ids = _ids(64)
    with pytest.raises(ValueError):
        dance_build_order(ids, bad.get("batch", 16), bad.get("frac", 0.5), 4, seed=0)


def test_all_unrouted_raises():
    with pytest.raises(ValueError):
        dance_build_order(torch.full((64,), -1), 16, 0.5, 4, seed=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_agreed_rotation_is_obeyed_exactly():
    ids = torch.cat([torch.zeros(8), torch.ones(96), torch.full((96,), 2.0)]).long()
    order, sched = dance_build_order(
        ids, 16, 1.0, 4, seed=9, suites_in_rotation=[0, 2], rotations=3
    )
    assert sched == [0, 2] * 3
    per_update = _suite_of(order, ids, 16)
    assert per_update == [[0], [2]] * 3


def test_agreed_rotation_with_locally_empty_suite_raises():
    ids = torch.cat([torch.zeros(64), torch.ones(64)]).long()  # suite 2 empty here
    with pytest.raises(ValueError, match="agreed rotation"):
        dance_build_order(ids, 16, 1.0, 4, seed=9, suites_in_rotation=[0, 2], rotations=1)
