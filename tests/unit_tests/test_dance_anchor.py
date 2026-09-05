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
"""Memory-Anchor augmentation of dance updates: the contracts that make it real.

If the majority-suite share silently shrinks, the update-level isolation dance exists
for is gone; if anchors are drawn from the update's own suite, the whole term is a
no-op wearing the method's name; if similarity retrieval is broken, anchors degrade to
random cross-suite mixing -- which is exactly the condition dance measured as harmful.
"""

import sys

import pytest
import torch

sys.path.insert(0, "/home/fanruochen/CL/RLinf")

from rlinf.workers.actor.fsdp_actor_worker import (  # noqa: E402
    anchor_obs_embed,
    dance_anchor_augment,
    dance_build_order,
)


def _setup(n_per_suite=64, num_suites=4, bspr=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.cat([torch.full((n_per_suite,), k) for k in range(num_suites)])
    ids = ids[torch.randperm(ids.numel(), generator=g)]
    order, sched = dance_build_order(ids, bspr, 0.5, num_suites, seed=seed)
    # embeddings: suite k centered at unit direction e_k with small noise -> similarity
    # structure is known exactly.
    d = num_suites + 4
    emb = torch.zeros(ids.numel(), d)
    for i, s in enumerate(ids.tolist()):
        emb[i, s] = 1.0
        emb[i, num_suites:] = 0.05 * torch.randn(4, generator=g)
    emb = torch.nn.functional.normalize(emb, dim=1)
    return ids, order, sched, emb


def test_majority_suite_preserved_and_anchors_cross_suite():
    ids, order, sched, emb = _setup()
    bspr, frac = 16, 0.25
    new, stats = dance_anchor_augment(order, sched, ids, bspr, frac, emb)
    n_a = round(bspr * frac)
    for u, k in enumerate(sched):
        sl = ids[new[u * bspr : (u + 1) * bspr]]
        assert (sl[: bspr - n_a] == k).all(), "majority slots must stay the update suite"
        assert (sl[bspr - n_a :] != k).all(), "anchor slots must be OTHER suites"
    assert sum(stats["anchor_counts"].values()) == n_a * len(sched)


def test_zero_frac_is_identity():
    ids, order, sched, emb = _setup()
    new, stats = dance_anchor_augment(order, sched, ids, 16, 0.0, emb)
    assert torch.equal(new, order) and stats["anchor_counts"] == {}


def test_frac_bounds_raise():
    ids, order, sched, emb = _setup()
    with pytest.raises(ValueError):
        dance_anchor_augment(order, sched, ids, 16, 0.6, emb)
    with pytest.raises(ValueError):
        dance_anchor_augment(order, sched, ids, 16, -0.1, emb)


def test_anchors_are_nearest_cross_suite_in_embed_space():
    # plant ONE cross-suite sample right on suite 0's centroid direction; it must be
    # picked for every suite-0 update.
    ids, order, sched, emb = _setup()
    planted = (ids == 1).nonzero(as_tuple=True)[0][0]
    emb[planted] = 0.0
    emb[planted, 0] = 1.0  # exactly suite-0 direction
    new, _ = dance_anchor_augment(order, sched, ids, 16, 0.25, emb)
    for u, k in enumerate(sched):
        if k == 0:
            assert planted in new[u * 16 : (u + 1) * 16], (
                "the planted most-similar cross-suite sample must be retrieved"
            )


def test_suite_weight_biases_selection():
    ids, order, sched, emb = _setup()
    # make every cross-suite candidate equally similar, then weight suite 3 up hard:
    emb = torch.nn.functional.normalize(torch.ones_like(emb), dim=1)
    w = torch.tensor([1.0, 1.0, 1.0, 2.0])
    new, stats = dance_anchor_augment(order, sched, ids, 16, 0.25, emb, suite_w=w)
    picked = stats["anchor_counts"]
    # suite 3 must dominate every update where it is a candidate (all k != 3 updates)
    n_updates_not3 = sum(1 for k in sched if k != 3)
    assert picked.get(3, 0) == n_updates_not3 * 4, picked


def test_single_suite_rollout_is_noop_not_crash():
    ids = torch.zeros(64, dtype=torch.long)
    order, sched = dance_build_order(ids, 16, 1.0, 4, seed=1)
    emb = torch.nn.functional.normalize(torch.randn(64, 8), dim=1)
    new, stats = dance_anchor_augment(order, sched, ids, 16, 0.25, emb)
    assert torch.equal(new, order) and stats["anchor_counts"] == {}


def test_update_count_and_length_unchanged():
    ids, order, sched, emb = _setup()
    new, _ = dance_anchor_augment(order, sched, ids, 16, 0.25, emb)
    assert new.numel() == order.numel(), "anchors REPLACE positions, never extend"


def test_obs_embed_text_only_and_combined():
    texts = ["pick up the bowl", "pick up the bowl", "open the drawer"]
    e = anchor_obs_embed(None, texts)
    assert e.shape[0] == 3
    assert torch.allclose(e[0], e[1])
    assert (e[0] @ e[2]) < (e[0] @ e[1]) - 0.1
    pv = torch.rand(3, 6, 32, 32)  # wrist-concat 6-channel
    e2 = anchor_obs_embed(pv, texts, img_weight=0.5)
    assert e2.shape[0] == 3
    assert torch.allclose(e2.norm(dim=1), torch.ones(3), atol=1e-4)


def test_obs_embed_bad_pixel_shape_raises():
    with pytest.raises(ValueError):
        anchor_obs_embed(torch.rand(2, 3), ["a", "b"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
