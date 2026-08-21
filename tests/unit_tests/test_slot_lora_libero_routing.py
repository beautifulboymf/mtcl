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

"""Route every REAL task of the R1 run and check it lands in its own suite's slot.

The unit tests in ``test_slot_lora_routing.py`` all use a four-entry toy table, and a
toy table cannot contain the failure this file exists to catch: LIBERO's own
instructions include a pair where one is a proper PREFIX of the other across two
different suites (``libero_goal``'s "turn on the stove" inside ``libero_10`` task 122's
"turn on the stove and put the moka pot on it"). Substring matching resolves that pair
by whichever key it looks at first, and nothing downstream can see the result: the
sample DOES match, so the actor's ``slot/route_fallback`` metric -- which counts only
``suite is None`` -- stays at exactly 0 while ~10% of the long suite is scored by the
goal expert and lands in the goal slot.

Everything here is rebuilt from the SHIPPING artifacts rather than restated: the routed
suites and the slot order come out of ``libero_mt4slot_6gpu.yaml``, the prompt table is
built by the same loop ``FSDPActorWorker._load_teacher_model`` runs, the match order by
the same rule ``_route_match_order`` uses, and the active task ids by the same
``expand_active_suites_to_task_ids`` the env calls. A reconstruction that drifts from
the real one would assert nothing.

Pure CPU, no simulator, no model.
"""

import os

import pytest

libero_benchmark = pytest.importorskip("libero.libero.benchmark")

from omegaconf import OmegaConf  # noqa: E402

from rlinf.envs.libero.utils import (  # noqa: E402
    expand_active_suites_to_task_ids,
    get_libero130_task_id_to_suite,
)
from rlinf.models.slot_lora.routing import match_suite_ids  # noqa: E402

_R1_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples",
    "embodiment",
    "config",
    "libero_mt4slot_6gpu.yaml",
)


def _r1_config():
    """The R1 YAML, read as plain data (no hydra defaults, no env resolution needed).

    Every key this test reads -- ``actor.teacher_map``, ``actor.model.slot_lora`` and
    ``env.train.active_suites`` -- is written literally in that file, so a bare
    ``OmegaConf.load`` is enough and the test does not need a hydra composition (or the
    ``EMBODIED_PATH`` env var the searchpath wants).
    """
    return OmegaConf.load(_R1_CONFIG)


def _prompt_to_suite(routed_suites):
    """The routing table, built by ``_load_teacher_model``'s loop, verbatim.

    Copied from ``rlinf/workers/actor/fsdp_actor_worker.py`` (the block under "TRUE
    multi-teacher routing table"): iterate ``benchmark.libero_suites`` in ITS order,
    keep only the suites that have a teacher, de-dup by task NAME across suites, key on
    the stripped lower-cased instruction. The suite iteration order is load-bearing for
    the bug under test -- ``libero_goal`` comes before ``libero_10`` -- so it is
    reproduced rather than approximated.
    """
    want = set(routed_suites)
    table: dict[str, str] = {}
    seen: set[str] = set()
    for suite_name in getattr(libero_benchmark, "libero_suites", []):
        if suite_name not in want:
            continue
        for task_name, task in libero_benchmark.task_maps.get(suite_name, {}).items():
            if task_name in seen:
                continue
            seen.add(task_name)
            language = getattr(task, "language", None)
            if not language:
                continue
            table[language.strip().lower()] = suite_name
    return table


def _match_order(slot_order, routed_suites, table):
    """``FSDPActorWorker._route_match_order``, verbatim: slots first, then the rest."""
    order = list(slot_order)
    extra = set(routed_suites) | set(table.values())
    order += sorted(s for s in extra if s not in order)
    return order


def _task_by_id():
    """``task_id -> Task`` for the aggregated ``libero_130`` benchmark.

    Same aggregation as ``get_benchmark_overridden("libero_130")`` -- suite order from
    ``benchmark.libero_suites``, de-dup by task name -- so the index here is the task_id
    ``LiberoEnv`` assigns and the one ``expand_active_suites_to_task_ids`` returns.
    """
    aggregated = {}
    for suite_name in getattr(libero_benchmark, "libero_suites", []):
        for task_name, task in libero_benchmark.task_maps.get(suite_name, {}).items():
            aggregated.setdefault(task_name, task)
    return dict(enumerate(aggregated.values()))


@pytest.fixture(scope="module")
def r1_routing():
    cfg = _r1_config()
    routed_suites = [str(s) for s in cfg.actor.teacher_map.keys()]
    slot_order = [str(s) for s in cfg.actor.model.slot_lora.slot_order]
    active_suites = [str(s) for s in cfg.env.train.active_suites]
    table = _prompt_to_suite(routed_suites)
    return {
        "table": table,
        "match_order": _match_order(slot_order, routed_suites, table),
        "task_ids": expand_active_suites_to_task_ids(active_suites),
        "id_to_suite": get_libero130_task_id_to_suite(),
        "task_by_id": _task_by_id(),
    }


def test_the_r1_config_still_describes_the_four_routed_suites(r1_routing):
    # Guards the fixture: if the config's suites, slots or teachers ever stop agreeing,
    # every assertion below would still "pass" while testing a different experiment.
    assert sorted(r1_routing["match_order"]) == [
        "libero_10",
        "libero_goal",
        "libero_object",
        "libero_spatial",
    ]
    assert len(r1_routing["task_ids"]) == 40
    assert len(r1_routing["table"]) == 40


def test_every_active_task_routes_to_its_own_suite(r1_routing):
    """THE test. Every one of the 40 tasks this run trains on must reach its own slot.

    A sample that reaches the wrong slot is invisible everywhere else: it matched, so
    the fallback fraction is 0; the two table keys differ in length, so the actor's
    exact-collision duplicate warning never fires; and the slot it wrongly trains still
    shows a moving ``slot/dw_norm_k``.
    """
    match_order = r1_routing["match_order"]
    misrouted = []
    for task_id in r1_routing["task_ids"]:
        task = r1_routing["task_by_id"][task_id]
        expected = r1_routing["id_to_suite"][task_id]
        got = match_suite_ids([task.language], r1_routing["table"], match_order)
        if got != [match_order.index(expected)]:
            landed = match_order[got[0]] if got[0] >= 0 else "NO SLOT"
            misrouted.append((task_id, task.language, expected, landed))
    assert not misrouted, (
        f"{len(misrouted)} of {len(r1_routing['task_ids'])} active LIBERO tasks route "
        f"to the wrong suite: {misrouted}"
    )


def test_the_prefix_pair_that_makes_this_necessary_is_still_in_libero(r1_routing):
    """Pin the specific collision, so a LIBERO upgrade that removes it is visible.

    If this ever fails, the test above has become vacuous on the shipped benchmark and
    the longest-key-first rule is no longer exercised by real data -- which is worth
    knowing, not worth silently losing.
    """
    table = r1_routing["table"]
    nested = [
        (short, long)
        for short in table
        for long in table
        if short != long and short in long and table[short] != table[long]
    ]
    assert nested == [
        ("turn on the stove", "turn on the stove and put the moka pot on it")
    ]
    assert table["turn on the stove"] == "libero_goal"
    assert table["turn on the stove and put the moka pot on it"] == "libero_10"


def test_routing_does_not_depend_on_the_tables_iteration_order(r1_routing):
    """Reversing the table must not change a single route.

    The table's order is an accident of ``benchmark.libero_suites`` (which puts
    ``libero_goal`` before ``libero_10``); a routing that depends on it is one LIBERO
    release away from moving 10% of a suite into another suite's slot.
    """
    match_order = r1_routing["match_order"]
    forward = dict(r1_routing["table"])
    backward = dict(reversed(list(forward.items())))
    prompts = [r1_routing["task_by_id"][t].language for t in r1_routing["task_ids"]]
    assert match_suite_ids(prompts, forward, match_order) == match_suite_ids(
        prompts, backward, match_order
    )
