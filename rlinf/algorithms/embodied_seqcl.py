# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sequential continual-learning schedule for LIBERO on-policy distillation (OPD).

CORE ALGORITHM (this file is the single source of truth for the recipe).

Setting
-------
Learn four LIBERO suites *one after another* without forgetting the earlier ones:

    stage 0: libero_spatial
    stage 1: libero_object   (+ rehearse spatial)
    stage 2: libero_goal     (+ rehearse spatial, object)
    stage 3: libero_10/long  (+ rehearse spatial, object, goal)

- **Student** = SFT generalist ``RLinf-OpenVLAOFT-LIBERO-130-Base-Lora`` (has some
  ability on every suite, so OPD has a non-zero on-policy signal to work with -- it
  is NOT a cold start). Each stage resumes from the previous stage's weights.
- **Teacher** = the single 130-GRPO generalist ``RLinf-OpenVLAOFT-LIBERO-130``, which
  is strong (~97%) on all four suites. We therefore load **one** teacher and route
  every suite to it (``teacher_map`` below). The routing table is kept explicit so a
  future experiment can point each suite at its own expert teacher without touching
  the training loop -- only the map changes.

Method at stage k
-----------------
The student rolls out on-policy over the *active* suites = {new suite} ∪ {old suites}
(one ``libero_130`` env restricted to those suites' task ids), and every rollout is
scored by the teacher; the differentiable forward-KL OPD loss then pulls the student
toward the teacher on ALL active suites at once. Learning the new suite and rehearsing
the old ones happen in the same optimisation, from one shared teacher.

Rehearsal weighting is deliberately MECHANICAL for a first cut: the new suite gets a
larger share of the on-policy data than each old suite (``suite_sample_weights``, e.g.
new=1.0, old=0.3). With a single teacher, "more data on suite S" is equivalent to
"more teacher weight on S" in the averaged OPD loss, so this data-mixture knob is the
per-teacher weight ratio. The env consumes the weights and oversamples the new suite;
nothing in the loss needs to change.

This module holds only the *policy* (order, active sets, weights, teacher routing). The
mechanisms live in:
  - env oversampling        : rlinf/envs/libero/libero_env.py (reads suite_sample_weights)
  - suite<->task_id mapping  : rlinf/envs/libero/utils.py (libero_130 aggregation order)
  - teacher load + OPD loss  : rlinf/workers/actor/fsdp_actor_worker.py (teacher_map, forward-KL)
It has no heavy imports so it can be run standalone to emit a stage plan for the driver:

    python rlinf/algorithms/embodied_seqcl.py <stage_idx>
"""

from __future__ import annotations

import json
import sys

# LIBERO suites in the order we learn them. "libero_10" is LIBERO-LONG. These names
# match both the env configs (env/libero_*.yaml -> task_suite_name) and the suite keys
# used by the libero_130 aggregated benchmark, so they can be expanded to task ids at
# runtime (rlinf/envs/libero/utils.py::expand_active_suites_to_task_ids).
SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# Mechanical rehearsal weights (data mixture): the new suite is learned "harder" than
# the old suites are rehearsed. Tune later; kept crude on purpose (user's call).
DEFAULT_CURRENT_WEIGHT = 1.0
DEFAULT_OLD_WEIGHT = 0.3

# The single generalist teacher every suite routes to for now.
DEFAULT_TEACHER_PATH = (
    "/share/fanruochen-local/checkpoints/RLinf-OpenVLAOFT-LIBERO-130"
)
# The student's starting weights for stage 0 (SFT generalist, non-RL).
DEFAULT_STUDENT_PATH = (
    "/share/fanruochen-local/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora"
)


def num_stages() -> int:
    return len(SUITE_ORDER)


def stage_suites(stage_idx: int) -> tuple[str, list[str]]:
    """Return ``(current_suite, old_suites)`` active at stage ``stage_idx``.

    ``old_suites`` is every suite learned before this stage (to be rehearsed).
    """
    if not 0 <= stage_idx < len(SUITE_ORDER):
        raise ValueError(
            f"stage_idx {stage_idx} out of range [0, {len(SUITE_ORDER) - 1}]"
        )
    current = SUITE_ORDER[stage_idx]
    old = list(SUITE_ORDER[:stage_idx])
    return current, old


def active_suites(stage_idx: int) -> list[str]:
    """All suites the env rolls out on at stage ``stage_idx`` (new first, then old)."""
    current, old = stage_suites(stage_idx)
    return [current] + old


def suite_sample_weights(
    stage_idx: int,
    current_weight: float = DEFAULT_CURRENT_WEIGHT,
    old_weight: float = DEFAULT_OLD_WEIGHT,
) -> dict[str, float]:
    """Data-mixture weight per active suite: new heavy, each old light."""
    current, old = stage_suites(stage_idx)
    weights = {current: float(current_weight)}
    for suite in old:
        weights[suite] = float(old_weight)
    return weights


def teacher_map(
    stage_idx: int, teacher_path: str = DEFAULT_TEACHER_PATH
) -> dict[str, str]:
    """Route each active suite -> a teacher checkpoint.

    NOW: every suite -> the single 130 generalist, so the actor de-dups by path and
    loads exactly one teacher. LATER: give a suite its own expert by changing its value
    here (e.g. ``{"libero_object": ".../GRPO-object", ...}``); no loop change needed.
    """
    return {suite: teacher_path for suite in active_suites(stage_idx)}


def stage_plan(
    stage_idx: int,
    teacher_path: str = DEFAULT_TEACHER_PATH,
    current_weight: float = DEFAULT_CURRENT_WEIGHT,
    old_weight: float = DEFAULT_OLD_WEIGHT,
) -> dict:
    """Full, self-contained description of one sequential stage."""
    current, old = stage_suites(stage_idx)
    return {
        "stage_idx": stage_idx,
        "current_suite": current,
        "old_suites": old,
        "active_suites": active_suites(stage_idx),
        "suite_sample_weights": suite_sample_weights(
            stage_idx, current_weight, old_weight
        ),
        "teacher_map": teacher_map(stage_idx, teacher_path),
    }


def _omega_list(items) -> str:
    """OmegaConf inline list with UNQUOTED elements: ``[a, b]``. The config parses these
    via ``${oc.decode:...}``, whose grammar REJECTS the double-quoted strings JSON emits
    (``GrammarParseError: mismatched input '"'``), so never use json.dumps here."""
    return "[" + ", ".join(str(x) for x in items) + "]"


def _omega_dict(d) -> str:
    """OmegaConf inline dict with UNQUOTED keys: ``{a: 1.0, b: 0.3}`` (same reason)."""
    return "{" + ", ".join(f"{k}: {v}" for k, v in d.items()) + "}"


def _emit_shell(stage_idx: int) -> str:
    """Print ``export VAR=...`` lines the bash driver can ``eval``.

    Emits the per-stage bits (the student path is threaded by the driver across stages,
    since it depends on the previous stage's output, so it is intentionally NOT here).
    """
    plan = stage_plan(stage_idx)
    lines = [
        f"export SEQCL_STAGE_IDX={stage_idx}",
        f"export SEQCL_CURRENT_SUITE={plan['current_suite']}",
        # OmegaConf ${oc.decode:...} grammar: UNQUOTED elements/keys only (NOT JSON).
        "export SEQCL_ACTIVE_SUITES='%s'" % _omega_list(plan["active_suites"]),
        "export SEQCL_SUITE_WEIGHTS='%s'" % _omega_dict(plan["suite_sample_weights"]),
        # informational only (the config hardcodes teacher_map); not oc.decoded, JSON ok.
        "export SEQCL_TEACHER_MAP='%s'" % json.dumps(plan["teacher_map"]),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(_emit_shell(idx))
