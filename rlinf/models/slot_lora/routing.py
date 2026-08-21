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

"""Prompt -> slot-index routing: one answer drives both the teacher and the slot.

``rlinf.workers.actor.fsdp_actor_worker.FSDPActorWorker._route_prepare`` decodes each
rollout sample's prompt once and calls :func:`match_suite_ids` to decide BOTH which
frozen expert TEACHER scores that sample and which suite's LoRA SLOT its gradient is
written into. One answer drives both, so the two can never disagree about a sample --
which is the whole reason the routing lives in one function instead of two loops.

MATCHING SEMANTICS: strip, lower-case, substring containment, LONGEST matching key
wins. The first three are copied verbatim from the teacher-side loop this replaced.
The fourth is a deliberate CORRECTION of it, and it changes results -- see below.

WHY LONGEST-FIRST, AND WHAT IT FIXES.
The routing table is ``{instruction: suite}`` built from LIBERO's own task list, and
LIBERO contains a pair where one instruction is a proper PREFIX of another IN A
DIFFERENT SUITE: ``libero_goal``'s "turn on the stove" sits inside ``libero_10`` task
122's "turn on the stove and put the moka pot on it". The loop this replaced took the
first key in the routing dict's own iteration order, and that order comes from
``benchmark.libero_suites``, which puts ``libero_goal`` before ``libero_10`` -- so task
122 resolved to ``libero_goal``. Measured against the installed ``libero`` package: of
the 40 tasks this experiment routes, exactly one resolved to the wrong suite, and
exactly one prompt was matched by more than one table key.

NOTHING DOWNSTREAM COULD SEE IT. The sample DID match, so the actor's fatal
``slot/route_fallback`` check -- which counts only ``suite is None`` -- stayed at
exactly 0.0. The actor's duplicate-key warning tests EXACT key collision, and two keys
of different lengths never collide. The wrongly-trained slot still showed a moving
``slot/dw_norm_k``. So one of the ten ``libero_10`` tasks -- ~10% of the long suite,
~2.5% of every micro-batch -- was scored by the goal expert and pushed its gradient
into the goal slot, while the long slot (rank 128 of R=256, the largest precisely
because long has the biggest deficit) silently never saw it.

THIS DEVIATES FROM THE PRIOR BASELINE, ON PURPOSE. The pre-existing teacher-side
router had these same first-match semantics, so the mt4 / mt4w2 baseline runs
mis-scored the same ~2.5% of samples. The actor now derives teacher routing from this
function too, so fixing the slot side necessarily changes teacher routing relative to
those runs. That is a deliberate, documented deviation in the direction of
correctness: a sample scored by another suite's expert was never intended behaviour in
either arm.

WHY A SILENT CORRECTION HERE RATHER THAN A LOUD REFUSAL AT TABLE-BUILD TIME.
The substring-chain case is not ambiguous -- it has a unique right answer. A key that
CONTAINS another key is strictly more specific, and the more specific instruction is by
construction the one the prompt was generated from; there is nothing for an operator to
decide. Refusing such a table would also refuse the shipping LIBERO benchmark, i.e.
make the feature unrunnable on the only model it exists for. What has no right answer
is a TIE: two keys of the SAME length, mapping to DIFFERENT suites, both contained in
one prompt. Longest-first cannot break that, dict order would break it by accident, and
the result would be a coin flip over which slot takes the gradient -- so that case
RAISES (see :func:`match_suite_ids`). Between them the two rules cover the whole space:
decidable is decided, undecidable is refused, and nothing is left to iteration order.

One deliberate behavioral difference from the teacher router: on no match, the teacher
router falls back to a default teacher (so every sample is always scored by *some*
expert); this function returns ``-1`` instead of a fallback slot, because "no slot owns
this sample" must be representable and never silently aliased to slot 0's gradient.
"""


def match_suite_ids(
    texts: list[str],
    prompt_to_suite: dict[str, str],
    suite_order: list[str],
) -> list[int]:
    """Map decoded rollout prompts to slot indices; ``-1`` = no slot owns this sample.

    Args:
        texts: Decoded prompts for one micro-batch (one string per sample).
        prompt_to_suite: ``{instruction_substring: suite_name}``. Matching checks
            whether a key is a substring of the (stripped, lower-cased) prompt, and the
            LONGEST matching key wins. The dict's own iteration order is NOT consulted:
            it is an accident of ``benchmark.libero_suites`` and letting it decide is
            what sent ``libero_10`` task 122 into the goal slot (see the module
            docstring).
        suite_order: The slot index order, e.g. ``["libero_10", "libero_goal",
            "libero_spatial", "libero_object"]``. The returned index is this list's
            position of the matched suite. A suite that ``prompt_to_suite`` maps to
            but that is absent from ``suite_order`` yields ``-1`` for that sample,
            same as no match at all.

    Returns:
        One int per input text, either a valid index into ``suite_order`` or ``-1``.
        Always the same length as ``texts``. Pure Python (no torch, no I/O, no
        logging) -- the caller converts this to a tensor and owns device placement.

    Raises:
        ValueError: if a prompt is matched by two DIFFERENT-suite keys of the same
            (maximal) length. Nothing can choose between them, so the sample's teacher
            and its slot would both be picked by dict order; the run must not start.
    """
    # Sorted ONCE per call, not once per text: the table is ~40 entries and identical
    # for every sample of the micro-batch. The secondary key is the instruction itself,
    # so equal-length keys have a fixed order too and the function is a pure function
    # of (texts, table contents, suite_order) -- never of how the table was built.
    ordered = sorted(prompt_to_suite.items(), key=lambda kv: (-len(kv[0]), kv[0]))
    ids: list[int] = []
    for text in texts:
        normalized = text.strip().lower()
        suite = None
        matched_key = None
        for instruction, candidate_suite in ordered:
            if instruction not in normalized:
                continue
            if suite is None:
                suite, matched_key = candidate_suite, instruction
                continue
            # Past the first hit the scan continues only while keys are the same
            # LENGTH; anything shorter is strictly less specific and is settled.
            if len(instruction) < len(matched_key):
                break
            if candidate_suite != suite:
                raise ValueError(
                    f"prompt {text!r} is matched by two equally specific routing keys "
                    f"that disagree: {matched_key!r} -> {suite!r} and "
                    f"{instruction!r} -> {candidate_suite!r}. Longest-key-first cannot "
                    "break a tie of equal lengths, so this sample's teacher and its "
                    "LoRA slot would both be chosen by the routing dict's iteration "
                    "order -- a coin flip that raises nothing and shows up in no "
                    "metric. Make the two instructions distinguishable, or route only "
                    "one of the two suites."
                )
        if suite is not None and suite in suite_order:
            ids.append(suite_order.index(suite))
        else:
            ids.append(-1)
    return ids
