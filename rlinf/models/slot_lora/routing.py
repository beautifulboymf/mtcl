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

"""Prompt -> slot-index routing, mirroring the teacher-side router exactly.

``rlinf.workers.actor.fsdp_actor_worker.FSDPActorWorker._teacher_forward`` decodes
each rollout sample's prompt and matches it against ``self.teacher_prompt_to_suite``
to pick which frozen expert TEACHER scores that sample (see the loop around
``for k, v in _route.items(): if k in tl: ...``). Slot-LoRI's student side needs the
same sample to land in the same suite's LoRA SLOT, so the gradient it produces is
written back into the slot that owns it.

The matching semantics here -- strip, lower-case, substring containment, first key
in the routing dict's own iteration order wins -- are copied VERBATIM from that
teacher-side loop, not reimplemented from a spec. This is deliberate: if the two
routers ever disagree on a single sample, that sample is silently scored by one
suite's teacher while its gradient is written into a different suite's slot. There
is no exception raised and no obviously wrong loss curve -- the run just quietly
trains the wrong slot on the wrong signal. Any future change to the teacher-side
loop must be mirrored here (or this function must be extracted and shared), or this
invariant breaks silently.

One deliberate behavioral difference from the teacher router: on no match, the
teacher router falls back to a default teacher (so every sample is always scored by
*some* expert); this function returns ``-1`` instead of a fallback slot, because
"no slot owns this sample" must be representable and never silently aliased to slot
0's gradient.
"""


def match_suite_ids(
    texts: list[str],
    prompt_to_suite: dict[str, str],
    suite_order: list[str],
) -> list[int]:
    """Map decoded rollout prompts to slot indices; ``-1`` = no slot owns this sample.

    Args:
        texts: Decoded prompts for one micro-batch (one string per sample).
        prompt_to_suite: ``{instruction_substring: suite_name}``, keyed and iterated
            in exactly the same way as the teacher router's routing table. Matching
            checks whether a key is a substring of the (stripped, lower-cased)
            prompt, and the FIRST key (in this dict's own iteration order) that
            matches wins -- later matches are never considered.
        suite_order: The slot index order, e.g. ``["libero_10", "libero_goal",
            "libero_spatial", "libero_object"]``. The returned index is this list's
            position of the matched suite. A suite that ``prompt_to_suite`` maps to
            but that is absent from ``suite_order`` yields ``-1`` for that sample,
            same as no match at all.

    Returns:
        One int per input text, either a valid index into ``suite_order`` or ``-1``.
        Always the same length as ``texts``. Pure Python (no torch, no I/O, no
        logging) -- the caller converts this to a tensor and owns device placement.
    """
    ids: list[int] = []
    for text in texts:
        normalized = text.strip().lower()
        suite = None
        for instruction, candidate_suite in prompt_to_suite.items():
            if instruction in normalized:
                suite = candidate_suite
                break
        if suite is not None and suite in suite_order:
            ids.append(suite_order.index(suite))
        else:
            ids.append(-1)
    return ids
