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

from rlinf.models.slot_lora.routing import match_suite_ids

_SUITE_ORDER = ["libero_10", "libero_goal", "libero_spatial", "libero_object"]

_PROMPT_TO_SUITE = {
    "pick up the black bowl": "libero_spatial",
    "put the bowl on the plate": "libero_object",
    "open the top drawer": "libero_goal",
    "turn on the stove": "libero_10",
}


def test_each_prompt_maps_to_its_suite_index():
    texts = [
        "In: What action should the robot take to pick up the black bowl "
        "and place it on the tray? Out:",
        "In: What action should the robot take to put the bowl on the plate? Out:",
        "In: What action should the robot take to open the top drawer? Out:",
        "In: What action should the robot take to turn on the stove? Out:",
    ]
    got = match_suite_ids(texts, _PROMPT_TO_SUITE, _SUITE_ORDER)
    assert got == [
        _SUITE_ORDER.index("libero_spatial"),
        _SUITE_ORDER.index("libero_object"),
        _SUITE_ORDER.index("libero_goal"),
        _SUITE_ORDER.index("libero_10"),
    ]


def test_instruction_is_a_substring_of_a_longer_prompt():
    # Realistic rollout prompt: the instruction is embedded inside a template,
    # not the whole string.
    text = (
        "In: What action should the robot take to pick up the black bowl "
        "between the plate and the ramekin and place it on the plate? Out:"
    )
    got = match_suite_ids([text], _PROMPT_TO_SUITE, _SUITE_ORDER)
    assert got == [_SUITE_ORDER.index("libero_spatial")]


def test_case_insensitive_and_whitespace_is_stripped():
    text = "   IN: WHAT ACTION SHOULD THE ROBOT TAKE TO OPEN THE TOP DRAWER? OUT:   "
    got = match_suite_ids([text], _PROMPT_TO_SUITE, _SUITE_ORDER)
    assert got == [_SUITE_ORDER.index("libero_goal")]


def test_unknown_prompt_gives_negative_one():
    text = "In: What action should the robot take to fly to the moon? Out:"
    got = match_suite_ids([text], _PROMPT_TO_SUITE, _SUITE_ORDER)
    assert got == [-1]


def test_suite_missing_from_suite_order_gives_negative_one():
    # "libero_90" is a real suite that is never routed/slotted.
    prompt_to_suite = {"pick up the black bowl": "libero_90"}
    text = "In: ... pick up the black bowl ... Out:"
    got = match_suite_ids([text], prompt_to_suite, _SUITE_ORDER)
    assert got == [-1]


def test_empty_texts_gives_empty_list():
    assert match_suite_ids([], _PROMPT_TO_SUITE, _SUITE_ORDER) == []


def test_empty_prompt_to_suite_gives_all_negative_one():
    texts = [
        "In: ... pick up the black bowl ... Out:",
        "In: ... open the top drawer ... Out:",
    ]
    got = match_suite_ids(texts, {}, _SUITE_ORDER)
    assert got == [-1, -1]


def test_first_match_wins_when_two_instructions_appear_in_one_prompt():
    # Dict insertion order controls which key is checked first: "pick up the
    # black bowl" (-> spatial) is inserted before "put the bowl on the plate"
    # (-> object) in _PROMPT_TO_SUITE, so a prompt containing both substrings
    # must resolve to spatial, not object.
    text = (
        "In: What action should the robot take to pick up the black bowl "
        "and put the bowl on the plate? Out:"
    )
    got = match_suite_ids([text], _PROMPT_TO_SUITE, _SUITE_ORDER)
    assert got == [_SUITE_ORDER.index("libero_spatial")]

    # Reversing the dict's insertion order flips which suite wins, proving the
    # result tracks iteration order rather than some fixed priority.
    reordered = {
        "put the bowl on the plate": "libero_object",
        "pick up the black bowl": "libero_spatial",
    }
    got_reordered = match_suite_ids([text], reordered, _SUITE_ORDER)
    assert got_reordered == [_SUITE_ORDER.index("libero_object")]


def test_every_returned_id_is_a_valid_slot_index_or_negative_one():
    texts = [
        "In: ... pick up the black bowl ... Out:",
        "In: ... nonsense that matches nothing ... Out:",
        "In: ... put the bowl on the plate ... Out:",
        "In: ... open the top drawer ... Out:",
        "In: ... turn on the stove ... Out:",
    ]
    got = match_suite_ids(texts, _PROMPT_TO_SUITE, _SUITE_ORDER)
    for idx in got:
        assert idx == -1 or 0 <= idx < len(_SUITE_ORDER)


def test_result_length_always_equals_input_length():
    texts = [
        "In: ... pick up the black bowl ... Out:",
        "In: ... nonsense ... Out:",
    ] * 5
    got = match_suite_ids(texts, _PROMPT_TO_SUITE, _SUITE_ORDER)
    assert len(got) == len(texts)


def test_returns_a_plain_list_of_ints():
    got = match_suite_ids(
        ["In: ... pick up the black bowl ... Out:"], _PROMPT_TO_SUITE, _SUITE_ORDER
    )
    assert isinstance(got, list)
    assert all(isinstance(x, int) for x in got)


def test_empty_string_text_gives_negative_one():
    got = match_suite_ids([""], _PROMPT_TO_SUITE, _SUITE_ORDER)
    assert got == [-1]


def test_empty_instruction_key_matches_everything_first():
    # Mirrors the teacher router's literal semantics: "" is a substring of
    # every string, so an empty key placed first wins for every sample. This
    # looks like a footgun, but the routing table is built from real LIBERO
    # instructions and never contains an empty key in practice; the test
    # documents the behavior rather than prescribing it.
    prompt_to_suite = {"": "libero_10", "pick up the black bowl": "libero_spatial"}
    got = match_suite_ids(["anything at all"], prompt_to_suite, _SUITE_ORDER)
    assert got == [_SUITE_ORDER.index("libero_10")]


def test_duplicate_suite_order_entries_use_first_occurrence_index():
    # Defensive: if suite_order ever contained a duplicate, list.index() finds
    # the first occurrence -- document that rather than leaving it implicit.
    suite_order = ["libero_spatial", "libero_goal", "libero_spatial"]
    got = match_suite_ids(
        ["In: ... pick up the black bowl ... Out:"], _PROMPT_TO_SUITE, suite_order
    )
    assert got == [0]
