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

import pytest

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


def test_the_longest_matching_instruction_wins():
    # Both keys are contained in this prompt. "put the bowl on the plate" (25
    # chars, -> object) is longer than "pick up the black bowl" (22, -> spatial),
    # so object wins -- even though spatial is inserted FIRST in _PROMPT_TO_SUITE.
    text = (
        "In: What action should the robot take to pick up the black bowl "
        "and put the bowl on the plate? Out:"
    )
    got = match_suite_ids([text], _PROMPT_TO_SUITE, _SUITE_ORDER)
    assert got == [_SUITE_ORDER.index("libero_object")]


def test_the_table_iteration_order_does_not_change_the_answer():
    # The routing dict's order is an accident of how the actor builds the table
    # (benchmark.libero_suites order); a routing that tracked it sent libero_10
    # task 122 into the goal slot. Reversing the table must change nothing.
    text = (
        "In: What action should the robot take to pick up the black bowl "
        "and put the bowl on the plate? Out:"
    )
    reordered = {
        "put the bowl on the plate": "libero_object",
        "pick up the black bowl": "libero_spatial",
    }
    forward = {
        "pick up the black bowl": "libero_spatial",
        "put the bowl on the plate": "libero_object",
    }
    assert match_suite_ids([text], reordered, _SUITE_ORDER) == match_suite_ids(
        [text], forward, _SUITE_ORDER
    )


def test_a_prefix_key_never_beats_the_longer_key_it_is_a_prefix_of():
    # The real LIBERO collision, in miniature: libero_goal's "turn on the stove"
    # is a proper prefix of libero_10 task 122's instruction, and the goal key
    # comes first in the table the actor builds.
    prompt_to_suite = {
        "turn on the stove": "libero_goal",
        "turn on the stove and put the moka pot on it": "libero_10",
    }
    long_prompt = (
        "In: What action should the robot take to turn on the stove and put "
        "the moka pot on it? Out:"
    )
    short_prompt = "In: What action should the robot take to turn on the stove? Out:"
    assert match_suite_ids([long_prompt], prompt_to_suite, _SUITE_ORDER) == [
        _SUITE_ORDER.index("libero_10")
    ]
    # ...and the shorter instruction still reaches its own suite.
    assert match_suite_ids([short_prompt], prompt_to_suite, _SUITE_ORDER) == [
        _SUITE_ORDER.index("libero_goal")
    ]


def test_an_equal_length_disagreement_raises_instead_of_flipping_a_coin():
    # Longest-first settles a substring CHAIN (the longer key is strictly more
    # specific). It cannot settle a TIE: two keys of the same length mapping to
    # different suites, both present in one prompt. Dict order would decide it,
    # invisibly, so this is refused rather than resolved.
    prompt_to_suite = {
        "open the drawer": "libero_goal",  # 15 characters
        "shut the fridge": "libero_10",  # also 15
    }
    text = "In: ... open the drawer then shut the fridge ... Out:"
    with pytest.raises(ValueError, match="equally specific"):
        match_suite_ids([text], prompt_to_suite, _SUITE_ORDER)


def test_an_equal_length_agreement_does_not_raise():
    # Same length, SAME suite: there is nothing to choose between, so it routes.
    prompt_to_suite = {
        "open the drawer": "libero_goal",
        "shut the fridge": "libero_goal",
    }
    text = "In: ... open the drawer then shut the fridge ... Out:"
    assert match_suite_ids([text], prompt_to_suite, _SUITE_ORDER) == [
        _SUITE_ORDER.index("libero_goal")
    ]


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


def test_an_empty_instruction_key_is_the_last_resort_not_the_first():
    # "" is a substring of every string, so it always matches -- but it is also
    # the SHORTEST possible key, so longest-first reaches it only when nothing
    # else matched. Under the old first-in-the-dict rule an empty key placed
    # first swallowed every sample; now it behaves like a catch-all, which is
    # the only sane reading of a zero-length instruction. The routing table is
    # built from real LIBERO instructions and never contains one in practice;
    # the test documents the behavior rather than prescribing it.
    prompt_to_suite = {"": "libero_10", "pick up the black bowl": "libero_spatial"}
    assert match_suite_ids(["anything at all"], prompt_to_suite, _SUITE_ORDER) == [
        _SUITE_ORDER.index("libero_10")
    ]
    assert match_suite_ids(
        ["In: ... pick up the black bowl ... Out:"], prompt_to_suite, _SUITE_ORDER
    ) == [_SUITE_ORDER.index("libero_spatial")]


def test_duplicate_suite_order_entries_use_first_occurrence_index():
    # Defensive: if suite_order ever contained a duplicate, list.index() finds
    # the first occurrence -- document that rather than leaving it implicit.
    suite_order = ["libero_spatial", "libero_goal", "libero_spatial"]
    got = match_suite_ids(
        ["In: ... pick up the black bowl ... Out:"], _PROMPT_TO_SUITE, suite_order
    )
    assert got == [0]
