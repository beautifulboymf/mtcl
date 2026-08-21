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

"""The actor's per-micro-batch slot routing: ``_route_prepare`` and its gate scope.

WHY THESE TESTS LOOK LIKE THIS. The real actor cannot be instantiated here -- it needs
ray, LIBERO and a 7B checkpoint -- so every test below either

* drives the REAL unbound methods against a hand-built instance
  (``EmbodiedFSDPActor.__new__``) carrying only the attributes those methods read, or
* asserts a STRUCTURAL fact about the real source with ``ast``.

Both kinds run the shipped code, not a copy of it. The one thing neither can check is
that FSDP + gradient checkpointing behave as documented at 7B scale; the structural
tests exist precisely because that is the part no CPU test can reach, and the failure
it protects against (the scope closing before ``backward()``) is silent under a
non-strict gate.
"""

import ast
import inspect
import textwrap

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.slot_lora import SlotGate, match_suite_ids
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
PATHS = {s: f"/ckpt/{s}::/adapters/{s}" for s in SUITES}
# instruction substring -> suite, exactly the shape _load_teacher_model builds
PROMPT_TO_SUITE = {
    "pick up the black bowl": "libero_spatial",
    "pick up the alphabet soup": "libero_object",
    "open the middle drawer": "libero_goal",
    "put both moka pots on the stove": "libero_10",
}


class _FakeTokenizer:
    """Returns canned prompts and counts how many times it was asked."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def batch_decode(self, ids, skip_special_tokens=True):
        self.calls += 1
        assert ids.shape[0] == len(self.texts), "decode saw the wrong micro-batch"
        return list(self.texts)


class _ExplodingTokenizer:
    def batch_decode(self, *a, **kw):  # pragma: no cover - must never be reached
        raise AssertionError("prompts were decoded twice for one micro-batch")


class _FakeTeacher:
    """Minimal stand-in for a frozen expert: returns per-sample logprobs it can be
    identified by, so a scatter that puts a group's rows in the wrong place shows up."""

    def __init__(self, tag):
        self.tag = tag

    def __call__(self, forward_inputs, **kwargs):
        n = forward_inputs["input_ids"].shape[0]
        return {"logprobs": torch.full((n, 3), float(self.tag))}


def _make_actor(
    texts,
    *,
    slot_enabled=True,
    slot_order=tuple(SUITES),
    tol=None,
    n_teachers=4,
):
    """A bare EmbodiedFSDPActor carrying only what the routing methods read."""
    actor = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    algorithm = {"adv_type": "opd"}
    if tol is not None:
        algorithm["slot_route_fallback_tol"] = tol
    actor.cfg = OmegaConf.create({"algorithm": algorithm})
    actor.teacher_prompt_to_suite = dict(PROMPT_TO_SUITE)
    actor.teacher_suite_to_path = {s: PATHS[s] for s in SUITES[:n_teachers]}
    actor.teacher_models = {
        PATHS[s]: _FakeTeacher(i) for i, s in enumerate(SUITES[:n_teachers])
    }
    actor.teacher_model = next(iter(actor.teacher_models.values()))
    actor._route_tokenizer = _FakeTokenizer(texts)
    actor._slot_enabled = slot_enabled
    actor._slot_order = tuple(slot_order) if slot_enabled else ()
    actor._slot_gate = None
    actor._slot_gate_ids = None
    actor._slot_fallback = 0.0
    actor._route_ready = False
    actor.warnings = []
    actor.infos = []
    actor.log_warning = actor.warnings.append
    actor.log_info = actor.infos.append
    return actor


def _inputs(n, device="cpu"):
    return {"input_ids": torch.arange(n * 4, device=device).reshape(n, 4)}


def _legacy_groups(texts, route, suite_to_path, default_path):
    """The grouping loop `_route_prepare` replaced, copied verbatim from git history.

    Kept as the reference for the part of the refactor that is meant to be a pure
    refactor: strip, lower-case, substring containment, and the fallback to the first
    teacher. The teacher side is a baseline other runs are compared against, and a
    silently changed grouping would move it.

    IT IS NO LONGER A REFERENCE FOR TIE-BREAKING, and must not be used as one. This
    loop takes the FIRST key in the routing dict's iteration order, which is how
    libero_10 task 122 ("turn on the stove and put the moka pot on it") ended up routed
    to libero_goal by its prefix "turn on the stove"; `match_suite_ids` now takes the
    LONGEST matching key instead, deliberately. See
    `rlinf/models/slot_lora/routing.py` and
    `tests/unit_tests/test_slot_lora_libero_routing.py`. The prompts below are all
    mutually non-nested, so the two agree on them and the refactor half stays testable.
    """
    groups = {}
    for i, t in enumerate(texts):
        tl = t.strip().lower()
        suite = None
        for k, v in route.items():
            if k in tl:
                suite = v
                break
        path = suite_to_path.get(suite, default_path) if suite else default_path
        groups.setdefault(path, []).append(i)
    return groups


# --------------------------------------------------------------------------------------
# the two routings must agree, sample for sample
# --------------------------------------------------------------------------------------


def test_slot_ids_and_teacher_groups_agree_sample_for_sample():
    texts = [
        "In: What action should the robot take to pick up the black bowl? Out:",
        "In: What action should the robot take to open the middle drawer? Out:",
        "In: What action should the robot take to pick up the alphabet soup? Out:",
        "In: What action should the robot take to put both moka pots on the stove? Out:",
        "In: What action should the robot take to pick up the black bowl? Out:",
    ]
    actor = _make_actor(texts)
    actor._route_prepare(_inputs(len(texts)))

    slot_ids = actor._slot_gate_ids.tolist()
    # invert the teacher grouping: sample index -> suite the expert belongs to
    path_to_suite = {v: k for k, v in actor.teacher_suite_to_path.items()}
    teacher_suite = [None] * len(texts)
    for path, idxs in actor._last_groups.items():
        for i in idxs:
            teacher_suite[i] = path_to_suite[path]

    for i, suite in enumerate(teacher_suite):
        assert slot_ids[i] == SUITES.index(suite), (
            f"sample {i}: scored by {suite}'s expert but routed to slot {slot_ids[i]} "
            f"({SUITES[slot_ids[i]]})"
        )


def test_teacher_grouping_matches_the_loop_it_replaced_on_unnested_prompts():
    # No key in PROMPT_TO_SUITE is a substring of another, so first-key-wins and
    # longest-key-wins cannot differ here and the refactor is comparable to the loop
    # it replaced. Where they DO differ is the point of the change; see _legacy_groups.
    texts = [
        "pick up the black bowl and place it on the plate",
        "open the middle drawer of the cabinet",
        "put both moka pots on the stove",
        "pick up the alphabet soup and put it in the basket",
    ]
    actor = _make_actor(texts, slot_enabled=False)
    actor._route_prepare(_inputs(len(texts)))
    expected = _legacy_groups(
        texts,
        PROMPT_TO_SUITE,
        actor.teacher_suite_to_path,
        next(iter(actor.teacher_models)),
    )
    assert actor._last_groups == expected


def test_matching_is_the_shared_pure_function():
    """The slot ids must be what match_suite_ids says, not a second implementation."""
    texts = [
        "open the middle drawer",
        "pick up the alphabet soup",
        "pick up the black bowl",
    ]
    actor = _make_actor(texts)
    actor._route_prepare(_inputs(len(texts)))
    assert actor._slot_gate_ids.tolist() == match_suite_ids(
        texts, PROMPT_TO_SUITE, list(SUITES)
    )


def test_one_decode_per_micro_batch():
    texts = ["pick up the black bowl", "open the middle drawer"]
    actor = _make_actor(texts)
    actor._route_prepare(_inputs(2))
    assert actor._route_tokenizer.calls == 1
    # the teacher forward must REUSE that, not decode again
    actor._route_tokenizer = _ExplodingTokenizer()
    actor._teacher_forward(_inputs(2), {})


def test_teacher_forward_standalone_still_routes():
    """Called without a prepare (the old standalone path), it derives the routing itself."""
    texts = ["pick up the black bowl", "open the middle drawer"]
    actor = _make_actor(texts)
    assert actor._route_ready is False
    out = actor._teacher_forward(_inputs(2), {})
    assert actor._last_groups is not None and len(actor._last_groups) == 2
    # row 0 came from the spatial expert (tag 0), row 1 from the goal expert (tag 2)
    assert out["logprobs"][0, 0].item() == 0.0
    assert out["logprobs"][1, 0].item() == 2.0


# --------------------------------------------------------------------------------------
# unmatched samples
# --------------------------------------------------------------------------------------


def test_unmatched_sample_gets_no_slot_and_is_counted():
    texts = [
        "pick up the black bowl",
        "assemble the flux capacitor",  # in no routing table
        "open the middle drawer",
        "polish the doorknob",  # in no routing table
    ]
    actor = _make_actor(texts, tol=1.0)
    actor._route_prepare(_inputs(4))
    ids = actor._slot_gate_ids.tolist()
    assert ids[1] == -1 and ids[3] == -1
    assert ids[0] == SUITES.index("libero_spatial")
    assert ids[2] == SUITES.index("libero_goal")
    assert actor._slot_fallback == pytest.approx(0.5)
    # the asymmetry this metric exists to expose: no slot, but a real expert still
    # scored them and folded them into the loss
    default_path = next(iter(actor.teacher_models))
    assert 1 in actor._last_groups[default_path]
    assert 3 in actor._last_groups[default_path]


def test_any_unmatched_sample_raises_by_default():
    texts = ["pick up the black bowl", "assemble the flux capacitor"]
    actor = _make_actor(texts)
    with pytest.raises(RuntimeError, match="route_fallback"):
        actor._route_prepare(_inputs(2))


def test_tolerance_downgrades_the_raise_to_one_warning():
    texts = ["pick up the black bowl", "assemble the flux capacitor"]
    actor = _make_actor(texts, tol=0.6)
    actor._route_prepare(_inputs(2))
    assert len(actor.warnings) == 1
    actor._route_prepare(_inputs(2))
    assert len(actor.warnings) == 1, "the warning must not spam every micro-batch"


def test_fully_routed_batch_is_silent():
    texts = ["pick up the black bowl", "open the middle drawer"]
    actor = _make_actor(texts)
    actor._route_prepare(_inputs(2))
    assert actor._slot_fallback == 0.0
    assert actor.warnings == []


def test_a_suite_with_a_teacher_but_no_slot_still_reaches_its_own_expert():
    """slot_order shorter than the teacher map: the sample gets slot -1 but must NOT be
    handed to whichever expert happens to be first."""
    texts = ["put both moka pots on the stove"]
    actor = _make_actor(texts, slot_order=SUITES[:3], tol=1.0)
    actor._route_prepare(_inputs(1))
    assert actor._slot_gate_ids.tolist() == [-1]
    assert actor._last_groups == {PATHS["libero_10"]: [0]}


# --------------------------------------------------------------------------------------
# the ids the gate is actually handed
# --------------------------------------------------------------------------------------


def test_ids_are_accepted_by_a_real_strict_gate():
    texts = ["pick up the black bowl", "open the middle drawer", "no such task here"]
    actor = _make_actor(texts, tol=1.0)
    actor._route_prepare(_inputs(3))
    gate = SlotGate(len(SUITES), strict=True)
    with gate.scoped(actor._slot_gate_ids):
        assert gate.current().tolist() == actor._slot_gate_ids.tolist()


def test_ids_live_on_the_activations_device():
    texts = ["pick up the black bowl"]
    actor = _make_actor(texts)
    fi = _inputs(1)
    actor._route_prepare(fi)
    assert actor._slot_gate_ids.device == fi["input_ids"].device
    assert actor._slot_gate_ids.dtype is torch.long
    assert actor._slot_gate_ids.ndim == 1


def test_no_gate_ids_when_slot_lora_is_off():
    texts = ["pick up the black bowl", "open the middle drawer"]
    actor = _make_actor(texts, slot_enabled=False)
    actor._route_prepare(_inputs(2))
    assert actor._slot_gate_ids is None
    assert actor._last_groups is not None  # the teacher side is unchanged


def test_single_teacher_prepares_nothing_and_marks_itself_ready():
    texts = ["pick up the black bowl"]
    actor = _make_actor(texts, slot_enabled=False, n_teachers=1)
    actor._route_prepare(_inputs(1))
    assert actor._last_groups is None
    assert actor._slot_gate_ids is None
    assert actor._route_ready is True, (
        "an unprepared-looking flag would make _teacher_forward decode a second time"
    )


def test_slot_scope_is_a_noop_when_slot_lora_is_off():
    actor = _make_actor(["pick up the black bowl"], slot_enabled=False)
    with actor._slot_scope():
        pass


def test_slot_scope_refuses_to_run_unrouted():
    actor = _make_actor(["pick up the black bowl"])
    actor._slot_gate = SlotGate(len(SUITES), strict=True)
    actor._slot_gate_ids = None
    with pytest.raises(RuntimeError, match="no slot routing"):
        actor._slot_scope()


def test_slot_scope_installs_and_restores():
    actor = _make_actor(["pick up the black bowl", "open the middle drawer"])
    gate = SlotGate(len(SUITES), strict=True)
    actor._slot_gate = gate
    actor._route_prepare(_inputs(2))
    with actor._slot_scope():
        assert gate.current().tolist() == [0, 2]
    with pytest.raises(RuntimeError):
        gate.current()


# --------------------------------------------------------------------------------------
# end to end on CPU: the ids this router produces really do isolate the gradients
# --------------------------------------------------------------------------------------


def test_router_ids_isolate_gradients_through_a_recomputed_forward():
    """_route_prepare -> _slot_scope -> checkpointed forward -> backward.

    The one thing this cannot use is the 7B student, so it stands a single real
    SlotLoRALinear in its place and wraps it in the same gradient checkpointing the
    trainer turns on. What it checks is the whole point of the task: the slot each
    sample's PROMPT selected is the only slot its gradient reaches, and that survives
    the forward being re-executed during backward.
    """
    from torch.utils.checkpoint import checkpoint

    from rlinf.models.slot_lora import SlotLoRALinear

    texts = [
        "pick up the black bowl",  # slot 0
        "put both moka pots on the stove",  # slot 3
    ]
    actor = _make_actor(texts)
    gate = SlotGate(len(SUITES), strict=True)
    actor._slot_gate = gate
    layer = SlotLoRALinear(
        torch.nn.Linear(8, 5).double(), (2, 2, 2, 2), scale=1.0, gate=gate
    )
    # B starts at zero, so every slot's gradient would be zero for the wrong reason
    torch.nn.init.normal_(layer.slot_B.weight, std=0.1)

    actor._route_prepare(_inputs(2))
    assert actor._slot_gate_ids.tolist() == [0, 3]

    x = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    with actor._slot_scope():
        out = checkpoint(layer, x, use_reentrant=False)
        # sample 1 only: whatever gradient appears must be slot 3's alone
        out[1].sum().backward()

    grad = layer.slot_B.weight.grad
    offsets, ranks = layer.slot_B.offsets, layer.slot_B.slot_ranks
    for k, (start, rank) in enumerate(zip(offsets, ranks)):
        block = grad[:, start : start + rank]
        if k == 3:
            assert block.abs().sum() > 0, "the owning slot got no gradient"
        else:
            assert block.abs().sum() == 0, (
                f"slot {k} took gradient from a sample routed to slot 3"
            )


def test_closing_the_scope_before_backward_is_loud():
    """The failure the actor's structure exists to prevent, at module scale."""
    from torch.utils.checkpoint import checkpoint

    from rlinf.models.slot_lora import SlotLoRALinear

    actor = _make_actor(["pick up the black bowl"])
    gate = SlotGate(len(SUITES), strict=True)
    actor._slot_gate = gate
    layer = SlotLoRALinear(
        torch.nn.Linear(8, 5).double(), (2, 2, 2, 2), scale=1.0, gate=gate
    )
    actor._route_prepare(_inputs(1))
    x = torch.randn(1, 8, dtype=torch.float64, requires_grad=True)
    with actor._slot_scope():
        out = checkpoint(layer, x, use_reentrant=False)
    with pytest.raises(RuntimeError, match="no slot routing"):
        out.sum().backward()


# --------------------------------------------------------------------------------------
# structural: the scope has to enclose the BACKWARD, and the routing has to precede
# the student forward
# --------------------------------------------------------------------------------------


def _run_training_ast():
    """EmbodiedFSDPActor's own run_training, as an AST.

    The module has two: FSDPActor's (the math/LLM path) and this one, which is what
    the OpenVLA-OFT run executes. Only this class's source is parsed, so the other one
    cannot be picked up by accident; ``[-1]`` is there in case a second definition ever
    shadows this one inside the class, which has happened in this file before.
    """
    src = textwrap.dedent(inspect.getsource(EmbodiedFSDPActor))
    tree = ast.parse(src)
    cls = tree.body[0]
    defs = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_training"
    ]
    assert defs, "run_training not found"
    return defs[-1]


def _slot_scope_with(fn):
    for node in ast.walk(fn):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_slot_scope"
            ):
                return node
    return None


def _body_span(with_node):
    """First and last source line of a ``with`` block's BODY.

    Walking the With node itself is wrong: ``withitem`` carries no lineno, and the
    context expression sits outside the body anyway.
    """
    lo = with_node.body[0].lineno
    hi = max(
        getattr(n, "end_lineno", n.lineno)
        for stmt in with_node.body
        for n in ast.walk(stmt)
        if hasattr(n, "lineno")
    )
    return lo, hi


def _calls(fn, predicate):
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and predicate(n)]


def _is_student_forward(node):
    f = node.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr == "model"
        and isinstance(f.value, ast.Name)
        and f.value.id == "self"
        and any(kw.arg == "compute_logprobs" for kw in node.keywords)
    )


def _is_backward(node):
    return isinstance(node.func, ast.Attribute) and node.func.attr == "backward"


def test_slot_scope_encloses_the_student_forward_and_the_backward():
    fn = _run_training_ast()
    scope = _slot_scope_with(fn)
    assert scope is not None, "run_training does not open a _slot_scope() block"
    lo, hi = _body_span(scope)

    forwards = _calls(fn, _is_student_forward)
    assert forwards, "no self.model(compute_logprobs=...) call found"
    for call in forwards:
        assert lo <= call.lineno <= hi, (
            "the student forward runs outside the slot scope; a strict gate raises "
            "there, a non-strict one trains every slot on every sample"
        )

    backwards = _calls(fn, _is_backward)
    assert backwards, "no .backward() call found"
    for call in backwards:
        assert lo <= call.lineno <= hi, (
            "backward() runs outside the slot scope. Gradient checkpointing re-runs "
            "the wrapped forward during backward, on the autograd engine's worker "
            "thread, and it would find no routing installed."
        )


def test_stale_ids_are_cleared_even_when_prepare_is_skipped():
    """The reset must not hide inside the `if` that calls _route_prepare.

    A micro-batch that skips the prepare (no forward_inputs) would otherwise reach the
    forward carrying the PREVIOUS one's ids: right shape, plausible values, every
    sample routed to the wrong suite's slot.
    """
    fn = _run_training_ast()
    prepares = _calls(
        fn,
        lambda n: isinstance(n.func, ast.Attribute) and n.func.attr == "_route_prepare",
    )
    prep_line = min(p.lineno for p in prepares)
    clears = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Attribute)
        and n.targets[0].attr == "_slot_gate_ids"
        and isinstance(n.value, ast.Constant)
        and n.value.value is None
    ]
    assert clears, "run_training never clears _slot_gate_ids"
    assert any(c.lineno < prep_line for c in clears), (
        "_slot_gate_ids is not cleared before the (conditional) _route_prepare call"
    )
    conditional = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.If):
            for stmt in node.body + node.orelse:
                conditional.update(id(sub) for sub in ast.walk(stmt))
    assert any(id(c) not in conditional for c in clears), (
        "every clear of _slot_gate_ids sits inside an `if`; it has to run for every "
        "micro-batch, including the ones that skip the prepare"
    )


def test_routing_is_prepared_before_the_student_forward():
    fn = _run_training_ast()
    prepares = _calls(
        fn,
        lambda n: isinstance(n.func, ast.Attribute) and n.func.attr == "_route_prepare",
    )
    assert prepares, "run_training never calls _route_prepare"
    forwards = _calls(fn, _is_student_forward)
    assert min(p.lineno for p in prepares) < min(f.lineno for f in forwards), (
        "_route_prepare must run BEFORE the student forward, or the gate carries the "
        "PREVIOUS micro-batch's routing"
    )


def test_teacher_forward_no_longer_decodes_on_its_own_hot_path():
    """The duplicate batch_decode is gone: exactly one decode site, in _route_prepare."""
    src = inspect.getsource(EmbodiedFSDPActor)
    assert src.count("batch_decode(") == 1
    assert "batch_decode(" in inspect.getsource(EmbodiedFSDPActor._route_prepare)


# --------------------------------------------------------------------------------------
# rollout worker: generation runs the merged policy, ungated
# --------------------------------------------------------------------------------------


def test_rollout_ungated_opens_a_window_on_a_strict_gate():
    worker = MultiStepRolloutWorker.__new__(MultiStepRolloutWorker)
    gate = SlotGate(2, strict=True)
    worker._slot_gates = [gate]
    with worker._ungated():
        assert gate.current() is None
    with pytest.raises(RuntimeError):
        gate.current()


def test_rollout_ungated_is_a_noop_without_slots():
    worker = MultiStepRolloutWorker.__new__(MultiStepRolloutWorker)
    worker._slot_gates = []
    with worker._ungated():
        pass


def test_every_rollout_model_call_is_ungated():
    """predict() is the only place the rollout worker runs a model; it must be wrapped."""
    src = inspect.getsource(MultiStepRolloutWorker.predict)
    tree = ast.parse(textwrap.dedent(src))
    fn = tree.body[0]
    ungated = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.With)
        and any(
            isinstance(i.context_expr, ast.Call)
            and isinstance(i.context_expr.func, ast.Attribute)
            and i.context_expr.func.attr == "_ungated"
            for i in n.items
        )
    ]
    assert ungated, "predict() does not open an _ungated() window"
    spans = [_body_span(w) for w in ungated]
    predicts = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "predict_action_batch"
    ]
    assert predicts, "no predict_action_batch call found"
    for call in predicts:
        assert any(lo <= call.lineno <= hi for lo, hi in spans), (
            "a rollout model forward runs outside the ungated window; its strict gate "
            "raises on the first gated linear"
        )
