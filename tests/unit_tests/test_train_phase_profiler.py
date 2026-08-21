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

"""Per-phase timing + in-step progress logging for ``EmbodiedFSDPActor.run_training``.

WHY THESE TESTS LOOK LIKE THIS. The thing being measured is a 7B FSDP training step on
two GPUs, which no CPU test can run. So the feature was split into two halves that CAN
be checked here, and one that cannot:

* ``TrainPhaseProfiler`` -- all the arithmetic (accumulation, denominators, projection,
  the interval-to-log-line decision) and all the formatting. Driven with a FAKE device
  clock, so the tests run the shipped class with zero CUDA involvement and can advance
  "device time" deterministically. This is where the bugs would be.
* the WIRING into ``run_training`` -- checked structurally with ``ast`` against the real
  source: that the instrumented method is the one that executes, and that every mark
  sits on the correct side of the call it is supposed to bracket. A mark placed one
  statement too early is invisible at runtime (the numbers still add up to the wall
  clock, they are just attributed to the wrong phase), so a structural test is the only
  thing that catches it.
* NOT covered: that ``torch.cuda.Event.elapsed_time`` attributes async GPU work the way
  the docstring claims. That is a property of CUDA, not of this code, and it needs a
  GPU. The mitigation is that the flush point was deliberately placed immediately after
  the ``loss.detach().item()`` that already drains the stream on every micro-batch, so
  the events being read are always already complete -- which IS checked structurally
  below (``test_micro_done_flushes_after_the_existing_item_sync``).
"""

import ast
import inspect
import textwrap

import pytest
from omegaconf import OmegaConf

from rlinf.workers.actor.fsdp_actor_worker import (
    _PROFILE_PHASES,
    EmbodiedFSDPActor,
    TrainPhaseProfiler,
    _eta_seconds,
    _fmt_dur,
    _phase_breakdown,
    _progress_line,
    _summary_line,
)

# --------------------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------------------


class FakeClock:
    """Stands in for the CUDA-event clock.

    Records how many events were ever created and how many synchronizations were asked
    for, which is how the "flag off => completely inert" test proves its point: an
    inert profiler must not so much as allocate one event.
    """

    def __init__(self):
        self.t_ms = 0.0
        self.created = 0
        self.syncs = 0

    def advance(self, ms):
        self.t_ms += float(ms)

    def event(self):
        self.created += 1
        return [None]

    def record(self, ev):
        ev[0] = self.t_ms

    def sync(self, ev):
        self.syncs += 1

    def elapsed_ms(self, a, b):
        return b[0] - a[0]


class ExplodingClock(FakeClock):
    def sync(self, ev):
        raise RuntimeError("device fell over")


class Wall:
    """Injectable host clock, in seconds."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_profiler(*, enabled=True, log_every=0, clock=None, wall=None):
    clock = clock if clock is not None else FakeClock()
    wall = wall if wall is not None else Wall()
    logs = []
    warns = []
    prof = TrainPhaseProfiler(
        enabled=enabled,
        log_every=log_every,
        log_fn=logs.append,
        warn_fn=warns.append,
        rank=0,
        step=7,
        clock=clock,
        now_fn=wall,
    )
    return prof, clock, wall, logs, warns


# Per-phase costs in milliseconds of "device time", chosen so every phase is distinct and
# their sum is a round number (100 ms per micro-batch).
PHASE_MS = {
    "prep": 1.0,
    "student_fwd": 30.0,
    "teacher_fwd": 20.0,
    "loss": 4.0,
    "backward": 45.0,
}
OPTIM_MS = 8.0
EMPTY_CACHE_MS = 2.0


def drive(prof, clock, wall, *, updates, mbs, wall_s_per_mb=1.0):
    """Replay the exact call sequence ``run_training`` makes, on fake clocks.

    Keeping this in one place means a change to the profiler's protocol breaks every
    test at once rather than silently drifting from what the actor does.
    """
    prof.begin_step(updates)
    for _ in range(updates):
        prof.begin_update(mbs)
        for _ in range(mbs):
            for phase, ms in PHASE_MS.items():
                clock.advance(ms)
                prof.mark(phase)
            wall.t += wall_s_per_mb
            prof.micro_done()
        clock.advance(EMPTY_CACHE_MS)
        prof.mark("empty_cache")
        clock.advance(OPTIM_MS)
        prof.mark("optim")
        prof.update_done()
    return prof.step_done()


# --------------------------------------------------------------------------------------
# the flag defaults to off, and off is completely inert
# --------------------------------------------------------------------------------------


def _bare_actor(algorithm):
    actor = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create({"algorithm": algorithm})
    actor._rank = 0
    actor.version = 3
    actor.log_info = lambda _m: None
    actor.log_warning = lambda _m: None
    return actor


def test_flag_defaults_to_off():
    """An untouched config must produce a disabled profiler.

    This is the guarantee that the production run is unaffected: the config it uses
    names none of these keys.
    """
    prof = _bare_actor({}).make_phase_profiler()
    assert prof.enabled is False


def test_flag_off_when_explicitly_false():
    prof = _bare_actor({"profile_train_phases": False}).make_phase_profiler()
    assert prof.enabled is False


def test_flag_on_reads_the_interval():
    prof = _bare_actor(
        {"profile_train_phases": True, "profile_log_every": 3}
    ).make_phase_profiler()
    assert prof.enabled is True
    assert prof._cfg_log_every == 3


def test_flag_on_defaults_the_interval_to_auto():
    prof = _bare_actor({"profile_train_phases": True}).make_phase_profiler()
    assert prof.enabled is True
    assert prof._cfg_log_every == 0  # 0 == derive it from the real loop bound


def test_disabled_profiler_creates_no_events_no_syncs_and_no_lines():
    """The whole point of the flag: with it off nothing at all happens.

    Not "cheap" -- nothing. No CUDA event is allocated, no synchronization is requested
    (the 5.6x slowdown this repo already measured came from per-micro-batch device
    syncs), and no line is logged.
    """
    prof, clock, wall, logs, warns = make_profiler(enabled=False)
    out = drive(prof, clock, wall, updates=4, mbs=12)
    assert clock.created == 0
    assert clock.syncs == 0
    assert logs == []
    assert warns == []
    assert out == {}
    assert all(v == 0.0 for v in prof.totals.values())


def test_enabled_profiler_does_create_events():
    """Guard against the inertness test passing for the wrong reason."""
    prof, clock, wall, logs, _ = make_profiler(enabled=True)
    drive(prof, clock, wall, updates=1, mbs=2)
    assert clock.created > 0
    assert clock.syncs > 0
    assert logs


# --------------------------------------------------------------------------------------
# accumulation
# --------------------------------------------------------------------------------------


def test_phase_totals_accumulate_per_step():
    prof, clock, wall, _, _ = make_profiler()
    updates, mbs = 3, 4
    totals = drive(prof, clock, wall, updates=updates, mbs=mbs)
    for phase, ms in PHASE_MS.items():
        assert totals[phase] == pytest.approx(ms * mbs * updates / 1000.0)
    assert totals["optim"] == pytest.approx(OPTIM_MS * updates / 1000.0)
    assert totals["empty_cache"] == pytest.approx(EMPTY_CACHE_MS * updates / 1000.0)


def test_update_totals_reset_between_updates():
    """The per-update breakdown must describe THAT update, not the step so far."""
    prof, clock, wall, logs, _ = make_profiler()
    drive(prof, clock, wall, updates=3, mbs=4)
    for phase, ms in PHASE_MS.items():
        assert prof.update_totals[phase] == pytest.approx(ms * 4)


def test_no_device_time_is_lost():
    """Every millisecond between two marks lands in exactly one phase.

    The profiler uses a single moving cursor rather than start/stop pairs precisely so
    that a gap is impossible; if someone reintroduces pairs, un-bracketed work would
    silently vanish from the breakdown and the phases would stop summing to the step.
    """
    prof, clock, wall, _, _ = make_profiler()
    totals = drive(prof, clock, wall, updates=2, mbs=5)
    per_mb = sum(PHASE_MS.values())
    expected = (per_mb * 5 + EMPTY_CACHE_MS + OPTIM_MS) * 2 / 1000.0
    assert sum(totals.values()) == pytest.approx(expected)


def test_a_missing_teacher_forward_folds_into_the_loss_phase():
    """Non-OPD runs never mark teacher_fwd; that time must not disappear."""
    prof, clock, wall, _, _ = make_profiler()
    prof.begin_step(1)
    prof.begin_update(1)
    clock.advance(5.0)
    prof.mark("prep")
    clock.advance(30.0)
    prof.mark("student_fwd")
    clock.advance(11.0)  # no teacher forward on this path
    prof.mark("loss")
    clock.advance(40.0)
    prof.mark("backward")
    prof.micro_done()
    totals = prof.step_done()
    assert totals["teacher_fwd"] == 0.0
    assert totals["loss"] == pytest.approx(0.011)


# --------------------------------------------------------------------------------------
# denominators: they must come from the real loop bounds
# --------------------------------------------------------------------------------------


def test_denominators_are_whatever_the_caller_passed():
    prof, clock, wall, logs, _ = make_profiler(log_every=1)
    drive(prof, clock, wall, updates=4, mbs=5)
    first = logs[0]
    assert "update 1/4" in first
    assert "micro 1/5 (update)" in first
    assert "1/20 (step" in first


def test_step_denominator_is_updates_times_micro_batches():
    prof, clock, wall, logs, _ = make_profiler(log_every=1)
    drive(prof, clock, wall, updates=3, mbs=7)
    last_progress = [ln for ln in logs if "(step," in ln][-1]
    assert "update 3/3" in last_progress
    assert "micro 7/7 (update) 21/21 (step, 100.0%)" in last_progress


def test_percentage_tracks_the_step_not_the_update():
    prof, clock, wall, logs, _ = make_profiler(log_every=1)
    drive(prof, clock, wall, updates=2, mbs=4)
    # 4th micro-batch of 8 == halfway through the STEP, but the end of update 1
    line = [ln for ln in logs if "micro 4/4 (update)" in ln][0]
    assert "4/8 (step, 50.0%)" in line


# --------------------------------------------------------------------------------------
# how often lines come out
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mbs,expected", [(48, 5), (12, 1), (8, 1), (100, 10), (1, 1), (0, 1), (25, 3)]
)
def test_auto_interval_targets_ten_lines_per_update(mbs, expected):
    assert TrainPhaseProfiler.auto_log_every(mbs) == expected


def test_auto_interval_yields_about_ten_lines_per_update():
    prof, clock, wall, logs, _ = make_profiler(log_every=0)
    drive(prof, clock, wall, updates=1, mbs=48)
    progress = [ln for ln in logs if "(step," in ln]
    # 48 // 5 == 9 scheduled lines, + the always-on first one (which is also the
    # liveness signal: it must not take 5 micro-batches to learn the step started)
    assert len(progress) == 10


def test_explicit_interval_wins():
    prof, clock, wall, logs, _ = make_profiler(log_every=4)
    drive(prof, clock, wall, updates=1, mbs=12)
    progress = [ln for ln in logs if "(step," in ln]
    # micro-batches 4, 8 and 12, plus the always-on first one
    assert len(progress) == 4
    assert "micro 1/12" in progress[0]
    assert "micro 4/12" in progress[1]
    assert "micro 12/12" in progress[3]


def test_the_very_first_micro_batch_always_logs():
    """Liveness beats tidiness: on a 3-hour step, waiting for micro-batch 5 to prove
    the step started is exactly the silence this feature exists to remove."""
    prof, clock, wall, logs, _ = make_profiler(log_every=100)
    drive(prof, clock, wall, updates=1, mbs=12)
    progress = [ln for ln in logs if "(step," in ln]
    assert len(progress) == 1
    assert "micro 1/12" in progress[0]


def test_each_update_emits_a_closing_line():
    prof, clock, wall, logs, _ = make_profiler(log_every=100)
    drive(prof, clock, wall, updates=3, mbs=4)
    closes = [ln for ln in logs if "update 3/3 DONE" in ln]
    assert len(closes) == 1
    assert "4 micro-batches" in closes[0]


# --------------------------------------------------------------------------------------
# the projection
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "elapsed,done,total,expected",
    [
        (100.0, 25, 100, 300.0),  # a quarter done -> three quarters left
        (60.0, 30, 60, 60.0),
        (10.0, 1, 2, 10.0),
        (10.0, 2, 2, 0.0),  # finished -> nothing left
        (10.0, 3, 2, 0.0),  # overshoot must not go negative
        (10.0, 0, 2, 0.0),  # nothing observed yet -> no rate to project from
        (10.0, 1, 0, 0.0),
    ],
)
def test_eta_is_linear_in_the_observed_rate(elapsed, done, total, expected):
    assert _eta_seconds(elapsed, done, total) == pytest.approx(expected)


def test_eta_in_the_line_uses_wall_clock_not_device_time():
    """The projection has to come from real elapsed time, because the phases only
    account for GPU work -- host-side stalls are exactly what a 3-hour step is made of
    and they must still show up in the ETA."""
    prof, clock, wall, logs, _ = make_profiler(log_every=1)
    prof.begin_step(2)
    prof.begin_update(4)
    for i in range(3):
        wall.t += 10.0  # 10 wall-seconds per micro-batch, zero device time
        prof.micro_done()
    line = logs[-1]
    # 3 of 8 micro-batches in 30s -> 10s each -> 50s left
    assert "elapsed 30.0s" in line
    assert "eta 50.0s" in line


def test_progress_line_before_any_micro_batch_does_not_divide_by_zero():
    assert "eta 0.0s" in _progress_line(
        rank=0,
        step=1,
        update_index=1,
        updates_per_step=2,
        mb_in_update=0,
        mbs_per_update=4,
        mb_done=0,
        mb_total=8,
        elapsed_s=3.0,
        totals_ms={},
    )


# --------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,text",
    [
        (0.0, "0.0s"),
        (0.25, "0.2s"),
        (59.94, "59.9s"),
        (60.0, "1m00s"),
        (61.5, "1m02s"),
        (3599.0, "59m59s"),
        (3600.0, "1h00m00s"),
        (11045.0, "3h04m05s"),
        (-5.0, "0.0s"),
        (float("nan"), "n/a"),
        (float("inf"), "n/a"),
    ],
)
def test_duration_formatting(seconds, text):
    assert _fmt_dur(seconds) == text


def test_progress_line_carries_everything_the_task_asks_for():
    line = _progress_line(
        rank=1,
        step=9,
        update_index=2,
        updates_per_step=4,
        mb_in_update=12,
        mbs_per_update=48,
        mb_done=60,
        mb_total=192,
        elapsed_s=3720.0,
        totals_ms={"student_fwd": 900_000.0, "backward": 1_500_000.0},
    )
    assert "[prof][r1]" in line
    assert "step 9" in line
    assert "update 2/4" in line  # which update, of how many
    assert "micro 12/48 (update) 60/192 (step, 31.2%)" in line  # count AND percentage
    assert "elapsed 1h02m00s" in line
    assert "eta 2h16m24s" in line  # 3720 * 132/60
    for phase in _PROFILE_PHASES:  # per-phase cumulative, fixed key set
        assert phase in line
    assert "student_fwd 15m00s" in line
    assert "backward 25m00s" in line


def test_breakdown_shows_percentages_when_a_wall_time_is_given():
    text = _phase_breakdown({"backward": 5000.0}, wall_s=10.0)
    assert "backward 5.0s (50.0%)" in text
    assert "student_fwd 0.0s (0.0%)" in text


def test_summary_reports_time_the_phases_do_not_explain():
    """`unaccounted` is the honesty check on the whole breakdown: if it is large, the
    marks are in the wrong places and the numbers should not be believed."""
    line = _summary_line(
        rank=0,
        step=3,
        updates_per_step=4,
        mbs_per_update=48,
        mb_total=192,
        elapsed_s=1000.0,
        totals_ms={"backward": 600_000.0, "student_fwd": 300_000.0},
    )
    assert "step 3 DONE" in line
    assert "4 updates x 48 micro-batches = 192" in line
    assert "wall 16m40s" in line
    assert "unaccounted 1m40s (10.0%)" in line


def test_summary_is_the_last_thing_logged_and_totals_are_returned_in_seconds():
    prof, clock, wall, logs, _ = make_profiler(log_every=100)
    totals = drive(prof, clock, wall, updates=2, mbs=3)
    assert "DONE" in logs[-1] and "updates x" in logs[-1]
    assert set(totals) == set(_PROFILE_PHASES)
    assert totals["backward"] == pytest.approx(PHASE_MS["backward"] * 6 / 1000.0)


# --------------------------------------------------------------------------------------
# a diagnostic must never take the run down with it
# --------------------------------------------------------------------------------------


def test_a_clock_failure_disables_profiling_instead_of_raising():
    prof, clock, wall, logs, warns = make_profiler(clock=ExplodingClock())
    totals = drive(prof, clock, wall, updates=2, mbs=3)  # must not raise
    assert prof.enabled is False
    assert warns and "disabled" in warns[0]
    assert totals == {}


def test_marks_before_begin_step_are_ignored():
    prof, clock, wall, logs, warns = make_profiler()
    prof.mark("backward")
    prof.micro_done()
    prof.update_done()
    assert clock.created == 0
    assert logs == []


# --------------------------------------------------------------------------------------
# structural: the instrumentation is wired into the run_training that actually executes,
# and each mark is on the correct side of the call it brackets
# --------------------------------------------------------------------------------------


def _run_training_ast():
    """``EmbodiedFSDPActor``'s own ``run_training``, as an AST.

    The module defines ``run_training`` twice: once on ``FSDPActor`` (the reasoning /
    LLM path, called with arguments by the reasoning runners) and once here. The
    OpenVLA-OFT embodied run instantiates ``EmbodiedFSDPActor``
    (``examples/embodiment/train_embodied_agent.py`` picks it for every ``loss_type``
    that is not ``embodied_dagger`` / ``embodied_nft``) and ``embodied_runner`` calls
    ``self.actor.run_training()`` with no arguments -- so THIS is the one that runs and
    the one that has to carry the instrumentation. Parsing only this class's source
    makes it impossible to check the other one by accident.
    """
    src = textwrap.dedent(inspect.getsource(EmbodiedFSDPActor))
    cls = ast.parse(src).body[0]
    defs = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run_training"
    ]
    assert defs, "EmbodiedFSDPActor has no run_training"
    return defs[-1]


def _profiler_calls(fn, method):
    return [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == method
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "prof"
    ]


def _marks(fn):
    """phase name -> source line of its ``prof.mark("phase")`` call."""
    out = {}
    for call in _profiler_calls(fn, "mark"):
        assert len(call.args) == 1 and isinstance(call.args[0], ast.Constant), (
            "prof.mark takes one literal phase name"
        )
        out[call.args[0].value] = call.lineno
    return out


def _call_line(fn, predicate):
    hits = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and predicate(n)]
    assert hits, "call not found in run_training"
    return min(n.lineno for n in hits), max(
        getattr(n, "end_lineno", n.lineno) for n in hits
    )


def _attr(name):
    return lambda n: isinstance(n.func, ast.Attribute) and n.func.attr == name


def test_the_executing_run_training_is_the_instrumented_one():
    fn = _run_training_ast()
    assert _profiler_calls(fn, "begin_step"), (
        "EmbodiedFSDPActor.run_training -- the definition the OpenVLA-OFT run actually "
        "executes -- is not instrumented"
    )
    # and the OTHER one is untouched: FSDPActor.run_training must not have grown a
    # profiler by copy-paste, which would time a path this feature was never measured on
    from rlinf.workers.actor.fsdp_actor_worker import FSDPActor

    other = textwrap.dedent(inspect.getsource(FSDPActor.run_training))
    assert "prof." not in other


def test_every_phase_is_marked():
    marks = _marks(_run_training_ast())
    assert set(marks) == set(_PROFILE_PHASES)


def test_marks_bracket_the_calls_they_name():
    fn = _run_training_ast()
    marks = _marks(fn)

    stu_lo, stu_hi = _call_line(
        fn,
        lambda n: isinstance(n.func, ast.Attribute)
        and n.func.attr == "model"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "self"
        and any(kw.arg == "compute_logprobs" for kw in n.keywords),
    )
    tea_lo, tea_hi = _call_line(fn, _attr("_teacher_forward"))
    bwd_lo, bwd_hi = _call_line(fn, _attr("backward"))
    opt_lo, opt_hi = _call_line(fn, _attr("optimizer_step"))

    assert marks["prep"] < stu_lo, "prep must close before the student forward starts"
    assert marks["student_fwd"] > stu_hi, (
        "student_fwd must be marked AFTER the forward call, or the forward lands in "
        "the next phase"
    )
    assert stu_hi < tea_lo, "sanity: the teacher forward follows the student forward"
    assert marks["teacher_fwd"] > tea_hi
    assert marks["loss"] > marks["teacher_fwd"], (
        "the loss phase is defined as everything between the teacher forward and the "
        "backward"
    )
    assert marks["loss"] < bwd_lo, "loss must close before backward() is entered"
    assert marks["backward"] > bwd_hi
    assert marks["empty_cache"] < opt_lo < marks["optim"], (
        "optim must bracket optimizer_step() alone; the empty_cache mark is what keeps "
        "the per-update cache flush out of it"
    )


def test_micro_done_flushes_after_the_existing_item_sync():
    """``prof.micro_done()`` reads the events, which needs them to be complete.

    It is placed after ``metrics_data["actor/total_loss"] = loss.detach().item()``,
    which already drains the stream on EVERY micro-batch whether profiling is on or
    not. That ordering is the whole reason the flush is free; if someone moves the
    flush above it, every micro-batch grows a real device stall and this feature
    becomes the 5.6x slowdown it was written to diagnose.
    """
    fn = _run_training_ast()
    item_lines = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Subscript)
        and isinstance(n.targets[0].slice, ast.Constant)
        and n.targets[0].slice.value == "actor/total_loss"
    ]
    assert item_lines, "the per-micro-batch total_loss .item() is gone"
    done = _profiler_calls(fn, "micro_done")
    assert len(done) == 1
    assert done[0].lineno > max(item_lines)


def test_denominators_are_taken_from_the_loop_bounds_not_from_config():
    """``updates_per_step`` is the variable the slot alternation already uses as the
    loop bound, and the micro-batch count is ``len(train_micro_batch)`` -- the actual
    iterable. Restating ``global_batch // micro_batch // world`` here would be a second
    source of truth that silently disagrees the day one of them changes."""
    fn = _run_training_ast()
    begin_step = _profiler_calls(fn, "begin_step")
    assert len(begin_step) == 1
    assert isinstance(begin_step[0].args[0], ast.Name)
    assert begin_step[0].args[0].id == "updates_per_step"

    begin_update = _profiler_calls(fn, "begin_update")
    assert len(begin_update) == 1
    arg = begin_update[0].args[0]
    assert isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
    assert arg.func.id == "len"
    assert isinstance(arg.args[0], ast.Name) and arg.args[0].id == "train_micro_batch"


def test_step_is_opened_and_closed_exactly_once():
    fn = _run_training_ast()
    for method in ("begin_step", "step_done", "begin_update", "update_done"):
        assert len(_profiler_calls(fn, method)) == 1, f"prof.{method} appears twice"
    begin = _profiler_calls(fn, "begin_step")[0].lineno
    end = _profiler_calls(fn, "step_done")[0].lineno
    assert begin < _profiler_calls(fn, "begin_update")[0].lineno < end
    assert _profiler_calls(fn, "update_done")[0].lineno < end


def test_the_profiler_is_built_from_config_inside_run_training():
    fn = _run_training_ast()
    built = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "make_phase_profiler"
    ]
    assert len(built) == 1
    src = inspect.getsource(EmbodiedFSDPActor.make_phase_profiler)
    assert 'get("profile_train_phases", False)' in src
    assert 'get("profile_log_every", 0)' in src
