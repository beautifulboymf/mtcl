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

"""Drive the real ``wait_for_gpus.sh`` with a fake sampler and assert on its decisions.

``examples/embodiment/incremental_sft/wait_for_gpus.sh`` blocks until enough cards on
this shared box are genuinely idle and then hands them to a target command.  It has
already shipped three bugs, and all three were found by RUNNING it, not by reading it:
the idle verdict was thrown away by ``read < <(probe_gpu)`` (process substitution
reports ``read``'s status, so every card counted as idle -- observed opening a window
on cards at 100% utilization); an empty ``NEED_ALL`` emitted ``GPUS=,1,2``; and
``exec``-ing the target let one load spike in the seconds between the waiter's check
and the target's end a twelve-hour wait.  All three are fixed here, and none of the
fixes had ever met a real window -- every card on the box has been busy, so the only
end-to-end runs used deliberately relaxed thresholds to force a fake window.

These tests close that gap without a GPU.  The script samples the cards through
``$NVIDIA_SMI`` (default ``nvidia-smi``), reads the gated volume from ``$DISK_PATH``
and the cgroup counters from ``$CGROUP_MEM_STAT``, so every input to a decision can be
faked from a temp dir.  ``PATH`` is additionally stapled to a guard that refuses to be
the real ``nvidia-smi``: if the seam ever regresses and the script reaches for the
machine's own cards, the guard fires and ``test_the_real_nvidia_smi_is_never_called``
fails rather than the suite quietly grading itself against whoever is on the box.

Pure CPU: no GPU, no ray, no model, seconds end to end.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
SCRIPT = _REPO / "examples" / "embodiment" / "incremental_sft" / "wait_for_gpus.sh"

# Stand-in for the two queries probe_gpu makes.  Reads a spec file of
# "<gpu> <memory.used> <util,util,...>" lines; consecutive utilization queries for one
# card walk the list and wrap, which is how a multi-sample probe sees a card that is
# busy on only some of its samples.
FAKE_SMI = r"""#!/bin/bash
gpu=""; field=""
while (( $# )); do
  case "$1" in
    -i) gpu="$2"; shift 2 ;;
    --query-gpu=*) field="${1#--query-gpu=}"; shift ;;
    *) shift ;;
  esac
done
line=$(grep -E "^${gpu}[[:space:]]" "$FAKE_SMI_SPEC" 2>/dev/null)
[ -n "$line" ] || { echo "No devices were found" >&2; exit 1; }
read -r _g mem utils <<< "$line"
case "$field" in
  memory.used) printf '%s\n' "$mem" ;;
  utilization.gpu)
    ctr="$FAKE_SMI_STATE/$gpu"
    n=0; [ -f "$ctr" ] && n=$(cat "$ctr")
    IFS=',' read -r -a arr <<< "$utils"
    printf '%s\n' "${arr[$(( n % ${#arr[@]} ))]}"
    printf '%s\n' "$(( n + 1 ))" > "$ctr"
    ;;
  *) exit 1 ;;
esac
"""

# First on PATH for every run, so a broken seam cannot silently fall through to the
# machine's own cards.
GUARD_SMI = r"""#!/bin/bash
echo "real nvidia-smi called: $*" >> "$SMI_GUARD_LOG"
exit 1
"""

# Targets.  The waiter runs them as a child with GPUS= set; they record the fact.
TARGET_OK = r"""#!/bin/bash
printf 'TARGET_RAN GPUS=%s\n' "$GPUS" >> "$SENTINEL"
"""

# Fails instantly the first time (a lost window: load spiked between the waiter's
# check and the target's own preflight), succeeds the second.
TARGET_FLAKY = r"""#!/bin/bash
printf 'TARGET_RAN GPUS=%s\n' "$GPUS" >> "$SENTINEL"
n=$(grep -c . "$SENTINEL")
(( n >= 2 )) || exit 7
"""

# Runs long enough to be past RETRY_IF_FAST_S, then fails for real.
TARGET_SLOW_FAIL = r"""#!/bin/bash
printf 'TARGET_RAN GPUS=%s\n' "$GPUS" >> "$SENTINEL"
sleep 1.3
exit 9
"""


def _write_exe(path: Path, body: str) -> Path:
    """Write ``body`` to ``path`` and make it executable."""
    path.write_text(body)
    path.chmod(0o755)
    return path


class Box:
    """A fake GPU box: card readings, disk, cgroup, log, lock and a target sentinel."""

    def __init__(self, tmp_path: Path):
        """Lay out the fake box under ``tmp_path``."""
        self.root = tmp_path
        self.spec = tmp_path / "gpus.spec"
        self.state = tmp_path / "smi_state"
        self.state.mkdir()
        self.log = tmp_path / "wait.log"
        self.lock = tmp_path / "wait.lock"
        self.sentinel = tmp_path / "sentinel"
        self.guard_log = tmp_path / "guard.log"
        self.cgroup = tmp_path / "memory.stat"
        self.cgroup.write_text("anon 0\nslab 0\nkernel_stack 0\n")
        self.smi = _write_exe(tmp_path / "fake-nvidia-smi", FAKE_SMI)
        guard_bin = tmp_path / "guardbin"
        guard_bin.mkdir()
        _write_exe(guard_bin / "nvidia-smi", GUARD_SMI)
        self.guard_bin = guard_bin
        self.target = _write_exe(tmp_path / "target.sh", TARGET_OK)
        self._procs: list[subprocess.Popen] = []

    def cards(self, spec: dict) -> None:
        """Declare the box: ``{gpu: (memory_used_mib, [util, util, ...])}``."""
        lines = []
        for gpu, (mem, utils) in sorted(spec.items()):
            if isinstance(utils, int):
                utils = [utils]
            lines.append(f"{gpu} {mem} {','.join(str(u) for u in utils)}")
        self.spec.write_text("\n".join(lines) + "\n")

    def env(self, **overrides) -> dict:
        """Base environment: the fake box, fast polling, and every gate open."""
        env = dict(os.environ)
        env.update(
            {
                "PATH": f"{self.guard_bin}:{os.environ['PATH']}",
                "SMI_GUARD_LOG": str(self.guard_log),
                "NVIDIA_SMI": str(self.smi),
                "FAKE_SMI_SPEC": str(self.spec),
                "FAKE_SMI_STATE": str(self.state),
                "WAIT_LOG": str(self.log),
                "WAIT_LOCK": str(self.lock),
                "SENTINEL": str(self.sentinel),
                "DISK_PATH": str(self.root),
                "CGROUP_MEM_STAT": str(self.cgroup),
                # the idle test, at its production defaults
                "NEED_ALL": "",
                "PICK_FROM": "0",
                "PICK_N": "1",
                "MEM_MAX_MIB": "5000",
                "UTIL_MAX_PCT": "20",
                "UTIL_N": "5",
                "UTIL_OK_N": "4",
                "MEM_BUCKET_MIB": "4096",
                # test-speed knobs
                "UTIL_SLEEP_S": "0",
                "POLL_S": "0.05",
                "MAX_WAIT_H": "1",
                # gates open unless a test closes one
                "NEED_GB": "0",
                "ANON_MAX_G": "999999999",
                "RETRY_IF_FAST_S": "180",
            }
        )
        env.update({k: str(v) for k, v in overrides.items()})
        return env

    def _popen(self, target: Path | None = None, **overrides) -> subprocess.Popen:
        cmd = [str(SCRIPT), str(target or self.target)]
        p = subprocess.Popen(
            cmd,
            env=self.env(**overrides),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._procs.append(p)
        return p

    def run(self, target: Path | None = None, timeout: int = 30, **overrides):
        """Run the waiter to completion; return the CompletedProcess."""
        p = self._popen(target, **overrides)
        out, _ = p.communicate(timeout=timeout)
        return subprocess.CompletedProcess(p.args, p.returncode, out, "")

    def run_until_round(self, n: int = 3, timeout: int = 20, **overrides) -> str:
        """Let the waiter poll until round ``n``, then stop it.  Return the log."""
        p = self._popen(**overrides)
        try:
            deadline = time.time() + timeout
            while time.time() < deadline:
                if f"round={n}" in self.log_text():
                    break
                if p.poll() is not None:
                    break
                time.sleep(0.02)
            else:
                pytest.fail(f"waiter never reached round={n}\n{self.log_text()}")
        finally:
            self.stop_all()
        return self.log_text()

    def stop_all(self) -> None:
        """Terminate any waiter still running."""
        for p in self._procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)

    def log_text(self) -> str:
        """The waiter's log so far ("" before the first line)."""
        return self.log.read_text() if self.log.exists() else ""

    def launched(self) -> list[str]:
        """The GPUS= line of every target launch, in order."""
        if not self.sentinel.exists():
            return []
        return [ln for ln in self.sentinel.read_text().splitlines() if ln]

    def gpus(self) -> str:
        """The GPUS value handed to the single expected launch."""
        runs = self.launched()
        assert len(runs) == 1, f"expected one launch, got {runs}"
        return runs[0].split("GPUS=", 1)[1]


@pytest.fixture
def box(tmp_path):
    """A fake box, torn down even if a waiter is still polling."""
    b = Box(tmp_path)
    try:
        yield b
    finally:
        b.stop_all()


# --------------------------------------------------------------------------------
# the idle test: memory AND sustained utilization
# --------------------------------------------------------------------------------


def test_low_memory_but_high_utilization_is_not_idle(box):
    """The exact job this project once co-launched on: <2% memory, 73-97% util.

    Memory alone would call this card free.  It was not: the tenant's job was
    compute-bound and ran that way for a quarter of an hour with our run on top.
    """
    box.cards({0: (1200, [73, 97, 85, 91, 88])})
    log = box.run_until_round(3)
    assert box.launched() == []
    assert "not ready" in log
    assert "0:1200MiB" in log


def test_memory_over_threshold_is_not_idle_even_at_zero_utilization(box):
    """A parked 9 GiB allocation is somebody's model, idle GPU or not."""
    box.cards({0: (9000, [0, 0, 0, 0, 0])})
    log = box.run_until_round(3)
    assert box.launched() == []
    assert "not ready" in log


def test_idle_card_opens_the_window(box):
    """Low memory and quiet on every sample: this is what a real window looks like."""
    box.cards({0: (1200, [0, 3, 1, 0, 2])})
    r = box.run()
    assert r.returncode == 0, r.stdout
    assert box.gpus() == "0"
    assert "WINDOW OPEN" in box.log_text()


@pytest.mark.parametrize(
    ("utils", "idle"),
    [
        ([0, 0, 0, 0, 0], True),  # 5/5 quiet
        ([99, 0, 0, 0, 0], True),  # 4/5 quiet == UTIL_OK_N: one sample between kernels
        ([99, 99, 0, 0, 0], False),  # 3/5 quiet: one short of the rule
        ([99, 99, 99, 0, 0], False),  # a majority busy
        ([99, 99, 99, 99, 99], False),
    ],
)
def test_util_ok_n_of_util_n_rule(box, utils, idle):
    """A minority of busy samples still counts as idle; UTIL_OK_N is the line.

    One instantaneous reading lands in the gap between kernels as often as not, which
    is the whole reason utilization is sampled more than once.
    """
    box.cards({0: (1200, utils)})
    if idle:
        r = box.run()
        assert r.returncode == 0, r.stdout
        assert box.gpus() == "0"
    else:
        box.run_until_round(3)
        assert box.launched() == []


def test_thresholds_are_strict_less_than(box):
    """A card exactly AT either threshold is busy: both tests are ``<``, not ``<=``."""
    box.cards({0: (5000, [0]), 1: (1200, [20])})  # mem == MEM_MAX / util == UTIL_MAX
    box.run_until_round(3, PICK_FROM="0,1")
    assert box.launched() == []


def test_unreadable_card_is_not_idle(box):
    """A card the sampler cannot report on is never assumed free."""
    box.cards({1: (1200, [0])})  # gpu 0 absent -> the sampler exits non-zero
    box.run_until_round(3)
    assert box.launched() == []


def test_non_numeric_reading_is_not_idle(box):
    """``[N/A]`` (MIG, vGPU, a driver hiccup) must read as busy, not as zero."""
    box.cards({0: ("[N/A]", ["[N/A]"])})
    log = box.run_until_round(3)
    assert box.launched() == []
    assert "syntax error" not in log


# --------------------------------------------------------------------------------
# NEED_ALL: the mandatory cards
# --------------------------------------------------------------------------------


def test_need_all_requires_every_listed_card(box):
    """One busy card in NEED_ALL shuts the window, however free the others are."""
    box.cards({0: (1200, [0]), 1: (1200, [95]), 2: (1200, [0])})
    log = box.run_until_round(3, NEED_ALL="0,1", PICK_FROM="2")
    assert box.launched() == []
    assert "not ready" in log


def test_need_all_idle_plus_one_pick_launches(box):
    """All mandatory cards idle plus PICK_N candidates: launch with the union."""
    box.cards({0: (1200, [0]), 1: (1200, [0]), 2: (1200, [0])})
    r = box.run(NEED_ALL="0,1", PICK_FROM="2")
    assert r.returncode == 0, r.stdout
    assert box.gpus() == "0,1,2"


def test_empty_need_all_means_nothing_is_mandatory(box):
    """``NEED_ALL=`` must not make the busy default 5,6,7 mandatory, or block on it."""
    box.cards({0: (1200, [0]), 5: (40000, [99]), 6: (40000, [99]), 7: (40000, [99])})
    r = box.run(NEED_ALL="")
    assert r.returncode == 0, r.stdout
    assert box.gpus() == "0"


def test_need_all_card_is_not_probed_twice_as_a_candidate(box):
    """A card in both lists is taken once, not double-counted toward PICK_N."""
    box.cards({0: (1200, [0]), 1: (1200, [0])})
    r = box.run(NEED_ALL="0", PICK_FROM="0,1")
    assert r.returncode == 0, r.stdout
    assert box.gpus() == "0,1"


# --------------------------------------------------------------------------------
# selection among PICK_FROM
# --------------------------------------------------------------------------------


def test_candidates_are_bucketed_by_memory_then_ranked_by_utilization(box):
    """The case that motivated bucketing: identical tenant footprint, MiB of noise.

    Every candidate carries the same ~1960 MiB, so raw memory order is decided by a few
    MiB of jitter -- and would take the card at 69% over the one at 41%.  Memory picks
    the tier; utilization picks within it, because compute is what we contend for.
    (UTIL_MAX_PCT is lifted here so that BOTH cards pass the idle test: this pins the
    ranking rule, not the idle rule.)
    """
    box.cards({0: (1953, [69]), 1: (1968, [41]), 2: (1961, [88])})
    r = box.run(PICK_FROM="0,1,2", UTIL_MAX_PCT="101")
    assert r.returncode == 0, r.stdout
    assert box.gpus() == "1"  # not 0, which raw memory order would have taken


def test_memory_bucket_outranks_utilization(box):
    """A quieter card a whole bucket higher in memory still loses to the low tier."""
    box.cards({0: (100, [15]), 1: (5000, [0])})
    r = box.run(PICK_FROM="0,1", MEM_MAX_MIB="8000")
    assert r.returncode == 0, r.stdout
    assert box.gpus() == "0"


def test_busy_candidate_is_never_chosen(box):
    """Being least loaded is a tie-break among idle cards, not a licence to take a busy one.

    GPU 0 is the least loaded of the busy cards AND holds the least memory; it still
    must lose to the one card that passes the idle test.
    """
    box.cards({0: (100, [25]), 1: (900, [90]), 2: (4000, [5])})
    r = box.run(PICK_FROM="0,1,2")
    assert r.returncode == 0, r.stdout
    assert box.gpus() == "2"


def test_all_candidates_busy_opens_no_window(box):
    """No idle candidate means no window, not the pick of a bad bunch."""
    box.cards({0: (100, [25]), 1: (100, [60]), 2: (100, [99])})
    box.run_until_round(3, PICK_FROM="0,1,2")
    assert box.launched() == []


def test_pick_n_zero_takes_exactly_the_need_all_cards(box):
    """PICK_N=0 is how you ask for exactly the mandatory cards -- the 2-GPU smoke test.

    Fourth bug: the launch was gated on ``[ -n "$chosen" ]``, and with nothing to pick
    ``chosen`` is empty however idle the cards are, so ``NEED_ALL=0,1 PICK_N=0`` waited
    forever on two cards that were sitting right there.  The window is a verdict about
    the cards, not a side effect of the pick list being non-empty.
    """
    box.cards({0: (1200, [0]), 1: (1200, [0])})
    r = box.run(NEED_ALL="0,1", PICK_FROM="2,3,4", PICK_N="0")
    assert r.returncode == 0, r.stdout
    assert box.gpus() == "0,1"


def test_pick_n_zero_still_requires_the_need_all_cards_to_be_idle(box):
    """Nothing to pick does not mean nothing to check."""
    box.cards({0: (1200, [0]), 1: (1200, [95])})
    box.run_until_round(3, NEED_ALL="0,1", PICK_FROM="", PICK_N="0")
    assert box.launched() == []


def test_nothing_to_select_is_a_usage_error(box):
    """Empty NEED_ALL with PICK_N=0 can never open a window: say so, do not hang."""
    box.cards({0: (1200, [0])})
    r = box.run(NEED_ALL="", PICK_N="0")
    assert r.returncode == 2, r.stdout
    assert "nothing to select" in r.stdout
    assert box.launched() == []


def test_fewer_idle_candidates_than_pick_n_opens_no_window(box):
    """PICK_N=2 with one idle candidate waits; it does not launch short-handed."""
    box.cards({0: (1200, [0]), 1: (1200, [95])})
    box.run_until_round(3, PICK_FROM="0,1", PICK_N="2")
    assert box.launched() == []


def test_pick_n_takes_the_n_least_loaded_idle_cards(box):
    """With PICK_N=2 the two quietest idle candidates go, in sorted order."""
    box.cards({0: (1200, [18]), 1: (1200, [1]), 2: (1200, [9]), 3: (1200, [95])})
    r = box.run(PICK_FROM="0,1,2,3", PICK_N="2")
    assert r.returncode == 0, r.stdout
    assert box.gpus() == "1,2"


# --------------------------------------------------------------------------------
# the emitted GPUS= list
# --------------------------------------------------------------------------------


def test_gpus_list_has_no_empty_field_when_need_all_is_empty(box):
    """Regression: an empty NEED_ALL used to join into ``GPUS=,1,2``."""
    box.cards({1: (1200, [0]), 2: (1200, [0])})
    r = box.run(NEED_ALL="", PICK_FROM="1,2", PICK_N="2")
    assert r.returncode == 0, r.stdout
    gpus = box.gpus()
    assert gpus == "1,2"
    assert not gpus.startswith(",") and ",," not in gpus and not gpus.endswith(",")


def test_gpus_list_is_sorted_and_deduplicated(box):
    """Unsorted, overlapping inputs still come out as a clean ascending CSV."""
    box.cards({0: (1200, [0]), 1: (1200, [0]), 2: (1200, [0])})
    r = box.run(NEED_ALL="2,0,2", PICK_FROM="1,0", PICK_N="1")
    assert r.returncode == 0, r.stdout
    gpus = box.gpus()
    assert gpus == "0,1,2"
    fields = gpus.split(",")
    assert all(fields), gpus
    assert len(fields) == len(set(fields)), gpus
    assert fields == sorted(fields, key=int), gpus


# --------------------------------------------------------------------------------
# the gates re-checked at the moment of launch
# --------------------------------------------------------------------------------


def test_holds_when_the_disk_is_below_need_gb(box):
    """The volume is shared and moves underneath a multi-hour wait: re-check at launch."""
    box.cards({0: (1200, [0])})
    log = box.run_until_round(2, NEED_GB="99999999")
    assert box.launched() == []
    assert "WINDOW OPEN" in log
    assert "HOLDING" in log and "free on" in log


def test_holds_when_cgroup_anon_is_above_anon_max_g(box):
    """Launching into an already-full cgroup is how the actor gets OOM-killed."""
    box.cards({0: (1200, [0])})
    box.cgroup.write_text("anon 322122547200\nslab 0\nkernel_stack 0\n")  # 300G
    log = box.run_until_round(2, ANON_MAX_G="200")
    assert box.launched() == []
    assert "HOLDING" in log
    assert "cgroup anon already 300G" in log


def test_launch_line_reports_the_gate_readings(box):
    """The launch line has to say what it checked, or a bad launch is unexplainable."""
    box.cards({0: (1200, [0])})
    r = box.run()
    assert r.returncode == 0, r.stdout
    log = box.log_text()
    assert "LAUNCH" in log and "disk=" in log and "anon=" in log


# --------------------------------------------------------------------------------
# the target: lost windows, and real failures
# --------------------------------------------------------------------------------


def test_target_that_fails_fast_resumes_the_wait(box):
    """A target that refuses its own preflight is a lost window, not the end of the wait.

    Load can spike in the seconds between the waiter's check and the target's.  With
    ``exec`` one such spike ended a twelve-hour wait; as a child, the waiter goes back
    to polling and catches the window next round.
    """
    box.cards({0: (1200, [0])})
    flaky = _write_exe(box.root / "flaky.sh", TARGET_FLAKY)
    r = box.run(flaky, timeout=30)
    assert r.returncode == 0, r.stdout
    assert len(box.launched()) == 2
    log = box.log_text()
    assert "lost window, resuming the wait" in log
    assert "target finished rc=0" in log


def test_target_that_fails_slowly_returns_its_exit_code(box):
    """Past RETRY_IF_FAST_S the target did real work: its exit code is the run's."""
    box.cards({0: (1200, [0])})
    slow = _write_exe(box.root / "slowfail.sh", TARGET_SLOW_FAIL)
    r = box.run(slow, timeout=60, RETRY_IF_FAST_S="1")
    assert r.returncode == 9, r.stdout
    assert len(box.launched()) == 1
    assert "target finished rc=9" in box.log_text()


def test_target_receives_gpus_in_its_environment(box):
    """GPUS= is the entire contract between the waiter and the target."""
    box.cards({0: (1200, [0]), 1: (1200, [0])})
    r = box.run(NEED_ALL="0", PICK_FROM="1")
    assert r.returncode == 0, r.stdout
    assert box.launched() == ["TARGET_RAN GPUS=0,1"]


# --------------------------------------------------------------------------------
# giving up, and refusing to double-book the window
# --------------------------------------------------------------------------------


def test_max_wait_expiry_exits_non_zero_without_launching(box):
    """An expired wait must fail loudly, never launch late onto cards it never checked."""
    box.cards({0: (1200, [0])})
    r = box.run(MAX_WAIT_H="0")
    assert r.returncode == 3, r.stdout
    assert box.launched() == []
    assert "GIVING UP" in box.log_text()


def test_second_waiter_refuses_to_start(box):
    """Two waiters would both fire into the same window and fight over the same cards."""
    box.cards({0: (1200, [95])})  # busy: the first waiter stays in its poll loop
    box._popen()
    deadline = time.time() + 15
    while "waiting:" not in box.log_text() and time.time() < deadline:
        time.sleep(0.02)
    assert "waiting:" in box.log_text(), "first waiter never started"

    second = subprocess.run(
        [str(SCRIPT), str(box.target)],
        env=box.env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert second.returncode == 1, second.stdout + second.stderr
    assert "refusing to start a second" in second.stdout + second.stderr
    assert box.launched() == []


def test_no_arguments_is_a_usage_error(box):
    """No target is a caller bug; do not sit on the lock for twelve hours over it."""
    r = subprocess.run(
        [str(SCRIPT)], env=box.env(), capture_output=True, text=True, timeout=30
    )
    assert r.returncode == 2
    assert "usage:" in r.stdout


# --------------------------------------------------------------------------------
# the seam itself
# --------------------------------------------------------------------------------


def test_the_real_nvidia_smi_is_never_called(box):
    """Every decision above came from the fake sampler, not from the machine's cards."""
    box.cards({0: (1200, [0])})
    r = box.run()
    assert r.returncode == 0, r.stdout
    guard = box.guard_log.read_text() if box.guard_log.exists() else ""
    assert guard == "", f"the script reached for the real nvidia-smi:\n{guard}"


def test_production_defaults_are_unchanged():
    """The seam must not move production: defaults are today's values.

    ``NVIDIA_SMI`` falls back to plain ``nvidia-smi``, the sample gap stays 1s, and the
    gated paths stay the real volume and the real cgroup file.
    """
    src = SCRIPT.read_text()
    for default in (
        'NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"',
        'UTIL_SLEEP_S="${UTIL_SLEEP_S:-1}"',
        'DISK_PATH="${DISK_PATH:-/share/fanruochen-local}"',
        'CGROUP_MEM_STAT="${CGROUP_MEM_STAT:-/sys/fs/cgroup/memory.stat}"',
    ):
        assert default in src, f"missing production default: {default}"
    assert 'NEED_ALL="${NEED_ALL-5,6,7}"' in src  # no colon: empty stays empty


def test_script_is_runnable_with_no_arguments_by_a_human():
    """It stays a plain executable bash script; the seam adds no required argument."""
    assert os.access(SCRIPT, os.X_OK)
    assert SCRIPT.read_text().startswith("#!/bin/bash")
    bash = shutil.which("bash")
    r = subprocess.run([bash, "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
