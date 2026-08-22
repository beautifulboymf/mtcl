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

"""Drive the real ``opd_mt4slot_upgrade.sh`` against a fake box and fake processes.

The supervisor's job is to move a live 2-GPU R1 run onto 4 cards when a window
opens, without losing the steps already done.  Everything it can get wrong is
expensive and most of it is silent:

* Killing the run **while a checkpoint is being written** leaves a DCP directory
  with data blocks and no ``.metadata`` -- unloadable, unconvertible, and this
  volume lost days to exactly that failure once.
* Killing a run that has produced **nothing** loses every step and gains nothing.
* Picking ``global_step_9`` over ``global_step_10`` because a lexical sort says
  ``"9" > "10"`` silently throws away a step.
* Reading the cards **once** lands between kernels as often as not: this project
  co-launched onto a tenant's compute-bound job showing <2% memory and 73-97%
  utilization and ran that way for a quarter of an hour.
* Relaunching without the resume actually taking effect looks perfectly healthy
  in the log while restarting from zero.

None of that needs a GPU to test.  The script reaches the cards only through
``$NVIDIA_SMI``, finds the run only through ``$AUTO_PAT``/``$JOB_PAT``, reads the
checkpoints under ``$CKPT_DIR`` and relaunches through ``$LAUNCHER`` -- so a temp
dir plus a couple of ``sleep``\\ s stands in for the whole box.

SAFETY.  A real 2-GPU run is usually live on this machine while these tests run.
Every test therefore passes process patterns anchored to its own ``tmp_path``,
which cannot match the real job's ``/home/.../examples/...`` command line, and
``_FakeRun`` asserts that the pattern matches exactly the pids it spawned before
the supervisor is allowed anywhere near a kill.  ``PATH`` is also stapled to a
guard that refuses to be the real ``nvidia-smi``.
"""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    _REPO / "examples" / "embodiment" / "incremental_sft" / "opd_mt4slot_upgrade.sh"
)

# Stand-in for the two queries the idle test makes, with one addition over the
# waiter's fake: cards listed in FAKE_SMI_HELD read FAKE_SMI_HELD_MIB for as long
# as FAKE_SMI_ALIVE_PID is alive.  That is what our own two cards do -- they hold
# the run's memory until the run dies and only then come free -- and it lets one
# spec file drive both the window probe (before the kill) and the release check
# (after it) without the test having to sequence anything by hand.
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
held=0
if [ -n "${FAKE_SMI_ALIVE_PID:-}" ] && kill -0 "$FAKE_SMI_ALIVE_PID" 2>/dev/null; then
  # A zombie is a corpse whose parent has not reaped it yet: `kill -0` says yes, but its
  # GPU memory is long gone.  The pinned process here is a pytest child, so it sits in Z
  # until the test calls poll() -- treat that as released, the way a real card does.
  st=$(sed -n 's/^State:[[:space:]]*\([A-Z]\).*/\1/p' "/proc/$FAKE_SMI_ALIVE_PID/status" 2>/dev/null)
  [ "$st" = "Z" ] || held=1
fi
if (( held )); then
  case ",${FAKE_SMI_HELD:-}," in
    *",$gpu,"*) mem="${FAKE_SMI_HELD_MIB:-70000}"; utils="${FAKE_SMI_HELD_UTIL:-99}" ;;
  esac
fi
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

# First on PATH for every run, so a broken seam cannot silently fall through to
# the machine's own cards.
GUARD_SMI = r"""#!/bin/bash
echo "real nvidia-smi called: $*" >> "$SMI_GUARD_LOG"
exit 1
"""

# Stands in for opd_mt4slot_auto.sh: the OOM-fallback loop.  The real one relaunches
# a fresh 2-GPU run at the next micro the moment the inner job exits non-zero, so
# the supervisor has to kill THIS before the job under it.  The fake does the same,
# and records it, so a supervisor that gets the order wrong fails a test instead of
# quietly starting a second job on the same cards.
FAKE_AUTO = r"""#!/bin/bash
trap 'echo "auto TERMINATED" >> "$FAKE_EVENTS"; exit 143' TERM
echo "auto UP $$" >> "$FAKE_EVENTS"
"$FAKE_SAFE" "$FAKE_EVENTS" bash "$FAKE_JOB" &
JOB=$!
wait "$JOB"
echo "auto INNER-EXIT rc=$?" >> "$FAKE_EVENTS"
# the real loop's dangerous move: a fresh 2-GPU run at the next micro size
echo "auto RELAUNCHED-AFTER-INNER-DEATH" >> "$FAKE_EVENTS"
while :; do sleep 5; done
"""

# Stands in for safe_run.sh.  The point of having it here at all: it carries the JOB's
# script path as an ARGUMENT on its own command line, so a plain `pgrep -f <job pattern>`
# matches it too.  Live on this box that made the supervisor report safe_run's pid as
# "the job" -- it then read the wrong process's environment and would have reported the
# wrong pid in its log.  The file is named so that the supervisor's "is my parent
# safe_run?" check recognises it.
FAKE_SAFE_RUN = r"""#!/bin/bash
shift                       # the logfile argument, as safe_run.sh takes
echo "safe_run UP $$" >> "$FAKE_EVENTS"
setsid "$@" &               # safe_run.sh setsids the job; the job is a session leader
JOB=$!
wait "$JOB"
while :; do sleep 5; done   # safe_run's watchdog outlives one inner exit
"""

# Stands in for opd_mt4slot.sh under safe_run.sh.  Two things about its shape matter:
# it forks a checkpoint reaper that shares its command line (so a bare pattern match
# returns two pids and picking either at random is a coin flip), and its ray session
# is reachable only through RAY_TMPDIR, which run_iso.sh -- a CHILD of the job, not the
# job itself -- is what exports.
FAKE_JOB = r"""#!/bin/bash
echo "job UP $$" >> "$FAKE_EVENTS"
( while :; do sleep 5; done ) &                       # the reaper twin: identical argv
RAY_TMPDIR="$FAKE_RAY_TMPDIR" bash "$FAKE_RAY" "$FAKE_RAY_TMPDIR" &
while :; do sleep 5; done
"""

# Stands in for the ray session: a process whose command line carries RAY_TMPDIR, which
# is the only thing the supervisor is allowed to match ray on.
FAKE_RAY = r"""#!/bin/bash
echo "ray UP $$ $1" >> "$FAKE_EVENTS"
while :; do sleep 5; done
"""

# Stands in for opd_mt4slot_auto.sh on the way back up.  Records the whole
# environment the supervisor handed it, and writes the runner's resume line into
# the driver log the way a real resumed run would.
FAKE_LAUNCHER_OK = r"""#!/bin/bash
{
  echo "LAUNCHED"
  for v in GPUS RESUME_DIR MICROS GRAD_CKPT TAG STEPS SAVE_INTERVAL KEEP_CKPTS PORT PROFILE PROFILE_EVERY; do
    echo "$v=${!v-<unset>}"
  done
} >> "$RELAUNCH_SENTINEL"
{
  echo '  "resume_dir": "'"$RESUME_DIR"'",'
  echo "[INFO 09:00:00 RLinf] Resuming training from checkpoint directory $RESUME_DIR."
} >> "$FAKE_DRIVER_LOG"
sleep 600
"""

# Launches, but never resumes -- the silent failure the verification exists for.
FAKE_LAUNCHER_NO_RESUME = r"""#!/bin/bash
echo "LAUNCHED" >> "$RELAUNCH_SENTINEL"
echo "[INFO 09:00:00 RLinf] Saving checkpoint at step 1." >> "$FAKE_DRIVER_LOG"
sleep 600
"""


def _write_exe(path: Path, body: str) -> Path:
    """Write ``body`` to ``path`` and make it executable."""
    path.write_text(body)
    path.chmod(0o755)
    return path


def _killpg(proc) -> None:
    """SIGKILL a child's whole process group -- but never our own.

    ``os.killpg(os.getpgid(pid), ...)`` on a child that was NOT started with
    ``start_new_session=True`` resolves to pytest's own group and kills the test
    runner.  Every child here is session-led, so a group that matches ours means
    something went wrong and falling back to a plain kill is the safe answer.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    if pgid == os.getpgid(0):
        proc.kill()
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


class _FakeRun:
    """The auto loop, the job, its reaper twin and a ray session -- as real processes."""

    def __init__(self, box, gpus="1,3", **env_extra):
        self.box = box
        self.ray_tmpdir = f"/tmp/rayiso_FAKE_{os.getpid()}_{abs(id(self))}"
        env = dict(os.environ)
        env.update(
            {
                "FAKE_EVENTS": str(box.events),
                "FAKE_JOB": str(box.job_script),
                "FAKE_SAFE": str(box.safe_script),
                "FAKE_RAY": str(box.ray_script),
                "FAKE_RAY_TMPDIR": self.ray_tmpdir,
                # the launcher environment the supervisor must carry across untouched
                "GPUS": gpus,
                "TAG": "faketag",
                "STEPS": "15",
                "SAVE_INTERVAL": "1",
                "KEEP_CKPTS": "2",
                "GRAD_CKPT": "True",
                "PROFILE": "1",
                "PROFILE_EVERY": "50",
            }
        )
        env.update({k: str(v) for k, v in env_extra.items()})
        self.auto = subprocess.Popen(
            ["bash", str(box.auto_script)], env=env, start_new_session=True
        )
        self._wait_for_events(("auto UP", "safe_run UP", "job UP", "ray UP"))
        # SAFETY: prove the patterns this test hands the supervisor match only processes
        # this test started, before the supervisor is allowed anywhere near a kill.
        job_pids = self.pgrep(box.job_pat)
        assert job_pids, "the job pattern matched nothing"
        for pid in job_pids + self.pgrep(box.auto_pat) + self.pgrep(self.ray_tmpdir):
            args = subprocess.run(
                ["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True
            ).stdout
            assert str(box.root) in args or self.ray_tmpdir in args, (
                f"pattern reached a process outside the fake box: {args!r}"
            )

    def _wait_for_events(self, needles, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            text = self.box.events_text()
            if all(n in text for n in needles):
                return
            time.sleep(0.02)
        pytest.fail(f"fake run never came up: {self.box.events_text()!r}")

    @staticmethod
    def pgrep(pat):
        r = subprocess.run(
            ["pgrep", "-u", str(os.getuid()), "-f", pat], capture_output=True, text=True
        )
        return [int(x) for x in r.stdout.split()]

    def alive(self):
        """Which parts of the fake run are still running."""
        out = []
        if self.auto.poll() is None:
            out.append("auto")
        # the job pattern also matches the safe_run wrapper, which is the point
        if self.pgrep(self.box.job_pat):
            out.append("job")
        if self.pgrep(self.ray_tmpdir):
            out.append("ray")
        return out

    def stop(self):
        if self.auto.poll() is None:
            _killpg(self.auto)
        try:
            self.auto.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        for pid in self.pgrep(self.box.job_pat) + self.pgrep(self.ray_tmpdir):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


class Box:
    """A fake box: cards, checkpoints, driver log, processes, lock and sentinels."""

    def __init__(self, tmp_path: Path):
        """Lay out the fake box under ``tmp_path``."""
        self.root = tmp_path
        self.spec = tmp_path / "gpus.spec"
        self.state = tmp_path / "smi_state"
        self.state.mkdir()
        self.log = tmp_path / "upgrade.log"
        self.lock = tmp_path / "upgrade.lock"
        self.events = tmp_path / "events"
        self.guard_log = tmp_path / "guard.log"
        self.ckpts = tmp_path / "checkpoints"
        self.ckpts.mkdir()
        self.driver_log = tmp_path / "driver.log"
        self.driver_log.write_text("[old run] some earlier line\n")
        self.relaunch_sentinel = tmp_path / "relaunch"
        self.smi = _write_exe(tmp_path / "fake-nvidia-smi", FAKE_SMI)
        guard_bin = tmp_path / "guardbin"
        guard_bin.mkdir()
        _write_exe(guard_bin / "nvidia-smi", GUARD_SMI)
        self.guard_bin = guard_bin
        # Anchored to tmp_path on purpose: these patterns CANNOT match the real
        # 2-GPU run that is usually live on this box.
        self.auto_script = _write_exe(tmp_path / "fake_auto.sh", FAKE_AUTO)
        self.job_script = _write_exe(tmp_path / "fake_job.sh", FAKE_JOB)
        # named for the supervisor's "is my parent safe_run?" check
        self.safe_script = _write_exe(tmp_path / "fake_safe_run.sh", FAKE_SAFE_RUN)
        self.ray_script = _write_exe(tmp_path / "fake_ray.sh", FAKE_RAY)
        self.launcher = _write_exe(tmp_path / "fake_launcher.sh", FAKE_LAUNCHER_OK)
        self.auto_pat = str(self.auto_script).replace(".", r"\.")
        self.job_pat = str(self.job_script).replace(".", r"\.")
        self._procs: list[subprocess.Popen] = []
        self._runs: list[_FakeRun] = []

    # ---- fake box state -------------------------------------------------------

    def cards(self, spec: dict) -> None:
        """Declare the box: ``{gpu: (memory_used_mib, [util, util, ...])}``."""
        lines = []
        for gpu, (mem, utils) in sorted(spec.items()):
            if isinstance(utils, int):
                utils = [utils]
            lines.append(f"{gpu} {mem} {','.join(str(u) for u in utils)}")
        self.spec.write_text("\n".join(lines) + "\n")

    def ckpt(self, step: int, *, metadata=True, distcp=True, weights="static"):
        """Build one ``global_step_<step>`` directory in a chosen state.

        ``weights``: ``"static"`` (finished), ``"growing"`` (rank 0 is still inside
        ``torch.save``), ``"missing"`` (the DCP part is done, the full weights are
        not started), ``"empty"`` (zero bytes on disk).
        """
        d = self.ckpts / f"global_step_{step}"
        dcp = d / "actor" / "dcp_checkpoint"
        dcp.mkdir(parents=True, exist_ok=True)
        if distcp:
            (dcp / "__0_0.distcp").write_bytes(b"x" * 64)
            (dcp / "__1_0.distcp").write_bytes(b"x" * 64)
        if metadata:
            (dcp / ".metadata").write_bytes(b"pickle")
        msd = d / "actor" / "model_state_dict"
        msd.mkdir(parents=True, exist_ok=True)
        fw = msd / "full_weights.pt"
        if weights == "static":
            fw.write_bytes(b"w" * 4096)
        elif weights == "empty":
            fw.write_bytes(b"")
        elif weights == "growing":
            fw.write_bytes(b"w" * 16)
            self._grow(fw)
        elif weights == "missing":
            pass
        else:
            raise ValueError(weights)
        return d

    def _grow(self, path: Path):
        """Append to ``path`` forever, i.e. a ``torch.save`` still in flight."""
        p = subprocess.Popen(
            [
                "bash",
                "-c",
                'while :; do printf "wwwwwwww" >> "$1"; sleep 0.05; done',
                "_",
                str(path),
            ],
            start_new_session=True,
        )
        self._procs.append(p)

    def run(self, gpus="1,3", **env_extra) -> _FakeRun:
        """Start the fake auto loop + job + ray session."""
        r = _FakeRun(self, gpus=gpus, **env_extra)
        self._runs.append(r)
        return r

    # ---- driving the supervisor ----------------------------------------------

    def env(self, **overrides) -> dict:
        """Base environment: the fake box, fast polling, every gate open."""
        env = dict(os.environ)
        env.update(
            {
                "PATH": f"{self.guard_bin}:{os.environ['PATH']}",
                "SMI_GUARD_LOG": str(self.guard_log),
                "NVIDIA_SMI": str(self.smi),
                "FAKE_SMI_SPEC": str(self.spec),
                "FAKE_SMI_STATE": str(self.state),
                "FAKE_EVENTS": str(self.events),
                "FAKE_DRIVER_LOG": str(self.driver_log),
                "RELAUNCH_SENTINEL": str(self.relaunch_sentinel),
                "UPGRADE_LOG": str(self.log),
                "UPGRADE_LOCK": str(self.lock),
                "CKPT_DIR": str(self.ckpts),
                "DRIVER_LOG": str(self.driver_log),
                "LAUNCHER": str(self.launcher),
                "AUTO_PAT": self.auto_pat,
                "JOB_PAT": self.job_pat,
                "DISK_PATH": str(self.root),
                # the idle test, at its production defaults
                "NEED_GPUS": "4",
                "PICK_FROM": "0,1,2,3,4,5,6,7",
                "MEM_MAX_MIB": "5000",
                "UTIL_MAX_PCT": "20",
                "UTIL_N": "5",
                "UTIL_OK_N": "4",
                "MEM_BUCKET_MIB": "4096",
                # test-speed knobs
                "UTIL_SLEEP_S": "0",
                "POLL_S": "0.05",
                "MAX_WAIT_H": "1",
                "STABLE_S": "0.4",
                "KILL_WAIT_S": "5",
                "REL_WAIT_S": "5",
                "VERIFY_S": "10",
                # gates open unless a test closes one
                "NEED_GB": "0",
            }
        )
        env.update({k: str(v) for k, v in overrides.items()})
        return env

    def sh(self, *args, timeout=60, **overrides):
        """Run the supervisor to completion; return the CompletedProcess."""
        return subprocess.run(
            [str(SCRIPT), *args],
            env=self.env(**overrides),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def popen(self, *args, **overrides) -> subprocess.Popen:
        """Run the supervisor in the background (for the lock and polling tests)."""
        p = subprocess.Popen(
            [str(SCRIPT), *args],
            env=self.env(**overrides),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # Its OWN session, always.  Without this the child shares pytest's process
            # group, and teardown's killpg takes pytest down with it -- which is exactly
            # what happened the first time this suite was run.
            start_new_session=True,
        )
        self._procs.append(p)
        return p

    def wait_for_log(self, needle, timeout=20, proc=None):
        """Block until ``needle`` shows up in the supervisor's log."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if needle in self.log_text():
                return self.log_text()
            if proc is not None and proc.poll() is not None:
                break
            time.sleep(0.02)
        pytest.fail(f"never saw {needle!r} in:\n{self.log_text()}")

    # ---- readouts -------------------------------------------------------------

    def log_text(self) -> str:
        """The supervisor's log so far."""
        return self.log.read_text() if self.log.exists() else ""

    def events_text(self) -> str:
        """What the fake processes recorded."""
        return self.events.read_text() if self.events.exists() else ""

    def relaunch_text(self) -> str:
        """The environment the fake launcher was handed."""
        return (
            self.relaunch_sentinel.read_text()
            if self.relaunch_sentinel.exists()
            else ""
        )

    def relaunch_env(self) -> dict:
        """The relaunch environment as a dict."""
        out = {}
        for ln in self.relaunch_text().splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                out[k] = v
        return out

    def stop_all(self) -> None:
        for r in self._runs:
            r.stop()
        for p in self._procs:
            if p.poll() is None:
                _killpg(p)
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass


@pytest.fixture
def box(tmp_path):
    """A fake box, torn down even if a supervisor or a fake run is still alive."""
    b = Box(tmp_path)
    # Six cards busy, two free: the shape of the real box while we wait.
    b.cards(
        {
            0: (70000, [99]),
            1: (1000, [0]),
            2: (70000, [99]),
            3: (1000, [0]),
            4: (70000, [99]),
            5: (70000, [99]),
            6: (70000, [99]),
            7: (70000, [99]),
        }
    )
    try:
        yield b
    finally:
        b.stop_all()


def _free_window(box):
    """Cards 1 and 3 are ours; 6 and 7 have just come free, 6 the quieter."""
    box.cards(
        {
            0: (70000, [99]),
            1: (1000, [0]),
            2: (70000, [99]),
            3: (1000, [0]),
            4: (70000, [99]),
            5: (70000, [99]),
            6: (1200, [0, 1, 0, 2, 0]),
            7: (1500, [5, 4, 6, 3, 5]),
        }
    )


# --------------------------------------------------------------------------------
# 1. what counts as a COMPLETE checkpoint
# --------------------------------------------------------------------------------


def test_complete_checkpoint_is_accepted(box):
    """Shards, ``.metadata`` and a settled ``full_weights.pt`` -- the finished shape."""
    d = box.ckpt(3)
    r = box.sh("newest")
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == str(d)


def test_missing_metadata_is_a_half_written_checkpoint(box):
    """Data blocks on disk, ``.metadata`` absent: the unloadable failure itself.

    ``dcp.save`` writes every ``.distcp`` shard first and only then renames
    ``.metadata.tmp`` into place, so this is precisely what a run killed mid-save
    leaves behind, and ``load_checkpoint`` cannot read it.
    """
    box.ckpt(3, metadata=False)
    r = box.sh("newest")
    assert r.returncode == 1
    assert r.stdout.strip() == ""


def test_metadata_without_shards_is_rejected(box):
    """``load_checkpoint`` globs ``*.distcp`` and raises when it finds none."""
    box.ckpt(3, distcp=False)
    assert box.sh("newest").returncode == 1


def test_full_weights_still_being_written_is_not_complete(box):
    """``.metadata`` is already there, but rank 0 is still inside ``torch.save``.

    The DCP part is safe to resume from, but the 15 GB ``full_weights.pt`` is the
    newest by mtime and is what the convert step at the end of the run picks up,
    so a truncated one is a real loss.  A size that is still moving says so.
    """
    box.ckpt(3, weights="growing")
    r = box.sh("newest")
    assert r.returncode == 1, r.stdout
    assert "still growing" in box.log_text() or "still growing" in r.stdout


def test_missing_full_weights_is_not_complete(box):
    """The DCP half finished, the full-weights half never started."""
    box.ckpt(3, weights="missing")
    assert box.sh("newest").returncode == 1


def test_empty_checkpoint_directory_is_not_complete(box):
    """A directory created by ``os.makedirs`` and nothing else yet."""
    (box.ckpts / "global_step_3").mkdir()
    assert box.sh("newest").returncode == 1


def test_no_checkpoints_at_all(box):
    """Nothing has been saved yet -- there is nothing to resume from."""
    assert box.sh("newest").returncode == 1


# --------------------------------------------------------------------------------
# 2. which checkpoint is the newest
# --------------------------------------------------------------------------------


def test_step_10_beats_step_9_numerically_not_lexically(box):
    """``sort`` without ``-n`` puts ``global_step_9`` after ``global_step_10``.

    Taking the lexical maximum here silently throws away a step, and the run saves
    every step, so this trap is hit the moment the run passes step 9.
    """
    box.ckpt(9)
    newest = box.ckpt(10)
    r = box.sh("newest")
    assert r.returncode == 0
    assert r.stdout.strip() == str(newest)


def test_falls_back_to_the_previous_complete_checkpoint(box):
    """The newest is mid-write, so the one before it is what we resume from.

    ``KEEP_CKPTS=2`` exists to guarantee this: the reaper keeps the directory being
    written AND the last complete one, so a save in flight never leaves the run
    with nothing loadable.
    """
    prev = box.ckpt(9)
    box.ckpt(10, metadata=False)
    r = box.sh("newest")
    assert r.returncode == 0
    assert r.stdout.strip() == str(prev)


def test_all_candidates_incomplete_means_nothing_to_resume_from(box):
    box.ckpt(9, metadata=False)
    box.ckpt(10, metadata=False)
    assert box.sh("newest").returncode == 1


def test_stray_directories_are_ignored(box):
    """``best/`` is not a step, and neither is anything else in there."""
    (box.ckpts / "best").mkdir()
    (box.ckpts / "global_step_notanumber").mkdir()
    d = box.ckpt(2)
    assert box.sh("newest").stdout.strip() == str(d)


# --------------------------------------------------------------------------------
# 3. the window: memory AND sustained utilization, ours counted as ours
# --------------------------------------------------------------------------------


def test_window_opens_with_two_spare_idle_cards(box):
    """Our two plus the two that just came free."""
    _free_window(box)
    r = box.sh("window", OURS_GPUS="1,3")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WINDOW_OPEN" in r.stdout
    assert "GPUS=1,3,6,7" in r.stdout


def test_our_own_cards_are_not_idle_tested(box):
    """They are full of our own run and still count as available to it."""
    box.cards(
        {
            0: (70000, [99]),
            1: (70000, [99]),  # ours, busy with our own training
            2: (70000, [99]),
            3: (70000, [99]),  # ours
            4: (70000, [99]),
            5: (70000, [99]),
            6: (1200, [0]),
            7: (1500, [0]),
        }
    )
    r = box.sh("window", OURS_GPUS="1,3")
    assert r.returncode == 0, r.stdout
    assert "GPUS=1,3,6,7" in r.stdout


def test_low_memory_but_high_utilization_does_not_open_a_window(box):
    """<2% memory, 73-97% utilization: the tenant job this project co-launched onto.

    Memory alone calls card 6 free.  It is not.
    """
    box.cards(
        {
            0: (70000, [99]),
            1: (1000, [0]),
            2: (70000, [99]),
            3: (1000, [0]),
            4: (70000, [99]),
            5: (70000, [99]),
            6: (1200, [73, 97, 85, 91, 88]),
            7: (1500, [0, 0, 0, 0, 0]),
        }
    )
    r = box.sh("window", OURS_GPUS="1,3")
    assert r.returncode == 1
    assert "WINDOW_CLOSED" in r.stdout


def test_one_busy_sample_out_of_five_still_counts_as_idle(box):
    """A single reading between kernels must not veto a real window."""
    box.cards(
        {
            0: (70000, [99]),
            1: (1000, [0]),
            2: (70000, [99]),
            3: (1000, [0]),
            4: (70000, [99]),
            5: (70000, [99]),
            6: (1200, [99, 0, 0, 0, 0]),
            7: (1500, [0, 0, 0, 0, 0]),
        }
    )
    r = box.sh("window", OURS_GPUS="1,3")
    assert r.returncode == 0, r.stdout
    assert "GPUS=1,3,6,7" in r.stdout


def test_only_one_spare_card_is_not_a_window(box):
    """Three cards is not four; waiting costs nothing, a bad switch costs the run."""
    box.cards(
        {
            0: (70000, [99]),
            1: (1000, [0]),
            2: (70000, [99]),
            3: (1000, [0]),
            4: (70000, [99]),
            5: (70000, [99]),
            6: (1200, [0]),
            7: (70000, [99]),
        }
    )
    r = box.sh("window", OURS_GPUS="1,3")
    assert r.returncode == 1
    assert "WINDOW_CLOSED" in r.stdout


def test_least_occupied_card_is_preferred(box):
    """Within a memory bucket, utilization breaks the tie."""
    box.cards(
        {
            0: (70000, [99]),
            1: (1000, [0]),
            2: (70000, [99]),
            3: (1000, [0]),
            4: (1900, [18, 17, 19, 18, 17]),  # idle by the rule, but working
            5: (70000, [99]),
            6: (1900, [0, 0, 0, 0, 0]),  # same bucket, genuinely quiet
            7: (70000, [99]),
        }
    )
    r = box.sh("window", OURS_GPUS="1,3", NEED_GPUS="3")
    assert r.returncode == 0, r.stdout
    assert "GPUS=1,3,6" in r.stdout


def test_unreadable_card_is_busy_never_free(box):
    """A card that answers ``[N/A]`` is not a card we understand."""
    box.spec.write_text("1 1000 0\n3 1000 0\n6 [N/A] 0\n7 1500 0\n")
    r = box.sh("window", OURS_GPUS="1,3", PICK_FROM="6,7")
    assert r.returncode == 1


def test_the_real_nvidia_smi_is_never_called(box):
    """If the seam regresses, the guard on PATH fires instead of the real cards."""
    _free_window(box)
    box.sh("window", OURS_GPUS="1,3")
    assert not box.guard_log.exists(), box.guard_log.read_text()


# --------------------------------------------------------------------------------
# 4. refusals -- the supervisor must stop rather than press on
# --------------------------------------------------------------------------------


def test_refuses_when_there_is_no_run_to_supervise(box):
    """Nothing is running and nothing says the run finished: that is not normal."""
    _free_window(box)
    box.ckpt(3)
    r = box.sh(timeout=60)
    assert r.returncode == 4, r.stdout
    assert "no run to supervise" in (r.stdout + box.log_text())
    assert box.relaunch_text() == ""


def test_waits_instead_of_killing_a_run_that_has_saved_nothing(box):
    """Killing a run that produced nothing loses everything and gains nothing."""
    _free_window(box)
    run = box.run()
    p = box.popen()
    box.wait_for_log("nothing to resume from", proc=p)
    assert "job" in run.alive()
    assert box.relaunch_text() == ""


def test_waits_while_the_only_checkpoint_is_mid_write(box):
    """The window is open, but the newest save is in flight and it is all we have.

    This is the failure that cost this machine days: never kill during a save.
    """
    _free_window(box)
    box.ckpt(4, metadata=False)
    run = box.run()
    p = box.popen()
    box.wait_for_log("nothing to resume from", proc=p)
    assert "job" in run.alive()


def test_does_not_fire_when_no_window_is_open(box):
    """Six cards busy: keep waiting, keep the run."""
    box.ckpt(3)
    run = box.run()
    p = box.popen()
    box.wait_for_log("no window", proc=p)
    assert "job" in run.alive()
    assert box.relaunch_text() == ""


def test_exits_quietly_when_the_run_already_finished_all_its_steps(box):
    """15 of 15 saved: the run is in its convert step.  Do not touch it."""
    _free_window(box)
    box.ckpt(15)
    run = box.run()
    r = box.sh(timeout=60)
    assert r.returncode == 0, r.stdout
    assert "already finished" in (r.stdout + box.log_text())
    assert "job" in run.alive()
    assert box.relaunch_text() == ""


def test_exits_quietly_when_the_run_finished_and_is_gone(box):
    """Nothing running, but the log says it got there on its own."""
    _free_window(box)
    box.ckpt(15)
    r = box.sh(timeout=60)
    assert r.returncode == 0, r.stdout
    assert box.relaunch_text() == ""


def test_holds_when_the_volume_is_too_full_to_take_the_relaunch(box):
    """The disk floor is re-checked at the moment of firing, not at startup."""
    _free_window(box)
    box.ckpt(3)
    run = box.run()
    p = box.popen(NEED_GB="99999999")
    box.wait_for_log("HOLDING", proc=p)
    assert "job" in run.alive()
    assert box.relaunch_text() == ""


def test_refuses_to_launch_when_the_cards_are_not_released(box):
    """The kill went through but a card still holds memory: do not launch on it.

    Modelled by pinning the fake sampler to a process that outlives the kill, so
    our two cards keep reading full.
    """
    _free_window(box)
    box.ckpt(3)
    run = box.run()
    # cards 1 and 3 read full for as long as THIS pytest process lives
    r = box.sh(
        timeout=90,
        FAKE_SMI_ALIVE_PID=str(os.getpid()),
        FAKE_SMI_HELD="1,3",
        FAKE_SMI_HELD_MIB="70000",
        FAKE_SMI_HELD_UTIL="0",
    )
    assert r.returncode == 5, r.stdout
    assert "not released" in (r.stdout + box.log_text())
    assert box.relaunch_text() == ""
    assert run.alive() == [] or "job" not in run.alive()


# --------------------------------------------------------------------------------
# 5. the lock
# --------------------------------------------------------------------------------


def test_a_second_supervisor_refuses_to_start(box):
    """Two supervisors would both fire into the same window and fight over cards."""
    box.ckpt(3)
    box.run()
    first = box.popen()
    box.wait_for_log("watching", proc=first)
    second = box.sh(timeout=30)
    assert second.returncode == 1
    assert "another" in second.stdout.lower()


def test_the_lock_is_released_when_the_supervisor_exits(box):
    """A refusal must not leave the lock wedged for the next attempt."""
    _free_window(box)
    box.ckpt(3)
    first = box.sh(timeout=60)
    assert first.returncode == 4
    second = box.sh(timeout=60)
    assert second.returncode == 4, second.stdout


# --------------------------------------------------------------------------------
# 6. the switch itself
# --------------------------------------------------------------------------------


def test_the_full_switch(box):
    """Window open, a complete checkpoint behind us: kill cleanly, relaunch on four."""
    _free_window(box)
    box.ckpt(9)
    ckpt10 = box.ckpt(10)
    run = box.run(gpus="1,3")
    r = box.sh(
        timeout=120,
        FAKE_SMI_ALIVE_PID=str(run.auto.pid),
        FAKE_SMI_HELD="1,3",
        FAKE_SMI_HELD_MIB="70000",
        FAKE_SMI_HELD_UTIL="0",
    )
    assert r.returncode == 0, r.stdout + box.log_text()

    env = box.relaunch_env()
    assert env["GPUS"] == "1,3,6,7"
    assert env["RESUME_DIR"] == str(ckpt10)
    # everything else carried across from the run we replaced
    assert env["TAG"] == "faketag"
    assert env["STEPS"] == "15"
    assert env["SAVE_INTERVAL"] == "1"
    assert env["KEEP_CKPTS"] == "2"
    assert env["PROFILE"] == "1"
    assert env["PROFILE_EVERY"] == "50"

    assert run.alive() == [], f"survivors: {run.alive()}"
    assert "RESUMED" in box.log_text()


def test_the_auto_loop_dies_before_the_job_it_would_relaunch(box):
    """Kill the inner job first and the loop starts a fresh 2-GPU run at micro=4.

    Then two jobs fight over the same cards.  The loop has to go first.
    """
    _free_window(box)
    box.ckpt(10)
    run = box.run()
    r = box.sh(
        timeout=120,
        FAKE_SMI_ALIVE_PID=str(run.auto.pid),
        FAKE_SMI_HELD="1,3",
        FAKE_SMI_HELD_MIB="70000",
        FAKE_SMI_HELD_UTIL="0",
    )
    assert r.returncode == 0, r.stdout + box.log_text()
    assert "RELAUNCHED-AFTER-INNER-DEATH" not in box.events_text()


def test_the_ray_session_is_matched_by_its_tmpdir(box):
    """Ray goes by RAY_TMPDIR, never by process group: other sessions are tenants'."""
    _free_window(box)
    box.ckpt(10)
    run = box.run()
    # a second, unrelated ray session that must survive
    bystander = subprocess.Popen(
        ["bash", str(box.ray_script), "/tmp/rayiso_SOMEONE_ELSE"],
        env={**os.environ, "FAKE_EVENTS": str(box.events)},
        start_new_session=True,
    )
    box._procs.append(bystander)
    try:
        r = box.sh(
            timeout=120,
            FAKE_SMI_ALIVE_PID=str(run.auto.pid),
            FAKE_SMI_HELD="1,3",
            FAKE_SMI_HELD_MIB="70000",
            FAKE_SMI_HELD_UTIL="0",
        )
        assert r.returncode == 0, r.stdout + box.log_text()
        assert _FakeRun.pgrep(run.ray_tmpdir) == []
        assert bystander.poll() is None, "an unrelated ray session was killed"
    finally:
        bystander.kill()


def test_a_relaunch_that_does_not_resume_is_reported_loudly(box):
    """A supervisor that restarts from zero while looking healthy is the worst case."""
    _free_window(box)
    ckpt = box.ckpt(10)
    _write_exe(box.launcher, FAKE_LAUNCHER_NO_RESUME)
    run = box.run()
    r = box.sh(
        timeout=120,
        FAKE_SMI_ALIVE_PID=str(run.auto.pid),
        FAKE_SMI_HELD="1,3",
        FAKE_SMI_HELD_MIB="70000",
        FAKE_SMI_HELD_UTIL="0",
    )
    assert r.returncode == 6, r.stdout + box.log_text()
    log = box.log_text()
    assert "COULD NOT VERIFY" in log
    assert str(ckpt) in log


def test_only_new_driver_log_lines_count_as_evidence_of_the_resume(box):
    """An old resume line from a previous attempt must not verify this one."""
    _free_window(box)
    ckpt = box.ckpt(10)
    box.driver_log.write_text(
        f"[INFO 01:00:00 RLinf] Resuming training from checkpoint directory {ckpt}.\n"
    )
    _write_exe(box.launcher, FAKE_LAUNCHER_NO_RESUME)
    run = box.run()
    r = box.sh(
        timeout=120,
        FAKE_SMI_ALIVE_PID=str(run.auto.pid),
        FAKE_SMI_HELD="1,3",
        FAKE_SMI_HELD_MIB="70000",
        FAKE_SMI_HELD_UTIL="0",
    )
    assert r.returncode == 6, r.stdout + box.log_text()


def test_the_job_is_not_confused_with_its_safe_run_wrapper(box):
    """``pgrep -f`` matches three processes; only one of them is the job.

    ``safe_run.sh`` carries the job's script path as an argument, and the checkpoint
    reaper is forked from the job so its command line is byte-identical.  Observed
    live: a plain ``pgrep -f`` picked safe_run and the supervisor then read the wrong
    process's environment.  ``status`` has to name the middle one.
    """
    run = box.run()
    r = box.sh("status")
    assert r.returncode == 0, r.stdout + r.stderr
    reported = int(
        [ln for ln in r.stdout.splitlines() if ln.startswith("job_pid=")][0]
        .split("=", 1)[1]
        .split()[0]
    )
    matches = run.pgrep(box.job_pat)
    assert len(matches) >= 3, f"fixture should present the ambiguity: {matches}"
    args = subprocess.run(
        ["ps", "-o", "args=", "-p", str(reported)], capture_output=True, text=True
    ).stdout
    assert "fake_safe_run.sh" not in args, f"picked the wrapper: {args!r}"
    assert "fake_job.sh" in args, args
    # and the environment it carries across comes from the job, not the wrapper
    assert "TAG=faketag" in r.stdout
    assert "ours=1,3" in r.stdout
