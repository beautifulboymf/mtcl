#!/bin/bash
# opd_mt4slot_upgrade.sh — move the LIVE R1 run from 2 cards onto 4 when a window opens.
#
#   opd_mt4slot_upgrade.sh            # supervise: wait for the window, then switch
#   opd_mt4slot_upgrade.sh status     # what it sees right now; changes nothing
#   opd_mt4slot_upgrade.sh newest     # the checkpoint it would resume from
#   opd_mt4slot_upgrade.sh window     # is a 4-card window open right now?
#
# WHY THIS EXISTS. Only two cards were free when R1 had to start, and starting on two was better
# than losing them to another tenant. Four cards is roughly 3x faster per step. The window opens
# without warning and closes the same way, so the switch has to be automatic -- but a switch is a
# KILL followed by a RESTART, and both halves can destroy the run.
#
# WHAT MAKES THE SWITCH LEGITIMATE. Only two knobs differ between 2 and 4 cards --
# actor.micro_batch_size and actor.fsdp_config.gradient_checkpointing -- and neither changes the
# training math: global_batch_size stays 192, so the optimizer sees the identical batch; only how
# a step is split and whether activations are recomputed change. The DCP checkpoint reshards
# across a different world size, which is what lets a 2-rank save load on 4 ranks.
#
# THE RED LINES, in the order they matter.
#
# 1. NEVER kill during a save. A half-written DCP directory -- data blocks present, `.metadata`
#    missing -- is unloadable AND unconvertible, and this volume lost days to exactly that class
#    of failure. The completeness rule is derived from the save path, not guessed; see
#    ckpt_complete() below.
# 2. NEVER kill a run with nothing to resume from. That loses every step and gains nothing, so
#    "no complete checkpoint yet" means keep waiting, however wide the window.
# 3. NEVER take a card on a reading. `probe_gpu` here is `wait_for_gpus.sh`'s idle test,
#    unchanged: memory AND a majority of repeated utilization samples. A single instantaneous
#    reading lands between kernels as often as not; this project once co-launched onto a tenant's
#    compute-bound job showing under 2% memory and 73-97% utilization.
# 4. NEVER kill ray by process group. Other ray sessions on this box belong to other jobs and
#    live in other process groups; this one is identified by its private RAY_TMPDIR, which is how
#    run_iso.sh's own cleanup does it.
# 5. NEVER assume the relaunch resumed. A supervisor that restarts from step 0 while the log
#    looks healthy is worse than one that never fires, so the resume is VERIFIED against the
#    driver log and a failure to verify is loud and non-zero.
#
# Anything it cannot make sense of -- no run, no checkpoint, cards that stay busy after the kill,
# a relaunch that cannot be shown to have resumed -- is a refusal with a reason, never a guess.
#
# TESTING. Every input to a decision comes from a command or a path named by a variable:
# NVIDIA_SMI (the sampler), CKPT_DIR / DRIVER_LOG / DISK_PATH (the state), AUTO_PAT / JOB_PAT
# (finding the run) and LAUNCHER (the way back up). All default to the real thing, so a human
# runs this with no arguments; tests/unit_tests/test_opd_mt4slot_upgrade.py points them at fakes.
set -uo pipefail

REPO="${REPO:-/home/fanruochen/CL/RLinf}"
O="${O:-/share/fanruochen-local/outputs}"
INC="$REPO/examples/embodiment/incremental_sft"

# ---- what we are supervising ------------------------------------------------------------------
# TAG names the run; everything else about where it keeps its state follows from it, exactly as
# opd_mt4slot.sh derives it (LOGP=$O/seqcl_$TAG, and the runner appends experiment_name again).
TAG_DEFAULT="${TAG:-mt4slot}"
CKPT_DIR="${CKPT_DIR:-$O/seqcl_${TAG_DEFAULT}/seqcl_${TAG_DEFAULT}/checkpoints}"
DRIVER_LOG="${DRIVER_LOG:-$O/opd_mt4slot_driver.log}"
LAUNCHER="${LAUNCHER:-$INC/opd_mt4slot_auto.sh}"
RELAUNCH_LOG="${RELAUNCH_LOG:-$O/opd_mt4slot_upgrade_relaunch.log}"
# Patterns that find the live run. The auto loop MUST be matched separately from the job it
# supervises: it relaunches a fresh 2-GPU run at the next micro_batch_size the moment the inner
# job exits non-zero, so killing the job without killing the loop first leaves two jobs fighting
# over the same four cards. `\.` is escaped so opd_mt4slot_auto.sh cannot match JOB_PAT.
AUTO_PAT="${AUTO_PAT:-incremental_sft/opd_mt4slot_auto\.sh}"
JOB_PAT="${JOB_PAT:-incremental_sft/opd_mt4slot\.sh}"
STEPS="${STEPS:-15}"                 # the run is done at this many; do not resume onto a finished run
PROC_ROOT="${PROC_ROOT:-/proc}"

# ---- the cards --------------------------------------------------------------------------------
# The idle test, verbatim from wait_for_gpus.sh. Same names, same defaults, same meaning.
NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"
NEED_GPUS="${NEED_GPUS:-4}"
PICK_FROM="${PICK_FROM:-0,1,2,3,4,5,6,7}"
MEM_MAX_MIB="${MEM_MAX_MIB:-5000}"
UTIL_MAX_PCT="${UTIL_MAX_PCT:-20}"
UTIL_N="${UTIL_N:-5}"
UTIL_OK_N="${UTIL_OK_N:-4}"
UTIL_SLEEP_S="${UTIL_SLEEP_S:-1}"
MEM_BUCKET_MIB="${MEM_BUCKET_MIB:-4096}"
# OURS_GPUS: the cards the run already holds. They are not idle-tested -- they are full of our
# own training and they are already ours. Discovered from the job's environment when unset.
OURS_GPUS="${OURS_GPUS:-}"

# ---- the switch -------------------------------------------------------------------------------
# 4-card settings. Both are training-math-neutral (see the header), and both are deliberately the
# CONSERVATIVE choice: 4 cards is strictly easier on memory than the 2 this run survives on today
# (total_num_envs is pinned at 48, so envs-per-rank falls from 24 to 12, and the FSDP shard and
# optimizer state shrink as world size grows), so micro=8 with recomputation on will fit. The
# launcher's own OOM loop walks MICROS_4G down if the cards turn out fuller than they read.
GRAD_CKPT_4G="${GRAD_CKPT_4G:-True}"
MICROS_4G="${MICROS_4G:-8 4 2}"
# A DIFFERENT ray port from the run we are replacing. The old head is dead by then, but a port in
# TIME_WAIT or one lingering worker turns a clean relaunch into a 6-minute "Failed to connect to
# GCS" hang. run_iso.sh derives its worker range as [PORT+10, PORT+300], so stay >=1000 away.
NEW_PORT="${NEW_PORT:-59000}"

# ---- gates and timing -------------------------------------------------------------------------
POLL_S="${POLL_S:-120}"
MAX_WAIT_H="${MAX_WAIT_H:-12}"
NEED_GB="${NEED_GB:-175}"            # opd_mt4slot.sh's own floor; re-checked at the moment of firing
DISK_PATH="${DISK_PATH:-/share/fanruochen-local}"
STABLE_S="${STABLE_S:-5}"            # full_weights.pt must not change size across this gap
KILL_WAIT_S="${KILL_WAIT_S:-60}"     # per stage of the kill, before escalating to SIGKILL
REL_WAIT_S="${REL_WAIT_S:-180}"      # how long to wait for the cards to actually come free
REL_MAX_MIB="${REL_MAX_MIB:-$MEM_MAX_MIB}"
VERIFY_S="${VERIFY_S:-2400}"         # the resumed run loads two 7B teacher bases first
LOG="${UPGRADE_LOG:-$O/opd_mt4slot_upgrade.log}"
LOCK="${UPGRADE_LOCK:-/tmp/opd_mt4slot_upgrade.lock}"

MODE="${1:-supervise}"
case "$MODE" in supervise|status|newest|window) ;; *)
  echo "usage: opd_mt4slot_upgrade.sh [supervise|status|newest|window]"; exit 2 ;; esac

# Single instance, for the same reason the waiter has one: two supervisors would both fire into
# the same window, and the second would kill the run the first had just relaunched.
if [ "$MODE" = "supervise" ]; then
  exec 9>"$LOCK" || { echo "cannot open lock $LOCK"; exit 1; }
  flock -n 9 || { echo "another opd_mt4slot_upgrade.sh already holds $LOCK -- refusing to start a second"; exit 1; }
fi

mkdir -p "$(dirname "$LOG")" 2>/dev/null
# Progress goes to the log and to STDERR, never to stdout: `newest` and `window` put their answer
# on stdout and a caller has to be able to read it without filtering commentary out of it.
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG" >&2; }

# ==================================================================================================
# processes
# ==================================================================================================

# `kill -0` says yes to a zombie, which is exactly what a killed process looks like until its
# parent reaps it -- so a kill that worked would look like a kill that failed and escalate to
# SIGKILL against a corpse. Ask the kernel for the state instead.
proc_alive() {
  local p="$1" st
  [ -n "$p" ] || return 1
  kill -0 "$p" 2>/dev/null || return 1
  st=$(sed -n 's/^State:[[:space:]]*\([A-Z]\).*/\1/p' "$PROC_ROOT/$p/status" 2>/dev/null)
  [ "$st" = "Z" ] && return 1
  return 0
}

proc_env() {  # proc_env <pid> <VAR> -> value on stdout
  local pid="$1" var="$2" f="$PROC_ROOT/$1/environ"
  [ -r "$f" ] || return 1
  local v
  v=$(tr '\0' '\n' < "$f" 2>/dev/null | sed -n "s/^${var}=//p" | head -1)
  [ -n "$v" ] || return 1
  printf '%s\n' "$v"
}

pgrep_pat() {  # every pid whose command line matches, minus this script and its own children
  local pat="$1" p out=()
  while read -r p; do
    [ -n "$p" ] || continue
    [ "$p" = "$$" ] && continue
    [ "$p" = "$PPID" ] && continue
    out+=("$p")
  done < <(pgrep -u "$USER" -f "$pat" 2>/dev/null)
  printf '%s\n' "${out[@]+"${out[@]}"}"
}

descendants() {  # every process under <pid>, breadth first
  local queue=("$1") out=() cur k kids
  while (( ${#queue[@]} )); do
    cur="${queue[0]}"; queue=("${queue[@]:1}")
    mapfile -t kids < <(pgrep -P "$cur" 2>/dev/null)
    for k in "${kids[@]+"${kids[@]}"}"; do
      [ -n "$k" ] || continue
      out+=("$k"); queue+=("$k")
    done
  done
  printf '%s\n' "${out[@]+"${out[@]}"}"
}

# argv[0] and argv[1] only -- the interpreter and the script it was asked to run.
cmd_head() { tr '\0' ' ' < "$PROC_ROOT/$1/cmdline" 2>/dev/null | awk '{print $1, $2}'; }

# Finding the job is fiddlier than one `pgrep -f` because TWO other processes carry its name:
#
#   safe_run.sh  runs it, so the job's path is an ARGUMENT on safe_run's own command line
#                (`bash safe_run.sh <log> bash .../opd_mt4slot.sh`) and `pgrep -f` matches it.
#                Observed live: this returned safe_run's pid as "the job".
#   the reaper   is forked from the job, so it has a byte-identical command line.
#
# Two filters, and both are needed. Matching argv[1] rather than anywhere in the command line
# drops the wrapper; dropping any match whose PARENT is also a match leaves the reaper behind
# with its twin. Neither filter alone is enough, and getting this wrong means killing the wrong
# process and reading the wrong environment.
match_pids() {
  local pat="$1" pids=() p q ppid isparent kept=() out=()
  mapfile -t pids < <(pgrep_pat "$pat")
  for p in "${pids[@]+"${pids[@]}"}"; do
    [ -n "$p" ] || continue
    cmd_head "$p" | grep -qE "$pat" && kept+=("$p")
  done
  for p in "${kept[@]+"${kept[@]}"}"; do
    ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    isparent=0
    for q in "${kept[@]+"${kept[@]}"}"; do [ "$q" = "$ppid" ] && isparent=1; done
    (( isparent )) || out+=("$p")
  done
  printf '%s\n' "${out[@]+"${out[@]}"}"
}

# run_iso.sh exports RAY_TMPDIR into the training process's environment; it is the only handle on
# THIS ray session that cannot also catch somebody else's.
find_ray_tmpdir() {
  local job="$1" p v
  v=$(proc_env "$job" RAY_TMPDIR) && { printf '%s\n' "$v"; return 0; }
  for p in $(descendants "$job"); do
    v=$(proc_env "$p" RAY_TMPDIR) && { printf '%s\n' "$v"; return 0; }
  done
  return 1
}

wait_gone() {  # all of <pids...> gone within KILL_WAIT_S
  local deadline=$(( $(date +%s) + KILL_WAIT_S )) p alive
  while :; do
    alive=0
    for p in "$@"; do proc_alive "$p" && alive=1; done
    (( alive )) || return 0
    (( $(date +%s) < deadline )) || return 1
    sleep 0.2
  done
}

kill_pids() {  # SIGTERM, then SIGKILL what is left. By PID -- never by process group.
  local p
  (( $# )) || return 0
  for p in "$@"; do proc_alive "$p" && kill -TERM "$p" 2>/dev/null; done
  wait_gone "$@" && return 0
  say "  still alive after SIGTERM, escalating: $*"
  for p in "$@"; do proc_alive "$p" && kill -KILL "$p" 2>/dev/null; done
  wait_gone "$@"
}

# ==================================================================================================
# checkpoints
# ==================================================================================================

# WHAT COUNTS AS COMPLETE, and why this is sound.
#
# The save path is FSDPStrategy.save_checkpoint (rlinf/hybrid_engines/fsdp/strategy/base.py):
#
#   1. dcp.save(...)                      -> <ckpt>/actor/dcp_checkpoint/
#   2. torch.distributed.barrier()
#   3. rank0 torch.save(...)              -> <ckpt>/actor/model_state_dict/full_weights.pt
#   4. torch.distributed.barrier()
#
# and inside step 1, torch 2.6's FileSystemWriter.finish() writes `.metadata.tmp`, fsyncs it, and
# RENAMES it to `.metadata`. Rename within a filesystem is atomic, so `.metadata` exists if and
# only if every `.distcp` shard was already written and fsynced. There is no state in which
# `.metadata` is half there. That is the marker -- not a file count, not an age, not a size.
#
# We additionally require full_weights.pt to exist and to hold its size across STABLE_S. Resume
# itself does not need it (load_checkpoint only globs dcp_checkpoint/*.distcp), but it is a ~15 GB
# torch.save that runs for minutes AFTER `.metadata` lands, it is the newest file by mtime, and it
# is what the convert step at the end of the run picks up -- so a truncated one is a real loss for
# the price of waiting a few seconds. A size that is still moving is a save still in flight.
#
# The `.distcp` check is not redundant: load_checkpoint raises if the glob comes back empty, so a
# directory with a metadata file and no shards would pass a metadata-only test and fail at load.
ckpt_complete() {
  local d="$1" dcp="$1/actor/dcp_checkpoint" fw="$1/actor/model_state_dict/full_weights.pt"
  [ -f "$dcp/.metadata" ] || { echo "no .metadata (DCP save did not finish)"; return 1; }
  compgen -G "$dcp/*.distcp" >/dev/null || { echo "no .distcp shards"; return 1; }
  [ -f "$fw" ] || { echo "no full_weights.pt yet (rank0 save not started)"; return 1; }
  local a b
  a=$(stat -c %s "$fw" 2>/dev/null) || { echo "cannot stat full_weights.pt"; return 1; }
  (( a > 0 )) || { echo "full_weights.pt is empty"; return 1; }
  sleep "$STABLE_S"
  b=$(stat -c %s "$fw" 2>/dev/null) || { echo "cannot stat full_weights.pt"; return 1; }
  [ "$a" = "$b" ] || { echo "full_weights.pt still growing ($a -> $b)"; return 1; }
  echo "complete"
  return 0
}

ckpt_steps() {  # every global_step_<N> under CKPT_DIR, ascending NUMERICALLY
  [ -d "$CKPT_DIR" ] || return 0
  find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -type d \
    -regextype posix-extended -regex '.*/global_step_[0-9]+' -printf '%f\n' 2>/dev/null \
    | sed 's/^global_step_//' | sort -n
}

# The newest COMPLETE checkpoint, walking down from the highest step. `sort -n`, not `sort`:
# lexically "global_step_9" sorts above "global_step_10", and the run saves every step, so a
# lexical maximum starts throwing away a step the moment it passes 9.
#
# Walking DOWN is what makes a mid-write newest checkpoint harmless: KEEP_CKPTS>=2 guarantees the
# last complete one is still on disk underneath it.
newest_complete() {
  local steps=() i s d why
  mapfile -t steps < <(ckpt_steps)
  for (( i = ${#steps[@]} - 1; i >= 0; i-- )); do
    s="${steps[i]}"; d="$CKPT_DIR/global_step_$s"
    why=$(ckpt_complete "$d")
    if [ "$why" = "complete" ]; then
      printf '%s\n' "$d"
      return 0
    fi
    say "  global_step_$s is not resumable: $why"
  done
  return 1
}

max_step() {  # highest step directory on disk, complete or not; "" if none
  ckpt_steps | tail -1
}

# ==================================================================================================
# cards
# ==================================================================================================

# wait_for_gpus.sh's probe, unchanged. Returns 0 if the card is idle by BOTH measures and echoes
# "mem util_mean n_busy". Call it with COMMAND substitution, never process substitution: with
# `read < <(probe_gpu)` the `$?` you get back is read's, the idle verdict is silently discarded,
# and every card counts as idle -- that bug was observed opening a window on cards at 100%.
probe_gpu() {
  local g="$1" mem util busy=0 sum=0 i
  mem=$("$NVIDIA_SMI" -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || return 2
  [[ "$mem" =~ ^[0-9]+$ ]] || return 2   # "[N/A]" is not a card we understand
  for (( i = 0; i < UTIL_N; i++ )); do
    util=$("$NVIDIA_SMI" -i "$g" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    [[ "$util" =~ ^[0-9]+$ ]] || util=100   # unreadable == busy, never == free
    sum=$(( sum + util ))
    (( util < UTIL_MAX_PCT )) || busy=$(( busy + 1 ))
    sleep "$UTIL_SLEEP_S"
  done
  echo "$mem $(( sum / UTIL_N )) $busy"
  (( mem < MEM_MAX_MIB )) || return 1
  (( UTIL_N - busy >= UTIL_OK_N )) || return 1
  return 0
}

gpu_mem() {
  local m
  m=$("$NVIDIA_SMI" -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  [[ "$m" =~ ^[0-9]+$ ]] || { echo 999999; return 1; }
  echo "$m"
}

# Our cards plus enough idle ones to reach NEED_GPUS. Prints the full csv on stdout and a
# per-card readout on WINDOW_STATUS; returns 1 when the window is not open.
WINDOW_STATUS=""
pick_window() {
  local ours="$1" want g out rc mem umean busy cands="" n_ok chosen
  want=$(( NEED_GPUS - $(printf '%s' "${ours//,/ }" | wc -w) ))
  WINDOW_STATUS=""
  if (( want <= 0 )); then printf '%s\n' "$ours"; return 0; fi
  for g in ${PICK_FROM//,/ }; do
    case ",$ours," in *",$g,"*) continue ;; esac
    out=$(probe_gpu "$g"); rc=$?
    read -r mem umean busy <<< "$out"
    WINDOW_STATUS+="$g:${mem:-?}MiB/${umean:-?}% "
    # Bucket memory before ranking, then rank by utilization inside the bucket: every candidate
    # here carries the same ~2 GiB tenant footprint, so raw memory order is a few MiB of noise and
    # would happily prefer a card at 69% over one at 41%. Compute is what we contend for.
    (( rc == 0 )) && cands+="$(( mem / MEM_BUCKET_MIB )) $umean $mem $g"$'\n'
  done
  n_ok=$(printf '%s' "$cands" | grep -c . || true)
  (( n_ok >= want )) || return 1
  chosen=$(printf '%s' "$cands" | sort -k1,1n -k2,2n | head -n "$want" | awk '{print $4}' | paste -sd, -)
  printf '%s\n%s\n' "${ours//,/$'\n'}" "${chosen//,/$'\n'}" \
    | grep -E '^[0-9]+$' | sort -n | uniq | paste -sd, -
  return 0
}

# ==================================================================================================
# discovery
# ==================================================================================================

AUTO_PID=""; JOB_PID=""; RAY_TMPDIR_FOUND=""; CARRY=()
discover() {
  local jobs=() autos=() v k
  AUTO_PID=""; JOB_PID=""; RAY_TMPDIR_FOUND=""; CARRY=()
  mapfile -t autos < <(match_pids "$AUTO_PAT")
  mapfile -t jobs  < <(match_pids "$JOB_PAT")
  [ "${#autos[@]}" -gt 0 ] && AUTO_PID="${autos[0]}"
  if (( ${#jobs[@]} > 1 )); then
    say "REFUSING: ${#jobs[@]} processes match JOB_PAT after removing their own children (${jobs[*]})."
    say "          That is two runs, or a pattern that is too broad. Sort it out by hand."
    return 2
  fi
  [ "${#jobs[@]}" -eq 1 ] && JOB_PID="${jobs[0]}"
  [ -n "$JOB_PID" ] || return 1
  RAY_TMPDIR_FOUND=$(find_ray_tmpdir "$JOB_PID" || true)
  # Everything the replacement run must inherit UNCHANGED. Read out of the live job rather than
  # restated here, so the relaunch cannot quietly differ from the run it replaces in some knob
  # nobody thought to copy.
  for k in TAG STEPS SAVE_INTERVAL KEEP_CKPTS STUDENT NEED_GB PROFILE PROFILE_EVERY LOG; do
    v=$(proc_env "$JOB_PID" "$k") && CARRY+=("$k=$v")
  done
  return 0
}

ours_gpus() {
  [ -n "$OURS_GPUS" ] && { printf '%s\n' "$OURS_GPUS"; return 0; }
  [ -n "$JOB_PID" ] && proc_env "$JOB_PID" GPUS && return 0
  return 1
}

# ==================================================================================================
# the switch
# ==================================================================================================

stop_the_run() {
  local desc=()
  # Collect the job's children BEFORE anything dies: once the parent goes they reparent to init
  # and the tree that identifies them is gone. The checkpoint reaper is in here, and an orphaned
  # reaper would go on deleting checkpoints belonging to the NEXT run with the same tag.
  mapfile -t desc < <(descendants "$JOB_PID")

  # THE LOOP FIRST. opd_mt4slot_auto.sh relaunches a fresh 2-GPU run at the next micro_batch_size
  # as soon as the job under it exits non-zero. Kill the job first and it helpfully starts a
  # second job on the cards we are about to take.
  if [ -n "$AUTO_PID" ]; then
    say "  stopping the OOM-fallback loop first (pid $AUTO_PID) so it cannot relaunch"
    kill_pids "$AUTO_PID" || { say "  the loop would not die (pid $AUTO_PID)"; return 1; }
  else
    say "  no auto-loop parent found; the job appears to have been launched directly"
  fi

  # safe_run.sh sits between the loop and the job and exits on its own once the job is gone, but
  # only if it is actually the parent -- check rather than assume.
  local safe_pid=""
  safe_pid=$(ps -o ppid= -p "$JOB_PID" 2>/dev/null | tr -d ' ')
  if [ -n "$safe_pid" ] && ! ps -o args= -p "$safe_pid" 2>/dev/null | grep -q "safe_run\.sh"; then
    safe_pid=""
  fi

  say "  stopping the job (pid $JOB_PID) and its ${#desc[@]} child process(es)"
  kill_pids "$JOB_PID" ${desc[@]+"${desc[@]}"} || {
    say "  the job would not die (pid $JOB_PID)"; return 1; }
  [ -n "$safe_pid" ] && kill_pids "$safe_pid"

  # THE RAY SESSION, by its private tmpdir and nothing else. Never `kill -- -$PGID` and never a
  # blanket `pkill -f ray`: other ray sessions on this box are other jobs, and normal ray workers
  # do not all share this job's process group anyway.
  if [ -n "$RAY_TMPDIR_FOUND" ]; then
    local rp=()
    mapfile -t rp < <(pgrep_pat "$RAY_TMPDIR_FOUND")
    if (( ${#rp[@]} )); then
      say "  stopping the ray session $RAY_TMPDIR_FOUND (${#rp[@]} process(es))"
      kill_pids "${rp[@]}" || say "  WARNING: some ray processes under $RAY_TMPDIR_FOUND survived"
    else
      say "  ray session $RAY_TMPDIR_FOUND already gone"
    fi
  else
    # Deliberately NOT falling back to a broad match. A pattern loose enough to find this ray
    # session without its tmpdir is loose enough to kill somebody else's, and the card-release
    # check below will catch anything that really is still holding the GPUs.
    say "  WARNING: no RAY_TMPDIR found for pid $JOB_PID; leaving ray alone and relying on the"
    say "           card-release check to catch anything still holding the GPUs"
  fi
  return 0
}

cards_released() {  # every target card back under REL_MAX_MIB within REL_WAIT_S
  local gpus="$1" deadline=$(( $(date +%s) + REL_WAIT_S )) g m busy
  while :; do
    busy=""
    for g in ${gpus//,/ }; do
      m=$(gpu_mem "$g")
      (( m < REL_MAX_MIB )) || busy+="$g:${m}MiB "
    done
    [ -z "$busy" ] && return 0
    (( $(date +%s) < deadline )) || { say "  cards still held: $busy"; return 1; }
    sleep 2
  done
}

relaunch() {  # relaunch <gpus> <resume_dir>
  local gpus="$1" resume="$2" kv
  say "RELAUNCH gpus=$gpus resume_dir=$resume micros='$MICROS_4G' grad_ckpt=$GRAD_CKPT_4G port=$NEW_PORT"
  say "  carrying over unchanged: ${CARRY[*]-<nothing discovered>}"
  # setsid so the run outlives this supervisor: the supervisor's job ends once the switch is
  # verified, and the training must not.
  setsid env "GPUS=$gpus" "RESUME_DIR=$resume" "MICROS=$MICROS_4G" \
    "GRAD_CKPT=$GRAD_CKPT_4G" "PORT=$NEW_PORT" ${CARRY[@]+"${CARRY[@]}"} \
    bash "$LAUNCHER" >>"$RELAUNCH_LOG" 2>&1 &
  say "  launcher started: $LAUNCHER (output -> $RELAUNCH_LOG)"
}

# DID IT ACTUALLY RESUME? A relaunch that silently starts from step 0 looks exactly like a healthy
# one for the first hour, and this project has been bitten repeatedly by mechanisms that failed
# silently while looking fine. Two independent witnesses, read only from bytes the driver log
# grew AFTER the relaunch so a previous attempt's line cannot vouch for this one:
#   * the resolved-config dump train_embodied_agent.py prints  -> the override reached hydra
#   * "Resuming training from checkpoint directory ..."        -> EmbodiedRunner.init_workers
#     actually took the resume branch and loaded the checkpoint
# The second is the authority; the first only localises the failure when there is one.
verify_resumed() {  # verify_resumed <offset> <resume_dir>
  local off="$1" resume="$2" deadline=$(( $(date +%s) + VERIFY_S )) new hydra_seen=0
  say "verifying the resume against $DRIVER_LOG (up to ${VERIFY_S}s)"
  while :; do
    new=$(tail -c "+$(( off + 1 ))" "$DRIVER_LOG" 2>/dev/null)
    if (( ! hydra_seen )) && printf '%s' "$new" | grep -qF "\"resume_dir\": \"$resume\""; then
      hydra_seen=1
      say "  the resume_dir override reached hydra"
    fi
    if printf '%s' "$new" | grep -qF "Resuming training from checkpoint directory $resume"; then
      say "RESUMED from $resume -- the runner took the resume branch, not a fresh start"
      return 0
    fi
    (( $(date +%s) < deadline )) || break
    sleep 5
  done
  say "COULD NOT VERIFY that the relaunch resumed from $resume within ${VERIFY_S}s."
  if (( hydra_seen )); then
    say "  the override DID reach hydra, so the checkpoint load is what did not happen (or is"
    say "  still in progress -- two 7B teacher bases load first)."
  else
    say "  no resolved-config line naming resume_dir either, so the override may not have"
    say "  reached the training process at all."
  fi
  say "  The run is LEFT RUNNING on 4 cards; it is NOT killed, because a slow start and a"
  say "  restart-from-zero look the same from here and only a human can tell them apart."
  say "  CHECK BY HAND before trusting the result: grep -n 'Resuming training' $DRIVER_LOG"
  return 1
}

# ==================================================================================================
# modes
# ==================================================================================================

if [ "$MODE" = "newest" ]; then
  d=$(newest_complete) || { say "no complete checkpoint under $CKPT_DIR"; exit 1; }
  printf '%s\n' "$d"
  exit 0
fi

if [ "$MODE" = "window" ]; then
  discover; drc=$?
  (( drc == 2 )) && exit 2
  ours=$(ours_gpus) || { echo "WINDOW_UNKNOWN: cannot tell which cards are already ours"; exit 2; }
  if sel=$(pick_window "$ours"); then
    echo "WINDOW_OPEN GPUS=$sel   ours=$ours  [$WINDOW_STATUS]"
    exit 0
  fi
  echo "WINDOW_CLOSED ours=$ours need=$NEED_GPUS  [$WINDOW_STATUS]"
  exit 1
fi

if [ "$MODE" = "status" ]; then
  discover; drc=$?
  echo "job_pid=${JOB_PID:-<none>} auto_pid=${AUTO_PID:-<none>} ray_tmpdir=${RAY_TMPDIR_FOUND:-<none>}"
  echo "carry=${CARRY[*]-<none>}"
  echo "ckpt_dir=$CKPT_DIR steps_on_disk=[$(ckpt_steps | paste -sd, -)]"
  if d=$(newest_complete); then echo "newest_complete=$d"; else echo "newest_complete=<none>"; fi
  ours=$(ours_gpus || echo "<unknown>")
  echo "ours=$ours"
  if [ "$ours" != "<unknown>" ] && sel=$(pick_window "$ours"); then
    echo "window=OPEN would_use=$sel"
  else
    echo "window=CLOSED [$WINDOW_STATUS]"
  fi
  exit 0
fi

# ==================================================================================================
# supervise
# ==================================================================================================

say "watching for a ${NEED_GPUS}-card window for the run under $CKPT_DIR"
say "  idle test: mem<${MEM_MAX_MIB}MiB AND >=${UTIL_OK_N}/${UTIL_N} samples under ${UTIL_MAX_PCT}%"
say "  on switch: micro walks '$MICROS_4G', gradient_checkpointing=$GRAD_CKPT_4G, ray port $NEW_PORT"
say "  poll ${POLL_S}s, give up after ${MAX_WAIT_H}h; log $LOG"

DEADLINE=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
ROUND=0

while :; do
  ROUND=$(( ROUND + 1 ))
  (( $(date +%s) < DEADLINE )) || { say "GIVING UP after ${MAX_WAIT_H}h -- no window ever opened"; exit 3; }

  discover; drc=$?
  (( drc == 2 )) && exit 4

  # FINISHED? Checked before anything else, and on the highest step DIRECTORY rather than the
  # highest complete one: if the last step is mid-save the run is busy finishing, and if it is
  # done the run is in its convert step. Either way there is nothing to upgrade and killing it
  # would throw away the ending.
  top=$(max_step)
  if [ -n "$top" ] && (( top >= STEPS )); then
    say "the run has already finished all $STEPS steps (global_step_$top on disk) -- nothing to"
    say "upgrade. Leaving it alone; if it is still running it is converting."
    exit 0
  fi

  if [ -z "$JOB_PID" ]; then
    say "REFUSING: no run to supervise -- nothing matches JOB_PAT ($JOB_PAT) and the checkpoints"
    say "          under $CKPT_DIR stop at step ${top:-<none>} of $STEPS, so it did not finish."
    say "          Not starting anything: a supervisor that launches a run nobody asked for is"
    say "          not a supervisor. Check whether the run died and why."
    exit 4
  fi

  ours=$(ours_gpus) || {
    say "REFUSING: found the run (pid $JOB_PID) but cannot tell which cards it holds -- no GPUS"
    say "          in its environment and OURS_GPUS not set. Refusing to guess."
    exit 4; }

  # NOTHING TO RESUME FROM is a reason to WAIT, not a reason to fire. Killing a run that has
  # saved nothing loses every step it has done and gains a faster restart from zero.
  if ! resume=$(newest_complete); then
    say "round=$ROUND holding: nothing to resume from yet under $CKPT_DIR (steps on disk: [$(ckpt_steps | paste -sd, -)])"
    sleep "$POLL_S"; continue
  fi

  if ! sel=$(pick_window "$ours"); then
    say "round=$ROUND no window (ours=$ours, need $NEED_GPUS) [$WINDOW_STATUS]"
    sleep "$POLL_S"; continue
  fi

  # The volume moves under us while we wait, and the relaunch writes checkpoints of its own.
  # Re-check here, at the moment of firing, not at startup.
  free=$(df -BG --output=avail "$DISK_PATH" 2>/dev/null | tail -1 | tr -dc '0-9')
  if (( ${free:-0} < NEED_GB )); then
    say "round=$ROUND HOLDING: window is open but only ${free:-0}G free on $DISK_PATH, need >=${NEED_GB}G."
    say "          Not switching: the relaunch would fill a shared volume. Free space."
    sleep "$POLL_S"; continue
  fi

  # LAST LOOK BEFORE THE KILL. Minutes may have passed probing the cards, and save_interval=1
  # means a new save can have started in that time. Re-derive the resume target from scratch:
  # if the newest is now mid-write, this drops back to the one under it rather than firing at a
  # directory that was complete when we looked and is being overwritten now.
  resume=$(newest_complete) || {
    say "round=$ROUND stand down: the checkpoint we were going to resume from is no longer complete"
    sleep "$POLL_S"; continue; }

  say "WINDOW OPEN round=$ROUND -> $sel (ours=$ours) [$WINDOW_STATUS]"
  say "SWITCHING: resume_dir=$resume  disk=${free}G"
  say "  job pid=$JOB_PID  auto pid=${AUTO_PID:-<none>}  ray=${RAY_TMPDIR_FOUND:-<none>}"

  OFFSET=$(stat -c %s "$DRIVER_LOG" 2>/dev/null || echo 0)

  if ! stop_the_run; then
    say "REFUSING to launch: the old run did not stop cleanly. Nothing has been started."
    exit 5
  fi

  if ! cards_released "$sel"; then
    say "REFUSING to launch: the cards were not released within ${REL_WAIT_S}s after the kill."
    say "          Something is still holding GPU memory -- a survived worker, or a tenant who"
    say "          took the window while we were killing. Launching now would OOM or co-run."
    say "          The old run is already stopped; resume it by hand with RESUME_DIR=$resume"
    exit 5
  fi
  say "  cards released: $sel"

  relaunch "$sel" "$resume"
  if verify_resumed "$OFFSET" "$resume"; then
    say "UPGRADE DONE: $ours -> $sel, resumed from $(basename "$resume")"
    exit 0
  fi
  exit 6
done
