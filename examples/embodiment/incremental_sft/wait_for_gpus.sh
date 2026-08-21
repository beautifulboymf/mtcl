#!/bin/bash
# wait_for_gpus.sh — block until enough GPUs are genuinely idle, then run a command with GPUS= set.
#
#   wait_for_gpus.sh <command> [args...]
#
# WHY THIS EXISTS. Every card on this box belongs to someone at any given moment, and the useful
# window opens without warning when a tenant's job ends. Sitting on `watch nvidia-smi` wastes the
# window; launching on top of a tenant wastes their compute for hours. This waits, checks, and
# only then hands over -- and it re-checks EVERY red line at the moment of launch, not once at the
# start of the wait, because a volume that had room when the wait began may not when it ends.
#
# WHAT "IDLE" MEANS HERE, and why it is not just memory. This project once co-launched on top of
# another tenant's compute-bound job that showed under 2% memory and 73-97% utilization, and ran
# that way for a quarter of an hour. So a card counts as idle only if BOTH hold:
#   * memory.used < MEM_MAX_MIB
#   * at least UTIL_OK_N of UTIL_N utilization samples are under UTIL_MAX_PCT
# Utilization is sampled repeatedly on purpose: one instantaneous reading lands in the gap between
# kernels as often as not, and reads as free.
#
# CARD SELECTION.
#   NEED_ALL   every one of these must be idle. Default 5,6,7.
#   PICK_FROM  the remaining slots are filled from here, least-loaded first, and each candidate
#              must pass the SAME idle test -- "least loaded" is a tie-break among idle cards, not
#              a licence to take a busy one. Default 0,1,2,3,4.
#   PICK_N     how many to take from PICK_FROM. Default 1, i.e. four cards total.
# Note GPU 4-7 are frequently one four-GPU job, so when it ends GPU4 usually becomes the least
# loaded card in PICK_FROM and gets chosen without anyone having to intervene.
#
# The command is run with GPUS=<csv> in its environment. Everything else about it is the
# command's own business -- this script does not know or care whether it is a smoke test or
# the real run.
#
# TESTING. Every input to a decision comes from a command or a path named by a variable:
# NVIDIA_SMI (the sampler), DISK_PATH and CGROUP_MEM_STAT (the launch gates). All three
# default to the real thing, so production behaviour is unchanged and a human still runs
# this with no arguments; tests/unit_tests/test_wait_for_gpus.py points them at fakes so the
# idle test and the card selection can be exercised on a box where every card is busy.
set -uo pipefail

NEED_ALL="${NEED_ALL-5,6,7}"   # no colon: an explicit empty string means "nothing is mandatory"
PICK_FROM="${PICK_FROM:-0,1,2,3,4}"
PICK_N="${PICK_N:-1}"
# FORCE_GPUS: cards taken UNCONDITIONALLY, with no idle test. For the case where a card carries a
# known, small, long-lived tenant process the operator has decided to co-run with -- a judgement
# only a human can make, so it must be typed on the command line, and every reading is logged.
# Never put a card here to work around a failing idle test.
FORCE_GPUS="${FORCE_GPUS-}"
# EXTRA_N: after PICK_N genuinely idle cards are found, top the selection up with this many MORE
# cards chosen as the least-occupied of whatever is left -- WITHOUT requiring them to be idle.
# The case it serves: a job needs four cards but only three are ever free at once, and a fourth
# carrying a small long-lived tenant is a better trade than not running. Ranked the same way as
# the idle candidates (memory bucket first, then utilization), and every pick is logged with its
# actual readings so a bad trade is visible rather than silent.
EXTRA_N="${EXTRA_N:-0}"
MEM_MAX_MIB="${MEM_MAX_MIB:-5000}"
UTIL_MAX_PCT="${UTIL_MAX_PCT:-20}"
UTIL_N="${UTIL_N:-5}"
UTIL_OK_N="${UTIL_OK_N:-4}"
UTIL_SLEEP_S="${UTIL_SLEEP_S:-1}"          # gap between utilization samples
MEM_BUCKET_MIB="${MEM_BUCKET_MIB:-4096}"   # candidates within one bucket are ranked by utilization
POLL_S="${POLL_S:-120}"
MAX_WAIT_H="${MAX_WAIT_H:-12}"
NEED_GB="${NEED_GB:-175}"
ANON_MAX_G="${ANON_MAX_G:-200}"
RETRY_IF_FAST_S="${RETRY_IF_FAST_S:-180}"  # a target that dies faster than this never started real work
DISK_PATH="${DISK_PATH:-/share/fanruochen-local}"
CGROUP_MEM_STAT="${CGROUP_MEM_STAT:-/sys/fs/cgroup/memory.stat}"
LOG="${WAIT_LOG:-/share/fanruochen-local/outputs/wait_for_gpus.log}"
LOCK="${WAIT_LOCK:-/tmp/wait_for_gpus.lock}"

# The ONLY thing in here that touches the cards, so it is the only seam a test needs. Must
# be a single executable that answers `-i <n> --query-gpu=<field> --format=csv,noheader,nounits`.
NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"

(( $# >= 1 )) || { echo "usage: wait_for_gpus.sh <command> [args...]"; exit 2; }
if [ -z "${NEED_ALL//,/}" ] && (( PICK_N <= 0 )); then
  echo "nothing to select: NEED_ALL is empty and PICK_N=$PICK_N -- there is no such thing as a window"; exit 2
fi

# Single instance. Two waiters would both fire into the same window and fight over the same cards.
exec 9>"$LOCK" || { echo "cannot open lock $LOCK"; exit 1; }
flock -n 9 || { echo "another wait_for_gpus.sh already holds $LOCK -- refusing to start a second"; exit 1; }

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Returns 0 if the card is idle by both measures. Echoes "mem util_mean n_busy" on stdout.
probe_gpu() {
  local g="$1" mem util busy=0 sum=0 i
  mem=$("$NVIDIA_SMI" -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || return 2
  # A card that answers "[N/A]" (MIG, vGPU, a driver hiccup) is not a card we understand,
  # and an unparsable reading must never be arithmetic-error its way into looking free.
  [[ "$mem" =~ ^[0-9]+$ ]] || return 2
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

say "waiting: NEED_ALL=$NEED_ALL  PICK_FROM=$PICK_FROM  PICK_N=$PICK_N"
say "idle test: mem<${MEM_MAX_MIB}MiB AND >=${UTIL_OK_N}/${UTIL_N} samples under ${UTIL_MAX_PCT}%"
say "gates at launch: disk>=${NEED_GB}G, cgroup anon<${ANON_MAX_G}G;  poll ${POLL_S}s, give up after ${MAX_WAIT_H}h"
say "target: $*"

DEADLINE=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
ROUND=0

while :; do
  ROUND=$(( ROUND + 1 ))
  (( $(date +%s) < DEADLINE )) || { say "GIVING UP after ${MAX_WAIT_H}h -- window never opened"; exit 3; }

  status=""; all_ok=1
  for g in ${NEED_ALL//,/ }; do
    # COMMAND substitution, not process substitution: `read < <(f); rc=$?` captures READ's
    # status, not f's, so the idle verdict was silently discarded and every card counted as
    # idle -- observed firing on cards at 100% utilization.
    out=$(probe_gpu "$g"); rc=$?
    read -r mem umean busy <<< "$out"
    status+="$g:${mem}MiB/${umean}% "
    (( rc == 0 )) || all_ok=0
  done

  chosen=""; ready=0
  if (( all_ok == 1 )); then
    # Rank the candidates by memory, then by mean utilization, and keep only the idle ones.
    cands=""
    for g in ${PICK_FROM//,/ }; do
      case ",$NEED_ALL," in *",$g,"*) continue ;; esac
      out=$(probe_gpu "$g"); rc=$?
      read -r mem umean busy <<< "$out"
      status+="($g:${mem}MiB/${umean}%)"
      # Bucket the memory before ranking. On this box every candidate carries the same ~1960 MiB
      # tenant footprint, so raw memory order is decided by a few MiB of noise and would happily
      # pick a card at 69% utilization over one at 41%. Memory picks the tier; utilization picks
      # within it -- which is the right way round, since compute is what we actually contend for.
      (( rc == 0 )) && cands+="$(( mem / MEM_BUCKET_MIB )) $umean $mem $g"$'\n'
    done
    n_ok=$(printf '%s' "$cands" | grep -c . || true)
    if (( n_ok >= PICK_N )); then
      # The window is open because the CARDS say so. Do not infer it from `chosen` being
      # non-empty: PICK_N=0 is the legitimate way to ask for exactly the NEED_ALL cards
      # (the 2-GPU smoke test is NEED_ALL=0,1 PICK_N=0), and that selects nothing to pick.
      ready=1
      if (( PICK_N > 0 )); then
        chosen=$(printf '%s' "$cands" | sort -k1,1n -k2,2n | head -n "$PICK_N" | awk '{print $4}' | paste -sd, -)
      fi
    fi
  fi

  extra=""
  if (( ready == 1 && EXTRA_N > 0 )); then
    pool=""
    for g in ${PICK_FROM//,/ }; do
      case ",$NEED_ALL,${chosen:+$chosen,}" in *",$g,"*) continue ;; esac
      out=$(probe_gpu "$g")
      read -r mem umean busy <<< "$out"
      pool+="$(( mem / MEM_BUCKET_MIB )) $umean $mem $g"$'\n'
    done
    n_pool=$(printf '%s' "$pool" | grep -c . || true)
    if (( n_pool >= EXTRA_N )); then
      extra=$(printf '%s' "$pool" | sort -k1,1n -k2,2n | head -n "$EXTRA_N" | awk '{print $4}' | paste -sd, -)
      for g in ${extra//,/ }; do
        read -r em eu _ <<< "$(probe_gpu "$g")"
        say "TOP-UP GPU$g taken as least-occupied, NOT idle-tested: ${em}MiB / ${eu}%"
      done
    else
      say "round=$ROUND idle cards found but only $n_pool card(s) left to top up with, need $EXTRA_N"
      ready=0
    fi
  fi

  if (( ready == 1 )); then
    GPUS_SEL=$(printf '%s\n%s\n%s\n%s\n' "${NEED_ALL//,/$'\n'}" "${chosen//,/$'\n'}" "${FORCE_GPUS//,/$'\n'}" "${extra//,/$'\n'}" | grep -E '^[0-9]+$' | sort -n | uniq | paste -sd, -)
    for g in ${FORCE_GPUS//,/ }; do
      read -r fm fu _ <<< "$(probe_gpu "$g")"
      say "FORCED GPU$g taken with no idle test: ${fm}MiB / ${fu}% -- operator's explicit choice"
    done
    say "WINDOW OPEN round=$ROUND -> GPUS=$GPUS_SEL   [$status]"

    # Re-check the red lines HERE, not at the top. The wait may have been hours; the volume and
    # the container's memory are shared and move underneath us.
    free=$(df -BG --output=avail "$DISK_PATH" 2>/dev/null | tail -1 | tr -dc '0-9')
    if (( ${free:-0} < NEED_GB )); then
      say "HOLDING: only ${free}G free on $DISK_PATH, need >=${NEED_GB}G. Free space; still waiting."
      sleep "$POLL_S"; continue
    fi
    anon=$(awk '/^anon /{a=$2} /^slab /{s=$2} /^kernel_stack /{k=$2} END{printf "%.0f",(a+s+k)/1073741824}' "$CGROUP_MEM_STAT" 2>/dev/null)
    if (( ${anon:-999} >= ANON_MAX_G )); then
      say "HOLDING: cgroup anon already ${anon}G (limit ${ANON_MAX_G}G). Still waiting."
      sleep "$POLL_S"; continue
    fi

    say "LAUNCH  disk=${free}G anon=${anon}G  GPUS=$GPUS_SEL  cmd: $*"
    # Run as a CHILD, not exec. The target re-checks the cards itself (it must -- this waiter is
    # not the only thing that can be wrong), and load can spike in the seconds between the two
    # checks. With exec, one such spike ended a twelve-hour wait. A target that dies faster than
    # RETRY_IF_FAST_S did not get as far as real work, so resume waiting instead of giving up;
    # anything slower than that started for real and its exit code is the run's, not ours.
    t0=$(date +%s)
    GPUS="$GPUS_SEL" "$@"
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if (( rc != 0 && dt < RETRY_IF_FAST_S )); then
      say "target refused after ${dt}s (rc=$rc) -- treating as a lost window, resuming the wait"
      sleep "$POLL_S"; continue
    fi
    say "target finished rc=$rc after ${dt}s"
    exit "$rc"
  fi

  say "round=$ROUND not ready  [$status]"
  sleep "$POLL_S"
done
