#!/bin/bash
# opd_mt4slot_auto.sh — run R1, falling back to a smaller micro-batch if it runs out of memory.
#
# WHY THIS EXISTS. micro_batch_size=8 is mt4's value, so running it is ZERO deviation from the
# control and it is the faster of the two. But R1's first attempt died of CUDA OOM inside the
# student forward with 36 MiB free on an 80 GB card, and whether 8 fits depends on how much of
# each card a tenant is holding at the moment we get it -- which nobody can know in advance.
#
# So: try 8, and only if that specific failure happens, retry at 4. Halving micro does NOT change
# the training math (global_batch_size stays 192, so the optimizer sees the identical batch); it
# only doubles the number of gradient-accumulation steps. Note the config comment records that
# micro was deliberately raised 4 -> 8 once before, "fewer grad-accum steps, fills the ~40G VRAM",
# so 4 is a known-good value that costs speed, not correctness.
#
# The fallback fires ONLY on an out-of-memory failure. Any other non-zero exit is a real problem
# and must surface as one rather than being retried into a different configuration -- retrying a
# crash with new settings is how a bug becomes a mystery.
set -uo pipefail

O=/share/fanruochen-local/outputs
LOG="${LOG:-$O/opd_mt4slot_driver.log}"
TAG="${TAG:-mt4slot}"
STEPS="${STEPS:-15}"
SAVE_INTERVAL="${SAVE_INTERVAL:-3}"    # mt4 used 5; 3 bounds what a crash can cost to ~2 steps
                                       # and is experimentally inert -- it changes nothing but
                                       # how often weights are written.
MICROS="${MICROS:-8 4}"                # tried in order, on OOM only

: "${GPUS:?GPUS must be set (the waiter exports it)}"

for m in $MICROS; do
  echo "======== R1 attempt: micro_batch_size=$m  gpus=$GPUS  steps=$STEPS  $(date '+%F %T') ========"
  MICRO="$m" TAG="$TAG" STEPS="$STEPS" SAVE_INTERVAL="$SAVE_INTERVAL" GPUS="$GPUS" \
    SR_PROC_MAX=1000 bash /share/fanruochen-local/dev/scripts/safe_run.sh "$LOG" \
      bash /home/fanruochen/CL/RLinf/examples/embodiment/incremental_sft/opd_mt4slot.sh
  rc=$?
  if (( rc == 0 )); then
    echo "R1_DONE micro=$m rc=0 $(date '+%F %T')"
    exit 0
  fi

  # Was it memory? Look only at this attempt's tail so an OOM from an earlier attempt in the same
  # log cannot make a different failure look like one.
  if tail -n 4000 "$LOG" | grep -qE "CUDA out of memory|OutOfMemoryError"; then
    echo "R1 attempt at micro=$m ran OUT OF MEMORY (rc=$rc) -- falling back to the next size"
    continue
  fi

  echo "R1 attempt at micro=$m failed with rc=$rc and it was NOT an OOM -- stopping."
  echo "Retrying a non-memory crash under different settings would only hide it."
  exit "$rc"
done

echo "R1 exhausted every micro_batch_size in '$MICROS' on OOM. The cards we were given are too"
echo "full for this configuration; wait for cleaner cards or add one."
exit 1
