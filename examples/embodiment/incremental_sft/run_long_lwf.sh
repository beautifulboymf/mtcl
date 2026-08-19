#!/bin/bash
# run_long_lwf.sh — stage 4 of the sequential CL chain: give the 3-suite student `long` (libero_10).
#
#   init/student = opd_init_dual1250   (BEST 3-suite model we have: spatial .86 / object .94 /
#                                       goal .88 @temp1.0, avg .893 -- higher than all three OPD
#                                       students fo3/.787 diag/.773 mt3b/.773)
#   new task     = libero_10_no_noops   ground-truth LoRA-SFT
#   anchor       = the SAME model, frozen, forward-KL on spatial+object+goal demo states
#                  (off-policy / offline distillation = LwF). All three suites MUST be listed:
#                  when the goal stage anchored spatial only, object fell 0.90 -> 0.02.
#   lambda 0.3   = the validated knob (lambda=0 forgets, lambda=1 won't learn).
#
# WHY 5000 STEPS. Aligned to RLinf/OpenVLA-OFT, which train on the FULL trajectory set
# (their published long SFT is `Openvla-oft-SFT-libero10-trajall`; the `-traj1` variant in every
# RLinf config is a deliberately WEAK 1-demo-per-task init -- its object twin measures SR 0.10).
# libero_10 is the biggest suite by transitions and the longest by episode:
#     spatial 432 traj / 52,970 trans / avg 123      object 454 / 66,984 / avg 148
#     long    379 traj / 101,469 trans / avg 268
# Our failed from-scratch long SFT ran 1500 steps x eff_batch 64 = 96k samples = 0.95 EPOCH and
# scored 0.04; the object run that reached 1.00 got 2.87 epochs. 5000 steps = 3.15 epochs here.
#
# CHECKPOINTS: SFT_ADAPTER_ONLY=1 -> every save writes ONLY the LoRA adapter (~0.2G) to
# <out>/<exp>/adapters/step_N, never a merged 15G model. Resume from any of them with
#     SFT_INIT_ADAPTER=<...>/adapters/step_N  (finetune_lwf.py honours it; optimizer state resets)
# which also lets this restart on MORE gpus if cards free up.
#
# Usage: run_long_lwf.sh [gpus=6] [max_steps=5000] [lambda=0.3]
set -o pipefail
GPUS="${1:-6}"; MAXSTEPS="${2:-5000}"; LAMBDA="${3:-0.3}"

REPO=/home/fanruochen/CL/RLinf
SCRIPTS=/share/fanruochen-local/dev/scripts
STUDENT=/share/fanruochen-local/outputs/inc_sft_opd/opd_init_dual1250
OUT=/share/fanruochen-local/outputs/inc_sft_opd/lwf_long_from3suite
LOG=/share/fanruochen-local/outputs/inc_sft_opd/lwf_long_driver.log

# RESUMING ON A DIFFERENT GPU COUNT -- two things bite, both handled here via RESUME_FROM_STEP:
#  1. SFT_INIT_ADAPTER restores the LoRA weights but NOT the step counter; finetune_lwf.py always
#     counts from 0. So max_steps must be the REMAINING steps, not the target total.
#  2. exp_id embeds b{batch_size*grad_accum}, so 2 gpus write to +b32+ and 4 gpus to +b16+. Segments
#     would silently interleave in one tree with step_N numbering restarting at 500 each time.
# Set RESUME_FROM_STEP=<already-completed steps> together with SFT_INIT_ADAPTER: the run gets its
# own <out>_r<N> tree and max_steps is reduced to the remainder, so `step_N` inside a segment always
# means "effective step RESUME_FROM_STEP + N".
RESUME_FROM_STEP="${RESUME_FROM_STEP:-0}"
if [ "$RESUME_FROM_STEP" -gt 0 ]; then
  [ -n "${SFT_INIT_ADAPTER:-}" ] || { echo "ABORT: RESUME_FROM_STEP=$RESUME_FROM_STEP needs SFT_INIT_ADAPTER"; exit 1; }
  [ -f "$SFT_INIT_ADAPTER/adapter_config.json" ] || { echo "ABORT: not an adapter dir: $SFT_INIT_ADAPTER"; exit 1; }
  OUT="${OUT}_r${RESUME_FROM_STEP}"
  LOG="${LOG%.log}_r${RESUME_FROM_STEP}.log"
  MAXSTEPS=$(( MAXSTEPS - RESUME_FROM_STEP ))
  [ "$MAXSTEPS" -gt 0 ] || { echo "ABORT: nothing left to train (remaining=$MAXSTEPS)"; exit 1; }
  echo "[run_long_lwf] RESUME from step $RESUME_FROM_STEP -> $MAXSTEPS remaining steps, out=$OUT"
fi

export LWF_NEW_DSET=libero_10_no_noops
export LWF_ANCHOR_DSET=libero_spatial_no_noops,libero_object_no_noops,libero_goal_no_noops
export SFT_ADAPTER_ONLY=1        # save LoRA adapters only -- do NOT merge
export SFT_SAVE_LATEST=False     # keep EVERY save_steps checkpoint, not just the last
export SFT_INIT_ADAPTER="${SFT_INIT_ADAPTER:-}"   # set to .../adapters/step_N to resume
# Each rank holds ~74.4 GiB on an 80 GiB card, so the margin is thin and TWO things eat into it as
# the rank count grows: (a) every OTHER rank leaves a ~898 MiB NCCL peer buffer on this card, so
# 4 ranks cost 3 x 898 MiB = 2.6 GiB before training starts (2 ranks cost only 0.9 GiB), and
# (b) any co-tenant on the card. That combination OOM'd the first 4-GPU attempt on GPU0 at 2026-08-18
# 00:10 ("Tried to allocate 340.00 MiB ... 137.62 MiB is free"), where the tenants held 1.94 GiB.
# expandable_segments reclaims the allocator's fragmented reserve -- that run reported 1.86 GiB
# "reserved but unallocated", i.e. 5x the allocation that failed. It changes no training math.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[ -f "$STUDENT/model.safetensors.index.json" ] || { echo "ABORT: student missing: $STUDENT"; exit 1; }
mkdir -p "$OUT"

# ---- RED-LINE PRE-FLIGHT: refuse to start rather than trip the watchdog (or the kernel) mid-run ----
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)

# DISK. Adapters are ~0.5G each (measured 463M, not the 0.2G I first guessed).
avail=$(df -BG --output=avail /share/fanruochen-local | tail -1 | tr -dc '0-9')
[ "$avail" -lt 100 ] && { echo "ABORT: only ${avail}G free on /share/fanruochen-local"; exit 1; }

# MEMORY. Each rank holds student + frozen teacher + FOUR RLDS pipelines whose shuffle buffers are
# 100k elements each. MEASURED: 1 rank = 37G RSS; 2 ranks = 83.4G cgroup anon => ~42G per rank, and
# it scales linearly. Project it against the container cap and refuse NOW: a container OOM kills
# every rank hours into the run, and (see the 2026-06-13 incident) memory/disk blowups here are the
# failures that cost days, not minutes.
cgmax=$(awk '{print int($1/1073741824)}' /sys/fs/cgroup/memory.max 2>/dev/null || echo 476)
cgnow=$(awk '/^anon /{a=$2} /^slab /{s=$2} /^kernel_stack /{k=$2} END{printf "%.0f",(a+s+k)/1073741824}' /sys/fs/cgroup/memory.stat)
proj=$(( 42 * NPROC ))
if [ "$proj" -gt $(( cgmax * 60 / 100 )) ]; then
  echo "ABORT: projected ${proj}G anon (42G x ${NPROC} ranks) > 60% of the ${cgmax}G container cap." >&2
  echo "       Use fewer ranks, or shrink the 100k-element shuffle buffers (that is what costs RAM)." >&2
  exit 1
fi

# IO. Never pile a heavy job onto an already-jammed disk -- an IO jam locks everyone out of the box.
iow=$(vmstat 1 2 | tail -1 | awk '{print $16}')
if [ "${iow:-0}" -ge 30 ]; then
  echo "ABORT: system iowait ${iow}% >= 30%, disk already busy -- wait for it to clear"; exit 1
fi

echo "[run_long_lwf] PREFLIGHT ok | disk=${avail}G | anon now=${cgnow}G proj=${proj}G cap=${cgmax}G | iowait=${iow}%"
echo "[run_long_lwf] gpus=$GPUS(np$NPROC) steps=$MAXSTEPS lambda=$LAMBDA resume='${SFT_INIT_ADAPTER:-none}'"

# safe_run's default CPU ceiling is 5000% (50 cores) sustained. ONE rank of this job measured
# 1541% (15.4 cores) -- almost all of it the num_workers=0 RLDS decode+augment pipelines, which
# are what starve the GPU (util 34%, iowait 0). N ranks scale that ~linearly, so 4 ranks would sit
# near the default ceiling and the watchdog would kill our OWN job. Raise it in proportion to the
# rank count; still far below a runaway (the 500-env crash was ~5-10x legit). NPROC is already
# computed in the pre-flight block above.
export SR_CPU_MAX="${SR_CPU_MAX:-$(( 2200 * NPROC + 1500 ))}"

exec bash "$SCRIPTS/safe_run.sh" "$LOG" \
  bash "$REPO/examples/embodiment/incremental_sft/incremental_sft_lwf.sh" \
    "$STUDENT" "$STUDENT" "$OUT" "$LAMBDA" "$MAXSTEPS" 500 "$GPUS"
