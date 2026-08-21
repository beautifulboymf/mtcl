#!/bin/bash
# opd_mt4slot_profile.sh — 2-GPU PROFILING run: where does a training step actually spend
# its time?
#
# THE QUESTION. The R1 slot-LoRI run at global_batch_size 192 / micro_batch_size 2 on two
# ranks takes over three hours per training step, and until now run_training printed nothing
# between step boundaries -- no way to tell slow from hung, no way to say which of the five
# things a micro-batch does is the expensive one. This script turns on
# algorithm.profile_train_phases and runs the SMALLEST configuration that still executes the
# same code path, so the breakdown comes out in minutes instead of hours.
#
# WHAT IT MEASURES, per micro-batch, accumulated per optimizer update and per step:
#   prep         batch -> device, before_micro_batch, slot routing (_route_prepare)
#   student_fwd  self.model(...)
#   teacher_fwd  self._teacher_forward(...)  -- the frozen routed expert
#   loss         advantage + KL/anchor construction, up to the backward
#   backward     grad_scaler.scale(loss).backward()   (re-runs the forward under GRAD_CKPT)
#   empty_cache  the per-update cache flush + the slot lr alternation
#   optim        optimizer_step()
# plus `unaccounted`, the wall time the phases do NOT explain -- if that is large, the
# breakdown is lying and should not be acted on.
#
# WHAT IS KEPT IDENTICAL TO THE REAL RUN, because it is what makes the real run slow:
#   * MICRO=2 and GRAD_CKPT=True. Gradient checkpointing re-runs the forward inside
#     backward(), which is precisely the attribution question, and micro_batch_size 2 is
#     what turns one update into dozens of micro-batches.
#   * the same config, the same student, the same routed 4-teacher setup.
# What is shrunk: envs, episode length, global_batch_size and step count. Those change how
# MANY micro-batches run, not what one of them costs.
#
# READ THE LAST STEP, NOT THE FIRST. Steps 1-2 of this pipeline are compile/warm-up
# dominated -- this project measured 86.6 and 77.9 minutes for steps 1-2 against 14.4 for
# steady state and burned a day misdiagnosing it as a sync bug. STEPS defaults to 3 so there
# is one honest step to read.
#
# THE PROFILING IS DIAGNOSTIC-ONLY AND OFF BY DEFAULT IN THE CONFIG. It reads back CUDA
# events, which needs a device synchronize; that is free only because the flush sits right
# after a .item() that already drains the stream every micro-batch. Do not turn it on for a
# production run just because it looked cheap here.
#
# NOTHING IS CHECKPOINTED: save_interval > max_steps, and val_check_interval is -1 in the
# config so no eval-driven `best` save fires either.
#
# Run it under the watchdog, never bare:
#   GPUS=0,1 bash /share/fanruochen-local/dev/scripts/safe_run.sh \
#     /share/fanruochen-local/outputs/opd_mt4slot_profile.log \
#     bash examples/embodiment/incremental_sft/opd_mt4slot_profile.sh
# then:
#   grep -F '[prof]' /share/fanruochen-local/outputs/opd_mt4slot_profile.log
set -uo pipefail

TAG="${TAG:-mt4slotprof}"
GPUS="${GPUS:-0,1}"
PORT="${PORT:-60500}"                 # >=1000 from the R1 port (58000) and the mini one (59500),
                                      # and PORT+300 <= 65535 as run_iso.sh requires
STEPS="${STEPS:-3}"                   # 1-2 are warm-up; the 3rd is the one to read
ENVS="${ENVS:-8}"                     # 8 / 2 ranks = 4 per rank, divisible by group_size 4
EP_STEPS="${EP_STEPS:-128}"           # short episodes: the rollout is not what is being profiled
GLOBAL_BATCH="${GLOBAL_BATCH:-32}"    # 32 / (micro 2 * 2 ranks) = 8 micro-batches per update
MICRO="${MICRO:-2}"                   # the real run's value -- do not "optimise" it here
GRAD_CKPT="${GRAD_CKPT:-True}"        # the real run's value; it is half the question
PROFILE_EVERY="${PROFILE_EVERY:-0}"   # 0 = auto, ~10 progress lines per optimizer update
MON_INT="${MON_INT:-10}"

O=/share/fanruochen-local/outputs
STUDENT="${STUDENT:-$O/inc_sft_opd/lwf_long_e1000_merged}"
REPO=/home/fanruochen/CL/RLinf
CFG=libero_mt4slot_6gpu
SCRIPTS=/share/fanruochen-local/dev/scripts
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
MON="${MON:-$O/opd_mt4slot_profile.mon}"

[ -f "$STUDENT/model.safetensors.index.json" ] || {
  echo "ABORT: student is not an HF model dir: $STUDENT"; exit 1; }

# ---- preflight -------------------------------------------------------------------------------
# Disk: nothing large is written, but ray spill and logs still need room, and this volume is the
# one that has actually taken the machine down (a full /share/fanruochen-local stopped every
# tenant's writes for days).
free=$(df -BG --output=avail /share/fanruochen-local | tail -1 | tr -dc '0-9')
(( free >= 50 )) || { echo "ABORT: only ${free}G free on /share/fanruochen-local"; exit 1; }

# Host memory: a CPU-snapshotting weight syncer once pushed the container cgroup past 477G and
# the kernel OOM-killed the actor. Count ANON only: page cache is reclaimable and memory.current
# is mostly cache on this box.
anon=$(awk '/^anon /{a=$2} /^slab /{s=$2} /^kernel_stack /{k=$2} END{printf "%.0f",(a+s+k)/1073741824}' /sys/fs/cgroup/memory.stat)
(( anon < 200 )) || { echo "ABORT: cgroup anon already ${anon}G before starting"; exit 1; }

# GPUs: memory AND utilization, three samples with a gap. A memory-only check is how this
# project once co-launched on top of another tenant's compute-bound job and ran on it for a
# quarter of an hour; a single instantaneous utilization reading catches idle ticks between
# kernels and reads as free.
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits) || {
    echo "ABORT: cannot query GPU$g"; exit 1; }
  (( used < 5000 )) || { echo "ABORT: GPU$g holds ${used} MiB -- not ours to take"; exit 1; }
  busy=0; last=0
  for _ in 1 2 3; do
    last=$(nvidia-smi -i "$g" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    (( last < 20 )) || busy=$((busy+1))
    sleep 2
  done
  if (( busy >= 2 )); then
    # ALLOW_BUSY_GPU exists so that co-running on a tenant's card is always a VISIBLE, typed
    # decision that stays in the shell history -- never a quietly relaxed threshold. Defensible
    # for a short profiling run on a card with plenty of free memory; NOT defensible for the
    # multi-hour R1 run, whose launcher has no such flag on purpose.
    #
    # BUT NOTE WHAT IT COSTS *THIS* SCRIPT SPECIFICALLY: sharing a card with another tenant
    # makes the absolute per-phase times meaningless, because your kernels queue behind theirs.
    # The phase SHARES stay roughly informative; treat every duration as an upper bound and say
    # so when reporting the result.
    if [ "${ALLOW_BUSY_GPU:-0}" = "1" ]; then
      echo "WARN: GPU$g at ${last}% util and ALLOW_BUSY_GPU=1 -- co-running on another tenant's"
      echo "      card. The per-phase SHARES are still readable; the absolute times are NOT."
    else
      echo "ABORT: GPU$g at ${last}% util -- not ours to take (set ALLOW_BUSY_GPU=1 to override)"
      exit 1
    fi
  fi
done

echo "[prof-run] gpus=$GPUS envs=$ENVS ep_steps=$EP_STEPS global_batch=$GLOBAL_BATCH"
echo "[prof-run] micro=$MICRO grad_ckpt=$GRAD_CKPT steps=$STEPS log_every=$PROFILE_EVERY (0=auto)"
echo "[prof-run] micro-batches per optimizer update = $GLOBAL_BATCH / ($MICRO * 2 ranks) = $((GLOBAL_BATCH / (MICRO * 2)))"
echo "[prof-run] disk=${free}G anon=${anon}G  monitor -> $MON"
echo "[prof-run] NO checkpoint will be written (save_interval > max_steps, val_check_interval=-1)"
echo "[prof-run] steps 1-2 are compile/warm-up dominated -- READ THE LAST STEP"

# ---- monitor ---------------------------------------------------------------------------------
# Everything the safety questions need, sampled on a fixed interval:
#   anon_G   cgroup anonymous memory -- the number that OOM-kills, unlike memory.current
#   psi_io   /proc/pressure/io "some avg10" -- the direct measure of IO stall, better than iowait
#   free_G   the shared volume
#   gpu_MiB  per-GPU memory of the cards we took
MAIN_PID=$$
: > "$MON"
(
  echo "# ts anon_G psi_io_avg10 free_G gpu_mem_MiB" >> "$MON"
  while kill -0 "$MAIN_PID" 2>/dev/null; do
    a=$(awk '/^anon /{a=$2} /^slab /{s=$2} /^kernel_stack /{k=$2} END{printf "%.1f",(a+s+k)/1073741824}' /sys/fs/cgroup/memory.stat 2>/dev/null)
    p=$(awk '/^some/{print $2}' /proc/pressure/io 2>/dev/null | head -1 | cut -d= -f2)
    f=$(df -BG --output=avail /share/fanruochen-local 2>/dev/null | tail -1 | tr -dc '0-9')
    m=$(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)
    echo "$(date +%H:%M:%S) ${a:-?} ${p:-?} ${f:-?} ${m:-?}" >> "$MON"
    sleep "$MON_INT"
  done
) &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null' EXIT

# ---- environment -----------------------------------------------------------------------------
# Same three sources, in the same order, as every other launcher here. gpu_render_env.sh must
# come after .rlinf-env.sh, which forces MUJOCO_GL=osmesa and would otherwise leave every LIBERO
# rollout rendering on the CPU.
set +u
source /home/fanruochen/.rlinf-env.sh 2>/dev/null
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
source /share/fanruochen-local/dev/gpu_render_env.sh
set -u
export EMBODIED_PATH="$REPO/examples/embodiment" REPO_PATH="$REPO"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$REPO:${PYTHONPATH:-}"
export RLINF_CONVERT_VALUE_HEAD=False
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- run -------------------------------------------------------------------------------------
# All geometry is a hydra override here and ONLY here: the config keeps mt4's values so the real
# R1 run stays a clean control, and a profiling run must not quietly edit that record.
# `++` on the two profiling keys because they are absent from every config in the repo -- that
# absence is the guarantee that the production run is unaffected.
MT4SLOT_GPUS="$GPUS" MT4SLOT_TAG="$TAG" MT4SLOT_STUDENT_PATH="$STUDENT" \
MT4SLOT_MAX_STEPS="$STEPS" MT4SLOT_SAVE_INTERVAL=999 \
ISO_RAY_PORT="$PORT" bash "$SCRIPTS/run_iso.sh" \
  "$PY" "$REPO/examples/embodiment/train_embodied_agent.py" --config-name "$CFG" \
    env.train.total_num_envs="$ENVS" \
    env.train.max_episode_steps="$EP_STEPS" \
    env.train.max_steps_per_rollout_epoch="$EP_STEPS" \
    actor.global_batch_size="$GLOBAL_BATCH" \
    actor.micro_batch_size="$MICRO" \
    ++actor.fsdp_config.gradient_checkpointing="$GRAD_CKPT" \
    ++algorithm.profile_train_phases=true \
    ++algorithm.profile_log_every="$PROFILE_EVERY" \
    algorithm.rollout_epoch=1
RC=$?
kill "$MON_PID" 2>/dev/null; trap - EXIT

echo "PROFILE_DONE rc=$RC $(date '+%F %T')"
echo "== peak anon / peak IO pressure / min free / peak GPU mem =="
awk 'NR>1{if($2+0>ma)ma=$2+0; if($3+0>mp)mp=$3+0; if(mf==0||$4+0<mf)mf=$4+0}
     END{printf "anon_peak=%.1fG  psi_io_peak=%.1f  free_min=%dG  samples=%d\n", ma, mp, mf, NR-1}' "$MON"
echo "== per-GPU peak =="
awk 'NR>1{n=split($5,g,","); for(i=1;i<=n;i++) if(g[i]+0>p[i]) p[i]=g[i]+0}
     END{for(i=1;i<=n;i++) printf "  gpu[%d] peak %d MiB\n", i-1, p[i]}' "$MON"
echo "== the profile: grep -F '[prof]' <this log> =="
echo "   per-step summaries only:  grep -F 'DONE |' <this log> | grep -F '[prof]'"
echo "   READ THE LAST STEP: steps 1-2 are compile/warm-up dominated."
echo "   Each rank logs its own lines, tagged [prof][r0] / [prof][r1] -- a rank that is"
echo "   consistently behind is a straggler, not noise."
exit "$RC"
