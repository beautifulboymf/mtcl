#!/bin/bash
# opd_mt4slot_mini.sh — 2-GPU smoke test for the slot-LoRI R1 pipeline.
#
# WHAT THIS ANSWERS: does the thing start and survive two training steps, and does it jam IO or
# blow up host memory. NOT whether it works -- 8 envs / rollout_epoch 1 / 2 steps produces no
# result worth reading, and rollout_epoch 1 is known to wreck distillation quality (goal fell
# .62 -> .44 on it). Success here means "the integration is sound enough to spend GPU-hours on".
#
# WHY IT IS CHEAP AND SAFE:
#   * SAVE_INTERVAL > MAX_STEPS, so NOTHING is checkpointed. A slot checkpoint is ~36G and the
#     shared volume is at 97%; a smoke test has no business writing one. The save/convert path is
#     therefore NOT covered here -- test it separately, after freeing space.
#   * save_video is already False in the config, so the rollout writes no frames either.
#   * 8 envs / 128-step episodes / 2 steps: minutes, not hours.
#   * A sampler records cgroup anon memory, IO pressure, disk free and GPU memory every 10s to
#     <log>.mon so "did it jam IO / eat RAM" is answered by data instead of by impression.
#
# Run it under the watchdog, never bare:
#   GPUS=0,1 bash /share/fanruochen-local/dev/scripts/safe_run.sh \
#     /share/fanruochen-local/outputs/opd_mt4slot_mini.log \
#     bash examples/embodiment/incremental_sft/opd_mt4slot_mini.sh
set -uo pipefail

TAG="${TAG:-mt4slotmini}"
GPUS="${GPUS:-0,1}"
PORT="${PORT:-59500}"                 # far from the R1 port (58000) and from any serial-cycle run
STEPS="${STEPS:-2}"
ENVS="${ENVS:-8}"                     # 8 / 2 ranks = 4 per rank, divisible by group_size 4
EP_STEPS="${EP_STEPS:-128}"           # short episodes: this is a smoke test, not a rollout study
GLOBAL_BATCH="${GLOBAL_BATCH:-32}"    # 32 / (micro 8 * 2 ranks) = 2 grad-accum steps
MON_INT="${MON_INT:-10}"

O=/share/fanruochen-local/outputs
STUDENT="${STUDENT:-$O/inc_sft_opd/lwf_long_e1000_merged}"
REPO=/home/fanruochen/CL/RLinf
CFG=libero_mt4slot_6gpu
SCRIPTS=/share/fanruochen-local/dev/scripts
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
MON="${MON:-$O/opd_mt4slot_mini.mon}"

[ -f "$STUDENT/model.safetensors.index.json" ] || {
  echo "ABORT: student is not an HF model dir: $STUDENT"; exit 1; }

# ---- preflight -------------------------------------------------------------------------------
# Disk: nothing large is written, but ray spill and logs still need room, and this volume is the
# one that has actually taken the machine down.
free=$(df -BG --output=avail /share/fanruochen-local | tail -1 | tr -dc '0-9')
(( free >= 50 )) || { echo "ABORT: only ${free}G free on /share/fanruochen-local"; exit 1; }

# Host memory: the failure this guards is real -- a CPU-snapshotting weight syncer once pushed the
# container cgroup past 477G and the kernel OOM-killed the actor. bucket_syncer is configured
# instead, but start from a known-clean floor anyway. Count ANON only: page cache is reclaimable
# and memory.current is mostly cache on this box.
anon=$(awk '/^anon /{a=$2} /^slab /{s=$2} /^kernel_stack /{k=$2} END{printf "%.0f",(a+s+k)/1073741824}' /sys/fs/cgroup/memory.stat)
(( anon < 200 )) || { echo "ABORT: cgroup anon already ${anon}G before starting"; exit 1; }

# GPUs: memory AND utilization, three samples. Same rule as the real launcher.
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
    # decision that stays in the shell history -- never a quietly relaxed threshold. It is
    # defensible only for a short smoke test on a card with plenty of free memory; it is NOT
    # defensible for the multi-hour R1 run, whose launcher has no such flag on purpose.
    if [ "${ALLOW_BUSY_GPU:-0}" = "1" ]; then
      echo "WARN: GPU$g at ${last}% util and ALLOW_BUSY_GPU=1 -- co-running on another tenant's"
      echo "      card. Justified only because this test is minutes long and needs ~35G of the"
      echo "      ~78G free there. Do NOT use this flag for the real run."
    else
      echo "ABORT: GPU$g at ${last}% util -- not ours to take (set ALLOW_BUSY_GPU=1 to override)"
      exit 1
    fi
  fi
done

echo "[mini] gpus=$GPUS envs=$ENVS ep_steps=$EP_STEPS global_batch=$GLOBAL_BATCH steps=$STEPS"
echo "[mini] disk=${free}G anon=${anon}G  monitor -> $MON"
echo "[mini] NO checkpoint will be written (save_interval > max_steps)"

# ---- monitor ---------------------------------------------------------------------------------
# Everything the two questions need, sampled on a fixed interval:
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
# The geometry knobs are hydra overrides here and ONLY here: the config keeps mt4's values so the
# real run stays a clean control, and a smoke test must not quietly edit that record.
MT4SLOT_GPUS="$GPUS" MT4SLOT_TAG="$TAG" MT4SLOT_STUDENT_PATH="$STUDENT" \
MT4SLOT_MAX_STEPS="$STEPS" MT4SLOT_SAVE_INTERVAL=999 \
ISO_RAY_PORT="$PORT" bash "$SCRIPTS/run_iso.sh" \
  "$PY" "$REPO/examples/embodiment/train_embodied_agent.py" --config-name "$CFG" \
    env.train.total_num_envs="$ENVS" \
    env.train.max_episode_steps="$EP_STEPS" \
    env.train.max_steps_per_rollout_epoch="$EP_STEPS" \
    actor.global_batch_size="$GLOBAL_BATCH" \
    algorithm.rollout_epoch=1
RC=$?
kill "$MON_PID" 2>/dev/null; trap - EXIT

echo "MINI_DONE rc=$RC $(date '+%F %T')"
echo "== peak anon / peak IO pressure / min free / peak GPU mem =="
awk 'NR>1{if($2+0>ma)ma=$2+0; if($3+0>mp)mp=$3+0; if(mf==0||$4+0<mf)mf=$4+0}
     END{printf "anon_peak=%.1fG  psi_io_peak=%.1f  free_min=%dG  samples=%d\n", ma, mp, mf, NR-1}' "$MON"
echo "== per-GPU peak =="
awk 'NR>1{n=split($5,g,","); for(i=1;i<=n;i++) if(g[i]+0>p[i]) p[i]=g[i]+0}
     END{for(i=1;i<=n;i++) printf "  gpu[%d] peak %d MiB\n", i-1, p[i]}' "$MON"
exit "$RC"
