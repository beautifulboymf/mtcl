#!/bin/bash
# run_mt4w2.sh — plan A relaunch of the KL-weighted 4-teacher OPD.
#
# WHY THIS EXISTS: the first attempt (tag mt4w) computed the per-suite distillation weights
# inside the per-micro-batch path. Every `.any()` / boolean-mask index there forces a device
# sync, the CPU could no longer run ahead of the GPU, and run_training measured 80.0 min
# against a 14.3 min baseline on a byte-identical config -- 5.6x, one step in 89.5 min.
# The weights now come from _dw_refresh_weights(), called once per training step; the hot
# path only gathers a precomputed tensor. Step 1 runs uniform (no KL accumulated yet),
# adaptive from step 2 on.
#
# Steps cut 15 -> 8 so this is ~2.7h on 6 GPUs instead of 22h.
set -uo pipefail
GPUS="${GPUS:-0,1,2,3,4,5}"
STEPS="${STEPS:-8}"
TAG="${TAG:-mt4w2}"
PORT="${PORT:-62000}"
O=/share/fanruochen-local/outputs
LOG="$O/opd_${TAG}_driver.log"

# refuse to start on cards someone else is already on -- never evict a tenant, pick other cards
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
  (( used < 5000 )) || { echo "ABORT: GPU$g already holds ${used} MiB -- not ours to take"; exit 1; }
done
echo "[launch] gpus=$GPUS steps=$STEPS tag=$TAG port=$PORT log=$LOG"

cd /home/fanruochen/CL/RLinf
SR_PROC_MAX=1000 exec bash /share/fanruochen-local/dev/scripts/safe_run2.sh "$LOG" \
  env OPD_STUDENT="$O/seqcl_mt4/converted/mt4" \
      OPD_GPUS="$GPUS" OPD_ENVS=48 OPD_ROLLOUT_EPOCH=3 OPD_MICRO=8 \
      OPD_GLOBAL_BATCH=192 OPD_RAY_PORT="$PORT" \
      OPD_EXTRA="+algorithm.distill_dyn_weight=1.0 +algorithm.distill_w_min=0.25 +algorithm.distill_w_max=4.0 +algorithm.distill_w_ema=0.9" \
    bash examples/embodiment/incremental_sft/opd_4teacher.sh "$TAG" "$STEPS"
