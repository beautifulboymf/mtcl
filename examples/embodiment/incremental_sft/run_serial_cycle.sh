#!/bin/bash
# run_serial_cycle.sh — SERIAL multi-task OPD: one teacher at a time, a->b->c->d, chained.
#
# WHY SERIAL. The joint form (all four teachers routed inside one batch, their KLs pooled into a
# single scalar, one backward) was run twice and CONSERVES capability without adding any:
#   mt4    long +0.28 / goal -0.20 -> mean 0.745
#   mt4w2  goal +0.12 / long -0.10 -> mean 0.750   (0.12 SE from its 0.745 start = nothing)
# Both times the gain and the loss cancelled, on opposite suites; spatial and object never moved.
# Four teachers from different lineages pulling one shared LoRA in the SAME backward pass have no
# way to be reconciled. This project's own best 4-suite result came from the SERIAL form (0.89).
#
# Each segment: single suite, single teacher, plus
#   - anchor_lambda: forward-KL to a FROZEN copy of the model this cycle started from, so the
#     three suites not being trained are held in place (the LwF recipe, lambda=0.3 validated).
#   - distill_fail_alpha: SOFT emphasis on trajectories the student FAILED -- failed positions
#     count (1+alpha)x, successful ones still count 1. Not distill_on_failure, which HARD-filters
#     successes away and throws out exactly the states where "don't break what works" is learned.
set -uo pipefail

START="${START:-/share/fanruochen-local/outputs/inc_sft_opd/lwf_long_e2000_merged}"
ORDER="${ORDER:-10 goal object spatial}"     # biggest deficit first: long .62 goal .84 object .94 spatial .94
STEPS="${STEPS:-3}"
CYCLES="${CYCLES:-1}"
GPUS="${GPUS:-0,1,2,3,4,5}"
# On a single GPU, student+teacher+anchor are THREE full 7B copies with no FSDP sharding to
# spread activations across ranks -- turn on grad checkpointing to trade speed for VRAM.
# Multi-GPU keeps it off on purpose (see libero_seqcl_opd_2gpu.yaml's own comment: FSDP sharding
# already leaves headroom, and the ~23min/step update is the bottleneck there, not memory).
GRAD_CKPT="${GRAD_CKPT:-False}"
ENVS="${ENVS:-48}"   # must stay a multiple of group_size=8
LAMBDA="${LAMBDA:-0.3}"
FAIL_ALPHA="${FAIL_ALPHA:-1.0}"
BASE_PORT="${BASE_PORT:-53000}"
RUN="${RUN:-ser1}"

O=/share/fanruochen-local/outputs
B130="$O/inc_sft_opd/base_stats130"
declare -A TEACHER=(
  [spatial]="$B130::$O/seqcl_rlspat_opd/spatial_cat_r160"
  [object]="$B130::$O/inc_sft_opd/sft_object_base3_cont/openvla-7b-base+libero_object_no_noops+b32+lr-0.0003+lora-r32+dropout-0.0--image_aug/adapters/step_500"
  [goal]="$B130::$O/inc_sft_opd/teachers_r160/goal_r160"
  [10]="$O/inc_sft_opd/lwf_long_e1000_merged::$O/seqcl_long_opd130/adapter/long_opd130"
)

[ -f "$START/model.safetensors.index.json" ] || { echo "ABORT: start is not an HF model dir: $START"; exit 1; }
for s in $ORDER; do
  t="${TEACHER[$s]:-}"; [ -n "$t" ] || { echo "ABORT: no teacher for suite '$s'"; exit 1; }
  b="${t%%::*}"; a="${t#*::}"
  [ -f "$b/model.safetensors.index.json" ] || { echo "ABORT: teacher base missing: $b"; exit 1; }
  [ -f "$a/adapter_model.safetensors" ]    || { echo "ABORT: teacher adapter missing: $a"; exit 1; }
done
# Memory-only checks miss a real failure mode: a job can be compute-bound with a tiny memory
# footprint (observed 2026-08-20: another tenant at <2% memory / 73-97% utilization on GPU0-4,
# sustained across 5 samples with the training job fully stopped -- our memory-only check would
# have happily launched on top of it and stolen their compute cycles). Check BOTH, and sample
# utilization 3x with a gap since a single instantaneous reading can catch an idle tick.
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
  (( used < 5000 )) || { echo "ABORT: GPU$g holds ${used} MiB -- not ours to take"; exit 1; }
  busy=0
  for _ in 1 2 3; do
    util=$(nvidia-smi -i "$g" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    (( util < 20 )) || busy=$((busy+1))
    sleep 2
  done
  (( busy < 2 )) || { echo "ABORT: GPU$g at ${util}% util (memory looked free, compute is not) -- not ours to take"; exit 1; }
done

# The anchor is frozen at the START of the whole chain: every segment is pulled back toward the
# 0.835 model, so drift cannot accumulate segment over segment.
ANCHOR="$START"
CUR="$START"
IDX=0
echo "======== SERIAL CYCLE  start=$(basename "$START")  order=[$ORDER] x${CYCLES}  steps/seg=$STEPS"
echo "         lambda=$LAMBDA  fail_alpha=$FAIL_ALPHA  gpus=$GPUS  anchor=$(basename "$ANCHOR") ========"

for c in $(seq 1 "$CYCLES"); do
for SUITE in $ORDER; do
  TAG="${RUN}_c${c}_${SUITE}"
  PORT=$(( BASE_PORT + IDX * 1000 )); IDX=$((IDX+1))
  free=$(df -BG --output=avail /share/fanruochen-local | tail -1 | tr -dc '0-9')
  (( free >= 120 )) || { echo "ABORT: only ${free}G free (each segment writes ~29G)"; exit 1; }
  echo ""
  echo "######## [cycle $c] SUITE=$SUITE  init=$(basename "$CUR")  port=$PORT  free=${free}G  $(date '+%F %T') ########"
  OPD_GPUS="$GPUS" OPD_ENVS="$ENVS" OPD_ROLLOUT_EPOCH=3 OPD_MICRO=8 OPD_GLOBAL_BATCH=192 \
  OPD_STEPS="$STEPS" OPD_RAY_PORT="$PORT" OPD_TEACHER="${TEACHER[$SUITE]}" \
  OPD_EXTRA="+algorithm.anchor_lambda=$LAMBDA +actor.base_model_path=$ANCHOR +algorithm.distill_fail_alpha=$FAIL_ALPHA ++actor.fsdp_config.gradient_checkpointing=$GRAD_CKPT" \
    bash /home/fanruochen/CL/RLinf/examples/embodiment/incremental_sft/opd_one.sh \
      "$SUITE" "$CUR" "$TAG"
  NEXT="$O/seqcl_${TAG}/converted/${TAG}"
  [ -f "$NEXT/model.safetensors.index.json" ] || { echo "ABORT: segment $TAG produced no model at $NEXT"; exit 1; }
  echo "######## [cycle $c] $SUITE done -> $NEXT  $(date '+%F %T') ########"
  CUR="$NEXT"
done
done
echo ""
echo "======== SERIAL CYCLE DONE  final=$CUR  $(date '+%F %T') ========"
echo "SERIAL_FINAL=$CUR"
