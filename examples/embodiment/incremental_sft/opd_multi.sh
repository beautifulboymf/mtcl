#!/bin/bash
# opd_multi.sh — TRUE MULTI-TEACHER on-policy distillation. The student rolls out over SEVERAL
# suites at once; every rollout sample is scored by the EXPERT TEACHER OF ITS OWN SUITE (routed in
# fsdp_actor_worker._teacher_forward via the prompt->suite table). Teachers only SCORE — they are
# frozen (eval, no_grad) and never act/roll out.
#
# Each teacher may be a full HF dir OR "<base_dir>::<adapter_dir>" (base + PEFT LoRA adapter), so N
# per-suite experts that share one base are N small adapters.
#
# Usage:
#   [OPD_GPUS=2-5] [OPD_STEPS=15] [OPD_RAY_PORT=28000] [OPD_ENVS=16] [OPD_ROLLOUT_EPOCH=4] \
#   [OPD_WEIGHTS='{libero_goal:1.0,libero_spatial:0.5,libero_object:0.5}'] \
#   opd_multi.sh <init_model_dir> <tag> <suite1=teacher1> [suite2=teacher2 ...]
# Example:
#   opd_multi.sh <init> mt3 \
#     libero_spatial=$BASE::/…/spatial_cat_r160 \
#     libero_object=$BASE::/…/object_r32_sft \
#     libero_goal=$BASE::/…/goal_r160
set -uo pipefail
INIT="${1:?init model dir}"; TAG="${2:?tag}"; shift 2
[ "$#" -ge 1 ] || { echo "ABORT: need at least one <suite>=<teacher> pair"; exit 1; }
REPO=/home/fanruochen/CL/RLinf
SCRIPTS=/share/fanruochen-local/dev/scripts
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
OPD_GPUS="${OPD_GPUS:-2-5}"; OPD_STEPS="${OPD_STEPS:-15}"
LOGP="/share/fanruochen-local/outputs/seqcl_${TAG}"
[ -f "$INIT/model.safetensors.index.json" ] || { echo "ABORT: init not an HF model dir: $INIT"; exit 1; }

# parse suite=teacher pairs -> hydra teacher_map overrides + active_suites list
SUITES=(); OVR=()
for pair in "$@"; do
  s="${pair%%=*}"; t="${pair#*=}"
  [ "$s" = "$pair" ] && { echo "ABORT: bad pair '$pair' (want suite=teacher)"; exit 1; }
  # validate teacher (both plain dir and base::adapter form)
  b="${t%%::*}"; a=""; [ "$t" != "$b" ] && a="${t#*::}"
  [ -f "$b/model.safetensors.index.json" ] || { echo "ABORT: teacher base not an HF model: $b"; exit 1; }
  [ -n "$a" ] && { [ -f "$a/adapter_model.safetensors" ] || { echo "ABORT: adapter missing: $a"; exit 1; }; }
  SUITES+=("$s"); OVR+=("actor.teacher_map.${s}=${t}")
done
ACTIVE="[$(IFS=,; echo "${SUITES[*]}")]"

set +u
source /home/fanruochen/.rlinf-env.sh 2>/dev/null
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa  # EGL banned 2026-09-03 (two host crashes); CPU render only
set -u
export EMBODIED_PATH="$REPO/examples/embodiment" REPO_PATH="$REPO"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH="$REPO:${PYTHONPATH:-}"
export RLINF_CONVERT_VALUE_HEAD=False

echo "== df =="; df -h /share/fanruochen-local | tail -1
echo "======== MULTI-TEACHER OPD [$ACTIVE] init=$(basename "$INIT") gpus=$OPD_GPUS steps=$OPD_STEPS  $(date '+%F %T') ========"
for o in "${OVR[@]}"; do echo "   teacher: $o"; done
SEQCL_GPUS="$OPD_GPUS" SEQCL_STUDENT_PATH="$INIT" \
SEQCL_ACTIVE_SUITES="$ACTIVE" SEQCL_SUITE_WEIGHTS="${OPD_WEIGHTS:-null}" \
SEQCL_MAX_STEPS="$OPD_STEPS" SEQCL_CURRENT_SUITE="$TAG" \
ISO_RAY_PORT="${OPD_RAY_PORT:-28000}" bash "$SCRIPTS/run_iso.sh" \
  "$PY" "$REPO/examples/embodiment/train_embodied_agent.py" --config-name libero_seqcl_opd_2gpu \
    algorithm.rollout_epoch="${OPD_ROLLOUT_EPOCH:-4}" \
    env.train.total_num_envs="${OPD_ENVS:-16}" \
    env.train.max_episode_steps="${OPD_EP_STEPS:-512}" \
    env.train.max_steps_per_rollout_epoch="${OPD_EP_STEPS:-512}" \
    actor.micro_batch_size="${OPD_MICRO:-8}" \
    ${OPD_GLOBAL_BATCH:+actor.global_batch_size=$OPD_GLOBAL_BATCH} \
    ${OPD_SHIFT_BETA:+"+algorithm.shift_beta=$OPD_SHIFT_BETA"} \
    ${OPD_FAIL_ONLY:+"+algorithm.distill_on_failure=$OPD_FAIL_ONLY"} \
    ${OPD_EXTRA:+$OPD_EXTRA} \
    "${OVR[@]}"
echo "OPD_TRAIN_DONE $(date '+%F %T')"

CKPT=$(find "$LOGP" -name full_weights.pt -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -z "$CKPT" ] && { echo "ABORT: no full_weights.pt under $LOGP"; exit 1; }
CONV="$LOGP/converted/${TAG}"
echo "======== CONVERT $CKPT -> $CONV (base=$(basename "$INIT")) ========"
bash "$SCRIPTS/convert_oft_lora_ckpt.sh" "$CKPT" "$CONV" "$INIT"
[ -f "$CONV/model.safetensors.index.json" ] || { echo "ABORT: convert failed"; exit 1; }
echo "OPD_CONVERTED=$CONV"
echo "OPD_${TAG}_DONE $(date '+%F %T')"
