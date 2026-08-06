#!/bin/bash
# opd_rehearse.sh — REHEARSED OPD: distill the 130 teacher on MULTIPLE suites at once (rollout the
# new + old suites, weighted) to LEARN the new suite while PRESERVING the old (anti-forgetting via
# rehearsal + teacher anchor). Reuses the seqcl OPD config; the env expands active_suites -> task
# ids and oversamples per suite weights. Prints OPD_CONVERTED=<dir>. Run under safe_run.
#
# Usage: [OPD_GPUS=4-7 OPD_ENVS=32 OPD_ROLLOUT_EPOCH=1 OPD_STEPS=15 \
#         OPD_SUITES='[libero_spatial,libero_object]' \
#         OPD_WEIGHTS='{libero_spatial:0.3,libero_object:1.0}'] opd_rehearse.sh <init_model_dir> <tag>
# NOTE: do NOT override env.train.max_episode_steps / max_steps_per_rollout_epoch — RLinf asserts
# they be divisible by num_action_chunks(8) AND 32; the config default 512 is safe. Speed comes from
# rollout_epoch=1 + 4 GPUs (envs must be 32 so 32//4//1 = 8 = group_size).
set -uo pipefail
INIT="${1:?init model dir}"; TAG="${2:?tag}"
REPO=/home/fanruochen/CL/RLinf
SCRIPTS=/share/fanruochen-local/dev/scripts
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
OPD_GPUS="${OPD_GPUS:-4-7}"; OPD_ENVS="${OPD_ENVS:-32}"
OPD_ROLLOUT_EPOCH="${OPD_ROLLOUT_EPOCH:-1}"; OPD_STEPS="${OPD_STEPS:-15}"
OPD_SUITES="${OPD_SUITES:-[libero_spatial,libero_object]}"
OPD_WEIGHTS="${OPD_WEIGHTS:-{libero_spatial:0.3,libero_object:1.0}}"
LOGP="/share/fanruochen-local/outputs/seqcl_${TAG}"
[ -f "$INIT/model.safetensors.index.json" ] || { echo "ABORT: init not an HF model dir: $INIT"; exit 1; }

set +u
source /home/fanruochen/.rlinf-env.sh 2>/dev/null
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
source /share/fanruochen-local/dev/gpu_render_env.sh
set -u
export EMBODIED_PATH="$REPO/examples/embodiment" REPO_PATH="$REPO"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$REPO:${PYTHONPATH:-}"
export RLINF_CONVERT_VALUE_HEAD=False

echo "== df =="; df -h /share/fanruochen-local | tail -1
echo "======== REHEARSE-OPD suites=$OPD_SUITES weights=$OPD_WEIGHTS init=$(basename "$INIT") teacher=130 gpus=$OPD_GPUS envs=$OPD_ENVS rollout_epoch=$OPD_ROLLOUT_EPOCH steps=$OPD_STEPS  $(date '+%F %T') ========"
SEQCL_GPUS="$OPD_GPUS" SEQCL_STUDENT_PATH="$INIT" \
SEQCL_ACTIVE_SUITES="$OPD_SUITES" SEQCL_SUITE_WEIGHTS="$OPD_WEIGHTS" \
SEQCL_MAX_STEPS="$OPD_STEPS" SEQCL_CURRENT_SUITE="$TAG" \
ISO_RAY_PORT="${OPD_RAY_PORT:-28000}" bash "$SCRIPTS/run_iso.sh" \
  "$PY" "$REPO/examples/embodiment/train_embodied_agent.py" --config-name libero_seqcl_opd_2gpu \
    algorithm.rollout_epoch="$OPD_ROLLOUT_EPOCH" \
    env.train.total_num_envs="$OPD_ENVS"
echo "OPD_TRAIN_DONE $(date '+%F %T')"

CKPT=$(find "$LOGP" -name full_weights.pt -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -z "$CKPT" ] && { echo "ABORT: no full_weights.pt under $LOGP"; exit 1; }
CONV="$LOGP/converted/${TAG}"
echo "======== CONVERT $CKPT -> $CONV (base=$(basename "$INIT")) ========"
bash "$SCRIPTS/convert_oft_lora_ckpt.sh" "$CKPT" "$CONV" "$INIT"
[ -f "$CONV/model.safetensors.index.json" ] || { echo "ABORT: convert failed"; exit 1; }
echo "OPD_CONVERTED=$CONV"
echo "OPD_${TAG}_DONE $(date '+%F %T')"
