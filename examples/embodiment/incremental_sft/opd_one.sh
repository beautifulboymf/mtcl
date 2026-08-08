#!/bin/bash
# opd_one.sh — ONE OPD stage: on-policy-distill the 130 teacher into <init_model> on ONE suite,
# then convert the LoRA training checkpoint to a merged HF model. Reuses the single-suite seqcl
# OPD config (active_suites=[one], no rehearsal, teacher_map -> the single 130). Prints
# OPD_CONVERTED=<dir>. Run under safe_run.
# Usage: [OPD_GPUS=4-5] [OPD_STEPS=15] [OPD_RAY_PORT=28000] opd_one.sh <spatial|object|goal|10> <init_model_dir> <tag>
set -uo pipefail
SUITE="${1:?suite}"; INIT="${2:?init model dir}"; TAG="${3:?tag}"
REPO=/home/fanruochen/CL/RLinf
SCRIPTS=/share/fanruochen-local/dev/scripts
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
OPD_GPUS="${OPD_GPUS:-4-5}"; OPD_STEPS="${OPD_STEPS:-15}"
LOGP="/share/fanruochen-local/outputs/seqcl_${TAG}"
[ -f "$INIT/model.safetensors.index.json" ] || { echo "ABORT: init not an HF model dir: $INIT"; exit 1; }

set +u   # venv activate references unbound PYTHONPATH under set -u
source /home/fanruochen/.rlinf-env.sh 2>/dev/null
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
source /share/fanruochen-local/dev/gpu_render_env.sh   # OPD env rollout renders via egl
set -u
export EMBODIED_PATH="$REPO/examples/embodiment" REPO_PATH="$REPO"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$REPO:${PYTHONPATH:-}"
export RLINF_CONVERT_VALUE_HEAD=False

echo "== df =="; df -h /share/fanruochen-local | tail -1
# Optional: override this suite's teacher (default config teacher_map = 130 generalist). Set
# OPD_TEACHER to a matched specialist (e.g. spatial_opd_student_098) to distill toward it instead.
TEACHER_OVR=""
[ -n "${OPD_TEACHER:-}" ] && TEACHER_OVR="actor.teacher_map.libero_${SUITE}=${OPD_TEACHER}"
echo "======== OPD [$SUITE] init=$(basename "$INIT") teacher=${OPD_TEACHER:-130(default)} gpus=$OPD_GPUS steps=$OPD_STEPS  $(date '+%F %T') ========"
SEQCL_GPUS="$OPD_GPUS" SEQCL_STUDENT_PATH="$INIT" \
SEQCL_ACTIVE_SUITES="[libero_${SUITE}]" SEQCL_SUITE_WEIGHTS='null' \
SEQCL_MAX_STEPS="$OPD_STEPS" SEQCL_CURRENT_SUITE="$TAG" \
ISO_RAY_PORT="${OPD_RAY_PORT:-28000}" bash "$SCRIPTS/run_iso.sh" \
  "$PY" "$REPO/examples/embodiment/train_embodied_agent.py" --config-name libero_seqcl_opd_2gpu \
    algorithm.rollout_epoch="${OPD_ROLLOUT_EPOCH:-4}" \
    env.train.total_num_envs="${OPD_ENVS:-16}" \
    env.train.max_episode_steps="${OPD_EP_STEPS:-512}" \
    env.train.max_steps_per_rollout_epoch="${OPD_EP_STEPS:-512}" \
    ${TEACHER_OVR}
echo "OPD_TRAIN_DONE $(date '+%F %T')"

CKPT=$(find "$LOGP" -name full_weights.pt -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -z "$CKPT" ] && { echo "ABORT: no full_weights.pt under $LOGP"; exit 1; }
CONV="$LOGP/converted/${TAG}"
echo "======== CONVERT $CKPT -> $CONV (base=$(basename "$INIT")) ========"
bash "$SCRIPTS/convert_oft_lora_ckpt.sh" "$CKPT" "$CONV" "$INIT"
[ -f "$CONV/model.safetensors.index.json" ] || { echo "ABORT: convert failed"; exit 1; }
echo "OPD_CONVERTED=$CONV"
echo "OPD_${TAG}_DONE $(date '+%F %T')"
