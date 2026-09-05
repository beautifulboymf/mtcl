#!/bin/bash
# eval_one_model.sh — GREEDY (temp=0) eval of ONE OFT model on ONE LIBERO suite.
# 50 env, libero_130 norm, is_lora=False (merged model), run_iso-isolated ray, video off.
# Auto-creates a per-GPU eval config from the g7 template if missing.
# Usage: [ISO_RAY_PORT=29000] eval_one_model.sh <model_dir> <spatial|object|goal|10> <gpu> <tag>
set -uo pipefail
MODEL="${1:?model dir}"; SUITE="${2:?suite}"; GPU="${3:-5}"; TAG="${4:-eval}"
REPO=/home/fanruochen/CL/RLinf
SCRIPTS=/share/fanruochen-local/dev/scripts
CFGDIR="$REPO/examples/embodiment/config"
OUT=/share/fanruochen-local/outputs/inc_sft_opd
mkdir -p "$OUT"
[ -f "$MODEL/model.safetensors.index.json" ] || { echo "ABORT: not an HF model dir: $MODEL"; exit 1; }

set +u   # venv activate references unbound PYTHONPATH under set -u
source /home/fanruochen/.rlinf-env.sh 2>/dev/null
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa  # EGL banned 2026-09-03 (two host crashes); CPU render only
set -u
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export RLINF_CONVERT_VALUE_HEAD=False

WANT="$CFGDIR/libero_${SUITE}_g${GPU}_eval.yaml"
if [ ! -f "$WANT" ]; then
  SRC="$CFGDIR/libero_${SUITE}_g7_eval.yaml"
  [ -f "$SRC" ] || { echo "ABORT: no template $SRC"; exit 1; }
  sed 's/actor,env,rollout: "7"/actor,env,rollout: "'"$GPU"'"/' "$SRC" > "$WANT"
  echo "[eval_one] created $(basename "$WANT")"
fi

# EVAL_TEMP: decoding temperature. Default 0 = greedy, which is what every historical number in
# this project was measured at -- do NOT change the default or past results stop being comparable.
# Worth sweeping because the distill loss is FORWARD KL (mode-covering): it makes the student cover
# the teacher's mass but does NOT preserve the argmax. The probes measured the teacher's top-1
# sitting at rank ~3.7 in the student, i.e. greedy decoding plays an action the teacher would not,
# so temp>0 can beat temp=0 for a mode-covering-distilled student.
EVAL_TEMP="${EVAL_TEMP:-0}"
echo "== df =="; df -h /share/fanruochen-local | tail -1
echo "======== EVAL temp=$EVAL_TEMP [$TAG] $SUITE | $(basename "$MODEL") | GPU$GPU  $(date '+%F %T') ========"
ISO_RAY_PORT="${ISO_RAY_PORT:-29000}" bash "$SCRIPTS/run_iso.sh" \
  bash "$REPO/examples/embodiment/eval_embodiment.sh" "libero_${SUITE}_g${GPU}_eval" LIBERO \
    rollout.model.model_path="$MODEL" actor.model.model_path="$MODEL" \
    actor.model.unnorm_key=libero_130_no_noops_trajall actor.model.is_lora=False \
    algorithm.sampling_params.temperature_eval="$EVAL_TEMP" \
    env.eval.total_num_envs=50 env.train.total_num_envs=50 \
    env.eval.video_cfg.save_video=False \
  2>&1 | tee "$OUT/eval_${TAG}_${SUITE}.log"
echo "EVAL_${TAG}_${SUITE}_DONE $(date '+%F %T')"
