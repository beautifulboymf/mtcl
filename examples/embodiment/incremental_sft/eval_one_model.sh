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

source /home/fanruochen/.rlinf-env.sh 2>/dev/null
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
source /share/fanruochen-local/dev/gpu_render_env.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export RLINF_CONVERT_VALUE_HEAD=False

WANT="$CFGDIR/libero_${SUITE}_g${GPU}_eval.yaml"
if [ ! -f "$WANT" ]; then
  SRC="$CFGDIR/libero_${SUITE}_g7_eval.yaml"
  [ -f "$SRC" ] || { echo "ABORT: no template $SRC"; exit 1; }
  sed 's/actor,env,rollout: "7"/actor,env,rollout: "'"$GPU"'"/' "$SRC" > "$WANT"
  echo "[eval_one] created $(basename "$WANT")"
fi

echo "== df =="; df -h /share/fanruochen-local | tail -1
echo "======== EVAL greedy [$TAG] $SUITE | $(basename "$MODEL") | GPU$GPU  $(date '+%F %T') ========"
ISO_RAY_PORT="${ISO_RAY_PORT:-29000}" bash "$SCRIPTS/run_iso.sh" \
  bash "$REPO/examples/embodiment/eval_embodiment.sh" "libero_${SUITE}_g${GPU}_eval" LIBERO \
    rollout.model.model_path="$MODEL" actor.model.model_path="$MODEL" \
    actor.model.unnorm_key=libero_130_no_noops_trajall actor.model.is_lora=False \
    algorithm.sampling_params.temperature_eval=0 \
    env.eval.total_num_envs=50 env.train.total_num_envs=50 \
    env.eval.video_cfg.save_video=False \
  2>&1 | tee "$OUT/eval_${TAG}_${SUITE}.log"
echo "EVAL_${TAG}_${SUITE}_DONE $(date '+%F %T')"
