#!/bin/bash
# eval_adapter.sh — GREEDY (temp=0) eval of a base+LoRA-adapter (NO merge, NO temp write -> IO-safe).
# Loads model_path=base_stats130 (openvla-7b-base symlinks + STATS130 dataset_statistics so unnorm_key
# libero_130 resolves) and lora_path=<adapter dir> via PeftModel on BOTH rollout+actor. 50 env, video off.
# Usage: [ISO_RAY_PORT=29000 EVAL_LORA_RANK=32] eval_adapter.sh <adapter_dir> <spatial|object|goal|10> <gpu> <tag>
set -uo pipefail
ADIR="${1:?adapter dir}"; SUITE="${2:?suite}"; GPU="${3:-5}"; TAG="${4:-aeval}"
REPO=/home/fanruochen/CL/RLinf
CFGDIR="$REPO/examples/embodiment/config"
OUT=/share/fanruochen-local/outputs/inc_sft_opd
# Base the adapter sits on. Defaults to openvla-7b-base+STATS130; override with EVAL_BASE when the
# adapter was trained on a different init (e.g. an LwF adapter on top of the CL student lwf_sponly2 --
# read it from the adapter's adapter_config.json "base_model_name_or_path").
BASE130="${EVAL_BASE:-/share/fanruochen-local/outputs/inc_sft_opd/base_stats130}"
RANK="${EVAL_LORA_RANK:-32}"
[ -f "$ADIR/adapter_model.safetensors" ] || { echo "ABORT: not an adapter dir: $ADIR"; exit 1; }
[ -f "$BASE130/model.safetensors.index.json" ] || { echo "ABORT: base_stats130 missing: $BASE130"; exit 1; }

set +u
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
fi

echo "== df =="; df -h /share/fanruochen-local | tail -1
echo "======== EVAL-ADAPTER greedy [$TAG] $SUITE | $(basename "$ADIR") | base130+LoRA r$RANK | GPU$GPU  $(date '+%F %T') ========"
ISO_RAY_PORT="${ISO_RAY_PORT:-29000}" bash /share/fanruochen-local/dev/scripts/run_iso.sh \
  bash "$REPO/examples/embodiment/eval_embodiment.sh" "libero_${SUITE}_g${GPU}_eval" LIBERO \
    rollout.model.model_path="$BASE130" actor.model.model_path="$BASE130" \
    +rollout.model.is_lora=True +rollout.model.lora_path="$ADIR" +rollout.model.lora_rank="$RANK" \
    actor.model.is_lora=True actor.model.lora_path="$ADIR" actor.model.lora_rank="$RANK" \
    actor.model.unnorm_key=libero_130_no_noops_trajall \
    algorithm.sampling_params.temperature_eval="${EVAL_TEMP:-0}" \
    env.eval.total_num_envs=50 env.train.total_num_envs=50 \
    env.eval.video_cfg.save_video=False \
  2>&1 | tee "$OUT/eval_${TAG}_${SUITE}.log"
echo "EVAL_${TAG}_${SUITE}_DONE $(date '+%F %T')"
