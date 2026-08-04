#!/bin/bash
# eval_inc_models.sh — GREEDY post-hoc eval of the incremental-SFT->OPD models on spatial+object.
# Reads model paths from $ROOT/MODELS.txt (produced by run_incremental_sft_opd.sh) or from args.
# All evals: greedy (temperature_eval=0), 50 env, libero_130 norm, run_iso-isolated ray, video off.
#
#   spatial student  -> eval on spatial        (upper reference right after learning spatial)
#   FINAL (object)   -> eval on spatial        (spatial RETENTION / forgetting)
#   FINAL (object)   -> eval on object         (object, just learned)
#
# Usage: [EVAL_GPU=4] bash eval_inc_models.sh   (or pass  <spatial_OPD_dir> <object_OPD_dir>)
set -uo pipefail
REPO=/home/fanruochen/CL/RLinf
SCRIPTS=/share/fanruochen-local/dev/scripts
ROOT=/share/fanruochen-local/outputs/inc_sft_opd
CFGDIR="$REPO/examples/embodiment/config"
EVAL_GPU="${EVAL_GPU:-4}"

source /home/fanruochen/.rlinf-env.sh 2>/dev/null
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
source /share/fanruochen-local/dev/gpu_render_env.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export RLINF_CONVERT_VALUE_HEAD=False

SPA_OPD="${1:-$(sed -n 's/^spatial_OPD=//p' "$ROOT/MODELS.txt" 2>/dev/null)}"
OBJ_OPD="${2:-$(sed -n 's/^object_OPD_FINAL=//p' "$ROOT/MODELS.txt" 2>/dev/null)}"
[ -d "$SPA_OPD" ] || { echo "ABORT: spatial_OPD model not found: $SPA_OPD"; exit 1; }
[ -d "$OBJ_OPD" ] || { echo "ABORT: object_OPD(final) model not found: $OBJ_OPD"; exit 1; }

# make sure a per-GPU eval config exists for this GPU (spatial/object only ship g1/g2/g5/g7).
ensure_cfg(){ # $1=suite
  local suite="$1" want="$CFGDIR/libero_${suite}_g${EVAL_GPU}_eval.yaml"
  [ -f "$want" ] && return 0
  local src="$CFGDIR/libero_${suite}_g7_eval.yaml"
  [ -f "$src" ] || { echo "ABORT: no template $src"; exit 1; }
  sed 's/actor,env,rollout: "7"/actor,env,rollout: "'"$EVAL_GPU"'"/' "$src" > "$want"
  echo "  [ensure_cfg] created $(basename "$want")"
}

geval(){ # $1=model $2=suite $3=tag
  local MODEL="$1" SUITE="$2" TAG="$3"
  ensure_cfg "$SUITE"
  echo "== df =="; df -h /share/fanruochen-local | tail -1
  echo "======== EVAL greedy [$TAG] on $SUITE | $(basename "$MODEL") | GPU$EVAL_GPU  $(date '+%F %T') ========"
  ISO_RAY_PORT="${EVAL_RAY_PORT:-29000}" bash "$SCRIPTS/run_iso.sh" \
    bash "$REPO/examples/embodiment/eval_embodiment.sh" "libero_${SUITE}_g${EVAL_GPU}_eval" LIBERO \
      rollout.model.model_path="$MODEL" actor.model.model_path="$MODEL" \
      actor.model.unnorm_key=libero_130_no_noops_trajall actor.model.is_lora=False \
      algorithm.sampling_params.temperature_eval=0 \
      env.eval.total_num_envs=50 env.train.total_num_envs=50 \
      env.eval.video_cfg.save_video=False \
    2>&1 | tee "$ROOT/eval_${TAG}_${SUITE}.log"
  echo "EVAL_${TAG}_${SUITE}_DONE $(date '+%F %T')"
}

geval "$SPA_OPD" spatial "spatialStudent"   # spatial right after learning it
geval "$OBJ_OPD" spatial "final"            # FINAL: spatial retention
geval "$OBJ_OPD" object  "final"            # FINAL: object (just learned)
echo "INC_EVAL_ALL_DONE $(date '+%F %T')"
