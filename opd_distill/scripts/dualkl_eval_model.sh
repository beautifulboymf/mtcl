#!/bin/bash
# dualkl_eval_model.sh <model_dir> <tag> <gpu_rank> [unnorm_key]
# Evals ONE model on: object-std (standard LIBERO) + object-PRO x3 (object/swap/lan). 50 env each.
# Caller must wrap in run_iso.sh (private ray) + safe_run. Env already activated by run_iso.sh.
MODEL="${1:?model dir}"; TAG="${2:?tag}"; GPU="${3:?gpu}"; UNNORM="${4:-libero_130_no_noops_trajall}"
cd /home/fanruochen/CL/RLinf
RES="/share/fanruochen-local/outputs/dualkl/eval_${TAG}"; mkdir -p "$RES"
CFG="libero_object_g${GPU}_eval"
OV="rollout.model.model_path=$MODEL actor.model.model_path=$MODEL actor.model.is_lora=False actor.model.unnorm_key=$UNNORM env.eval.total_num_envs=50 algorithm.eval_rollout_epoch=${EVAL_EPOCH:-1}"
grab(){ grep -aE "eval/success_at_end" "$1" 2>/dev/null | tail -1 | grep -aoE "'eval/(success_once|success_at_end)': array\([0-9.]+" | tr '\n' ' '; }

# 1) object-std (standard libero; NO pro pythonpath -> matches BASE baseline conditions)
unset LIBERO_TYPE LIBERO_SUFFIX
echo "======== EVAL $TAG/object-std start=$(date '+%F %T') df=$(df -BG --output=avail /share/fanruochen-local|tail -1|tr -dc 0-9)G ========"
bash examples/embodiment/eval_embodiment.sh "$CFG" LIBERO $OV 2>&1 | tee "$RES/object_std.log"
echo "======== $TAG/object-std done | $(grab "$RES/object_std.log") ========"

# 2) object-PRO x3 (liberopro source + prompt-fix sitecustomize on PYTHONPATH)
export PYTHONPATH="/share/fanruochen-local/dev/scripts/pro_prompt_fix_inject:/share/fanruochen-local/dev/envs/rlinf-openpi/libero_pro:${PYTHONPATH:-}"
for P in object swap lan; do
  export LIBERO_TYPE=pro LIBERO_SUFFIX="$P"
  echo "======== EVAL $TAG/PRO-$P start=$(date '+%F %T') ========"
  bash examples/embodiment/eval_embodiment.sh "$CFG" LIBERO $OV 2>&1 | tee "$RES/pro_$P.log"
  echo "======== $TAG/PRO-$P done | $(grab "$RES/pro_$P.log") ========"
done
echo "EVAL_DONE_${TAG} $(date '+%F %T')"
