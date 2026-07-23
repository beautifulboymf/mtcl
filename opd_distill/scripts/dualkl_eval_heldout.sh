#!/bin/bash
# dualkl_eval_heldout.sh <model_dir> <tag> <gpu> [unnorm] — held-out suites spatial/goal/long (standard). 50 env.
# The REAL forgetting test (tasks the model never trained on). Caller wraps in run_iso.sh + safe_run.
MODEL="${1:?model}"; TAG="${2:?tag}"; GPU="${3:?gpu}"; UNNORM="${4:-libero_130_no_noops_trajall}"
cd /home/fanruochen/CL/RLinf
RES="/share/fanruochen-local/outputs/dualkl/eval_${TAG}"; mkdir -p "$RES"
OV="rollout.model.model_path=$MODEL actor.model.model_path=$MODEL actor.model.is_lora=False actor.model.unnorm_key=$UNNORM env.eval.total_num_envs=50 algorithm.eval_rollout_epoch=${EVAL_EPOCH:-1}"
grab(){ grep -aE "eval/success_at_end" "$1" 2>/dev/null|tail -1|grep -aoE "'eval/(success_once|success_at_end)': array\([0-9.]+"|tr '\n' ' '; }
unset LIBERO_TYPE LIBERO_SUFFIX
for S in spatial goal 10; do
  EXTRA=""; [ "$S" = "10" ] && EXTRA="env.train.total_num_envs=50"
  echo "======== EVAL $TAG/$S start=$(date '+%F %T') ========"
  bash examples/embodiment/eval_embodiment.sh "libero_${S}_g${GPU}_eval" LIBERO $OV $EXTRA 2>&1 | tee "$RES/heldout_${S}.log"
  echo "======== $TAG/$S done | $(grab "$RES/heldout_${S}.log") ========"
done
echo "HELDOUT_DONE_${TAG} $(date '+%F %T')"
