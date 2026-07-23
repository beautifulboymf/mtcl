#!/bin/bash
# dualkl_eval_core.sh <model_dir> <tag> <gpu> [unnorm] — object-std + held-out(spatial/goal/long).
# EVAL_EPOCH env (default 1=50 trials, 3=150 trials) multiplies rollout epochs; parallel envs stay 50.
MODEL="${1:?model}"; TAG="${2:?tag}"; GPU="${3:?gpu}"; UNNORM="${4:-libero_130_no_noops_trajall}"
cd /home/fanruochen/CL/RLinf
RES="/share/fanruochen-local/outputs/dualkl/eval150_${TAG}_s${SEED:-1234}"; mkdir -p "$RES"
OV="rollout.model.model_path=$MODEL actor.model.model_path=$MODEL actor.model.is_lora=False actor.model.unnorm_key=$UNNORM env.eval.total_num_envs=50 algorithm.eval_rollout_epoch=1 env.eval.seed=${SEED:-1234}"
grab(){ grep -aE "eval/success_at_end" "$1" 2>/dev/null|tail -1|grep -aoE "'eval/(success_once|success_at_end)': array\([0-9.]+|num_trajectories': [0-9]+"|tr '\n' ' '; }
unset LIBERO_TYPE LIBERO_SUFFIX
# object-std
echo "======== $TAG/object-std start=$(date '+%T') EVAL_EPOCH=${EVAL_EPOCH:-1} ========"
bash examples/embodiment/eval_embodiment.sh "libero_object_g${GPU}_eval" LIBERO $OV 2>&1 | tee "$RES/object_std.log"
echo "======== $TAG/object-std done | $(grab "$RES/object_std.log") ========"
# held-out spatial/goal/long
for S in spatial goal 10; do
  EXTRA=""; [ "$S" = "10" ] && EXTRA="env.train.total_num_envs=50"
  echo "======== $TAG/$S start=$(date '+%T') ========"
  bash examples/embodiment/eval_embodiment.sh "libero_${S}_g${GPU}_eval" LIBERO $OV $EXTRA 2>&1 | tee "$RES/heldout_${S}.log"
  echo "======== $TAG/$S done | $(grab "$RES/heldout_${S}.log") ========"
done
echo "CORE150_DONE_${TAG} $(date '+%T')"
