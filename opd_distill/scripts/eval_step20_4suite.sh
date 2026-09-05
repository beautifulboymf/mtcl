#!/bin/bash
# Sequential post-hoc eval of the 130-full BIG OPD step_20 checkpoint on 4 LIBERO suites.
# 50 env each (NOT 500 -> red-line), GPU2 (g2 configs), EGL GPU render, one-at-a-time (no ray port clash).
# NOTE: no `set -u` — the sourced env scripts (rlinf-env / conda activate / gpu_render_env) are not -u clean.
cd /home/fanruochen/CL/RLinf
source /home/fanruochen/.rlinf-env.sh                                   # render-libs + proxy
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate   # conda env -> python on PATH
source /share/fanruochen-local/dev/gpu_render_env.sh   # osmesa shim (EGL quarantined)

CONV=/share/fanruochen-local/outputs/opd_130full_big/converted/step_20
UNNORM=libero_130_no_noops_trajall
OV="rollout.model.model_path=$CONV actor.model.model_path=$CONV actor.model.is_lora=False actor.model.unnorm_key=$UNNORM env.eval.total_num_envs=50"
RES=/share/fanruochen-local/outputs/opd_130full_big/posthoc_step20
mkdir -p "$RES"

for SUITE in object spatial goal 10; do
  DF=$(df -h /share/fanruochen-local | tail -1 | awk '{print $4}')
  echo "======== EVAL $SUITE  start=$(date '+%F %T')  df_free=$DF ========"
  EXTRA=""
  [ "$SUITE" == "10" ] && EXTRA="env.train.total_num_envs=50"   # libero_10: avoid None//int crash
  bash examples/embodiment/eval_embodiment.sh "libero_${SUITE}_g2_eval" LIBERO $OV $EXTRA 2>&1 | tee "$RES/eval_${SUITE}.log"
  echo "======== EVAL $SUITE  done=$(date '+%F %T')  exit=${PIPESTATUS[0]} ========"
done
echo "ALL_EVAL_DONE $(date '+%F %T')"
