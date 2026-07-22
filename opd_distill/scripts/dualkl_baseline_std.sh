#!/bin/bash
# Stage-0 baseline: eval the 130-generalist BASE (student init) on 4 standard LIBERO suites.
# is_lora=False evals the dir's safetensors directly. object runs FIRST as a probe:
#   ~0.7 => safetensors ARE the 130 generalist (baseline setup correct)
#   ~0.0 => safetensors are raw base; the 130 knowledge is in lora_adapter (must handle)
# 50 env, GPU7 (single; leaves 2-3/4-5 as clean training pairs), EGL, sequential.
cd /home/fanruochen/CL/RLinf
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/fanruochen/.rlinf-env.sh
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
source /share/fanruochen-local/dev/gpu_render_env.sh

BASE=/share/fanruochen-local/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora
UNNORM=libero_130_no_noops_trajall
OV="rollout.model.model_path=$BASE actor.model.model_path=$BASE actor.model.is_lora=False actor.model.unnorm_key=$UNNORM env.eval.total_num_envs=50"
RES=/share/fanruochen-local/outputs/dualkl/baseline_BASE
mkdir -p "$RES"

for SUITE in object spatial goal 10; do
  DF=$(df -BG --output=avail /share/fanruochen-local | tail -1 | tr -dc 0-9)
  echo "======== BASE-std $SUITE start=$(date '+%F %T') df=${DF}G ========"
  EXTRA=""; [ "$SUITE" = "10" ] && EXTRA="env.train.total_num_envs=50"
  bash examples/embodiment/eval_embodiment.sh "libero_${SUITE}_g7_eval" LIBERO $OV $EXTRA 2>&1 | tee "$RES/eval_${SUITE}.log"
  sr=$(grep -aE "eval/success_at_end" "$RES/eval_${SUITE}.log" | tail -1 | grep -aoE "'eval/(success_once|success_at_end)': array\([0-9.]+" | tr '\n' ' ')
  echo "======== BASE-std $SUITE done=$(date '+%F %T') | $sr ========"
done
echo "BASE_STD_DONE $(date '+%F %T')"
