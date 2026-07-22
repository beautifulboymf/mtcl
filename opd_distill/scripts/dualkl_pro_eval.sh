#!/bin/bash
# Reusable OpenVLA-OFT LIBERO-PRO eval (object suite, 3 perturbations).
# Recipe (from the 130-generalist PRO run): liberopro on PYTHONPATH + LIBERO_SUFFIX + prompt-fix sitecustomize.
# Usage: dualkl_pro_eval.sh <model_dir> <tag> <gpu_rank> [is_lora=False] [unnorm_key]
cd /home/fanruochen/CL/RLinf
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/fanruochen/.rlinf-env.sh
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
source /share/fanruochen-local/dev/gpu_render_env.sh

MODEL="${1:?model dir}"; TAG="${2:?tag}"; GPU="${3:?gpu rank}"
IS_LORA="${4:-False}"; UNNORM="${5:-libero_130_no_noops_trajall}"

# OFT+PRO recipe: liberopro source (parent so `import liberopro.liberopro` works) + prompt-fix sitecustomize
export PYTHONPATH="/share/fanruochen-local/dev/scripts/pro_prompt_fix_inject:/share/fanruochen-local/dev/envs/rlinf-openpi/libero_pro:${PYTHONPATH:-}"
export LIBERO_TYPE=pro

RES="/share/fanruochen-local/outputs/dualkl/pro_${TAG}"
mkdir -p "$RES"
# g<gpu>_eval config pins the GPU; use object suite for PRO
CFG="libero_object_g${GPU}_eval"
OV="rollout.model.model_path=$MODEL actor.model.model_path=$MODEL actor.model.is_lora=$IS_LORA actor.model.unnorm_key=$UNNORM env.eval.total_num_envs=50"

for PERTURB in object swap lan; do
  export LIBERO_SUFFIX="$PERTURB"    # overrides eval_embodiment.sh's forced LIBERO_PERTURBATION=all (SUFFIX wins, libero_env.py:191)
  DF=$(df -BG --output=avail /share/fanruochen-local | tail -1 | tr -dc 0-9)
  echo "======== PRO $TAG/$PERTURB start=$(date '+%F %T') df=${DF}G ========"
  bash examples/embodiment/eval_embodiment.sh "$CFG" LIBERO $OV 2>&1 | tee "$RES/pro_${PERTURB}.log"
  sr=$(grep -aE "eval/success_at_end" "$RES/pro_${PERTURB}.log" | tail -1 | grep -aoE "'eval/(success_once|success_at_end)': array\([0-9.]+" | tr '\n' ' ')
  echo "======== PRO $TAG/$PERTURB done=$(date '+%F %T') | $sr ========"
done
echo "PRO_DONE_${TAG} $(date '+%F %T')"
