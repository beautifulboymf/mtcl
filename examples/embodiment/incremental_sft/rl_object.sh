#!/bin/bash
# rl_object.sh — GRPO RL on an OBJECT-SFT model -> produces an OBJECT teacher.
# Sibling of rl_spatial.sh; uses libero_object_grpo_openvlaoft (adv=grpo, env action-level reward,
# full-FT FSDP). unnorm_key forced to libero_130 (our SFT models are 130-norm; the config default
# libero_object_no_noops would mis-decode -> 0 reward). GPU via config placement RANGE (RL_PLACEMENT),
# NOT CUDA_VISIBLE_DEVICES (ray sees all 8). n_gpu in range = env_world_size; total_num_envs //
# n_gpu // pipeline(1) must be divisible by group_size(8): 4 GPU x 32 env -> 32//4=8 OK, dproc~550<800.
#
# The user's TWO object teachers = this SAME launcher with a different init model:
#   (b) spatial-then-object-SFT -> RL:  init = sft_object_weak/...--1500_chkpt   (EXISTS)
#   (a) object-ONLY-SFT -> RL:          init = an object-only SFT of openvla-7b-base (must SFT first:
#         bash incremental_sft/incremental_sft.sh object /share/.../openvla-7b-base <out>  -> then RL that)
#
# Usage: [RL_PLACEMENT=4-7 RL_MICRO=32 RL_ENVS=32 RL_EVAL_ENVS=8 RL_MAX_EPOCHS=200 RL_SAVE=25 \
#         RL_ROLLOUT_EPOCH=16 RL_RAY_PORT=35000] rl_object.sh <object_sft_model_dir> <tag>
set -uo pipefail
INIT="${1:?object-SFT model dir}"; TAG="${2:?tag}"
REPO=/home/fanruochen/CL/RLinf
SCRIPTS=/share/fanruochen-local/dev/scripts
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
RL_PLACEMENT="${RL_PLACEMENT:-4-7}"
RL_MICRO="${RL_MICRO:-32}"; RL_ENVS="${RL_ENVS:-32}"; RL_EVAL_ENVS="${RL_EVAL_ENVS:-8}"
RL_MAX_EPOCHS="${RL_MAX_EPOCHS:-200}"; RL_SAVE="${RL_SAVE:-25}"
RL_UNNORM="${RL_UNNORM:-libero_130_no_noops_trajall}"
LOGP="/share/fanruochen-local/outputs/rl_object_${TAG}"
[ -f "$INIT/model.safetensors.index.json" ] || { echo "ABORT: init not an HF model: $INIT"; exit 1; }

set +u
source /home/fanruochen/.rlinf-env.sh 2>/dev/null
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
source /share/fanruochen-local/dev/gpu_render_env.sh
set -u
export EMBODIED_PATH="$REPO/examples/embodiment" REPO_PATH="$REPO"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$REPO:${PYTHONPATH:-}"
export ROBOT_PLATFORM=LIBERO
export RL_PLACEMENT
if [[ "$RL_PLACEMENT" == *-* ]]; then NGPU=$(( ${RL_PLACEMENT#*-} - ${RL_PLACEMENT%-*} + 1 )); else NGPU=$(echo "$RL_PLACEMENT" | tr ',' '\n' | grep -c .); fi

mkdir -p "$LOGP"
echo "== df =="; df -h /share/fanruochen-local | tail -1
echo "======== GRPO-RL [object] init=$(basename "$INIT") unnorm=$RL_UNNORM placement=$RL_PLACEMENT(n$NGPU) micro=$RL_MICRO envs=$RL_ENVS max_epochs=$RL_MAX_EPOCHS save=$RL_SAVE  $(date '+%F %T') ========"
ISO_RAY_PORT="${RL_RAY_PORT:-35000}" bash "$SCRIPTS/run_iso.sh" \
  "$PY" "$REPO/examples/embodiment/train_embodied_agent.py" --config-name libero_object_grpo_openvlaoft \
    rollout.model.model_path="$INIT" \
    actor.model.model_path="$INIT" \
    actor.model.unnorm_key="$RL_UNNORM" \
    runner.logger.log_path="$LOGP" \
    runner.save_interval="$RL_SAVE" \
    runner.max_epochs="$RL_MAX_EPOCHS" \
    actor.micro_batch_size="$RL_MICRO" \
    env.train.total_num_envs="$RL_ENVS" \
    env.eval.total_num_envs="$RL_EVAL_ENVS" \
    ${RL_ROLLOUT_EPOCH:+algorithm.rollout_epoch=$RL_ROLLOUT_EPOCH}
echo "RL_OBJECT_DONE rc=${PIPESTATUS[0]} $(date '+%F %T')"
