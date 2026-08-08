#!/bin/bash
# rl_spatial.sh — GRPO RL on a spatial-SFT model -> produces the FINAL spatial teacher.
# Uses the shipped libero_spatial_grpo_openvlaoft config (adv_type=grpo, env action-level reward,
# full-FT FSDP). Overrides model_path -> our spatial-SFT and unnorm_key -> libero_130 (our model was
# SFT'd in 130 norm; the config default libero_spatial_no_noops would mis-decode actions -> 0 reward).
# GPUs via CUDA_VISIBLE_DEVICES (config component_placement=all uses the visible set). run_iso isolates
# ray ports. Run under safe_run + live io_guard. Prints RL checkpoints under $LOGP/checkpoints/.
#
# Usage: [RL_GPUS=4,5,6,7 RL_MICRO=32 RL_ENVS=64 RL_MAX_EPOCHS=200 RL_SAVE=25 RL_RAY_PORT=32000] \
#          rl_spatial.sh <spatial_sft_model_dir> <tag>
set -uo pipefail
INIT="${1:?spatial-SFT model dir}"; TAG="${2:?tag}"
REPO=/home/fanruochen/CL/RLinf
SCRIPTS=/share/fanruochen-local/dev/scripts
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
RL_GPUS="${RL_GPUS:-4,5,6,7}"
RL_MICRO="${RL_MICRO:-32}"; RL_ENVS="${RL_ENVS:-32}"     # 32 train env -> dproc~550<800 (64 blew the 800 proc red line)
RL_EVAL_ENVS="${RL_EVAL_ENVS:-8}"                        # cap eval envs (config default 500 = proc explosion if instantiated)
RL_MAX_EPOCHS="${RL_MAX_EPOCHS:-200}"; RL_SAVE="${RL_SAVE:-25}"
RL_UNNORM="${RL_UNNORM:-libero_130_no_noops_trajall}"
LOGP="/share/fanruochen-local/outputs/rl_spatial_${TAG}"
[ -f "$INIT/model.safetensors.index.json" ] || { echo "ABORT: init not an HF model: $INIT"; exit 1; }

set +u   # venv activate references unbound vars
source /home/fanruochen/.rlinf-env.sh 2>/dev/null
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
source /share/fanruochen-local/dev/gpu_render_env.sh    # EGL GPU render for LIBERO rollout
set -u
export EMBODIED_PATH="$REPO/examples/embodiment" REPO_PATH="$REPO"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$REPO:${PYTHONPATH:-}"
export ROBOT_PLATFORM=LIBERO
export CUDA_VISIBLE_DEVICES="$RL_GPUS"
NGPU=$(echo "$RL_GPUS" | tr ',' '\n' | grep -c .)

mkdir -p "$LOGP"
echo "== df =="; df -h /share/fanruochen-local | tail -1
echo "======== GRPO-RL [spatial] init=$(basename "$INIT") unnorm=$RL_UNNORM gpus=$RL_GPUS(n$NGPU) micro=$RL_MICRO envs=$RL_ENVS max_epochs=$RL_MAX_EPOCHS save=$RL_SAVE  $(date '+%F %T') ========"
# NB: total_num_envs//n_gpu//pipeline_stage(1) must be divisible by group_size(8): 64//4=16=2*8 OK.
ISO_RAY_PORT="${RL_RAY_PORT:-32000}" bash "$SCRIPTS/run_iso.sh" \
  "$PY" "$REPO/examples/embodiment/train_embodied_agent.py" --config-name libero_spatial_grpo_openvlaoft \
    rollout.model.model_path="$INIT" \
    actor.model.model_path="$INIT" \
    actor.model.unnorm_key="$RL_UNNORM" \
    runner.logger.log_path="$LOGP" \
    runner.save_interval="$RL_SAVE" \
    runner.max_epochs="$RL_MAX_EPOCHS" \
    actor.micro_batch_size="$RL_MICRO" \
    env.train.total_num_envs="$RL_ENVS" \
    env.eval.total_num_envs="$RL_EVAL_ENVS"
echo "RL_SPATIAL_DONE rc=${PIPESTATUS[0]} $(date '+%F %T')"
