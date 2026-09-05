#!/bin/bash
# VLA-OPD training launcher — historical name "_egl"; NOW OSMESA-ONLY (EGL banned
# 2026-09-03: host crashes 2026-08-27/29). Name kept because run_iso_train.sh + watchdog reference it.
# Args: CONFIG_NAME [hydra overrides...]
set -x
CONFIG_NAME="${1:?need config name}"; shift
source ~/.rlinf-env.sh                       # render-libs + proxy
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
source /share/fanruochen-local/dev/gpu_render_env.sh   # osmesa shim (EGL quarantined)
cd /home/fanruochen/CL/RLinf
export EMBODIED_PATH="/home/fanruochen/CL/RLinf/examples/embodiment"
export REPO_PATH="/home/fanruochen/CL/RLinf"
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"
# CPU osmesa render, unconditionally (EGL banned on this machine).
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python "${SRC_FILE}" --config-path "${EMBODIED_PATH}/config/" --config-name "${CONFIG_NAME}" "$@"
