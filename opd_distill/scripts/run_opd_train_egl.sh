#!/bin/bash
# VLA-OPD training launcher — GPU-render (EGL) variant. Same as run_opd_train.sh
# but sources the isolated 535.179 GL libs and forces MUJOCO_GL=egl (NVIDIA render).
# Args: CONFIG_NAME [hydra overrides...]
set -x
CONFIG_NAME="${1:?need config name}"; shift
source ~/.rlinf-env.sh                       # render-libs + proxy
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
source /share/fanruochen-local/dev/gpu_render_env.sh   # 535.179 GL + MUJOCO_GL=egl (must be AFTER rlinf-env)
cd /home/fanruochen/CL/RLinf
export EMBODIED_PATH="/home/fanruochen/CL/RLinf/examples/embodiment"
export REPO_PATH="/home/fanruochen/CL/RLinf"
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"
# EGL GPU render (gpu_render_env.sh already set MUJOCO_GL=egl / PYOPENGL_PLATFORM=egl); re-assert.
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python "${SRC_FILE}" --config-path "${EMBODIED_PATH}/config/" --config-name "${CONFIG_NAME}" "$@"
