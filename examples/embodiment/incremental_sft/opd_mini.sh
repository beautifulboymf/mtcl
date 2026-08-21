#!/bin/bash
# opd_mini.sh — 迷你计时测试:最小合法配置跑 2 步,只为量时间,不为出结果。
#
# 用途:隔离"4 教师 / 2 底座" vs "3 教师 / 1 底座"的速度差。主实验的唯一结构性未知就是
# long 教师挂在另一个底座上(其余几何已对齐 mt3b),这个对照能直接回答它值不值得用 SVD 统一。
#
# 注意:rollout_epoch=1 会破坏蒸馏质量(记忆里 goal 因此从 0.62 掉到 0.44)——
# 这里只测时间,不看 SR,所以可以用;真实验绝不能用 1。
#
# 用法: opd_mini.sh <gpu> <port> <tag> <3|4>
set -uo pipefail
GPU="${1:?gpu}"; PORT="${2:?port}"; TAG="${3:?tag}"; NT="${4:-4}"
O=/share/fanruochen-local/outputs
STUDENT="$O/inc_sft_opd/lwf_long_e1000_merged"
B130="$O/inc_sft_opd/base_stats130"
T_SPATIAL="$B130::$O/seqcl_rlspat_opd/spatial_cat_r160"
T_OBJECT="$B130::$O/inc_sft_opd/sft_object_base3_cont/openvla-7b-base+libero_object_no_noops+b32+lr-0.0003+lora-r32+dropout-0.0--image_aug/adapters/step_500"
T_GOAL="$B130::$O/inc_sft_opd/teachers_r160/goal_r160"
T_LONG="$O/inc_sft_opd/lwf_long_e1000_merged::$O/seqcl_long_opd130/adapter/long_opd130"

PAIRS=(libero_spatial="$T_SPATIAL" libero_object="$T_OBJECT" libero_goal="$T_GOAL")
[ "$NT" = "4" ] && PAIRS+=(libero_10="$T_LONG")

free=$(df -BG --output=avail /share/fanruochen-local | tail -1 | tr -dc '0-9')
(( free >= 150 )) || { echo "ABORT: disk ${free}G"; exit 1; }
anon=$(awk '/^anon /{a=$2} /^slab /{s=$2} /^kernel_stack /{k=$2} END{printf "%.0f",(a+s+k)/1073741824}' /sys/fs/cgroup/memory.stat)
(( anon < 300 )) || { echo "ABORT: anon ${anon}G"; exit 1; }
echo "[mini-$TAG] $NT 教师  GPU$GPU  port=$PORT  disk=${free}G anon=${anon}G"

cd /home/fanruochen/CL/RLinf
exec env OPD_GPUS="$GPU" OPD_ENVS=8 OPD_MICRO=8 OPD_ROLLOUT_EPOCH=1 OPD_STEPS=2 \
     OPD_RAY_PORT="$PORT" SEQCL_SAVE_INTERVAL=999 \
     OPD_EXTRA="actor.global_batch_size=32" \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash examples/embodiment/incremental_sft/opd_multi.sh "$STUDENT" "$TAG" "${PAIRS[@]}"
