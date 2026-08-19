#!/bin/bash
# rl_long.sh — GRPO RL on libero_10 (long), starting from the CL student's LoRA adapter.
#
#   init   = opd_init_dual1250  +  adapters/step_1000   (long .56 / spatial .76 / object .96 / goal .68)
#   config = libero_10_grpo_long_lora.yaml
#
# WHY LoRA-RL AND NOT FULL-FT RL (the earlier rl_spatial.sh / rl_object.sh path):
#   1. MEMORY. Full-FT RL ships the whole 7B at every sync_model_to_rollout. Even with bucket_syncer
#      that peaked at 408G of the 477G cgroup cap -- 69G of headroom, and patch_syncer blew straight
#      through it (kernel OOM -> worker killed -> "NCCL watchdog stuck"). LoRA-RL syncs ~110M params.
#   2. IO. Full-FT checkpoints are 15G of full_weights.pt each; a LoRA checkpoint is adapter-sized,
#      so save_interval can be 5 instead of 25 without flooding the disk.
#   3. IT PRESERVES THE OTHER SUITES. This is a continual-learning student -- spatial/object/goal must
#      survive. Measured previously: LoRA+RL kept off-task mean .66 vs full-FT RL .54.
#   RLinf supports it directly: rlinf/models/__init__.py does
#   PeftModel.from_pretrained(model, cfg.lora_path, is_trainable=True), i.e. GRPO keeps training OUR
#   step_1000 adapter in place rather than starting a fresh one.
#
# Usage: [RL_PLACEMENT=0,1,2,6 RL_ENVS=32 RL_MICRO=32 RL_MAX_EPOCHS=30 RL_SAVE=5 RL_TAG=t1] rl_long.sh
set -uo pipefail
REPO=/home/fanruochen/CL/RLinf
SCRIPTS=/share/fanruochen-local/dev/scripts
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python

# GPU selection is the config's component_placement RANGE/LIST, NOT CUDA_VISIBLE_DEVICES (ray ignores
# that and sees all 8). A non-contiguous comma list is legal; only unique+ascending is required.
export RL_PLACEMENT="${RL_PLACEMENT:-0,1,2,6}"
export RL_ENVS="${RL_ENVS:-32}"            # 32//4gpu//1 = 8 = group_size (64 blew the 800-proc cap)
export RL_EVAL_ENVS="${RL_EVAL_ENVS:-8}"
export RL_MICRO="${RL_MICRO:-32}"
export RL_ROLLOUT_EPOCH="${RL_ROLLOUT_EPOCH:-4}"
export RL_GLOBAL_BATCH="${RL_GLOBAL_BATCH:-8192}"   # rollout_size 4*512=2048 % (8192/4gpu)=0
export RL_MAX_EPOCHS="${RL_MAX_EPOCHS:-30}"
export RL_SAVE="${RL_SAVE:-5}"
export RL_TAG="${RL_TAG:-t1}"
# Stage 2 starts from the OPD adapter, not the SFT one. Whatever adapter is named here also DECIDES
# THE RANK: RLinf loads it with PeftModel.from_pretrained and inherits the adapter's own r, ignoring
# lora_rank. So an r128 OPD adapter gives r128 RL with nothing merged.
export RL_INIT_MODEL="${RL_INIT_MODEL:-/share/fanruochen-local/outputs/inc_sft_opd/lwf_long_e1000_merged}"
export RL_INIT_ADAPTER="${RL_INIT_ADAPTER:?set RL_INIT_ADAPTER=<adapter dir> (its rank is the RL rank)}"
export RL_LR="${RL_LR:-1.0e-4}"
LOGP="/share/fanruochen-local/outputs/rl_long_${RL_TAG}"
[ -f "$RL_INIT_ADAPTER/adapter_model.safetensors" ] || { echo "ABORT: not an adapter dir: $RL_INIT_ADAPTER"; exit 1; }
[ -f "$RL_INIT_MODEL/model.safetensors.index.json" ] || { echo "ABORT: not an HF model dir: $RL_INIT_MODEL"; exit 1; }
ADAPTER_R=$(python3 -c "import json;print(json.load(open('$RL_INIT_ADAPTER/adapter_config.json'))['r'])")
ADAPTER_BASE=$(python3 -c "import json;print(json.load(open('$RL_INIT_ADAPTER/adapter_config.json'))['base_model_name_or_path'])")
[ "$(readlink -f "$ADAPTER_BASE")" = "$(readlink -f "$RL_INIT_MODEL")" ] \
  || { echo "ABORT: adapter was trained on $ADAPTER_BASE, not $RL_INIT_MODEL"; exit 1; }

if [[ "$RL_PLACEMENT" == *-* ]]; then NGPU=$(( ${RL_PLACEMENT#*-} - ${RL_PLACEMENT%-*} + 1 ))
else NGPU=$(echo "$RL_PLACEMENT" | tr ',' '\n' | grep -c .); fi

# ---- constraint arithmetic, checked here so a violation is a clear message not a mid-run assert ----
(( RL_ENVS % NGPU == 0 ))            || { echo "ABORT: RL_ENVS=$RL_ENVS not divisible by n_gpu=$NGPU"; exit 1; }
(( (RL_ENVS / NGPU) % 8 == 0 ))      || { echo "ABORT: RL_ENVS/n_gpu = $((RL_ENVS/NGPU)) not divisible by group_size 8"; exit 1; }
(( RL_GLOBAL_BATCH % NGPU == 0 ))    || { echo "ABORT: RL_GLOBAL_BATCH not divisible by n_gpu"; exit 1; }
(( (RL_ROLLOUT_EPOCH * 512) % (RL_GLOBAL_BATCH / NGPU) == 0 )) \
  || { echo "ABORT: rollout_size $((RL_ROLLOUT_EPOCH*512)) not divisible by batch_per_rank $((RL_GLOBAL_BATCH/NGPU))"; exit 1; }

# ---- RED-LINE PRE-FLIGHT ----
avail=$(df -BG --output=avail /share/fanruochen-local | tail -1 | tr -dc '0-9')
(( avail >= 150 )) || { echo "ABORT: only ${avail}G free on /share/fanruochen-local"; exit 1; }
anon=$(awk '/^anon /{a=$2} /^slab /{s=$2} /^kernel_stack /{k=$2} END{printf "%.0f",(a+s+k)/1073741824}' /sys/fs/cgroup/memory.stat)
(( anon < 150 )) || { echo "ABORT: cgroup anon already ${anon}G -- something else is running"; exit 1; }
iow=$(vmstat 1 2 | tail -1 | awk '{print $16}')
(( ${iow:-0} < 25 )) || { echo "ABORT: system iowait ${iow}% -- disk already busy"; exit 1; }
oom0=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events)

set +u
source /home/fanruochen/.rlinf-env.sh 2>/dev/null
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
source /share/fanruochen-local/dev/gpu_render_env.sh     # GPU-EGL render; CPU render makes rollout 4x slower
set -u
export EMBODIED_PATH="$REPO/examples/embodiment" REPO_PATH="$REPO"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$REPO:${PYTHONPATH:-}"
export ROBOT_PLATFORM=LIBERO
# Ray's memory monitor reads cgroup memory.current, which INCLUDES reclaimable page cache -- it
# false-fired at 98.5% while true MemAvailable was 235G and killed a worker mid-collective. safe_run
# still guards true MemAvailable, and the pre-flight above guards anon.
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$LOGP"
echo "======== GRPO-RL [long/libero_10] LoRA-on-adapter  $(date '+%F %T') ========"
echo "  init  = $(basename "$RL_INIT_MODEL") + $(basename "$RL_INIT_ADAPTER")  (adapter r=$ADAPTER_R -> RL rank=$ADAPTER_R)"
echo "  lr    = $RL_LR   (2e-5 was a full-FT lr and left the policy frozen: grad_norm .027, ratio_abs .016)"
echo "  placement=$RL_PLACEMENT (n$NGPU)  envs=$RL_ENVS  micro=$RL_MICRO  rollout_epoch=$RL_ROLLOUT_EPOCH  gb=$RL_GLOBAL_BATCH"
echo "  max_epochs=$RL_MAX_EPOCHS  save_interval=$RL_SAVE  out=$LOGP"
echo "  preflight: disk=${avail}G  anon=${anon}G  iowait=${iow}%  oom_kill_baseline=$oom0"
df -h /share/fanruochen-local | tail -1

ISO_RAY_PORT="${RL_RAY_PORT:-42000}" bash "$SCRIPTS/run_iso.sh" \
  "$PY" "$REPO/examples/embodiment/train_embodied_agent.py" --config-name libero_10_grpo_long_lora \
    runner.max_epochs="$RL_MAX_EPOCHS"
rc=$?
oom1=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events)
echo "RL_LONG_DONE rc=$rc  oom_kill: $oom0 -> $oom1  $(date '+%F %T')"
# oom_kill incrementing is THE signature of the container-OOM crash; NCCL "watchdog stuck" is only
# its symptom. Check this line first on any hang.
