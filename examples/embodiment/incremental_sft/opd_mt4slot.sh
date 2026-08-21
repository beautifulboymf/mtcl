#!/bin/bash
# opd_mt4slot.sh — R1 of the slot-LoRI study: mt4's 4-teacher routed OPD, with the student's ONE
# shared rank-128 LoRA replaced by FOUR per-suite LoRA slots on orthogonal input subspaces.
#
#   student  = inc_sft_opd/lwf_long_e1000_merged      (byte-for-byte mt4's starting point)
#   teachers = spatial | object | goal | long, routed per suite (mt4's exact set)
#   slots    = read from the config's actor.model.slot_lora (never restated here)
#   config   = examples/embodiment/config/libero_mt4slot_6gpu.yaml
#
# WHY THE CONTROLS COST NOTHING. mt4 (4-suite mean 0.745) and mt4w2 (0.750) already ran this
# setup with the shared LoRA from this same student, and both said the joint routed form only
# REDISTRIBUTES capability -- mt4 bought long +0.28 with goal -0.20, mt4w2 bought goal +0.12 with
# long -0.10, spatial and object never moved. So R1 needs no baseline run of its own, PROVIDED
# nothing but the student's LoRA structure changes. Everything else is pinned in the YAML at
# mt4's resolved values, and this script adds no hydra overrides on top of it -- on purpose. If
# you need to change a knob, change it in the config where it is visible next to the comment
# explaining what it is pinned to, not as an invisible override here.
#
# Run it under the watchdog, never bare:
#   SR_PROC_MAX=1000 bash /share/fanruochen-local/dev/scripts/safe_run.sh \
#     /share/fanruochen-local/outputs/opd_mt4slot_driver.log \
#     bash examples/embodiment/incremental_sft/opd_mt4slot.sh
set -uo pipefail

TAG="${TAG:-mt4slot}"
STEPS="${STEPS:-15}"                       # mt4 ran 15
GPUS="${GPUS:-0,1,2,3,4,5}"                # six, as mt4
PORT="${PORT:-58000}"                      # isolated ray head; keep >=1000 from any other job
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"        # mt4 ran 5
O=/share/fanruochen-local/outputs
STUDENT="${STUDENT:-$O/inc_sft_opd/lwf_long_e1000_merged}"
REPO=/home/fanruochen/CL/RLinf
CFG=libero_mt4slot_6gpu
SCRIPTS=/share/fanruochen-local/dev/scripts
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
LOGP="$O/seqcl_${TAG}"

# ---- student ---------------------------------------------------------------------------------
# Not just "does the directory exist": mt4 and mt4w2 are controls for this run only because the
# student is bit-identical to what mt4 started from, and a half-copied or wrong-shaped directory
# would still start a 5-hour job that means nothing.
[ -f "$STUDENT/model.safetensors.index.json" ] || {
  echo "ABORT: student is not an HF model dir (no model.safetensors.index.json): $STUDENT"; exit 1; }

# ---- disk ------------------------------------------------------------------------------------
# THE one red line that has actually taken this machine down: a full /share/fanruochen-local
# stopped every tenant's writes for days, and it happened by letting checkpoints pile up without
# ever looking at df.
#
# THE FLOOR IS THE UNCLEANED PEAK. The previous 120G was justified by "the runner keeps the
# latest plus best". It does not: embodied_runner._save_checkpoint writes
# checkpoints/global_step_<N>/ for every periodic save and only ever rmtree's the TAGGED path,
# so the global_step_* directories ACCUMULATE. mt4's own driver log shows saves at steps 5, 10
# and 15 plus `best` overwritten repeatedly, and mt4's surviving directories measure 32G each
# (17G DCP shards + 15G full_weights.pt) -- 128G concurrent, already over the old floor.
#
#   R1 arithmetic: R = 288 is ~2.25x mt4's rank-128 LoRA, so ~36G per directory.
#     STEPS/SAVE_INTERVAL = 15/5 = 3 periodic saves, + 1 for `best`   -> 4 x 36G = 144G peak
#     + the merged HF model the convert step writes (~15G, measured)  -> 159G total after
#   Floor 175G = 159G + ~16G of headroom for tensorboard/logs and a checkpoint that comes out
#   bigger than the estimate. At today's ~800G free this never bites; it exists for the tight
#   case, where the old 120G would have PASSED at 121G free and then filled a shared volume at
#   about step 10.
#
# The reaper below usually keeps the real high-water mark near 108G, but the floor deliberately
# does NOT assume it ran: a best-effort background loop must never be load-bearing for the guard
# that stops this volume from filling.
NEED_GB="${NEED_GB:-175}"
free=$(df -BG --output=avail /share/fanruochen-local | tail -1 | tr -dc '0-9')
(( free >= NEED_GB )) || {
  echo "ABORT: only ${free}G free on /share/fanruochen-local (need >=${NEED_GB}G:"
  echo "       4 checkpoint dirs x ~36G concurrent = 144G, + ~15G for the converted model)"
  echo "       Free space first -- filling this volume takes down every tenant, not just this job."
  exit 1; }

# ---- GPUs ------------------------------------------------------------------------------------
# BOTH memory and utilization. A memory-only check is how this project once co-launched on top of
# another tenant's compute-bound job: it showed under 2% memory and 73-97% utilization for a
# quarter of an hour before a human noticed. Sample utilization three times with a gap, because a
# single instantaneous reading catches idle ticks between kernels and reads as free.
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits) || {
    echo "ABORT: cannot query GPU$g"; exit 1; }
  (( used < 5000 )) || { echo "ABORT: GPU$g holds ${used} MiB -- not ours to take"; exit 1; }
  busy=0
  for _ in 1 2 3; do
    util=$(nvidia-smi -i "$g" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    (( util < 20 )) || busy=$((busy+1))
    sleep 2
  done
  (( busy < 2 )) || {
    echo "ABORT: GPU$g at ${util}% util (memory looked free, compute is not) -- not ours to take"
    exit 1; }
done

# ---- environment -----------------------------------------------------------------------------
# Same three sources as opd_one.sh / opd_multi.sh, in the same order. gpu_render_env.sh must come
# after .rlinf-env.sh: the latter forces MUJOCO_GL=osmesa and would leave every LIBERO rollout
# rendering on the CPU (~28 ms/step of GPU-EGL render becomes the rollout bottleneck on CPU).
# run_iso.sh re-applies the same stack for the raylet, which forks the env workers.
set +u   # the venv activate script reads unbound PYTHONPATH
source /home/fanruochen/.rlinf-env.sh 2>/dev/null
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
source /share/fanruochen-local/dev/gpu_render_env.sh
set -u
export EMBODIED_PATH="$REPO/examples/embodiment" REPO_PATH="$REPO"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$REPO:${PYTHONPATH:-}"
export RLINF_CONVERT_VALUE_HEAD=False

# ---- teachers --------------------------------------------------------------------------------
# Read straight out of the config rather than repeated here, so the launcher and the YAML cannot
# drift into naming different teachers. Each entry is "<base_dir>::<adapter_dir>"; two distinct
# bases across the four (long is an adapter on the student's own merged model), which the actor
# handles by grouping teachers by base -- 2 bases + 4 adapters, not four full 7B models.
mapfile -t TEACHERS < <("$PY" - "$REPO/examples/embodiment/config/$CFG.yaml" <<'PYEOF'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
for suite, path in cfg.actor.teacher_map.items():
    print(f"{suite}\t{path}")
PYEOF
) || { echo "ABORT: could not read actor.teacher_map out of $CFG.yaml"; exit 1; }
(( ${#TEACHERS[@]} == 4 )) || {
  echo "ABORT: expected 4 teachers in $CFG.yaml, got ${#TEACHERS[@]}"; exit 1; }
for row in "${TEACHERS[@]}"; do
  s="${row%%$'\t'*}"; t="${row#*$'\t'}"
  b="${t%%::*}"; a="${t#*::}"
  [ -f "$b/model.safetensors.index.json" ] || { echo "ABORT: teacher base missing: $b"; exit 1; }
  [ "$a" = "$t" ] || [ -f "$a/adapter_model.safetensors" ] || {
    echo "ABORT: teacher adapter missing: $a"; exit 1; }
  r=$("$PY" -c "import json;print(json.load(open('$a/adapter_config.json'))['r'])" 2>/dev/null || echo "?")
  echo "  teacher ok  $s  r=$r  $(basename "$b")::$(basename "$a")"
done

# ---- slot layout -----------------------------------------------------------------------------
# Derived from the config, never restated. The final review counted FOUR copies of this rank list
# (YAML, this script, the converter's default, the converter wrapper's default) and only the SUM
# is checkable against a checkpoint -- a wrong ORDER is undetectable because the merge sums over
# every slot and is blind to where the partition falls. One source of truth is the only fix.
read -r SLOT_ORDER_CSV SLOT_RANKS_CSV SLOT_R < <("$PY" - "$REPO/examples/embodiment/config/$CFG.yaml" <<'PYEOF'
import sys
from omegaconf import OmegaConf
sl = OmegaConf.load(sys.argv[1]).actor.model.slot_lora
order = list(sl.slot_order)
ranks = [int(sl.slot_ranks[s]) for s in order]
print(",".join(order), ",".join(map(str, ranks)), sum(ranks))
PYEOF
) || { echo "ABORT: could not read actor.model.slot_lora out of $CFG.yaml"; exit 1; }
[ -n "$SLOT_RANKS_CSV" ] || { echo "ABORT: empty slot layout in $CFG.yaml"; exit 1; }

echo "[preflight] disk=${free}G  gpus=$GPUS  student=$(basename "$STUDENT")  steps=$STEPS  port=$PORT"
echo "== df =="; df -h /share/fanruochen-local | tail -1
echo "======== SLOT-LORI R1 [$TAG] config=$CFG init=$(basename "$STUDENT") gpus=$GPUS steps=$STEPS  $(date '+%F %T') ========"
echo "         slots: $SLOT_ORDER_CSV = $SLOT_RANKS_CSV  (R=$SLOT_R)"
echo "         controls: mt4 0.745 (opd_mt4i_driver.log) / mt4w2 0.750 (opd_mt4w2_driver.log)"

# ---- checkpoint reaper -------------------------------------------------------------------------
# The runner never deletes a global_step_* directory, so a 15-step run at save_interval 5 ends
# holding three of them plus `best` -- ~144G that nothing but this loop will free while the job is
# still running. Best-effort ONLY: the disk floor above is sized for the uncleaned peak, so if this
# loop never starts, dies, or is switched off, the run is still safe. Set KEEP_CKPTS=0 to disable.
#
# The safety rules it must not break, in the order they matter:
#   * NEVER the newest. It keeps the newest KEEP_CKPTS (>=2) by STEP NUMBER, so the directory
#     currently being written is kept AND so is the last complete one -- a crash mid-write can
#     never leave the run with zero loadable checkpoints (the "half-written DCP, missing
#     .metadata, unconvertible" failure).
#   * NEVER `best`, and never anything outside this run's own checkpoints/ directory. The find
#     regex matches only `global_step_<digits>`, and CKPT_DIR is asserted to live under $O first.
#   * NEVER outlive its parent. If safe_run.sh SIGKILLs this script the EXIT trap does not fire,
#     so the loop also checks that the launcher's PID is still alive and exits within one tick --
#     an orphan reaper would otherwise eat the checkpoints of the NEXT run with the same TAG.
KEEP_CKPTS="${KEEP_CKPTS:-2}"
CKPT_DIR="$LOGP/seqcl_${TAG}/checkpoints"
REAP_PID=""
case "$CKPT_DIR" in
  "$O"/*) ;;
  *) echo "WARN: refusing to reap outside $O (CKPT_DIR=$CKPT_DIR)"; KEEP_CKPTS=0 ;;
esac
(( KEEP_CKPTS == 0 || KEEP_CKPTS >= 2 )) || KEEP_CKPTS=2
if (( KEEP_CKPTS > 0 )); then
  MAIN_PID=$$
  (
    while kill -0 "$MAIN_PID" 2>/dev/null; do
      sleep 120
      [ -d "$CKPT_DIR" ] || continue
      mapfile -t steps < <(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -type d \
        -regextype posix-extended -regex '.*/global_step_[0-9]+' -printf '%f\n' 2>/dev/null \
        | sed 's/^global_step_//' | sort -n)
      n=${#steps[@]}
      (( n > KEEP_CKPTS )) || continue
      for (( i = 0; i < n - KEEP_CKPTS; i++ )); do
        victim="$CKPT_DIR/global_step_${steps[i]}"
        [ -d "$victim" ] || continue
        echo "[reap] removing superseded $victim ($(du -sh "$victim" 2>/dev/null | cut -f1))"
        rm -rf -- "$victim"
      done
    done
  ) &
  REAP_PID=$!
  trap 'kill "$REAP_PID" 2>/dev/null' EXIT
  echo "[reap] keeping the newest $KEEP_CKPTS global_step_* under $CKPT_DIR (pid $REAP_PID)"
fi

# ---- train -----------------------------------------------------------------------------------
# No hydra overrides: everything this run needs is in the config. run_iso.sh puts the job on its
# OWN ray head so it cannot join (or be joined by) another job's cluster -- two concurrent RLinf
# jobs on the default ports collide on ray's internal port range and both die.
MT4SLOT_GPUS="$GPUS" MT4SLOT_TAG="$TAG" MT4SLOT_STUDENT_PATH="$STUDENT" \
MT4SLOT_MAX_STEPS="$STEPS" MT4SLOT_SAVE_INTERVAL="$SAVE_INTERVAL" \
ISO_RAY_PORT="$PORT" bash "$SCRIPTS/run_iso.sh" \
  "$PY" "$REPO/examples/embodiment/train_embodied_agent.py" --config-name "$CFG"
RC=$?
# The reaper's only job was to hold the mid-run peak down; past this point every save is done
# and the convert step below reads the newest checkpoint, so stop it before that walk starts.
[ -n "$REAP_PID" ] && { kill "$REAP_PID" 2>/dev/null; trap - EXIT; }
echo "OPD_TRAIN_DONE rc=$RC $(date '+%F %T')"
echo "== df =="; df -h /share/fanruochen-local | tail -1
(( RC == 0 )) || exit "$RC"

# ---- convert ---------------------------------------------------------------------------------
# A slot checkpoint is NOT a PEFT checkpoint, so it needs the slot converter, and that converter
# needs the SAME slot layout the run trained with -- a wrong scale is caught by the checkpoint's
# own record, a wrong rank ORDER is not (only the sum is checkable), so slot_order's ranks are
# passed explicitly in slot_order. Non-fatal on purpose: the training result is already on disk,
# and failing the whole 5-hour job over a merge step would be the wrong trade.
CKPT=$(find "$LOGP" -name full_weights.pt -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$CKPT" ] || { echo "WARN: no full_weights.pt under $LOGP -- nothing to convert"; exit 0; }
CONV="$LOGP/converted/${TAG}"
CONVERTER="$REPO/opd_distill/scripts/convert_oft_slot_ckpt.sh"
if [ ! -x "$CONVERTER" ]; then
  echo "WARN: $CONVERTER not found; convert by hand from $CKPT"
  exit 0
fi
echo "======== CONVERT $CKPT -> $CONV (base=$(basename "$STUDENT")) ========"
SLOT_RANKS="$SLOT_RANKS_CSV" SLOT_SCALE_MODE=match_mt4 SLOT_REF_RANK=128 SLOT_EPS=1e-6 \
  bash "$CONVERTER" "$CKPT" "$CONV" "$STUDENT" libero_130_no_noops_trajall \
  || { echo "WARN: slot conversion failed; the training checkpoint is intact at $CKPT"; exit 0; }
[ -f "$CONV/model.safetensors.index.json" ] || {
  echo "WARN: conversion produced no model at $CONV"; exit 0; }
echo "OPD_CONVERTED=$CONV"
echo "OPD_${TAG}_DONE $(date '+%F %T')"
