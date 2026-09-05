#!/bin/bash
# opd_dance_anchor.sh -- dance (single-suite updates + trajectory thinning) PLUS Memory
# Anchors (ANCHORER, arXiv:2608.26545 transferred): the tail ANCHOR_FRAC of every
# single-suite update is filled with the other-suite samples most similar to the update's
# observations, weighted toward the suite currently drifting furthest from its teacher
# (per-suite KL EMA). Each anchor distills toward its OWN routed teacher = targeted
# rehearsal inside the OPD update. See dance_anchor_augment in fsdp_actor_worker.py.
#
# Derived from opd_dance.sh with three deliberate changes:
#   * anchor knobs passed as hydra overrides (config stays libero_dance_2gpu);
#   * convert step uses the PEFT-LoRA converter (dance checkpoints are plain PEFT --
#     the slot converter FATAL'd on exactly this at the end of the first dance run);
#   * defaults sized for the 50-step run (the 15-step run conserved at 0.745).
#
# Run under the watchdog:
#   SR_PROC_MAX=1000 bash /share/fanruochen-local/dev/scripts/safe_run.sh \
#     /share/fanruochen-local/outputs/opd_dancea_driver.log \
#     bash examples/embodiment/incremental_sft/opd_dance_anchor.sh
set -uo pipefail

TAG="${TAG:-dancea}"
STEPS="${STEPS:-50}"
GPUS="${GPUS:-0,1,2,3}"
PORT="${PORT:-59000}"                      # >=1000 from any other job's ISO_RAY_PORT
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
ANCHOR_FRAC="${ANCHOR_FRAC:-0.15}"         # paper's 10-20% band
ANCHOR_BETA="${ANCHOR_BETA:-0.5}"          # KL-EMA suite-weight exponent (0 = similarity only)
ANCHOR_IMG_W="${ANCHOR_IMG_W:-0.5}"        # image vs instruction weight in the retrieval cosine
# Rollout-side edge coverage (user-ordered 2026-09-04), THREE stacked mechanisms:
#   * TEMP_TRAIN 1.6 -- hotter rollout sampling visits off-nominal (edge) states inside
#     every episode; post-hoc eval stays at 1.0 (TEMP_EVAL).
#   * EDGE_W -- guaranteed rollout QUOTA for the conflicting suites, via the env's
#     existing suite_sample_weights pool (training only; eval stays uniform). The
#     conflict axis is measured, not guessed: goal<->long traded in BOTH mt4 runs
#     (+0.28/-0.20, then the mirror) and both sit below M0 in dance step-8's post-hoc.
#     1.5x weight -> goal 30% / long 30% / spatial 20% / object 20% of episodes.
#   * anchor_frac above -- update-side guarantee that those edge states get trained on.
TEMP_TRAIN="${TEMP_TRAIN:-1.6}"
TEMP_EVAL="${TEMP_EVAL:-1.0}"
# FULL METHOD (user design 2026-09-05): entropy-adaptive KL + entropy chunk selection,
# on top of dance (routing + gradient staggering) + anchors. Set DISTILL_KL=forward and
# CHUNK_SELECT=0 to fall back to the previous dance+anchor-only run for the ablation.
DISTILL_KL="${DISTILL_KL:-entropy_adaptive}"  # low teacher-entropy->reverse, high->+forward
ENT_FWD_SCALE="${ENT_FWD_SCALE:-1.0}"         # forward superposition scale at max teacher entropy
# Chunk importance (our up-weight design): borrows ONLY TIP's parameter-free soft-OR
#   s = 1-(1-h)(1-d), h=norm student entropy, d=KL(student||teacher), per-batch min-max,
# then applies OUR soft reweight w = 1 + kappa*(s/mean s) on mtok (no token dropped).
# CHUNK_SELECT=on -> enabled; =off -> disabled (ablation). kappa scales the emphasis.
CHUNK_SELECT="${CHUNK_SELECT:-on}"
CHUNK_KAPPA="${CHUNK_KAPPA:-1.0}"             # emphasis strength on important chunks
CONF_TAU="${CONF_TAU:-0.0}"                   # redundant under entropy_adaptive; keep 0
EDGE_W="${EDGE_W:-{libero_spatial:1.0,libero_object:1.0,libero_goal:1.5,libero_10:1.5}}"
# RENDER=egl -- typed per-job opt-in for GPU render (user re-approved 2026-09-04).
# gpu_render_env.sh + run_iso.sh + the Python fuses each enforce their own gates and
# fall back to osmesa if any fail; the GPU preflight above already requires FULLY idle
# cards, which this job then occupies (~66G/80G) -- the sanctioned Bug-4905391 pattern.
RENDER="${RENDER:-osmesa}"
O=/share/fanruochen-local/outputs
STUDENT="${STUDENT:-$O/inc_sft_opd/lwf_long_e1000_merged}"
REPO=/home/fanruochen/CL/RLinf
CFG=libero_dance_2gpu
SCRIPTS=/share/fanruochen-local/dev/scripts
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
LOGP="$O/seqcl_${TAG}"

# ---- student ---------------------------------------------------------------------------------
[ -f "$STUDENT/model.safetensors.index.json" ] || {
  echo "ABORT: student is not an HF model dir (no model.safetensors.index.json): $STUDENT"; exit 1; }

# ---- disk ------------------------------------------------------------------------------------
# 50 steps at save_interval 10 -> 5 periodic saves (~32G each: 17G DCP + 15G full_weights)
# + best + the converted model (~15G) = ~222G uncleaned peak. The reaper below trims
# superseded ones but is best-effort; the floor assumes it never ran.
NEED_GB="${NEED_GB:-240}"
free=$(df -BG --output=avail /share/fanruochen-local | tail -1 | tr -dc '0-9')
(( free >= NEED_GB )) || {
  echo "ABORT: only ${free}G free on /share/fanruochen-local (need >=${NEED_GB}G)."
  echo "       Filling this volume takes down every tenant, not just this job."
  exit 1; }

# ---- GPUs (memory AND utilization, 3 samples) ------------------------------------------------
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits) || {
    echo "ABORT: cannot query GPU$g"; exit 1; }
  gfree=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits) || {
    echo "ABORT: cannot query GPU$g"; exit 1; }
  NEED_FREE_MIB="${NEED_FREE_MIB:-45000}"
  (( gfree >= NEED_FREE_MIB )) || {
    echo "ABORT: GPU$g has only ${gfree} MiB free (${used} MiB held), need >=${NEED_FREE_MIB}."; exit 1; }
  busy=0
  for _ in 1 2 3; do
    util=$(nvidia-smi -i "$g" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    (( util < 20 )) || busy=$((busy+1))
    sleep 2
  done
  (( busy < 2 )) || {
    echo "ABORT: GPU$g at ${util}% util -- post-crash rule: training goes ONLY on fully idle cards."
    exit 1; }
done

# ---- environment (default osmesa; RENDER=egl is the keyed opt-in) ----------------------------
set +u
source /home/fanruochen/.rlinf-env.sh 2>/dev/null
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
if [ "$RENDER" = "egl" ]; then
  source /share/fanruochen-local/dev/gpu_render_env.sh   # gated; falls back to osmesa on any gate failure
else
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
fi
set -u
export EMBODIED_PATH="$REPO/examples/embodiment" REPO_PATH="$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export RLINF_CONVERT_VALUE_HEAD=False

# ---- teachers (read from the config; launcher and YAML cannot drift) --------------------------
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
  echo "  teacher ok  $s  $(basename "$b")::$(basename "$a")"
done

echo "[preflight] disk=${free}G  gpus=$GPUS  student=$(basename "$STUDENT")  steps=$STEPS  port=$PORT"
echo "== df =="; df -h /share/fanruochen-local | tail -1
echo "======== DANCE+ANCHOR [$TAG] config=$CFG init=$(basename "$STUDENT") gpus=$GPUS steps=$STEPS  $(date '+%F %T') ========"
echo "         anchor_frac=$ANCHOR_FRAC beta=$ANCHOR_BETA img_w=$ANCHOR_IMG_W  render=${MUJOCO_GL:-osmesa}"
echo "         distill_kl=$DISTILL_KL ent_fwd_scale=$ENT_FWD_SCALE chunk_select=$CHUNK_SELECT kappa=$CHUNK_KAPPA temp_train=$TEMP_TRAIN"
echo "         controls (temp 1.0): M0 macro .783 (150eps) / dance15 best-step8 .745 / mt4w2 .750"

# ---- checkpoint reaper (best-effort; floor above assumes it never ran) ------------------------
KEEP_CKPTS="${KEEP_CKPTS:-2}"
KEEP_STEPS="${KEEP_STEPS:-$(seq -s, "$SAVE_INTERVAL" "$SAVE_INTERVAL" "$STEPS")}"
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
        case ",${KEEP_STEPS}," in
          *",${steps[i]},"*) continue ;;
        esac
        victim="$CKPT_DIR/global_step_${steps[i]}"
        [ -d "$victim" ] || continue
        echo "[reap] removing superseded $victim ($(du -sh "$victim" 2>/dev/null | cut -f1))"
        rm -rf -- "$victim"
      done
    done
  ) &
  REAP_PID=$!
  trap 'kill "$REAP_PID" 2>/dev/null' EXIT
  echo "[reap] keeping newest $KEEP_CKPTS + round-end steps {$KEEP_STEPS} under $CKPT_DIR (pid $REAP_PID)"
fi

# ---- train -----------------------------------------------------------------------------------
EXTRA=("++algorithm.anchor_frac=$ANCHOR_FRAC"
       "++algorithm.anchor_beta=$ANCHOR_BETA"
       "++algorithm.anchor_img_weight=$ANCHOR_IMG_W"
       "algorithm.sampling_params.temperature_train=$TEMP_TRAIN"
       "algorithm.sampling_params.temperature_eval=$TEMP_EVAL"
       "env.train.suite_sample_weights=$EDGE_W"
       "algorithm.distill_kl=$DISTILL_KL"
       "++algorithm.ent_adaptive_fwd_scale=$ENT_FWD_SCALE"
       "++algorithm.chunk_select=$CHUNK_SELECT"
       "++algorithm.chunk_select_kappa=$CHUNK_KAPPA"
       "++algorithm.distill_conf_tau=$CONF_TAU")
[ -n "${GRAD_CKPT:-}" ] && EXTRA+=("++actor.fsdp_config.gradient_checkpointing=$GRAD_CKPT")
[ -n "${MICRO:-}" ]     && EXTRA+=("actor.micro_batch_size=$MICRO")
[ -n "${PROFILE:-}" ]   && EXTRA+=("++algorithm.profile_train_phases=true" "++algorithm.profile_log_every=${PROFILE_EVERY:-50}")
[ -n "${RESUME_DIR:-}" ] && EXTRA+=("++runner.resume_dir=$RESUME_DIR")
[ -n "${ENVS:-}" ] && EXTRA+=("env.train.total_num_envs=$ENVS")
echo "         overrides: ${EXTRA[*]}"

MT4SLOT_GPUS="$GPUS" MT4SLOT_TAG="$TAG" MT4SLOT_STUDENT_PATH="$STUDENT" \
MT4SLOT_MAX_STEPS="$STEPS" MT4SLOT_SAVE_INTERVAL="$SAVE_INTERVAL" \
ISO_RAY_PORT="$PORT" ISO_RENDER="$RENDER" bash "$SCRIPTS/run_iso.sh" \
  "$PY" "$REPO/examples/embodiment/train_embodied_agent.py" --config-name "$CFG" \
    ${EXTRA[@]+"${EXTRA[@]}"}
RC=$?
[ -n "$REAP_PID" ] && { kill "$REAP_PID" 2>/dev/null; trap - EXIT; }
echo "OPD_TRAIN_DONE rc=$RC $(date '+%F %T')"
echo "== df =="; df -h /share/fanruochen-local | tail -1
(( RC == 0 )) || exit "$RC"

# ---- convert (PEFT-LoRA converter -- dance checkpoints are plain PEFT, NOT slot) --------------
CKPT=$(find "$LOGP" -name full_weights.pt -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$CKPT" ] || { echo "WARN: no full_weights.pt under $LOGP -- nothing to convert"; exit 0; }
CONV="$LOGP/converted/${TAG}"
CONVERTER="$REPO/opd_distill/scripts/convert_oft_lora_ckpt.sh"
if [ ! -x "$CONVERTER" ] && [ ! -f "$CONVERTER" ]; then
  echo "WARN: $CONVERTER not found; convert by hand from $CKPT"
  exit 0
fi
echo "======== CONVERT $CKPT -> $CONV (base=$(basename "$STUDENT")) ========"
bash "$CONVERTER" "$CKPT" "$CONV" "$STUDENT" libero_130_no_noops_trajall \
  || { echo "WARN: LoRA conversion failed; the training checkpoint is intact at $CKPT"; exit 0; }
[ -f "$CONV/model.safetensors.index.json" ] || {
  echo "WARN: conversion produced no model at $CONV"; exit 0; }
echo "OPD_CONVERTED=$CONV"
echo "OPD_${TAG}_DONE $(date '+%F %T')"
