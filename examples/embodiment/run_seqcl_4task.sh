#!/bin/bash
# Sequential continual learning on 4 LIBERO suites via OPD (spatial -> object -> goal -> long).
# Chains 4 training stages; each stage resumes from the previous stage's weights and rehearses
# all earlier suites (single 130 generalist teacher). Schedule/weights come from the core
# algorithm module rlinf/algorithms/embodied_seqcl.py.
#
#   student(stage0) = RLinf-OpenVLAOFT-LIBERO-130-Base-Lora  (SFT generalist)
#   teacher(all)    = RLinf-OpenVLAOFT-LIBERO-130            (loaded once, routed to every suite)
#
# Between stages: convert stage-k DCP LoRA ckpt -> merged HF model dir = stage-(k+1) student.
#
# USAGE (run manually; NOT auto-run):
#   run_seqcl_4task.sh            # all 4 stages, sequentially
#   run_seqcl_4task.sh 0 1        # only stages 0..1
# Tunables (env): SEQCL_GPUS=6-7  SEQCL_MAX_STEPS=50  SEQCL_SAVE_INTERVAL=25
#
# Post-hoc per-suite SR (the forgetting study) is done SEPARATELY after each stage with the
# existing eval scripts (in-training eval is unreliable) -- see notes at the bottom.
set -euo pipefail

# ---- environment (mirror datafree_train.sh) --------------------------------------------
cd /home/fanruochen/CL/RLinf
source /home/fanruochen/.rlinf-env.sh
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
source /share/fanruochen-local/dev/gpu_render_env.sh
export EMBODIED_PATH="/home/fanruochen/CL/RLinf/examples/embodiment"
export REPO_PATH="/home/fanruochen/CL/RLinf"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export RLINF_CONVERT_VALUE_HEAD=False   # OPD student has no value head

CONFIG=libero_seqcl_opd_2gpu
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
SCRIPTS=/share/fanruochen-local/dev/scripts
SEQCL_MODULE=rlinf/algorithms/embodied_seqcl.py
BASE_STUDENT=/share/fanruochen-local/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora
OUT_ROOT=/share/fanruochen-local/outputs
export SEQCL_GPUS="${SEQCL_GPUS:-6-7}"

STAGE_START="${1:-0}"
STAGE_END="${2:-3}"

# student for the NEXT stage; stage 0 starts from the SFT generalist.
NEXT_STUDENT="$BASE_STUDENT"

for STAGE in $(seq "$STAGE_START" "$STAGE_END"); do
  echo "================ SEQCL STAGE $STAGE ================"

  # per-stage plan (current suite, active suites, suite weights, teacher map) from the
  # single source of truth. Exports SEQCL_CURRENT_SUITE / SEQCL_ACTIVE_SUITES / SEQCL_SUITE_WEIGHTS.
  eval "$("$PY" "$SEQCL_MODULE" "$STAGE")"
  export SEQCL_STUDENT_PATH="$NEXT_STUDENT"
  echo "  current_suite = $SEQCL_CURRENT_SUITE"
  echo "  active_suites = $SEQCL_ACTIVE_SUITES"
  echo "  suite_weights = $SEQCL_SUITE_WEIGHTS"
  echo "  student       = $SEQCL_STUDENT_PATH"

  # RED LINE: never start a checkpointing job without confirming disk headroom.
  echo "== df /share/fanruochen-local =="; df -h /share/fanruochen-local | tail -1
  FREE_G=$(df -BG --output=avail /share/fanruochen-local | tail -1 | tr -dc '0-9')
  if [ "${FREE_G:-0}" -lt 60 ]; then
    echo "ABORT: <60G free on /share/fanruochen-local (stage ckpts ~15-40G). Clean up first."
    exit 1
  fi

  LOG_PATH="$OUT_ROOT/seqcl_${SEQCL_CURRENT_SUITE}"
  LOGFILE="$OUT_ROOT/seqcl_stage${STAGE}_${SEQCL_CURRENT_SUITE}.log"

  # ---- train this stage under the hard watchdog ----
  SR_PROC_MAX=800 "$SCRIPTS/safe_run.sh" "$LOGFILE" \
    "$PY" examples/embodiment/train_embodied_agent.py --config-name "$CONFIG"
  echo "SEQCL_STAGE_${STAGE}_TRAIN_DONE $(date '+%F %T')"

  # ---- locate the last DCP ckpt (val disabled -> no 'best', use newest global_step) ----
  CKPT=$(find "$LOG_PATH" -name full_weights.pt -printf '%T@ %p\n' 2>/dev/null \
           | sort -rn | head -1 | cut -d' ' -f2-)
  if [ -z "$CKPT" ]; then
    echo "ABORT: no full_weights.pt found under $LOG_PATH after stage $STAGE."
    exit 1
  fi
  echo "  stage $STAGE ckpt = $CKPT"

  # ---- convert LoRA ckpt -> merged HF model dir = next stage's student ----
  # (OFT defaults: --lora-rank 128 --num-action-chunks 8; base only supplies config/norm)
  CONVERTED="$LOG_PATH/converted/stage${STAGE}_${SEQCL_CURRENT_SUITE}"
  "$SCRIPTS/convert_oft_lora_ckpt.sh" "$CKPT" "$CONVERTED" "$BASE_STUDENT"
  echo "SEQCL_STAGE_${STAGE}_CONVERT_DONE -> $CONVERTED"

  # DCP raw checkpoint is large; once converted it can be reclaimed. Do NOT auto-delete
  # here -- deleting checkpoints needs explicit user OK (RL-chain red line). Reminder only:
  echo "  (reminder) after verifying $CONVERTED, the raw DCP under $LOG_PATH can be reclaimed"

  NEXT_STUDENT="$CONVERTED"
done

echo "SEQCL_ALL_DONE $(date '+%F %T')"
echo
echo "NEXT: post-hoc per-suite SR for the forgetting study (in-training eval is unreliable)."
echo "  For each stage's converted model, eval EVERY learned suite separately, e.g.:"
echo "    datafree_eval_spatial.sh <gpu> <converted_dir> 50   # + analogous object/goal/long"
echo "  Compare each suite's SR across stages to measure learning vs forgetting."
