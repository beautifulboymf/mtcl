#!/bin/bash
# Sequential CL over 3 LIBERO suites with PER-SUITE EXPERT teachers (spatial -> object -> goal).
#
# This replicates the recipe that WORKED for the 4-task run (run_seqcl_4task.sh: staged, each
# stage rehearses all earlier suites with new=1.0/old=0.3 oversampling, convert between stages)
# but swaps the SINGLE 130 generalist teacher for THREE per-suite experts, routed per sample.
# The one-shot "all 3 suites at once, uniform weights, 3 experts" version collapsed the two
# suites the student was already good at (object .82->.24, spatial .74->.26) while goal rose
# .38->.80 -- i.e. the largest-KL suite ate the shared LoRA capacity. Staging + rehearsal
# weights are the two ingredients that were missing.
#
# USAGE (manual):
#   [S3_GPUS=4-6 S3_STEPS=10 S3_ENVS=24 S3_MICRO=8 S3_GB=96 S3_RE=3 S3_PORT=53000] \
#     run_seqcl_3teacher.sh <init_student_dir> <tag> [stage_start] [stage_end]
set -uo pipefail
INIT="${1:?init student HF dir}"; TAG="${2:?tag}"
STAGE_START="${3:-0}"; STAGE_END="${4:-2}"
REPO=/home/fanruochen/CL/RLinf
SCRIPTS=/share/fanruochen-local/dev/scripts
B130=/share/fanruochen-local/outputs/inc_sft_opd/base_stats130
T=/share/fanruochen-local/outputs/inc_sft_opd/teachers_r160
SPA_T="${B130}::/share/fanruochen-local/outputs/seqcl_rlspat_opd/spatial_cat_r160"
OBJ_T="${B130}::${T}/object_r32_sft"
GOA_T="${B130}::${T}/goal_r160"

SUITES=(libero_spatial libero_object libero_goal)
declare -A TEACHER=( [libero_spatial]="$SPA_T" [libero_object]="$OBJ_T" [libero_goal]="$GOA_T" )
NEW_W=1.0; OLD_W=0.3      # the 4-task recipe's rehearsal weights

[ -f "$INIT/model.safetensors.index.json" ] || { echo "ABORT: init not an HF model: $INIT"; exit 1; }
NEXT_STUDENT="$INIT"

for STAGE in $(seq "$STAGE_START" "$STAGE_END"); do
  CUR="${SUITES[$STAGE]}"
  ACTIVE=""; WEIGHTS=""; TARGS=()
  for i in $(seq 0 "$STAGE"); do
    s="${SUITES[$i]}"
    ACTIVE="${ACTIVE:+$ACTIVE,}$s"
    w=$([ "$s" = "$CUR" ] && echo "$NEW_W" || echo "$OLD_W")
    WEIGHTS="${WEIGHTS:+$WEIGHTS,}$s:$w"
    TARGS+=("$s=${TEACHER[$s]}")
  done

  echo "================ SEQCL-3T STAGE $STAGE ($CUR) ================"
  echo "  active  = [$ACTIVE]"
  echo "  weights = {$WEIGHTS}"
  echo "  student = $NEXT_STUDENT"
  echo "== df =="; df -h /share/fanruochen-local | tail -1
  FREE_G=$(df -BG --output=avail /share/fanruochen-local | tail -1 | tr -dc '0-9')
  [ "${FREE_G:-0}" -lt 60 ] && { echo "ABORT: <60G free (RED LINE)"; exit 1; }

  STAGE_TAG="${TAG}_s${STAGE}_${CUR#libero_}"
  DRV="/share/fanruochen-local/outputs/inc_sft_opd/opd_${STAGE_TAG}_driver.log"
  OPD_GPUS="${S3_GPUS:-4-6}" OPD_STEPS="${S3_STEPS:-10}" OPD_ENVS="${S3_ENVS:-24}" \
  OPD_MICRO="${S3_MICRO:-8}" OPD_GLOBAL_BATCH="${S3_GB:-96}" OPD_ROLLOUT_EPOCH="${S3_RE:-3}" \
  OPD_RAY_PORT="$(( ${S3_PORT:-53000} + STAGE * 1000 ))" \
  OPD_WEIGHTS="{$WEIGHTS}" \
  SR_PROC_MAX=800 RAY_memory_monitor_refresh_ms=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash "$SCRIPTS/safe_run.sh" "$DRV" \
    bash "$REPO/examples/embodiment/incremental_sft/opd_multi.sh" \
      "$NEXT_STUDENT" "$STAGE_TAG" "${TARGS[@]}"
  rc=$?
  echo "SEQCL3T_STAGE_${STAGE}_DONE rc=$rc $(date '+%F %T')"
  [ "$rc" -ne 0 ] && { echo "ABORT: stage $STAGE failed"; exit "$rc"; }

  CONV="/share/fanruochen-local/outputs/seqcl_${STAGE_TAG}/converted/${STAGE_TAG}"
  [ -f "$CONV/model.safetensors.index.json" ] || { echo "ABORT: no converted model at $CONV"; exit 1; }
  echo "SEQCL3T_STAGE_${STAGE}_CONVERTED=$CONV"
  NEXT_STUDENT="$CONV"
done
echo "SEQCL3T_ALL_DONE final_student=$NEXT_STUDENT $(date '+%F %T')"
