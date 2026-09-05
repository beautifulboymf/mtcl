#!/bin/bash
# sft_4suite_student.sh -- build a SAME-LINEAGE multi-suite SFT student (2026-09-04).
#
# Same base and same demo data the four per-suite teachers were SFT'd from:
#   base = base_stats130 (= openvla-7b-base + libero_130 norm stats, symlinked), the exact
#          init spatial/object/goal teachers used.
#   data = libero_4suite_no_noops -- the NEW equal-weight RLDS mixture over
#          {spatial, object, goal, 10}_no_noops (added to OXE_NAMED_MIXTURES), i.e. the
#          same demos, interleaved ~25% each per batch.
#   norm = FORCED to libero_130 (SFT_NORM_OVERRIDE) so this student shares the teachers'
#          and the OPD student's action tokenization (unnorm_key=libero_130_no_noops_trajall).
#
# This is a from-base SFT baseline / alternative student, NOT part of the running dance job.
# It reuses incremental_sft.sh unchanged -- that script's DSET is just "libero_${SUITE}_no_noops",
# so passing SUITE=4suite yields DSET=libero_4suite_no_noops, which now resolves to the mix.
#
# Usage (run under the watchdog):
#   SR_PROC_MAX=1000 bash /share/fanruochen-local/dev/scripts/safe_run.sh \
#     /share/fanruochen-local/outputs/sft_4suite_driver.log \
#     bash examples/embodiment/incremental_sft/sft_4suite_student.sh
set -o pipefail

BASE="${BASE:-/share/fanruochen-local/outputs/inc_sft_opd/base_stats130}"
OUT="${OUT:-/share/fanruochen-local/outputs/inc_sft_opd/sft_4suite_student}"
MAXSTEPS="${MAXSTEPS:-3000}"
SAVESTEPS="${SAVESTEPS:-500}"         # snapshot a LoRA adapter every 500 steps
GPUS="${GPUS:-4,5}"                   # dance owns 0-3; SFT is CPU-render-free, GPU-only compute
REPO=/home/fanruochen/CL/RLinf

# ADAPTER-ONLY: each save_step snapshots just the LoRA adapter (~0.5G) under
# $OUT/*/adapters/step_N -- NO merged HF model, no RAM spike, no 15G/ckpt disk bloat.
# save_latest_only irrelevant in this mode (per-step dirs all survive). Merge post-hoc to eval.
# SUITE=4_task_suites -> DSET=libero_4_task_suites_no_noops, the equal-weight 4-suite
# mixture ALREADY present in the venv's prismatic OXE_NAMED_MIXTURES (the copy the
# training process actually imports). No library edit needed.
SFT_ADAPTER_ONLY=1 SFT_SAVE_LATEST="${SFT_SAVE_LATEST:-False}" \
  bash "$REPO/examples/embodiment/incremental_sft/incremental_sft.sh" \
  4_task_suites "$BASE" "$OUT" "$MAXSTEPS" "$SAVESTEPS" "$GPUS"
