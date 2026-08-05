#!/bin/bash
# run_incremental_sft_opd.sh — TRUE incremental continual learning: SFT is done PER-TASK and
# interleaved with OPD, instead of one up-front 130-SFT (the current Base-Lora) before OPD.
#
#   openvla-7b-base
#     --[A1 SFT spatial]-->  spatial-SFT
#     --[A2 OPD spatial ]-->  spatial student            (teacher = 130 generalist)
#     --[B1 SFT object ]-->  object-SFT   (on top of the spatial student)
#     --[B2 OPD object  ]-->  object student = FINAL
#
# Every SFT (A1,B1) uses libero_130 action norm (incremental_sft.sh => sft_aligned.py monkeypatch)
# so the student always shares the 130 teacher's action tokenization. OPD (A2,B2) reuses the
# existing single-suite seqcl OPD config. Compare the final model to the "SFT-all-130-upfront->OPD"
# baseline (the prior sequential-CL run) on spatial+object (greedy).
#
# NOT auto-run. Launch under safe_run.sh. Pick FREE, contiguous GPUs.
#   SFT_GPUS=4,5 OPD_GPUS=4-5 bash run_incremental_sft_opd.sh
set -uo pipefail
REPO=/home/fanruochen/CL/RLinf
INC="$REPO/examples/embodiment/incremental_sft"
SCRIPTS=/share/fanruochen-local/dev/scripts
ROOT=/share/fanruochen-local/outputs/inc_sft_opd
BASE0=/share/fanruochen-local/checkpoints/openvla-7b-base
TEACHER=/share/fanruochen-local/checkpoints/RLinf-OpenVLAOFT-LIBERO-130
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
mkdir -p "$ROOT"

SFT_GPUS="${SFT_GPUS:-4,5}"     # SFT: torchrun, comma list
OPD_GPUS="${OPD_GPUS:-4-5}"     # OPD: RLinf contiguous range
SFT_STEPS="${SFT_STEPS:-15000}"   # verl/OFT recipe, B-lite budget (LR constant 5e-4 = OFT below 30K)
OPD_STEPS="${OPD_STEPS:-15}"
DF_MIN_G="${DF_MIN_G:-120}"     # abort a stage if < this many GB free on the shared disk

banner(){ echo; echo "################ $* $(date '+%F %T') ################"; }
dfree(){ df -BG --output=avail /share/fanruochen-local | tail -1 | tr -dc '0-9'; }
guard_df(){ local a; a=$(dfree); echo "== df: ${a}G free =="; [ "${a:-0}" -lt "$DF_MIN_G" ] && { echo "ABORT: <${DF_MIN_G}G free"; exit 1; }; }

# ---- SFT one suite (norm=libero_130); echo the merged model dir on success ----
sft_stage(){ # $1=suite $2=base_model $3=out_subdir  -> sets SFT_OUT
  local SUITE="$1" BASE="$2" SUB="$3" OUTD="$ROOT/$3"
  guard_df
  banner "[SFT $SUITE] base=$(basename "$BASE")"
  bash "$INC/incremental_sft.sh" "$SUITE" "$BASE" "$OUTD" "$SFT_STEPS" "$SFT_STEPS" "$SFT_GPUS"
  SFT_OUT=$(ls -dt "$OUTD"/*/ 2>/dev/null | grep -v adapter-tmp | while read -r d; do
    [ -f "${d}model.safetensors.index.json" ] && { echo "${d%/}"; break; }; done)
  [ -z "$SFT_OUT" ] && { echo "ABORT: SFT $SUITE produced no merged model"; exit 1; }
  echo "SFT_OUT[$SUITE]=$SFT_OUT"
}

# ---- OPD one suite via the single-suite seqcl config; echo converted model dir ----
opd_stage(){ # $1=suite $2=student_init $3=tag  -> sets OPD_OUT
  local SUITE="$1" INIT="$2" TAG="$3"
  local LOGP="/share/fanruochen-local/outputs/seqcl_${TAG}"
  # the seqcl OPD config resolves ${oc.env:EMBODIED_PATH} in its hydra searchpath; run_iso does
  # not set these, so export them here (train_embodied_agent.py, unlike eval_embodiment.sh, does not).
  export EMBODIED_PATH="$REPO/examples/embodiment" REPO_PATH="$REPO"
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$REPO:${PYTHONPATH:-}"
  guard_df
  banner "[OPD $SUITE] init=$(basename "$INIT") teacher=130"
  SEQCL_GPUS="$OPD_GPUS" SEQCL_STUDENT_PATH="$INIT" \
  SEQCL_ACTIVE_SUITES="[libero_${SUITE}]" SEQCL_SUITE_WEIGHTS='null' \
  SEQCL_MAX_STEPS="$OPD_STEPS" SEQCL_CURRENT_SUITE="$TAG" \
  RLINF_CONVERT_VALUE_HEAD=False \
  ISO_RAY_PORT="${OPD_RAY_PORT:-28000}" bash "$SCRIPTS/run_iso.sh" \
    "$PY" "$REPO/examples/embodiment/train_embodied_agent.py" --config-name libero_seqcl_opd_2gpu
  local CKPT CONV
  CKPT=$(find "$LOGP" -name full_weights.pt -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  [ -z "$CKPT" ] && { echo "ABORT: OPD $SUITE no full_weights.pt under $LOGP"; exit 1; }
  CONV="$LOGP/converted/${TAG}"
  banner "[CONVERT $SUITE] $CKPT -> $CONV (base=$(basename "$INIT"))"
  bash "$SCRIPTS/convert_oft_lora_ckpt.sh" "$CKPT" "$CONV" "$INIT"
  [ -f "$CONV/model.safetensors.index.json" ] || { echo "ABORT: convert $SUITE failed"; exit 1; }
  OPD_OUT="$CONV"
  echo "OPD_OUT[$SUITE]=$OPD_OUT"
}

banner "INCREMENTAL SFT->OPD (spatial then object)  SFT_GPUS=$SFT_GPUS OPD_GPUS=$OPD_GPUS"

# ===== A: spatial =====
sft_stage spatial "$BASE0" "sft_spatial";              SPA_SFT="$SFT_OUT"
opd_stage spatial "$SPA_SFT" "inc_spatial";            SPA_OPD="$OPD_OUT"

# ===== B: object (on top of the spatial student) =====
sft_stage object  "$SPA_OPD" "sft_object";             OBJ_SFT="$SFT_OUT"
opd_stage object  "$OBJ_SFT" "inc_object";             OBJ_OPD="$OPD_OUT"

# record the model paths for the (separate) eval step
{ echo "spatial_SFT=$SPA_SFT";  echo "spatial_OPD=$SPA_OPD";
  echo "object_SFT=$OBJ_SFT";   echo "object_OPD_FINAL=$OBJ_OPD"; } | tee "$ROOT/MODELS.txt"
banner "CHAIN DONE -> models in $ROOT/MODELS.txt"
echo "Next: greedy eval with  bash $INC/eval_inc_models.sh"
banner "INC_SFT_OPD_TRAIN_ALL_DONE"
