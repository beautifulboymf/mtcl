#!/bin/bash
# seqcl_lori_chain.sh — sequential CL with an ITERATIVELY-UPDATED LoRI adapter, fully offline.
#
# Difference from seqcl_chain.sh (last night's plain-LoRA baseline):
#   * LoRI  : lora_A frozen (random down-projection), only B trains  [SFT_LORI=1]
#   * ITERATIVE: ONE adapter is carried across the whole task sequence via SFT_INIT_ADAPTER,
#                and the STUDENT BASE STAYS S1 FOREVER (never merged into). So the original
#                model is structurally intact -- dropping the adapter returns exactly S1.
#   * The teacher for the KL anchor at step k is merge(S1 + adapter_{k-1}), i.e. "the model as it
#     was after the previous task" -- materialised only to serve as a frozen teacher / for eval.
#
# Everything else is matched to the baseline on purpose (same suites, order, lambda, steps, rank,
# eff-batch, norm) so this is a clean A/B against seqcl_v2.
set -uo pipefail
O=/share/fanruochen-local/outputs
REPO=/home/fanruochen/CL/RLinf
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
MERGE=$REPO/opd_distill/scripts/merge_peft_adapter.py
LWF=$REPO/examples/embodiment/incremental_sft/incremental_sft_lwf.sh
SAFE=/share/fanruochen-local/dev/scripts/safe_run.sh
NORM=libero_130_no_noops_trajall
S1=$O/spatial_teacher_v2_sft/merged_step1000     # frozen base for the WHOLE chain
LAMBDA=${LAMBDA:-0.3}; STEPS=${STEPS:-2000}; SAVE=${SAVE:-500}; GPUS=${GPUS:-6,7}
FLOOR_GB=150
TAG=${TAG:-lori}
LOG=$O/seqcl_${TAG}_chain.log
say(){ echo "[${TAG} $(date '+%F %T')] $*" | tee -a "$LOG"; }
dfree(){ df -BG /share/fanruochen-local | tail -1 | awk '{print $4}' | tr -d 'G'; }
guard(){ local f; f=$(dfree); [ "${f:-0}" -lt "$FLOOR_GB" ] && { say "ABORT disk ${f}G<${FLOOR_GB}G"; exit 3; }; say "disk ${f}G free"; }

# $1 tag  $2 new_dset  $3 anchor_csv  $4 teacher_dir  $5 init_adapter("" = fresh)
# echoes nothing; sets ADAPTER_OUT / MERGED_OUT
step(){
  local tag="$1" new="$2" anchors="$3" teacher="$4" prev_adapter="$5"
  local out="$O/seqcl_${TAG}_${tag}"
  guard
  say "STEP $tag  new=$new  anchor=$anchors  teacher=$(basename "$teacher")  resume_adapter=${prev_adapter:-<fresh>}"
  SFT_LORI=1 SFT_INIT_ADAPTER="$prev_adapter" \
  LWF_NEW_DSET="$new" LWF_ANCHOR_DSET="$anchors" SFT_ADAPTER_ONLY=1 SR_PROC_MAX=1000 \
    bash "$SAFE" "$O/seqcl_${TAG}_${tag}_driver.log" \
    bash "$LWF" "$S1" "$teacher" "$out" "$LAMBDA" "$STEPS" "$SAVE" "$GPUS" 2>&1 | tee -a "$LOG"
  grep -aq 'LWF_SFT_DONE' "$O/seqcl_${TAG}_${tag}_driver.log" || { say "ABORT: step $tag no LWF_SFT_DONE"; exit 7; }
  ADAPTER_OUT=$(ls -d "$out"/*/adapters/step_${STEPS} 2>/dev/null | head -1)
  [ -n "$ADAPTER_OUT" ] || { say "ABORT: no step_${STEPS} adapter under $out"; exit 5; }
  say "STEP $tag done -> adapter $ADAPTER_OUT"
  # merge S1 + CUMULATIVE adapter -> the model "after task k" (teacher for k+1, and eval target)
  MERGED_OUT="$O/seqcl_${TAG}_M_${tag}/merged"
  guard
  if [ ! -f "$MERGED_OUT/model.safetensors.index.json" ]; then
    say "merging S1 + $tag adapter -> $MERGED_OUT (CPU)"
    CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" nice -n 10 ionice -c3 "$PY" "$MERGE" \
      --adapter "$ADAPTER_OUT" --base "$S1" --out "$MERGED_OUT" --unnorm-key "$NORM" 2>&1 | tee -a "$LOG"
  fi
  [ -f "$MERGED_OUT/model.safetensors.index.json" ] || { say "ABORT: merge failed $MERGED_OUT"; exit 6; }
  say "merged ok -> $MERGED_OUT"
}

say "=== iterative-LoRI chain START (base=S1 frozen, ONE adapter carried through) ==="
# k=2 object : teacher = S1 itself (the model before this task)
step object libero_object_no_noops "libero_spatial_no_noops" "$S1" ""
A_OBJ="$ADAPTER_OUT"; M_OBJ="$MERGED_OUT"
# k=3 goal   : resume the SAME adapter, teacher = model-after-object
step goal libero_goal_no_noops "libero_spatial_no_noops,libero_object_no_noops" "$M_OBJ" "$A_OBJ"
A_GOAL="$ADAPTER_OUT"; M_GOAL="$MERGED_OUT"
# k=4 long   : resume again, teacher = model-after-goal
step long libero_10_no_noops "libero_spatial_no_noops,libero_object_no_noops,libero_goal_no_noops" "$M_GOAL" "$A_GOAL"
say "LORI_CHAIN_DONE  M_object=$M_OBJ  M_goal=$M_GOAL  M_long=$MERGED_OUT"
