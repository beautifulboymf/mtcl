#!/bin/bash
# incremental_sft_lwf.sh — object-SFT + OFFLINE spatial-distillation (LwF) in ONE run.
#
# Learn OBJECT (libero_object) via LoRA-SFT on top of <init_model>, while ANCHORING spatial with a
# forward-KL to a FROZEN spatial specialist teacher on spatial demo states (both forced to libero_130
# action-norm). Goal: one model that learns object WITHOUT catastrophically forgetting spatial
# (plain object-SFT wiped spatial 0.98->0.00; the KL anchor should hold it). Everything OFFLINE ->
# SFT cadence (~1500 steps), no env rollout. finetune_lwf.py MERGES + saves a full HF model; we then
# copy the libero_130 dataset_statistics into every merged dir so it is directly OPD/eval-ready.
#
# Usage: incremental_sft_lwf.sh <init_model_dir> <spatial_teacher_dir> <out_dir> \
#            [lambda=1.0] [max_steps=1500] [save_steps=500] [gpus=4,5,6,7]
#   init & teacher are typically the SAME spatial_opd_student_098 (self-distillation LwF).
set -o pipefail   # NOT -u: venv activate references unbound vars
INIT="${1:?init model dir (e.g. spatial_opd_student_098)}"
TEACHER="${2:?spatial teacher dir (frozen spatial specialist)}"
OUT="${3:?out dir}"
LAMBDA="${4:-1.0}"; MAXSTEPS="${5:-1500}"; SAVESTEPS="${6:-500}"; GPUS="${7:-4,5,6,7}"
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
OBJ_DSET="libero_object_no_noops"      # NEW task (SFT, ground-truth)
SPA_DSET="libero_spatial_no_noops"     # anchor (KL to frozen spatial teacher)
GACC=$(( 64 / (8 * NPROC) )); [ "$GACC" -lt 1 ] && GACC=1

REPO=/home/fanruochen/CL/RLinf
NORM130=/share/fanruochen-local/checkpoints/norm_override_libero130.json
STATS130=/share/fanruochen-local/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora/dataset_statistics.json
DROOT=/share/fanruochen-local/datasets/rlds
FINE_LWF="$REPO/examples/embodiment/incremental_sft/finetune_lwf.py"

[ -f "$INIT/model.safetensors.index.json" ]    || { echo "ABORT: init not an HF model: $INIT"; exit 1; }
[ -f "$TEACHER/model.safetensors.index.json" ] || { echo "ABORT: teacher not an HF model: $TEACHER"; exit 1; }
[ -d "$DROOT/$OBJ_DSET" ] || { echo "ABORT: object RLDS missing: $DROOT/$OBJ_DSET"; exit 1; }
[ -d "$DROOT/$SPA_DSET" ] || { echo "ABORT: spatial RLDS missing: $DROOT/$SPA_DSET"; exit 1; }
[ -f "$NORM130" ]         || { echo "ABORT: norm override missing: $NORM130"; exit 1; }

source /home/fanruochen/.rlinf-env.sh 2>/dev/null
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
export PYTHONPATH="${PYTHONPATH:-}"
export WANDB_MODE=offline WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPUS"
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
export SFT_NORM_OVERRIDE="$NORM130"    # forces libero_130 action norm for BOTH object + spatial data

mkdir -p "$OUT"
echo "== df =="; df -h /share/fanruochen-local | tail -1
echo "==== LwF-SFT object=$OBJ_DSET + spatial-anchor(KL) teacher=$(basename "$TEACHER") lambda=$LAMBDA init=$(basename "$INIT") steps=$MAXSTEPS gpus=$GPUS(np$NPROC) batch=8 gacc=$GACC eff=$((8*NPROC*GACC)) lr=5e-4  $(date '+%F %T') ===="
cd /tmp   # neutral cwd so repo prismatic/ never shadows venv OFT prismatic
torchrun --standalone --nnodes 1 --nproc-per-node "$NPROC" "$FINE_LWF" \
  --vla_path "$INIT" \
  --data_root_dir "$DROOT" \
  --dataset_name "$OBJ_DSET" \
  --spatial_teacher_path "$TEACHER" \
  --spatial_dataset_name "$SPA_DSET" \
  --distill_lambda "$LAMBDA" \
  --run_root_dir "$OUT" \
  --adapter_tmp_dir "${OUT}/adapter-tmp" \
  --lora_rank 32 --lora_dropout 0.0 \
  --batch_size 8 --grad_accumulation_steps "$GACC" \
  --learning_rate 5e-4 --image_aug True \
  --max_steps "$MAXSTEPS" --save_steps "$SAVESTEPS" \
  --save_latest_checkpoint_only "${SFT_SAVE_LATEST:-False}" \
  --wandb_project inc_sft_lwf --wandb_entity none
rc=${PIPESTATUS[0]}
echo "LWF_SFT_TRAIN_DONE rc=$rc $(date '+%F %T')"
[ "$rc" -ne 0 ] && exit "$rc"

FOUND=0
for d in $(ls -dt "$OUT"/*/ 2>/dev/null | grep -v adapter-tmp); do
  [ -f "${d}model.safetensors.index.json" ] || continue
  cp "$STATS130" "${d}dataset_statistics.json"
  echo "LWF_SFT_MERGED=${d%/}"
  FOUND=$((FOUND+1))
done
[ "$FOUND" -eq 0 ] && { echo "ABORT: no merged HF model under $OUT"; exit 1; }
echo "LWF_SFT_DONE $(date '+%F %T')"
