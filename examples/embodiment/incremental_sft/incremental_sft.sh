#!/bin/bash
# incremental_sft.sh — ONE incremental, norm-ALIGNED SFT stage of the SFT->OPD continual pipeline.
#
# SFT a fresh LoRA (r32) on a SINGLE LIBERO suite's RLDS demos, on top of <base_model>, with the
# action normalization FORCED to libero_130 (via sft_aligned.py monkeypatch) so the student shares
# the 130 teacher's action tokenization. OpenVLA-OFT's finetune.py MERGES the LoRA and saves a full
# HF model, so the output is directly loadable as the next OPD stage's student init. We then copy
# the libero_130 dataset_statistics into it so OPD/eval use unnorm_key=libero_130_no_noops_trajall.
#
# Usage: incremental_sft.sh <spatial|object> <base_model_dir> <out_dir> [max_steps=2000] [save_steps=2000] [gpus=4,5]
set -o pipefail   # NOT -u: venv activate references unbound vars
SUITE="${1:?spatial|object}"; BASE="${2:?base model dir}"; OUT="${3:?out dir}"
MAXSTEPS="${4:-15000}"; SAVESTEPS="${5:-$MAXSTEPS}"; GPUS="${6:-4,5}"
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
DSET="libero_${SUITE}_no_noops"
# Match verl/OpenVLA-OFT effective batch = 64 (their 8 GPU x batch 8). grad_accum = 64/(8*NPROC):
# 2 GPU -> 4, 4 GPU -> 2, 8 GPU -> 1.  (batch/GPU stays 8; lr 5e-4 constant = OFT below 30K steps.)
GACC=$(( 64 / (8 * NPROC) )); [ "$GACC" -lt 1 ] && GACC=1

REPO=/home/fanruochen/CL/RLinf
NORM130=/share/fanruochen-local/checkpoints/norm_override_libero130.json
STATS130=/share/fanruochen-local/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora/dataset_statistics.json
DROOT=/share/fanruochen-local/datasets/rlds
FINE_ALIGNED="$REPO/examples/embodiment/incremental_sft/sft_aligned.py"

[ -d "$DROOT/$DSET" ] || { echo "ABORT: RLDS data missing: $DROOT/$DSET"; exit 1; }
[ -f "$NORM130" ]     || { echo "ABORT: norm override missing: $NORM130"; exit 1; }

source /home/fanruochen/.rlinf-env.sh 2>/dev/null
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
export PYTHONPATH="${PYTHONPATH:-}"
export WANDB_MODE=offline WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPUS"
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3   # tf(RLDS) must not grab GPU mem
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
export SFT_NORM_OVERRIDE="$NORM130"    # <-- forces libero_130 action norm in the RLDS pipeline

mkdir -p "$OUT"
echo "== df =="; df -h /share/fanruochen-local | tail -1
echo "==== INC-SFT [$SUITE] base=$(basename "$BASE") norm=libero_130 steps=$MAXSTEPS gpus=$GPUS(np$NPROC) batch=8 grad_accum=$GACC eff_batch=$((8*NPROC*GACC)) lr=5e-4  $(date '+%F %T') ===="
cd /tmp   # neutral cwd so the repo's prismatic/ never shadows the venv's OFT prismatic
torchrun --standalone --nnodes 1 --nproc-per-node "$NPROC" "$FINE_ALIGNED" \
  --vla_path "$BASE" \
  --data_root_dir "$DROOT" \
  --dataset_name "$DSET" \
  --run_root_dir "$OUT" \
  --adapter_tmp_dir "${OUT}/adapter-tmp" \
  --lora_rank 32 --lora_dropout 0.0 \
  --batch_size 8 --grad_accumulation_steps "$GACC" \
  --learning_rate 5e-4 --image_aug True \
  --max_steps "$MAXSTEPS" --save_steps "$SAVESTEPS" \
  --save_latest_checkpoint_only "${SFT_SAVE_LATEST:-True}" \
  --wandb_project inc_sft --wandb_entity none
rc=${PIPESTATUS[0]}
echo "INC_SFT_TRAIN_DONE rc=$rc $(date '+%F %T')"
[ "$rc" -ne 0 ] && exit "$rc"

# finetune.py MERGES + saves a full HF model (model.safetensors.index.json) at each save_step
# (and run_dir). Copy the libero_130 dataset_statistics into EVERY merged dir so any checkpoint
# (incl. intermediate ones, for picking a target-SR weak init) is directly eval-ready.
FOUND=0
for d in $(ls -dt "$OUT"/*/ 2>/dev/null | grep -v adapter-tmp); do
  [ -f "${d}model.safetensors.index.json" ] || continue
  cp "$STATS130" "${d}dataset_statistics.json"
  echo "INC_SFT_MERGED=${d%/}"
  FOUND=$((FOUND+1))
done
[ "$FOUND" -eq 0 ] && { echo "ABORT: no merged HF model under $OUT"; exit 1; }
echo "INC_SFT_DONE $(date '+%F %T')"
