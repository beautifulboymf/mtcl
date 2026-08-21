#!/usr/bin/env bash
# Reusable wrapper: merge an OpenVLA-OFT RLinf slot-LoRI checkpoint into a loadable HF
# model dir. CPU-only (CUDA_VISIBLE_DEVICES="") so it never touches any GPU.
#
# Usage:
#   convert_oft_slot_ckpt.sh <ckpt_full_weights.pt> <out_dir> [base_dir] [unnorm_key]
#
# The slot knobs come from the environment, so the positional arguments stay identical
# to convert_oft_lora_ckpt.sh. They MUST match the training run's
# actor.model.slot_lora block -- SLOT_RANKS in its slot_order, and the scale pair as
# configured; a wrong scale is rejected by the checkpoint's own record, a wrong rank
# ORDER is not (only the sum is checkable). Defaults are the R1 run's:
#
#   SLOT_RANKS=128,64,48,16   # slot_order = libero_10, libero_goal, libero_spatial, libero_object
#   SLOT_SCALE_MODE=match_mt4 # actor.model.slot_lora.a_scale_mode
#   SLOT_REF_RANK=128         # actor.model.slot_lora.a_scale_ref_rank
#   SLOT_EPS=1e-6             # actor.model.slot_lora.orth_eps
#   MIN_FREE_G=30             # abort unless this many GB are free (output is ~15G)
#
# Examples:
#   # best checkpoint of a slot run:
#   convert_oft_slot_ckpt.sh \
#     /share/fanruochen-local/outputs/opd_mt4slot/converted/_snap_full_weights.pt \
#     /share/fanruochen-local/outputs/opd_mt4slot/converted/best
#
#   # a run with a different slot layout:
#   SLOT_RANKS=64,64,64,64 convert_oft_slot_ckpt.sh /path/step25.pt /path/out
set -euo pipefail

CKPT="${1:?usage: convert_oft_slot_ckpt.sh <ckpt.pt> <out_dir> [base_dir] [unnorm_key]}"
OUT="${2:?usage: convert_oft_slot_ckpt.sh <ckpt.pt> <out_dir> [base_dir] [unnorm_key]}"
BASE="${3:-/share/fanruochen-local/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora}"
UNNORM="${4:-}"

SLOT_RANKS="${SLOT_RANKS:-128,64,48,16}"
SLOT_SCALE_MODE="${SLOT_SCALE_MODE:-match_mt4}"
SLOT_REF_RANK="${SLOT_REF_RANK:-128}"
SLOT_EPS="${SLOT_EPS:-1e-6}"
MIN_FREE_G="${MIN_FREE_G:-30}"

REPO=/home/fanruochen/CL/RLinf
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Safety: the merged model is ~15G of safetensors. A write that runs the filesystem out
# of space does not just lose this conversion -- on the shared volume it takes down
# every other tenant's writes with it -- so this ABORTS rather than only reporting.
# Measured against the nearest existing ancestor of $OUT, since $OUT itself may not
# exist yet.
DF_TARGET="$OUT"
while [[ ! -d "$DF_TARGET" && "$DF_TARGET" != "/" && "$DF_TARGET" != "." ]]; do
  DF_TARGET="$(dirname "$DF_TARGET")"
done
echo "== df $DF_TARGET =="
df -h "$DF_TARGET" | tail -1
FREE_G="$(df -P -BG "$DF_TARGET" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')"
if [[ -z "$FREE_G" ]]; then
  echo "ABORT: could not read free space for $DF_TARGET" >&2
  exit 1
fi
if (( FREE_G < MIN_FREE_G )); then
  echo "ABORT: only ${FREE_G}G free at $DF_TARGET; the merged model is ~15G and this" >&2
  echo "       needs >= ${MIN_FREE_G}G. Free space first (raise MIN_FREE_G only if you" >&2
  echo "       know the target is small)." >&2
  exit 1
fi

EXTRA=()
if [[ -n "$UNNORM" ]]; then EXTRA+=(--unnorm-key "$UNNORM"); fi

echo "== converting: $CKPT -> $OUT (base=$BASE) =="
echo "== slots: ranks=$SLOT_RANKS scale_mode=$SLOT_SCALE_MODE ref_rank=$SLOT_REF_RANK eps=$SLOT_EPS =="
CUDA_VISIBLE_DEVICES="" \
PYTHONPATH="$REPO:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false \
  "$PY" "$HERE/convert_oft_slot_ckpt.py" \
    --ckpt "$CKPT" \
    --base "$BASE" \
    --out  "$OUT" \
    --slot-ranks      "$SLOT_RANKS" \
    --slot-scale-mode "$SLOT_SCALE_MODE" \
    --slot-ref-rank   "$SLOT_REF_RANK" \
    --slot-eps        "$SLOT_EPS" \
    "${EXTRA[@]}"
