#!/usr/bin/env bash
# Reusable wrapper: merge an OpenVLA-OFT RLinf LoRA checkpoint into a loadable HF model dir.
# CPU-only (CUDA_VISIBLE_DEVICES="") so it never touches any GPU.
#
# Usage:
#   convert_oft_lora_ckpt.sh <ckpt_full_weights.pt> <out_dir> [base_dir] [unnorm_key]
#
# Examples:
#   # goal (best) checkpoint:
#   convert_oft_lora_ckpt.sh \
#     /share/fanruochen-local/outputs/opd_baselora_goal/converted/_snap_full_weights.pt \
#     /share/fanruochen-local/outputs/opd_baselora_goal/converted/best
#
#   # later, step_25 checkpoint (snapshot it first, then):
#   convert_oft_lora_ckpt.sh \
#     /share/fanruochen-local/outputs/opd_baselora_goal/converted/_snap_step25.pt \
#     /share/fanruochen-local/outputs/opd_baselora_goal/converted/step_25
set -euo pipefail

CKPT="${1:?usage: convert_oft_lora_ckpt.sh <ckpt.pt> <out_dir> [base_dir] [unnorm_key]}"
OUT="${2:?usage: convert_oft_lora_ckpt.sh <ckpt.pt> <out_dir> [base_dir] [unnorm_key]}"
BASE="${3:-/share/fanruochen-local/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora}"
UNNORM="${4:-}"

REPO=/home/fanruochen/CL/RLinf
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Safety: disk check before writing ~15GB output.
echo "== df /share/fanruochen-local =="
df -h /share/fanruochen-local | tail -1

EXTRA=()
if [[ -n "$UNNORM" ]]; then EXTRA+=(--unnorm-key "$UNNORM"); fi

echo "== converting: $CKPT -> $OUT (base=$BASE) =="
CUDA_VISIBLE_DEVICES="" \
PYTHONPATH="$REPO:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false \
  "$PY" "$HERE/convert_oft_lora_ckpt.py" \
    --ckpt "$CKPT" \
    --base "$BASE" \
    --out  "$OUT" \
    "${EXTRA[@]}"
