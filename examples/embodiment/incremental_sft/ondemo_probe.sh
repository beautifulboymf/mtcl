#!/bin/bash
# ondemo_probe.sh — launcher for ondemo_probe.py (forward passes only; no ray, no rendering).
#
# Runs from /tmp so the repo's prismatic/ never shadows the venv OFT prismatic (the probe refuses
# to start otherwise), and forces the libero_130 action norm that every model here was trained under.
# This is a plain single-process torch job, so CUDA_VISIBLE_DEVICES is the right way to pin a GPU
# (unlike the ray eval path, where the gN config carries the ABSOLUTE index and CUDA_VISIBLE_DEVICES
# desyncs ray's view -- see multiseed_eval.sh).
set -uo pipefail
GPU=${GPU:-6}
SUITE=${SUITE:-libero_spatial_no_noops}
BATCHES=${BATCHES:-25}
OUT=${OUT:-/share/fanruochen-local/outputs/ondemo_probe_${SUITE}.json}
REPO=/home/fanruochen/CL/RLinf
O=/share/fanruochen-local/outputs

# refuse to share a GPU: a squatter landing later would OOM us mid-probe
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU")
if [ "$used" -gt 2000 ] || [ "$util" -gt 10 ]; then
  echo "ABORT: GPU $GPU busy (${used}MiB, ${util}% util) — pick another"; exit 1
fi
echo "[probe] GPU $GPU free (${used}MiB, ${util}%)"

# the venv activate script dereferences PYTHONPATH/LD_LIBRARY_PATH without a default, which aborts
# under `set -u` -- predefine them empty (same guard run_iso_lite.sh uses).
export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source ~/.rlinf-env.sh
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
export SFT_NORM_OVERRIDE=/share/fanruochen-local/checkpoints/norm_override_libero130.json
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4        # a 50-env eval may be running: do not take all cores

# tag=path:measured_rollout_SR   (first entry is the reference for the KL/concentration numbers)
MODELS=${MODELS:-"\
S1=$O/spatial_teacher_v2_sft/merged_step1000:0.90 \
A1_noanchor=$O/shallowA1_object_noanchor/merged_step1000+libero_object_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug:0.68 \
A2_M50=$O/shallowA2_M50/merged:0.64 \
A2_M100=$O/shallowA2_M100/merged:0.84 \
A2_M200=$O/shallowA2_M200/merged:0.80 \
v2_S2_object=$O/seqcl_v2_S2_object/merged:0.84 \
lori_M_long=$O/seqcl_lori_M_long/merged:0.88"}

cd /tmp
nice -n 5 python "$REPO/examples/embodiment/incremental_sft/ondemo_probe.py" \
  --models "$MODELS" --dataset "$SUITE" --batches "$BATCHES" --out "$OUT"
echo "PROBE_DONE rc=$? out=$OUT"
