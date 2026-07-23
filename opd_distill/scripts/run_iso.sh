#!/bin/bash
# run_iso.sh — run ANY RLinf job (train or eval command) on a PRIVATE ray head so it
# never connects to (collides with) another job's cluster. Targeted cleanup only.
# Usage: ISO_RAY_PORT=6530 run_iso.sh <command...>
export PYTHONPATH="${PYTHONPATH:-}"; export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
# limit per-worker threads (eval env workers else thread-explode -> load inflation)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_NUM_THREADS=1
source ~/.rlinf-env.sh
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
RAY_BIN="$(command -v ray)"
export RAY_TMPDIR="/tmp/rayiso_$$"; PORT="${ISO_RAY_PORT:-6530}"
cleanup(){ pkill -u "$USER" -9 -f "$RAY_TMPDIR" >/dev/null 2>&1 || true; rm -rf "$RAY_TMPDIR" >/dev/null 2>&1 || true; }
trap cleanup EXIT
pkill -u "$USER" -9 -f "$RAY_TMPDIR" >/dev/null 2>&1 || true; rm -rf "$RAY_TMPDIR"
"$RAY_BIN" start --head --temp-dir="$RAY_TMPDIR" --port="$PORT" \
    --include-dashboard=false --disable-usage-stats >"/tmp/iso_rayhead_$$.log" 2>&1 || true
export RAY_ADDRESS="127.0.0.1:$PORT"
"$RAY_BIN" status >/dev/null 2>&1 && echo "[iso] head up port=$PORT tmp=$RAY_TMPDIR" || echo "[iso] WARN head unclear port=$PORT"
"$@"
