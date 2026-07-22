#!/bin/bash
# run_iso_train.sh — run an RLinf training job on a PRIVATE ray head (concurrency-safe).
# RLinf does ray.init(address="auto") which else connects to ANY existing cluster and
# collides. We pre-start a private head (unique port+tmpdir), point RAY_ADDRESS at it,
# and clean up by TARGETED pkill on our tmpdir only (NEVER `ray stop` = node-global).
# Usage: ISO_RAY_PORT=6520 run_iso_train.sh <config_name> [hydra overrides...]
# NOTE: no `set -u` — the sourced env/activate scripts reference unset PYTHONPATH etc.
export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source ~/.rlinf-env.sh
source /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/activate
RAY_BIN="$(command -v ray)"
export RAY_TMPDIR="/tmp/raytrain_$$"
PORT="${ISO_RAY_PORT:-6520}"

cleanup() { pkill -u "$USER" -9 -f "$RAY_TMPDIR" >/dev/null 2>&1 || true; rm -rf "$RAY_TMPDIR" >/dev/null 2>&1 || true; }
trap cleanup EXIT

pkill -u "$USER" -9 -f "$RAY_TMPDIR" >/dev/null 2>&1 || true; rm -rf "$RAY_TMPDIR"
"$RAY_BIN" start --head --temp-dir="$RAY_TMPDIR" --port="$PORT" \
    --include-dashboard=false --disable-usage-stats >"/tmp/iso_rayhead_$$.log" 2>&1 || true
export RAY_ADDRESS="127.0.0.1:$PORT"
if "$RAY_BIN" status >/dev/null 2>&1; then
  echo "[iso-train] private ray head up: port=$PORT tmp=$RAY_TMPDIR"
else
  echo "[iso-train] WARN: private head status unclear (port=$PORT)"; fi

# run the training job; it inherits RAY_ADDRESS -> connects to OUR private head only
bash /share/fanruochen-local/dev/scripts/run_opd_train_egl.sh "$@"
rc=$?
echo "[iso-train] job exit rc=$rc; cleaning private head $RAY_TMPDIR"
exit $rc
