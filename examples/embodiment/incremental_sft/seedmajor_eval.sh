#!/bin/bash
# seedmajor_eval.sh — evaluate every (model, suite) cell at ONE seed, then the next seed, ...
#
# WHY not just multiseed_eval.sh: that one is job-major (all 3 seeds of cell 1, then cell 2, ...),
# so with 9 cells x 3 seeds you learn nothing about cell 9 until ~6 hours in. Seed-major finishes a
# complete picture of ALL cells after the first ~1/3 of the wall clock, then tightens the error bars
# on a second and third pass. Same total work, answers arrive in a useful order -- and if the run has
# to be stopped early, what you have is a complete low-precision table instead of a partial one.
#
# Every cell still goes through multiseed_eval.sh (tested path: port ladder, GPU pinning, isolated
# ray head), one seed at a time; after each pass the running mean/SE over the seeds done SO FAR is
# recomputed from the shared TSV.
#
# Usage:
#   JOBS="A=/path:spatial,object,goal  B=/path2:spatial,object,goal" SEEDS="1235 7 4242" \
#   GPUCFG=g3 PORT_START=53000 bash seedmajor_eval.sh
set -uo pipefail
O=/share/fanruochen-local/outputs
REPO=/home/fanruochen/CL/RLinf
SEEDS=${SEEDS:-"1235 7 4242"}
JOBS=${JOBS:?JOBS="tag=/model/dir:suite[,suite] ..."}
GPUCFG=${GPUCFG:-g3}
PORT_START=${PORT_START:-53000}
RES=$O/multiseed_results.tsv
LOG=$O/seedmajor.log
say(){ echo "[sm $(date '+%F %T')] $*" | tee -a "$LOG"; }

pass=0
for seed in $SEEDS; do
  pass=$((pass+1))
  say "================ PASS $pass  (seed=$seed) ================"
  # a fresh port ladder per pass; orphan dashboard agents from a dead run squat PORT+4/5/6 forever
  PORT_START=$((PORT_START + (pass-1)*3000)) SEEDS="$seed" GPUCFG="$GPUCFG" JOBS="$JOBS" \
    bash "$REPO/examples/embodiment/incremental_sft/multiseed_eval.sh" 2>&1 | tail -2
  # clear this pass's ray leftovers before the next one (they ignore SIGTERM and squat ports)
  pkill -u "$USER" -9 -f 'ray::DashboardAgen[t]|ray::RuntimeEnvAgen[t]' >/dev/null 2>&1 || true
  say "---- running table after $pass seed(s) ----"
  /share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python - "$RES" "$SEEDS" <<'PY' 2>&1 | tee -a "$LOG"
import sys, collections, statistics as st
rows = collections.defaultdict(list)
for line in open(sys.argv[1]):
    p = line.rstrip("\n").split("\t")
    if len(p) == 4 and p[3] not in ("NA", ""):
        rows[(p[0], p[1])].append(float(p[3]))
print(f"{'model':<18}{'suite':<10}{'mean':>8}{'SE':>8}{'n':>4}")
for (tag, suite), v in sorted(rows.items()):
    se = st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")
    print(f"{tag:<18}{suite:<10}{st.mean(v):>8.3f}{se:>8.3f}{len(v):>4}")
PY
done
say "SEEDMAJOR_DONE"
