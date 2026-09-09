#!/bin/bash
# multiseed_eval.sh — evaluate (model, suite) pairs over SEVERAL SEEDS and report mean +/- SE.
#
# WHY: every number we have so far is a single 50-env run. At p~0.75 that is SE ~= 0.061, i.e. a
# 95% interval of +/-0.12 -- wider than most of the effects we were trying to read (anchor +0.16,
# BWT +0.04, the whole "recovery curve"). Nothing is claimable until the key points carry error
# bars. This runs the seeds SEQUENTIALLY (the RLinf eval driver talks to a FIXED dashboard port
# 8265, so two evals must never overlap) and prints mean/SE at the end.
#
# Usage:
#   JOBS="tag=/path/to/model:suite[,suite...]  tag2=/path2:suite" SEEDS="1235 7 4242" \
#   GPUCFG=g5 bash multiseed_eval.sh
# suite in {spatial,object,goal,long}; GPUCFG=gN pins ABSOLUTE physical GPU N
# (do NOT also set CUDA_VISIBLE_DEVICES -- that desyncs ray's GPU view and the model lands on a
#  tenant GPU or on CPU; this was diagnosed the hard way on 2026-09-07).
set -uo pipefail
O=/share/fanruochen-local/outputs
REPO=/home/fanruochen/CL/RLinf
RUNISO=/share/fanruochen-local/dev/scripts/run_iso_lite.sh
EVAL=$REPO/examples/embodiment/eval_embodiment.sh
NORM=libero_130_no_noops_trajall
GPUCFG=${GPUCFG:-g5}
SEEDS=${SEEDS:-"1235 7 4242"}
ENVS=${ENVS:-50}
TEMP=${TEMP:-1.0}
JOBS=${JOBS:?JOBS="tag=/model/dir:suite[,suite]  ..."}
LOG=$O/multiseed_eval.log
RES=$O/multiseed_results.tsv
say(){ echo "[ms $(date '+%F %T')] $*" | tee -a "$LOG"; }
cfg_of(){ case "$1" in spatial) echo libero_spatial_${GPUCFG}_eval;; object) echo libero_object_${GPUCFG}_eval;;
    goal) echo libero_goal_${GPUCFG}_eval;; long) echo libero_10_${GPUCFG}_eval;; *) echo "";; esac; }

# Start port is configurable because orphan ray DashboardAgents squat the DERIVED agent ports
# (PORT+4/+5/+6). If an earlier run died, its agents keep those three ports and EVERY later attempt
# on that PORT fails with "node timed out during startup" -- this silently ate 2 cells on 2026-09-08
# (52000 and 59000 were both squatted). Check with `ss -ltnp | grep ray` and pick a PORT_START whose
# whole +700*n ladder is clear.
port=${PORT_START:-52000}
say "=== multiseed START  seeds=[$SEEDS] envs=$ENVS temp=$TEMP gpu=$GPUCFG ==="
for job in $JOBS; do
  tag="${job%%=*}"; rest="${job#*=}"; M="${rest%%:*}"; suites="${rest##*:}"
  [ -f "$M/model.safetensors.index.json" ] || { say "SKIP $tag: no model at $M"; continue; }
  IFS=',' read -ra SU <<< "$suites"
  for s in "${SU[@]}"; do
    cfg=$(cfg_of "$s"); [ -n "$cfg" ] || { say "SKIP $tag/$s: bad suite"; continue; }
    vals=""
    for seed in $SEEDS; do
      log="$O/ms_${tag}_${s}_seed${seed}.log"
      say "EVAL $tag / $s / seed=$seed (port=$port)"
      ( MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_NUM_THREADS=1 \
        ISO_RAY_PORT=$port bash "$RUNISO" \
          bash "$EVAL" "$cfg" LIBERO \
            rollout.model.model_path="$M" actor.model.model_path="$M" actor.model.is_lora=False \
            actor.model.unnorm_key="$NORM" env.eval.total_num_envs=$ENVS env.train.total_num_envs=$ENVS \
            algorithm.sampling_params.temperature_eval=$TEMP ++actor.seed=$seed ) > "$log" 2>&1
      v=$(grep -aoE "success_once': array\([0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")
      say "  -> $tag $s seed$seed = ${v:-FAILED}"
      printf "%s\t%s\t%s\t%s\n" "$tag" "$s" "$seed" "${v:-NA}" >> "$RES"
      [ -n "$v" ] && vals="$vals $v"
      port=$((port+700)); [ $port -gt 64000 ] && port=52000
    done
    # mean +/- SE over seeds
    if [ -n "$vals" ]; then
      read -r m se n < <(python3 -c "
import sys,statistics as st
v=[float(x) for x in '''$vals'''.split()]
m=st.mean(v); se=(st.stdev(v)/len(v)**0.5) if len(v)>1 else float('nan')
print(f'{m:.3f} {se:.3f} {len(v)}')")
      say "SUMMARY $tag $s: mean=$m  SE=$se  (n=$n seeds)"
    fi
  done
done
say "MULTISEED_DONE  (raw rows in $RES)"
