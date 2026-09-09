#!/bin/bash
# seqcl_chain.sh -- UNATTENDED OFFLINE sequential continual-learning chain + per-step eval.
#
#   S1(spatial, already exists) --[object]--> S2 --[goal]--> S3 --[long]--> S4
#
# Each step = incremental_sft_lwf.sh: LoRA-SFT (r32) on the NEW suite's demos
# + lambda*KL( S_{k-1} || S_k ) OFFLINE on ALL previously-learned suites' demo states.
# Teacher = the previous cumulative student = SELF-DISTILLATION (single teacher).
# NO rollout, NO env, NO rendering in training (pure SFT + offline KL) -> zero EGL / crash risk.
#
# After EACH step's training completes and is merged, we EVAL that student on ALL suites it
# should now know (spatial+object for S2, +goal for S3, +long for S4) -> the forgetting curve.
# Eval = osmesa (CPU render, NO EGL), temp 1.0 (greedy lies for KL-distilled), 50 env, 1 seed,
# on a FREE card only, wrapped in `timeout` and NON-FATAL so a flaky eval never stalls the chain.
#
# Step 2 (object) is ALREADY RUNNING when this launches: the chain WAITS for it, merges S2,
# evals S2, then runs step 3 (goal) and step 4 (long) with the same merge+eval each.
# HARD disk floor before every save/merge so we can never fill /share (see 2026-06-13 crash).
set -uo pipefail
O=/share/fanruochen-local/outputs
REPO=/home/fanruochen/CL/RLinf
PY=/share/fanruochen-local/dev/envs/rlinf-openvlaoft/bin/python
MERGE=$REPO/opd_distill/scripts/merge_peft_adapter.py
LWF=$REPO/examples/embodiment/incremental_sft/incremental_sft_lwf.sh
EVAL=$REPO/examples/embodiment/eval_embodiment.sh
RUNISO=/share/fanruochen-local/dev/scripts/run_iso.sh
SAFE=/share/fanruochen-local/dev/scripts/safe_run.sh
NORM=libero_130_no_noops_trajall
LAMBDA=0.3; STEPS=2000; SAVE=500; GPUS=4,5,6,7
FLOOR_GB=150                                   # abort before any save/merge if free < this
EVAL_ENVS=50; EVAL_TEMP=1.0; EVAL_SEED=1235; EVAL_TIMEOUT=2700   # 45min/suite cap
S1=$O/spatial_teacher_v2_sft/merged_step1000   # spatial 0.92, the chain's starting student

LOG=$O/seqcl_chain.log
say(){ echo "[chain $(date '+%F %T')] $*" | tee -a "$LOG"; }
dfree_gb(){ df -BG /share/fanruochen-local | tail -1 | awk '{print $4}' | tr -d 'G'; }
guard_disk(){ local f; f=$(dfree_gb); if [ "${f:-0}" -lt "$FLOOR_GB" ]; then say "ABORT disk floor: ${f}G < ${FLOOR_GB}G free"; exit 3; fi; say "disk ok: ${f}G free"; }

gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }
gpus_free(){ local ok=1 g u; for g in 4 5 6 7; do u=$(gpu_used "$g"); [ -z "$u" ] && u=99999; [ "$u" -gt 2000 ] && ok=0; done; return $((1-ok)); }
wait_gpus(){ local i; for i in $(seq 1 40); do gpus_free && { say "gpus 4-7 free"; return 0; }
    say "gpus 4-7 busy, waiting ($i/40)"; sleep 30; done
    say "ABORT: gpus 4-7 still busy after 20min"; exit 4; }
pick_eval_gpu(){ local g u; for g in 4 5 6 7; do u=$(gpu_used "$g"); [ -z "$u" ] && u=99999; [ "$u" -lt 2000 ] && { echo "$g"; return 0; }; done; echo ""; return 1; }

# suite short-name -> eval config
cfg_of(){ case "$1" in spatial) echo libero_spatial_g2_eval;; object) echo libero_object_g2_eval;;
    goal) echo libero_goal_g2_eval;; long) echo libero_10_g2_eval;; *) echo "";; esac; }
# suite short-name -> a distinct ISO ray port (evals are sequential; distinct ports = safety)
port_of(){ case "$1" in spatial) echo 50000;; object) echo 51000;; goal) echo 52000;; long) echo 53000;; *) echo 50500;; esac; }

eval_model(){ # $1 tag(e.g. S2)  $2 model_dir  $3 suites_csv(spatial,object,...)
    local tag="$1" M="$2" suites="$3" egpu s cfg port log v
    [ -f "$M/model.safetensors.index.json" ] || { say "EVAL $tag SKIP: no model at $M"; return 0; }
    egpu=$(pick_eval_gpu); [ -n "$egpu" ] || { say "EVAL $tag SKIP: no free eval GPU"; return 0; }
    say "EVAL $tag on gpu=$egpu, suites=$suites, osmesa temp=$EVAL_TEMP env=$EVAL_ENVS"
    IFS=',' read -ra arr <<< "$suites"
    for s in "${arr[@]}"; do
        cfg=$(cfg_of "$s"); port=$(port_of "$s")
        [ -n "$cfg" ] || { say "EVAL $tag $s SKIP: no cfg"; continue; }
        log="$O/seqcl_eval_${tag}_${s}.log"
        CUDA_VISIBLE_DEVICES=$egpu ISO_RAY_PORT=$port MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_NUM_THREADS=1 \
        timeout "$EVAL_TIMEOUT" bash "$RUNISO" \
            bash "$EVAL" "$cfg" LIBERO \
            rollout.model.model_path="$M" actor.model.model_path="$M" actor.model.is_lora=False \
            actor.model.unnorm_key="$NORM" env.eval.total_num_envs=$EVAL_ENVS \
            algorithm.sampling_params.temperature_eval=$EVAL_TEMP ++actor.seed=$EVAL_SEED > "$log" 2>&1
        v=$(grep -aoE "success_once': array\([0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")
        say "EVAL_RESULT $tag $s = ${v:-FAILED}"
    done
    say "EVAL $tag DONE"
}

do_merge(){ # $1 adapter_glob  $2 base(full HF)  $3 out(full HF)
    guard_disk
    local adpt; adpt=$(ls -d $1 2>/dev/null | head -1)
    [ -n "$adpt" ] || { say "ABORT: no adapter matching $1"; exit 5; }
    if [ -f "$3/model.safetensors.index.json" ]; then say "merge already exists: $3"; return 0; fi
    say "merging $adpt onto $(basename "$2") -> $3 (CPU, nice/ionice)"
    CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" nice -n 10 ionice -c3 "$PY" "$MERGE" \
        --adapter "$adpt" --base "$2" --out "$3" --unnorm-key "$NORM" 2>&1 | tee -a "$LOG"
    [ -f "$3/model.safetensors.index.json" ] || { say "ABORT: merge failed $3"; exit 6; }
    say "merged ok -> $3"; }

run_step(){ # $1 tag  $2 init  $3 teacher  $4 new_dset  $5 anchor_csv  $6 outdir
    guard_disk; wait_gpus
    say "STEP $1: SFT new=$4 anchor=$5 init=$(basename "$2") teacher=$(basename "$3") lambda=$LAMBDA steps=$STEPS"
    LWF_NEW_DSET="$4" LWF_ANCHOR_DSET="$5" SFT_ADAPTER_ONLY=1 SR_PROC_MAX=1000 \
        bash "$SAFE" "$O/seqcl_chain_${1}_driver.log" \
        bash "$LWF" "$2" "$3" "$6" "$LAMBDA" "$STEPS" "$SAVE" "$GPUS" 2>&1 | tee -a "$LOG"
    grep -aq 'LWF_SFT_DONE' "$O/seqcl_chain_${1}_driver.log" || { say "ABORT: step $1 never reached LWF_SFT_DONE"; exit 7; }
    say "STEP $1 SFT done"; }

say "=== seqcl_chain START (offline self-distill + per-step eval: object->goal->long from S1=spatial) ==="
guard_disk

# ---- STEP 2 (object): ALREADY RUNNING -> wait, merge S2, eval S2 ----
OUT2=$O/seqcl_v2_step2_object_anchor
S2=$O/seqcl_v2_S2_object/merged
say "waiting for the running step2 (object) to finish (up to 4h) ..."
for i in $(seq 1 240); do
    grep -aq 'LWF_SFT_DONE' "$O/seqcl_v2_step2_driver.log" 2>/dev/null && { say "step2 signalled DONE"; break; }
    ls $OUT2/*/adapters/step_2000/adapter_model.safetensors >/dev/null 2>&1 && { say "step2 step_2000 adapter present"; break; }
    sleep 60
done
ls -d $OUT2/*/adapters/step_2000 >/dev/null 2>&1 || { say "ABORT: step2 never produced step_2000 adapter"; exit 2; }
do_merge "$OUT2/*/adapters/step_2000" "$S1" "$S2"
eval_model S2 "$S2" "spatial,object"

# ---- STEP 3 (goal): init=teacher=S2, anchor spatial+object -> merge S3, eval S3 ----
OUT3=$O/seqcl_v2_step3_goal_anchor
S3=$O/seqcl_v2_S3_goal/merged
run_step goal "$S2" "$S2" libero_goal_no_noops "libero_spatial_no_noops,libero_object_no_noops" "$OUT3"
do_merge "$OUT3/*/adapters/step_2000" "$S2" "$S3"
eval_model S3 "$S3" "spatial,object,goal"

# ---- STEP 4 (long): init=teacher=S3, anchor spatial+object+goal -> merge S4, eval S4 ----
OUT4=$O/seqcl_v2_step4_long_anchor
S4=$O/seqcl_v2_S4_long/merged
run_step long "$S3" "$S3" libero_10_no_noops "libero_spatial_no_noops,libero_object_no_noops,libero_goal_no_noops" "$OUT4"
do_merge "$OUT4/*/adapters/step_2000" "$S3" "$S4"
eval_model S4 "$S4" "spatial,object,goal,long"

say "SEQCL_CHAIN_DONE  S2=$S2  S3=$S3  S4=$S4"
