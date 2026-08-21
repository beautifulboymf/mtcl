#!/bin/bash
# opd_4teacher.sh — the 4-teacher routed OPD: one student, four per-suite expert teachers.
#
#   student  = lwf_long_e1000_merged   (the long-SFT CL student: .56/.76/.96/.68)
#   teachers = spatial .94 | object 1.00 | goal .94 | long .88   (long is the one we just built;
#              before it existed this set was blocked and long had to be routed to the 130 generalist)
#
# TWO BASES, ON PURPOSE. spatial/object/goal are adapters on base_stats130; the long teacher is an
# adapter on the student's own merged model. fsdp_actor_worker used to demand ONE base for its
# shared-base fast path and would otherwise load four full 7B teachers (a 78.5 GiB OOM). It now
# groups by base, so this costs 2 bases + 4 adapters.
#
# KNOWN RISK, stated up front: this exact shape (per-suite experts distilled into one student LoRA)
# collapsed twice before -- one-shot 3-teacher took the mean .65 -> .43, staged took it to .23 --
# and the cause was diagnosed as TEACHER/STUDENT LINEAGE MISMATCH, not a bug (routing, adapter
# switching, teacher SR, norms and suite balance were each verified first). Three of these four
# teachers are still foreign-lineage; only long shares the student's lineage. Run short first.
set -uo pipefail
TAG="${1:-mt4}"; STEPS="${2:-10}"

O=/share/fanruochen-local/outputs
STUDENT="${OPD_STUDENT:-$O/inc_sft_opd/lwf_long_e1000_merged}"
B130="$O/inc_sft_opd/base_stats130"
T_SPATIAL="$B130::$O/seqcl_rlspat_opd/spatial_cat_r160"
T_OBJECT="$B130::$O/inc_sft_opd/sft_object_base3_cont/openvla-7b-base+libero_object_no_noops+b32+lr-0.0003+lora-r32+dropout-0.0--image_aug/adapters/step_500"
T_GOAL="$B130::$O/inc_sft_opd/teachers_r160/goal_r160"
T_LONG="$O/inc_sft_opd/lwf_long_e1000_merged::$O/seqcl_long_opd130/adapter/long_opd130"

# ---- RED LINES. Disk is the one that has actually taken this machine down (a full /share stopped
# every tenant for days), and each OPD checkpoint here is ~29G, so refuse on a tight disk.
free=$(df -BG --output=avail /share/fanruochen-local | tail -1 | tr -dc '0-9')
(( free >= 200 )) || { echo "ABORT: only ${free}G free on /share/fanruochen-local (need >=200G; each ckpt ~29G)"; exit 1; }
anon=$(awk '/^anon /{a=$2} /^slab /{s=$2} /^kernel_stack /{k=$2} END{printf "%.0f",(a+s+k)/1073741824}' /sys/fs/cgroup/memory.stat)
(( anon < 150 )) || { echo "ABORT: cgroup anon already ${anon}G -- something else is running"; exit 1; }
iow=$(vmstat 1 2 | tail -1 | awk '{print $16}')
(( ${iow:-0} < 25 )) || { echo "ABORT: iowait ${iow}% -- disk already busy"; exit 1; }

for t in "$T_SPATIAL" "$T_OBJECT" "$T_GOAL" "$T_LONG"; do
  b="${t%%::*}"; a="${t#*::}"
  [ -f "$b/model.safetensors.index.json" ] || { echo "ABORT: teacher base missing: $b"; exit 1; }
  [ -f "$a/adapter_model.safetensors" ]    || { echo "ABORT: teacher adapter missing: $a"; exit 1; }
  r=$(python3 -c "import json;print(json.load(open('$a/adapter_config.json'))['r'])")
  echo "  teacher ok  r=$r  $(basename "$b")::$(basename "$a")"
done
echo "[preflight] disk=${free}G anon=${anon}G iowait=${iow}%  student=$(basename "$STUDENT")  steps=$STEPS"

cd /home/fanruochen/CL/RLinf
exec env OPD_GPUS="${OPD_GPUS:-0,1,2,4}" OPD_ENVS="${OPD_ENVS:-32}" OPD_STEPS="$STEPS" \
     OPD_ROLLOUT_EPOCH="${OPD_ROLLOUT_EPOCH:-4}" OPD_RAY_PORT="${OPD_RAY_PORT:-52000}" \
     OPD_MICRO="${OPD_MICRO:-8}" \
     ${OPD_WEIGHTS:+OPD_WEIGHTS="$OPD_WEIGHTS"} \
  bash examples/embodiment/incremental_sft/opd_multi.sh "$STUDENT" "$TAG" \
    libero_spatial="$T_SPATIAL" \
    libero_object="$T_OBJECT" \
    libero_goal="$T_GOAL" \
    libero_10="$T_LONG"
