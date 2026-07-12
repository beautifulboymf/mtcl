# OpenVLA-OFT On-Policy Distillation (VLA-OPD) — Continual-Learning Experiments

Fork of [RLinf](https://github.com/RLinf/RLinf). This branch adds **on-policy reverse-KL
distillation (VLA-OPD)** of an RL-trained teacher into an OpenVLA-OFT student, for continual
learning (learn a new task while preserving old skills). Everything here is a **diff on top of
upstream RLinf** — `git diff <upstream-base>..<this-branch>` shows exactly the added code.

## TL;DR result

Distilling the `object` task via OPD (reverse-KL, on-policy) into three different students,
**same method** (matched norm + LoRA rank 32 + rollout_epoch 2 + lr 5e-6):

| Experiment (config) | Student (object start) | object result | note |
|---|---|---|---|
| **1-traj SFT** `libero_object_opd_faithful_2gpu.yaml` | weak SFT-traj1 (0.10) | **0.10 → 0.45** | works from a weak start |
| **distill-130** `libero_object_opd_faithful130_2gpu.yaml` | 130 generalist (0.70) | **0.70 → 0.98** | reaches teacher; avg-once over 4 suites **0.62 → 0.67 (net gain)**, only mild forgetting |
| **cross-embodiment** `libero_object_opd_faithfulBASE_2gpu.yaml` | raw OpenX base (0) | **0 → 0** | bootstrap wall: on-policy OPD needs a non-zero start |

Contrast — the SAME strong 130 student **collapses** (goal 0.5→0, avg 0.62→0.24) when the method
is wrong (norm-**mismatch** + rank **128**). So the make-or-break is the **method**, not student strength.

## 130-task full OPD (distill ALL tasks, not one suite)

`libero_object_opd_faithful130_2gpu.yaml` distills a single-suite (object) teacher; to distill the
**whole 130-task generalist** roll out across all 130 tasks (`env/libero_130`) with the 130-GRPO
teacher. Config: **`libero_130full_opd_fromsft_2gpu.yaml`** (student = 130-Base-Lora SFT, teacher =
`RLinf-OpenVLAOFT-LIBERO-130` GRPO, matched norm `libero_130_no_noops_trajall`, rank 32, rollout_epoch 2,
`total_num_envs: 32`, 10 steps). In-training 130-task average success **climbs 0.62 → 0.67 → 0.81**
over the first 3 steps (toward the teacher's ~0.97) — OPD distillation works at full-130 scale, not just
single-suite. (10 steps is enough to see the rising trajectory; post-hoc 4-suite eval confirms.)

## The 3 make-or-break factors

1. **Norm alignment (crux).** Student and teacher must share the **same `unnorm_key`**. In
   discrete OFT the OPD reverse-KL matches action-token distributions; if student/teacher use
   different action normalizations, the same token = a different physical action → the distill
   target is decoded wrong → the student's existing skill collapses.
   Code: `actor.teacher_unnorm_key` handled in `rlinf/workers/actor/fsdp_actor_worker.py`
   (+ expert/teacher unnorm in `rlinf/workers/rollout/hf/huggingface_worker.py`).
2. **Low LoRA rank (32, not 128).** Lower rank = less drift / less entropy-collapse of the
   student's other-task behavior. Config: `actor.model.lora_rank: 32`.
   RLinf LoRA config (r, alpha=r, target_modules): `rlinf/models/__init__.py`.
3. **Non-zero student start.** On-policy OPD scores the *student's* rollouts with the teacher;
   with 0 success there is no signal (bootstrap wall). 0.10 already bootstraps; raw base (0) does not.

**NOT the factors:** learning rate (all runs `lr: 5.0e-6`), and rollout_epoch (winners used *less*
data, 2 vs the failed run's 4).

## Training method (VLA-OPD)

- `algorithm.adv_type: opd`, `algorithm.opd_mode: distill`, `loss_type: actor`.
- The OPD advantage = per-token reverse-KL reward `r_t = log π_teacher(a_t) − log π_student(a_t)`,
  registered in `rlinf/algorithms/advantages.py` (`@register_advantage("opd")`).
- Frozen RL'd teacher scores the on-policy student rollout; student updated with a LoRA adapter.

## Key hyperparameters (all three configs)

```
adv_type: opd        opd_mode: distill      loss_type: actor
lora_rank: 32        lr: 5.0e-6             rollout_epoch: 2
student unnorm_key == teacher_unnorm_key    (MATCHED — the crux)
2-GPU FSDP (component_placement "2-3"), enable_offload: True, temperature 1.6
```

## How to run

```bash
# train (2 GPUs, EGL GPU render). CONFIG = one of the 3 faithful configs.
bash opd_distill/scripts/run_opd_train_egl.sh libero_object_opd_faithful130_2gpu

# convert the LoRA checkpoint (full_weights.pt, PEFT-named) -> merged HF dir (CPU-only).
# IMPORTANT: pass --lora-rank 32 (script default is 128) and the student's unnorm_key.
python opd_distill/scripts/convert_oft_lora_ckpt.py \
  --ckpt <run>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt \
  --base <student_hf_dir> --out <run>/converted/step_N \
  --lora-rank 32 --unnorm-key libero_130_no_noops_trajall

# eval (per-suite, 50 env). unnorm_key must match the student's.
bash examples/embodiment/eval_embodiment.sh libero_object_g7_eval LIBERO \
  rollout.model.model_path=<converted> actor.model.model_path=<converted> \
  actor.model.is_lora=False actor.model.unnorm_key=libero_130_no_noops_trajall \
  env.eval.total_num_envs=50
```

## Files added / changed vs RLinf

**Core code (the method):**
- `rlinf/algorithms/advantages.py` — OPD reverse-KL advantage (`@register_advantage("opd")`).
- `rlinf/workers/actor/fsdp_actor_worker.py` — teacher loading + `teacher_unnorm_key` (norm align).
- `rlinf/workers/rollout/hf/huggingface_worker.py` — teacher/expert unnorm handling.
- `rlinf/models/__init__.py` — OpenVLA-OFT LoRA config (rank/alpha/target_modules).
- `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py`, `rlinf/runners/embodied_runner.py`, `rlinf/envs/libero/libero_env.py`, `examples/embodiment/eval_embodiment.sh`.

**Experiment configs:** `examples/embodiment/config/libero_object_opd_faithful{,130,BASE}_2gpu.yaml`
and the per-suite eval configs `libero_{object,spatial,goal,10}_{g7,grpo_openvlaoft}_eval.yaml`.

**Scripts:** `opd_distill/scripts/{run_opd_train_egl.sh, convert_oft_lora_ckpt.py, convert_oft_lora_ckpt.sh}`.
