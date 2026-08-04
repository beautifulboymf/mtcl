# Incremental SFT → OPD continual learning (pilot: spatial → object)

**Question.** The current pipeline SFTs ALL 130 tasks up front (the `RLinf-OpenVLAOFT-LIBERO-130-Base-Lora`
generalist) and only then does OPD. This experiment instead does **SFT per task, interleaved with OPD**
— truer to real continual learning, where you get tasks one at a time and cannot SFT on data you have
not seen yet. We measure the effect on a 2-suite pilot (spatial, then object).

```
openvla-7b-base
  --[A1 SFT spatial ]-->  spatial-SFT
  --[A2 OPD spatial ]-->  spatial student          (teacher = single 130 generalist)
  --[B1 SFT object  ]-->  object-SFT   (on top of the spatial student)
  --[B2 OPD object  ]-->  object student = FINAL
```
Baseline to compare against: the prior "SFT-all-130-upfront → OPD" sequential-CL run.

## Design decisions (please review)

1. **Base `B0` we LoRA ourselves** = `openvla-7b-base` (this is the model `130-Base-Lora`'s adapter is
   trained on: its `lora_adapter/adapter_config.json` says `base_model = .../openvla/openvla-7b`).
2. **SFT tool** = OpenVLA's `finetune.py`, run **inside the `rlinf-openvlaoft` venv from a neutral cwd
   (`/tmp`)** so its `import prismatic` resolves to the venv's **OFT** prismatic. That OFT prismatic
   trains **8-step action chunks** (`future_action_window_size = NUM_ACTIONS_CHUNK-1`), matching the
   8-chunk teacher/wrapper. `sft_aligned.py` asserts the venv prismatic loaded (fails loud otherwise)
   so we never silently train a 1-chunk model.
3. **Norm alignment = libero_130 throughout** (per your call — align with the previous experiment).
   Per-suite action bounds differ materially from libero_130 (e.g. action dim 5 range 0.607 vs
   spatial 0.253, ~2.4×), so alignment is required, not cosmetic: otherwise the student's 256-bin
   action grid ≠ the teacher's and OPD's forward-KL compares mismatched distributions.
   `sft_aligned.py` monkeypatches the RLDS pipeline to FORCE `make_dataset_from_rlds`'s statistics to
   the libero_130 values (`norm_override_libero130.json`) — the openvla repo is **not** edited. After
   SFT we copy the 130 `dataset_statistics.json` into the merged model and use
   `unnorm_key=libero_130_no_noops_trajall` everywhere.
4. **OPD** reuses the existing single-suite `libero_seqcl_opd_2gpu.yaml` (active_suites=[one suite],
   no rehearsal, teacher_map → the single 130). No new OPD config.
5. `finetune.py` **merges** the LoRA and saves a full HF model, so each SFT output is directly the
   next stage's student init — no separate merge step.

## Files
| file | role |
|---|---|
| `sft_aligned.py`   | norm-aligned launcher: monkeypatch RLDS stats → libero_130, assert venv OFT prismatic, run `finetune.py` |
| `incremental_sft.sh` | one SFT stage (suite, base, out) → merged 8-chunk OFT model with libero_130 stats |
| `make_norm_override.py` | regenerate `norm_override_libero130.json` from the 130-Base-Lora stats |
| `run_incremental_sft_opd.sh` | the chain: SFT→OPD→SFT→OPD; writes `MODELS.txt` |
| `eval_inc_models.sh` | GREEDY (temp=0) post-hoc eval on spatial+object, 50 env, libero_130 norm |

## Run (only after review; pick FREE GPUs; wrap in safe_run)
```bash
# 1) (one-time) norm override
python examples/embodiment/incremental_sft/make_norm_override.py
# 2) the chain (SFT torchrun on SFT_GPUS, OPD RLinf on OPD_GPUS)
SFT_GPUS=4,5 OPD_GPUS=4-5 SFT_STEPS=2000 OPD_STEPS=15 \
  bash /share/fanruochen-local/dev/scripts/safe_run.sh \
    /share/fanruochen-local/outputs/inc_sft_opd/driver.log \
    bash examples/embodiment/incremental_sft/run_incremental_sft_opd.sh
# 3) greedy eval
EVAL_GPU=4 bash examples/embodiment/incremental_sft/eval_inc_models.sh
```

## Safety
FREE GPUs only (never other tenants); OPD/eval ray isolated via `run_iso` (dedicated ports); SFT is
2-GPU torchrun with `TF_FORCE_GPU_ALLOW_GROWTH`; `df` guard before each stage; 50-env evals; every
heavy launch under `safe_run.sh`. Intermediate DCP checkpoints should be cleaned after conversion.
