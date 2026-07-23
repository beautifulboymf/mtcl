# dual-KL BASE anchor — 150-trial results (2026-07-23)

OpenVLA-OFT + LIBERO. Distill the object task (OPD, 130-GRPO teacher on object rollouts) into the
130-generalist student. **Only variable = the BASE anchor** (`anchor_lambda` 0 vs 1); everything else
identical (rank32, lr5e-6, 10 steps, micro_batch2, matched norm `libero_130_no_noops_trajall`).

Metric = `success_once` (field standard). **150 trials = 3 seeds {1234,2234,3234} × 50 env** (per-suite
tight across seeds → differences are real, not noise). object = the NEW/trained task (learn?);
spatial/goal/long = held-out suites, NOT distilled (forget?). GenRet = held-out avg ÷ BASE(0.593).

| model | object | spatial | goal | long | **held-out avg** | GenRet |
|---|---|---|---|---|---|---|
| BASE (ref, @50) | 0.64 | 0.66 | 0.40 | 0.72 | 0.593 | 1.00 |
| plain-OPD (no anchor) | 0.893 | 0.62 | 0.45 | 0.64 | 0.571 | 0.96 (net FORGETS) |
| RETAIN (WiSE-FT merge, α=.5) | 0.907 | 0.64 | 0.53 | 0.67 | 0.613 | 1.03 |
| **dual-KL (ours)** | **0.913** | **0.69** | **0.59** | **0.71** | **0.664** | **1.12** |

**Findings (H-A confirmed, confident at 150-trial):**
1. dual-KL learns object EQUALLY well (0.913 ≈ RETAIN 0.907 > plain-OPD 0.893) — the anchor does not sacrifice acquisition.
2. dual-KL preserves held-out BEST (0.664) — beats plain-OPD (0.571, +0.093) AND the strong RETAIN merge baseline (0.613, +0.051), on ALL three held-out suites.
3. dual-KL **Pareto-dominates** every baseline on (learn × preserve); held-out GenRet 1.12 → improves generalization 12% ABOVE base (plain-OPD is 0.96 = net forgets).

**object-PRO robustness (@50, once):** BASE object-appearance 0.38 / swap 0.68 / lan 0.70; dual-KL 0.52 / 0.88 / 0.90 (best of all models on unseen-object).

**Mechanism:** `loss = KL(πθ‖π_T)[reverse, learn] + λ·g(H_base)·KL(π_B‖πθ)[forward/mode-covering, preserve]`,
data-free (anchor on student's own rollout states + frozen base outputs). anchor_loss GROWS as student
drifts (3e-4→3e-2). Code: `fsdp_actor_worker.py` (_load_base_model + anchor term), config
`libero_object_dualkl_2gpu.yaml`. Eval: `dualkl_eval_core.sh` (3-seed), `run_iso.sh` (private ray/job).

**Caveat / limitation (next direction):** "generalization" here = held-out LIBERO suites — same domain /
objects / robot. NOT truly broad open-world generalization. Next: make BASE carry real broad
generalization + design a test and a preservation mechanism for THAT.
