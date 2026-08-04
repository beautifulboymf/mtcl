# OpenVLA Generalization Benchmark (for data-free preservation, "problem-2") — 2026-07-24

## Purpose
Measure a GENERAL base VLA's (OpenVLA-7B, OpenX-pretrained) broad generalization on a
credible, published real2sim benchmark, so that "problem-2" (does learning a new task via
data-free distillation/anchor erode the base's generalization, and does the anchor preserve it?)
has a concrete, quantifiable measuring stick. This replaces the failed LIBERO route (base ~1%
on Bridge = nothing to preserve).

## Why SimplerEnv (not native ManiSkill / LIBERO)
OpenVLA only knows its OpenX training embodiments: **google_robot (fractal)** + **widowx (bridge)**.
- Native ManiSkill tasks (Franka PickCube, etc.) → OpenVLA ≈ 0 (never trained). Out.
- LIBERO base zero-shot → weak. Out.
- **SimplerEnv real2sim digital twins** replicate exactly google_robot + widowx → OpenVLA works
  (validated: coke-can 40%). This IS the benchmark. It is peer-reviewed (SimplerEnv, CoRL'24
  ManiSkill3 / the original ManiSkill2 real2sim), so using it is defensible.

Infra: runs on `simpler_ms2` conda env (ManiSkill2_real2sim + sapien 2.2.2) on **CPU-Vulkan**
via the custom `fake_extsemfd` layer (GPU Vulkan blocked, see project memory). ~50s/episode.

## Environments (OpenVLA-native, 8 task families)
**google_robot (fractal, base is strong here):**
1. Pick Coke Can (`GraspSingleOpenedCokeCanInScene-v0`)
2. Move Near (`MoveNearGoogleBakedTexInScene-v0`)
3. Open/Close Drawer (`Open/CloseTopDrawerCustomInScene-v0` …)
4. Put in Drawer (`PlaceIntoClosedTopDrawerCustomInScene-v0`)

**widowx (bridge, base is WEAK ~0-4% — include for breadth, not the main signal):**
5. Put Carrot on Plate · 6. Put Eggplant in Basket · 7. Put Spoon on Towel · 8. Stack Green Cube

## The generalization axes = SimplerEnv's two eval modes
- **Visual Matching (VM)** — sim rendered to match a real photo (rgb-overlay). = in-distribution SR.
- **Variant Aggregation (OOD)** — pure-sim variants perturbing conditions the base never saw.
  **These axes ARE the operational definition of "generalization" (no need to invent one):**
  - **lighting**: `slightly_darker_lighting=True` / `slightly_brighter_lighting=True`
  - **distractors**: `...DistractorInScene-v0` + `distractor_config=more` (vs `no_distractor=True`)
  - **background/scene**: `..._alt_background`, `Baked_sc1_staging_objaverse_cabinet1_h870`, etc.
  - **table texture**: `urdf_version=recolor_tabletop_visual_matching_{1,2}`
  - **camera pose**: `...AltGoogleCamera{,2}InScene-v0`

**Generalization metric** = per-task per-axis success_once. Base's "generalization ability" =
variant-aggregation mean SR (and the VM→OOD drop). For problem-2: measure this whole profile on
BASE, then after distilling a new task (with vs without the data-free anchor) → does the OOD
profile erode (plain distill) vs hold (anchor)?

## v1 (this run): tight profile to establish the base anchor point
CPU is slow, so v1 = a representative subset (expand later, or after GPU-Vulkan speedup):
- Tasks: **Pick Coke Can, Move Near** (the 2 strong google_robot tasks validated so far).
- Modes: **VM baseline + OOD{darker-lighting, more-distractor, alt-background}**.
- ~9-10 episodes/config. Coke VM baseline already known = 0.40 (upright, 25ep).
- Output: SR per (task, mode) → base generalization profile.
Driver: `dev/scripts/openvla_genbench_v1.sh`; results → `/tmp/genbench_v1.log`.

## v2 (later)
Add Drawer + Put-in-Drawer + all 4 Bridge tasks; add camera + texture axes; ≥24 ep/config;
all 3 coke orientations. Full 8-task × 5-axis profile. Needs GPU-Vulkan for reasonable walltime.

## Then: problem-2 experiment
BASE profile (this) → distill a new task into OpenVLA (data-free OPD + action/visual anchor) →
re-measure profile → compare erosion (plain) vs preservation (anchor). The anchor's win = keeping
the variant-aggregation OOD SR while acquiring the new task.
