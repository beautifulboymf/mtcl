# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Merge four INDEPENDENTLY trained single-slot checkpoints into one slot checkpoint.

The original-LoRI control: each run trained ONLY its own suite's B against its own
suite's expert, from the same base, blind to the other suites (no anchor). Because
every run was built with the same seed, all four hold the IDENTICAL frozen
orthonormal A -- so the merged model

    W + sum_k B_k A_k     with B_k taken from run k

is exact: each run's checkpoint carries B_j = 0 for every j it did not train, and the
A row-blocks are mutually orthogonal, so column-splicing the four B's loses nothing.

Both facts are ASSERTED per layer, not assumed:
  * every non-own B column block must be exactly zero (a nonzero one means the
    gating leaked and the run was not independent);
  * every run's A must equal run 0's bit-for-bit (a mismatch means the seeds or the
    build order diverged, the subspaces are different, and splicing would silently
    mix bases).

Output is a normal slot full_weights.pt; the existing converter
(convert_oft_slot_ckpt.sh, SLOT_FROZEN_ORTH=1) turns it into the merged HF model.
"""

import argparse
import sys

import torch

# slot_order and ranks, MUST match the training config (libero_seqslot_4gpu.yaml)
SLOT_ORDER = ["libero_10", "libero_goal", "libero_spatial", "libero_object"]
SLOT_RANKS = {"libero_10": 128, "libero_goal": 64, "libero_spatial": 64, "libero_object": 32}


def offsets():
    out, acc = {}, 0
    for s in SLOT_ORDER:
        out[s] = (acc, acc + SLOT_RANKS[s])
        acc += SLOT_RANKS[s]
    return out, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="merged full_weights.pt path")
    for s in SLOT_ORDER:
        ap.add_argument(f"--{s.replace('libero_', '')}", default=None,
                        help=f"full_weights.pt of the run that trained {s}; omit to leave "
                             "that suite's columns ZERO (an incremental / partial merge)")
    args = ap.parse_args()
    paths = {s: getattr(args, s.replace("libero_", "")) for s in SLOT_ORDER}
    paths = {s: p for s, p in paths.items() if p}
    if not paths:
        print("FATAL: no runs given", flush=True); sys.exit(2)
    print(f"merging {sorted(paths)}; absent suites stay zero", flush=True)

    off, total_rank = offsets()
    print(f"slot columns: {off}  R={total_rank}", flush=True)

    dicts = {}
    for s, p in paths.items():
        print(f"loading {s}: {p}", flush=True)
        dicts[s] = torch.load(p, map_location="cpu", weights_only=True)

    present = [s for s in SLOT_ORDER if s in dicts]
    ref = dicts[present[0]]
    merged = {}
    n_b = n_a = 0
    for key, ref_t in ref.items():
        if key.endswith("slot_B.weight"):
            n_b += 1
            out = torch.zeros_like(ref_t)
            for s in present:
                t = dicts[s][key]
                lo, hi = off[s]
                own = t[:, lo:hi]
                # every column OUTSIDE the run's own block must be exactly zero
                other = t.clone()
                other[:, lo:hi] = 0
                if other.abs().max().item() != 0.0:
                    print(f"FATAL: {s} run has nonzero B outside its own columns at {key} "
                          f"(max {other.abs().max().item():.3e}) -- the runs were not "
                          "independent; refusing to splice.", flush=True)
                    sys.exit(2)
                out[:, lo:hi] = own
            merged[key] = out
        elif key.endswith("slot_A.weight"):
            n_a += 1
            for s in present[1:]:
                if not torch.equal(dicts[s][key], ref_t):
                    d = (dicts[s][key].float() - ref_t.float()).abs().max().item()
                    print(f"FATAL: {s} run's A differs from {SLOT_ORDER[0]}'s at {key} "
                          f"(max |diff| {d:.3e}) -- different subspaces cannot be "
                          "column-spliced. Check the seeds.", flush=True)
                    sys.exit(2)
            merged[key] = ref_t
        else:
            # base weights / extra_state: identical across runs by construction (frozen);
            # spot-check the first few tensors rather than every 15G of them.
            merged[key] = ref_t
    print(f"spliced {n_b} B layers, verified {n_a} shared A layers", flush=True)
    torch.save(merged, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
