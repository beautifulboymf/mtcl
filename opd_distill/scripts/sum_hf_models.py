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
"""Delta-sum several converted HF models onto their shared base:  out = base + sum(add_i - base).

The LoRI-independent merge when the runs do NOT share an A. Column-splicing four slot
checkpoints requires bit-identical A matrices; the four runs came up with different ones
(the config seed does not pin SlotProj's draw -- measured max|A_i - A_j| = 0.206), so the
merge has to happen in DELTA space instead: each run's converted model is base + its own
s*B_k@A_k (its checkpoint holds zero B columns for every other suite), and the deltas add.
Cross-run subspaces are then random rather than exactly orthogonal -- which is exactly
original LoRI's regime (independent random frozen A per task).

Accumulates in fp32, casts back to the base dtype per tensor. Non-tensor files (config,
tokenizer, dataset statistics, the safetensors index) are copied from the base; every
model must therefore come from the SAME converter and base, which also guarantees an
identical shard layout -- asserted per shard, per key.
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def shard_names(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return sorted(set(json.load(f)["weight_map"].values()))
    return ["model.safetensors"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--add", action="append", required=True,
                    help="converted model dir; repeatable, one per run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    shards = shard_names(args.base)
    for m in args.add:
        assert shard_names(m) == shards, f"{m} shard layout differs from base"
    os.makedirs(args.out, exist_ok=True)

    for shard in shards:
        print(f"[sum] {shard}: base + {len(args.add)} delta(s)", flush=True)
        out_tensors, dtypes = {}, {}
        with safe_open(os.path.join(args.base, shard), framework="pt") as fb:
            for key in fb.keys():
                t = fb.get_tensor(key)
                dtypes[key] = t.dtype
                out_tensors[key] = t.float()
        for m in args.add:
            with safe_open(os.path.join(m, shard), framework="pt") as fa:
                assert set(fa.keys()) == set(out_tensors), f"{m}/{shard} key set differs"
                with safe_open(os.path.join(args.base, shard), framework="pt") as fb:
                    for key in fa.keys():
                        out_tensors[key] += fa.get_tensor(key).float() - fb.get_tensor(key).float()
        save_file(
            {k: v.to(dtypes[k]) for k, v in out_tensors.items()},
            os.path.join(args.out, shard),
            metadata={"format": "pt"},
        )

    for name in os.listdir(args.base):
        if name.endswith(".safetensors"):
            continue
        src = os.path.join(args.base, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, name))
    print(f"[sum] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
