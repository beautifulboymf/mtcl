#!/usr/bin/env python
"""WiSE-FT / RETAIN weight-space interpolation baseline (CPU-only, no training).

  theta_merged = (1 - alpha) * theta_base + alpha * theta_ft     (per tensor)

RETAIN (arXiv:2512.08333) = this exact form on VLA (base x finetuned), the ICLR-2026
must-beat. alpha in {0.25,0.5,0.75}. Operates on two merged HF OpenVLA-OFT dirs
(safetensors); copies config/tokenizer/*.py from the finetuned dir.

Usage:
  merge_wise.py --base <base_hf> --ft <plainOPD_hf> --alpha 0.5 --out <dir>
"""
import argparse, json, os, shutil, glob
import torch
from safetensors.torch import load_file, save_file


def load_sharded(hf_dir):
    """Return {tensor_name: tensor} across all safetensors shards + the shard map."""
    idx = os.path.join(hf_dir, "model.safetensors.index.json")
    tensors, shard_of = {}, {}
    if os.path.exists(idx):
        with open(idx) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
        for s in shards:
            d = load_file(os.path.join(hf_dir, s))
            for k, v in d.items():
                tensors[k] = v
                shard_of[k] = s
    else:  # single file
        s = "model.safetensors"
        d = load_file(os.path.join(hf_dir, s))
        for k, v in d.items():
            tensors[k] = v
            shard_of[k] = s
    return tensors, shard_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a = args.alpha
    print(f"[wise] alpha={a}  base={args.base}  ft={args.ft}")

    tb, _ = load_sharded(args.base)
    tf, shard_of = load_sharded(args.ft)
    os.makedirs(args.out, exist_ok=True)

    merged, n_interp, n_copied = {}, 0, 0
    for k, vf in tf.items():
        if k in tb and tb[k].shape == vf.shape and vf.is_floating_point():
            merged[k] = ((1.0 - a) * tb[k].float() + a * vf.float()).to(vf.dtype)
            n_interp += 1
        else:
            merged[k] = vf  # shape mismatch / non-float / base-missing -> keep ft
            n_copied += 1
    print(f"[wise] interpolated {n_interp} tensors, kept {n_copied} as-is")

    # regroup by shard and save (preserve original sharding + index)
    by_shard = {}
    for k, s in shard_of.items():
        by_shard.setdefault(s, {})[k] = merged[k]
    for s, d in by_shard.items():
        save_file(d, os.path.join(args.out, s), metadata={"format": "pt"})
    # copy index + all non-weight files (config, tokenizer, *.py, dataset_statistics)
    for f in os.listdir(args.ft):
        if f.endswith(".safetensors"):
            continue
        src = os.path.join(args.ft, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, f))
    print(f"[wise] saved merged HF -> {args.out}")


if __name__ == "__main__":
    main()
