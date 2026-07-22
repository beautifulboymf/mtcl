#!/usr/bin/env python
"""reverse-LiNeS post-hoc baseline (CPU-only, no training).

LiNeS (arXiv:2410.17146) scales the fine-tuning task-vector linearly by layer DEPTH.
Standard LiNeS keeps DEEP layers full, shrinks SHALLOW. REVERSE-LiNeS (our hypothesis:
skill+forgetting live in the DEEP action expert) does the opposite — keep SHALLOW full,
pull DEEP back toward base:

  tv(k)      = theta_ft(k) - theta_base(k)
  lambda(l)  = 1 - (1-alpha) * l/(L-1)     # l=0 -> 1.0 (shallow full);  l=L-1 -> alpha (deep shrunk)
  theta(k)   = theta_base(k) + lambda(layer_of(k)) * tv(k)

Non-layer tensors (embeddings / final norm / action head) keep lambda=1 (full task update).
Use --forward for standard LiNeS (deep full, shallow shrunk) as an additional baseline.

Usage:
  merge_lines.py --base <base_hf> --ft <plainOPD_hf> --alpha 0.5 --out <dir> [--forward]
"""
import argparse, json, os, re, shutil
import torch
from safetensors.torch import load_file, save_file

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def load_sharded(hf_dir):
    idx = os.path.join(hf_dir, "model.safetensors.index.json")
    tensors, shard_of = {}, {}
    if os.path.exists(idx):
        weight_map = json.load(open(idx))["weight_map"]
        for s in sorted(set(weight_map.values())):
            for k, v in load_file(os.path.join(hf_dir, s)).items():
                tensors[k], shard_of[k] = v, s
    else:
        for k, v in load_file(os.path.join(hf_dir, "model.safetensors")).items():
            tensors[k], shard_of[k] = v, "model.safetensors"
    return tensors, shard_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--forward", action="store_true", help="standard LiNeS (deep full) instead of reverse")
    args = ap.parse_args()

    tb, _ = load_sharded(args.base)
    tf, shard_of = load_sharded(args.ft)
    # discover L (max layer index) across ft tensors
    L = 1 + max((int(m.group(1)) for k in tf if (m := LAYER_RE.search(k))), default=0)
    print(f"[lines] {'forward(standard)' if args.forward else 'reverse'}  alpha={args.alpha}  L={L} layers")

    os.makedirs(args.out, exist_ok=True)
    merged, n_ramp, n_full = {}, 0, 0
    for k, vf in tf.items():
        if k in tb and tb[k].shape == vf.shape and vf.is_floating_point():
            tv = vf.float() - tb[k].float()
            m = LAYER_RE.search(k)
            if m and L > 1:
                l = int(m.group(1))
                frac = l / (L - 1)
                lam = (args.alpha + (1 - args.alpha) * frac) if args.forward \
                    else (1 - (1 - args.alpha) * frac)
                n_ramp += 1
            else:
                lam = 1.0  # non-layer (embed/norm/head) -> full task update
                n_full += 1
            merged[k] = (tb[k].float() + lam * tv).to(vf.dtype)
        else:
            merged[k] = vf
            n_full += 1
    print(f"[lines] depth-ramped {n_ramp} layer tensors, {n_full} kept full")

    by_shard = {}
    for k, s in shard_of.items():
        by_shard.setdefault(s, {})[k] = merged[k]
    for s, d in by_shard.items():
        save_file(d, os.path.join(args.out, s), metadata={"format": "pt"})
    for f in os.listdir(args.ft):
        if not f.endswith(".safetensors") and os.path.isfile(os.path.join(args.ft, f)):
            shutil.copy2(os.path.join(args.ft, f), os.path.join(args.out, f))
    print(f"[lines] saved -> {args.out}")


if __name__ == "__main__":
    main()
