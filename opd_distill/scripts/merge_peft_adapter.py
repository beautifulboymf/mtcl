#!/usr/bin/env python
# merge_peft_adapter.py -- merge a STANDARD PEFT adapter dir (adapter_model.safetensors +
# adapter_config.json, as OpenVLA-OFT finetune.py writes them) onto its base into a plain
# loadable HF model dir. CPU-only (never touches a GPU). This is the official OFT merge path
# (PeftModel.from_pretrained -> merge_and_unload), unlike convert_oft_lora_ckpt.py which is
# for RLinf's own full_weights.pt state dicts.
#
# Usage:
#   CUDA_VISIBLE_DEVICES="" python merge_peft_adapter.py \
#     --adapter <adapter_dir> --base <base_hf_dir> --out <out_dir> [--unnorm-key ...]
import argparse
import json
import os
import shutil

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="PEFT adapter dir (adapter_model.safetensors)")
    ap.add_argument("--base", required=True, help="base HF model dir")
    ap.add_argument("--out", required=True, help="output dir for merged HF model")
    ap.add_argument("--unnorm-key", default="libero_130_no_noops_trajall")
    args = ap.parse_args()

    assert os.path.isfile(os.path.join(args.adapter, "adapter_model.safetensors")), (
        f"not a PEFT adapter dir: {args.adapter}"
    )
    assert os.path.isfile(os.path.join(args.base, "model.safetensors.index.json")), (
        f"not an HF model dir: {args.base}"
    )

    from peft import PeftModel  # noqa: E402
    from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor  # noqa: E402

    print(f"[1/5] loading base on CPU: {args.base}", flush=True)
    cfg = AutoConfig.from_pretrained(args.base, trust_remote_code=True)
    # fold dataset_statistics into norm_stats + set unnorm_key so the merged model evals
    # with the same action normalization as the teachers / OPD student.
    ds = os.path.join(args.base, "dataset_statistics.json")
    if os.path.isfile(ds):
        ns = getattr(cfg, "norm_stats", {}) or {}
        with open(ds) as f:
            ns.update(json.load(f))
        setattr(cfg, "norm_stats", ns)
    setattr(cfg, "unnorm_key", args.unnorm_key)

    base = AutoModelForVision2Seq.from_pretrained(
        args.base, config=cfg, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"[2/5] attaching adapter: {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    print("[3/5] merge_and_unload() ...", flush=True)
    model = model.merge_and_unload()
    print(f"[4/5] save_pretrained -> {args.out}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)

    print("[5/5] copying config/tokenizer/*.py from base (not overwriting weights) ...", flush=True)
    for fn in os.listdir(args.base):
        if fn.startswith("model") and fn.endswith((".safetensors", ".safetensors.index.json")):
            continue  # keep the merged weights we just wrote
        src = os.path.join(args.base, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, fn))
    # ensure dataset_statistics present in the out dir for eval unnorm lookup
    if os.path.isfile(ds):
        shutil.copy2(ds, os.path.join(args.out, "dataset_statistics.json"))
    assert os.path.isfile(os.path.join(args.out, "model.safetensors.index.json")), (
        "merge produced no sharded model index"
    )
    print(f"MERGE_DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()
