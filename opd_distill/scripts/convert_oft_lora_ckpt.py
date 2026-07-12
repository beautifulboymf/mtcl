# Copyright (c) 2026. Reusable converter: merge an OpenVLA-OFT RLinf LoRA training
# checkpoint (PEFT full_weights.pt state dict) into a plain HuggingFace model dir
# that the RLinf eval pipeline can load via `model.model_path`.
#
# Faithful to RLinf's own construction:
#   * base model built exactly like rlinf/models/embodiment/openvla_oft/rlinf/__init__.py:get_model
#     (OpenVLAOFTForRLActionPrediction.from_pretrained + set_num_images_in_input)
#   * LoRA wrapping = the EXACT LoraConfig + get_peft_model from rlinf/models/__init__.py:228
#     for model_type=openvla_oft, is_lora=True, lora_path=None, use_film=False:
#         r=lora_rank=128, lora_alpha=lora_rank=128  ->  merge scaling alpha/r = 1.0
#     target_modules and init_lora_weights are copied verbatim from that source.
#
# CPU-only by design (CUDA_VISIBLE_DEVICES="" set by the wrapper) so no GPU is touched.
#
# Usage:
#   python convert_oft_lora_ckpt.py \
#       --ckpt   /path/to/full_weights.pt \
#       --base   /share/fanruochen-local/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora \
#       --out    /path/to/output/dir \
#       [--lora-rank 128] [--unnorm-key libero_130_no_noops_trajall]

import argparse
import glob
import json
import os
import shutil
import sys

import torch


# --- EXACT LoRA target modules from rlinf/models/__init__.py:228 (openvla_oft path) ---
LORA_TARGET_MODULES = [
    "proj",
    "qkv",
    "fc1",
    "fc2",  # vision
    "q",
    "kv",
    "fc3",
    "out_proj",  # project
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",  # llm
]

# Non-weight files to copy from the base model dir so the output is a complete loadable model.
AUX_FILES = [
    "config.json",
    "dataset_statistics.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
]


def build_base_model(base_dir, unnorm_key, action_dim, num_action_chunks,
                     max_prompt_length, num_images_in_input, center_crop):
    """Build OpenVLAOFTForRLActionPrediction exactly like the RLinf loader (CPU, eager attn)."""
    from prismatic.extern.hf.configuration_prismatic import (
        OpenVLAConfig as OpenVLAOFTConfig,
    )
    from transformers import AutoConfig

    from rlinf.models.embodiment.openvla_oft.rlinf.openvla_oft_action_model import (
        OpenVLAOFTForRLActionPrediction,
    )

    try:
        AutoConfig.register("openvla", OpenVLAOFTConfig)
    except Exception:
        pass  # already registered

    cfg_obj = AutoConfig.from_pretrained(base_dir, trust_remote_code=True)

    # Merge dataset_statistics.json into norm_stats (as the RLinf loader does).
    ds_path = os.path.join(base_dir, "dataset_statistics.json")
    if os.path.isfile(ds_path):
        ns = getattr(cfg_obj, "norm_stats", {}) or {}
        with open(ds_path) as f:
            ns.update(json.load(f))
        setattr(cfg_obj, "norm_stats", ns)

    # Behavioral config attrs used at build time (do NOT change the nn.Module param tree).
    for k, v in {
        "unnorm_key": unnorm_key,
        "center_crop": center_crop,
        "use_film": False,
        "use_proprio": False,
        "num_images_in_input": num_images_in_input,
        "value_type": "action_level",
    }.items():
        setattr(cfg_obj, k, v)

    model = OpenVLAOFTForRLActionPrediction.from_pretrained(
        pretrained_model_name_or_path=base_dir,
        torch_dtype=torch.bfloat16,
        config=cfg_obj,
        action_dim=action_dim,
        num_action_chunks=num_action_chunks,
        add_value_head=False,
        max_prompt_length=max_prompt_length,
        trust_remote_code=True,
        attn_implementation="eager",  # CPU-safe; irrelevant to a weight-only merge
    )
    model.vision_backbone.set_num_images_in_input(num_images_in_input)
    model.to(torch.bfloat16)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="PEFT full_weights.pt state dict")
    ap.add_argument("--base", required=True, help="base student HF model dir")
    ap.add_argument("--out", required=True, help="output dir for merged HF model")
    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--action-dim", type=int, default=7)
    ap.add_argument("--num-action-chunks", type=int, default=8)
    ap.add_argument("--max-prompt-length", type=int, default=128)
    ap.add_argument("--num-images-in-input", type=int, default=1)
    ap.add_argument("--center-crop", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--unnorm-key", default=None,
                    help="default: sole key in base/dataset_statistics.json")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    assert os.path.isfile(args.ckpt), f"ckpt not found: {args.ckpt}"
    assert os.path.isdir(args.base), f"base dir not found: {args.base}"

    if args.unnorm_key is None:
        with open(os.path.join(args.base, "dataset_statistics.json")) as f:
            keys = list(json.load(f).keys())
        assert len(keys) >= 1, "empty dataset_statistics.json"
        args.unnorm_key = keys[0]
    print(f"[cfg] lora_rank(r)={args.lora_rank}  lora_alpha={args.lora_rank}  "
          f"scaling=alpha/r={args.lora_rank/args.lora_rank:.3f}  unnorm_key={args.unnorm_key}",
          flush=True)

    # 1) Build base model (RLinf-faithful).
    print("[1/6] building base OpenVLAOFTForRLActionPrediction on CPU ...", flush=True)
    model = build_base_model(
        args.base, args.unnorm_key, args.action_dim, args.num_action_chunks,
        args.max_prompt_length, args.num_images_in_input, args.center_crop,
    )

    # 2) Wrap with the EXACT LoraConfig RLinf uses for openvla_oft (rlinf/models/__init__.py:228).
    print("[2/6] applying get_peft_model (EXACT RLinf LoraConfig) ...", flush=True)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,      # == lora_rank  -> alpha=128
        lora_dropout=0.0,
        target_modules=LORA_TARGET_MODULES,
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora_config)

    # 3) Load the trained checkpoint state dict.
    print(f"[3/6] loading state dict: {args.ckpt}", flush=True)
    sd = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=True)
    n_ckpt = len(sd)
    n_lora = sum(1 for k in sd if "lora_" in k)
    n_base_layer = sum(1 for k in sd if "base_layer" in k)
    report = model.load_state_dict(sd, strict=False)
    missing = list(report.missing_keys)
    unexpected = list(report.unexpected_keys)

    # LoRA + base_layer keys from the ckpt MUST all be consumed (0 unexpected among them).
    bad_unexpected = [k for k in unexpected if ("lora_" in k or "base_layer" in k)]
    missing_lora = [k for k in missing if ("lora_" in k or "base_layer" in k)]
    print(f"    ckpt keys={n_ckpt} (lora={n_lora}, base_layer={n_base_layer})", flush=True)
    print(f"    load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    print(f"    -> unexpected LoRA/base_layer keys: {len(bad_unexpected)} "
          f"(MUST be 0)", flush=True)
    print(f"    -> missing LoRA/base_layer keys:    {len(missing_lora)} "
          f"(MUST be 0)", flush=True)
    if missing:
        print(f"    missing (first 20): {missing[:20]}", flush=True)
    if unexpected:
        print(f"    unexpected (first 20): {unexpected[:20]}", flush=True)
    if bad_unexpected or missing_lora:
        print("FATAL: LoRA/base_layer key mismatch -> module tree does not match ckpt. Aborting.",
              flush=True)
        sys.exit(2)

    # 4) Merge and unload.
    print("[4/6] merge_and_unload() (base + (B@A)*alpha/r) ...", flush=True)
    merged = model.merge_and_unload()

    # 5) Save merged weights.
    os.makedirs(args.out, exist_ok=True)
    print(f"[5/6] save_pretrained -> {args.out}", flush=True)
    merged.save_pretrained(args.out, safe_serialization=True)

    # 6) Copy aux (non-weight) files + all *.py; never overwrite merged safetensors/index.
    print("[6/6] copying config/tokenizer/*.py from base (not overwriting weights) ...", flush=True)
    copied = []
    for fn in AUX_FILES:
        src = os.path.join(args.base, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, fn))
            copied.append(fn)
    for src in glob.glob(os.path.join(args.base, "*.py")):
        fn = os.path.basename(src)
        shutil.copy2(src, os.path.join(args.out, fn))
        copied.append(fn)
    print(f"    copied: {copied}", flush=True)

    print("DONE_CONVERT", flush=True)


if __name__ == "__main__":
    main()
