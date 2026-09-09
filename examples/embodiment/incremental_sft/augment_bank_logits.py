"""augment_bank_logits.py -- precompute the teacher's action distribution into an anchor bank.

WHY
---
The off-demo anchor bank is a FIXED set of states, so the teacher's output on it never changes --
yet training currently re-runs a 7B teacher forward on every step to recompute it. Storing the
distribution once removes that forward from the training loop entirely.

Cost of storing: 56 action tokens x 256 bins x 2 bytes = 28 KB per state (~230 MB per 8k-state
suite) -- negligible next to the pixel tensors already in the bank. This is the standard
"trade storage for compute" move (DER stores logits instead of re-running a model), applied to a
rollout-state bank instead of a replay buffer.

It also makes MULTI-TEACHER anchoring nearly free: each extra checkpoint adds one stored tensor,
not one forward pass per step. That is the only reason multi-checkpoint anchoring has been
considered too expensive to try.

Writes key "teacher_logp" [B, 56, 256] fp16 = log-softmax over the action bins at the action
positions, in order. finetune_lwf.py uses it when present and skips the teacher forward.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import prismatic  # noqa: E402

_pf = getattr(prismatic, "__file__", "") or ""
if "site-packages" not in _pf:
    raise SystemExit(f"[aug] REFUSING: prismatic resolved to {_pf!r} (not the venv OFT prismatic)")

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor  # noqa: E402
from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help=".pt written by build_offdemo_anchor.py")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", default="", help="default: overwrite --bank")
    args = ap.parse_args()
    out = args.out or args.bank

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(args.teacher, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    action_bin_start = action_tokenizer.action_token_begin_idx + 1

    teacher = AutoModelForVision2Seq.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    num_patches = teacher.vision_backbone.featurizer.patch_embed.num_patches

    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    print(f"[aug] {len(bank)} batches from {args.bank}", flush=True)

    n_pos = None
    with torch.no_grad():
        for i, b in enumerate(bank):
            ids = b["input_ids"].to(device)
            attn = b["attention_mask"].to(device)
            pix = b["pixel_values"].to(torch.bfloat16).to(device)
            lbl = b["labels"].to(device)
            out_t = teacher(input_ids=ids, attention_mask=attn, pixel_values=pix, labels=lbl)
            logits = out_t.logits[:, num_patches:-1, action_bin_start:]
            mask = lbl[:, 1:] > action_tokenizer.action_token_begin_idx
            per = int(mask[0].sum())
            if n_pos is None:
                n_pos = per
                print(f"[aug] {per} action positions per sample, {logits.shape[-1]} bins", flush=True)
            if int(mask.sum()) != per * mask.shape[0]:
                raise SystemExit("[aug] samples disagree on action-token count -- cannot pack")
            lp = F.log_softmax(logits.float(), dim=-1)[mask].view(mask.shape[0], per, -1)
            b["teacher_logp"] = lp.to(torch.float16).cpu()
            if (i + 1) % 100 == 0:
                print(f"[aug] {i+1}/{len(bank)}", flush=True)

    torch.save(bank, out)
    mb = sum(b["teacher_logp"].numel() * 2 for b in bank) / 1e6
    print(f"[aug] wrote {out}  (+{mb:.0f} MB of stored teacher distributions)", flush=True)
    print("AUGMENT_DONE", flush=True)


if __name__ == "__main__":
    main()
