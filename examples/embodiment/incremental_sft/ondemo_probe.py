"""ondemo_probe.py -- is forgetting ON or OFF the demo manifold?  (multi-model, paired design)

WHY THIS EXISTS
---------------
Our "the anchor is in the wrong place" story rests on one assumption: when a VLA loses an old
suite, it does NOT start mispredicting the expert action on that suite's own demo states -- it
still matches the demos, and loses the task somewhere the demos never go (the classic BC
covariate-shift failure). If that is false -- if demo-state accuracy falls together with the
success rate -- then plain replay of the old data is sufficient and the story is dead.

A single before/after pair cannot settle this: one pair is one observation and both models differ
in many ways. So this runs a LIST of models whose rollout success rate on the suite is ALREADY
measured, over ONE FIXED set of demo batches (paired -- every model sees byte-identical inputs, so
differences cannot come from data sampling), and reports on-demo accuracy for each. With 7 models
spanning SR 0.64 -> 0.90 the question becomes a curve and a correlation, not an anecdote:

    accuracy FLAT while SR spans 26 points -> forgetting is OFF-manifold. Anchoring on demo states
        is redundant with the demo data itself, and belongs on off-demo states instead.
    accuracy TRACKS SR                     -> forgetting is ON-manifold. Replay suffices. Story dead.

Forward passes only. No rollouts. No training. No gradients.

Against the first model in the list (the reference) it also reports the two numbers the
"critical states" idea needs:
    (a) CONCENTRATION -- share of total model-vs-reference disagreement carried by the top 5% of
        action tokens. ~5% means disagreement is uniform and per-state weighting buys nothing.
    (b) WHICH ACTION DIMENSION those top tokens sit on (dim 6 = gripper), i.e. whether the
        disagreement lands on physically meaningful moments.

Usage: see ondemo_probe.sh (must run from /tmp, rlinf-openvlaoft venv, SFT_NORM_OVERRIDE set).
    --models  "tag=/path/to/merged:SR  tag2=/path2:SR ..."   (SR optional, "?" if unmeasured)
"""

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- identical guard + action-norm monkeypatch to finetune_lwf.py -----------------------------
# Every model here was trained under the libero_130 action normalisation; reading the demos under
# the per-suite norm would shift the target action bins and make all accuracies meaningless.
import prismatic  # noqa: E402

_pf = getattr(prismatic, "__file__", "") or ""
if "site-packages" not in _pf:
    raise SystemExit(f"[probe] REFUSING: prismatic resolved to {_pf!r} (not the venv OFT prismatic)")

_OVERRIDE = os.environ.get("SFT_NORM_OVERRIDE", "").strip()
if not _OVERRIDE:
    raise SystemExit("[probe] REFUSING: SFT_NORM_OVERRIDE unset -> norm would not match training")
import prismatic.vla.datasets.rlds.dataset as _rlds  # noqa: E402

with open(_OVERRIDE) as _f:
    _OVERRIDE_STATS = json.load(_f)
_orig_make = _rlds.make_dataset_from_rlds


def _make_aligned(*args, dataset_statistics=None, **kwargs):  # noqa: ANN001
    return _orig_make(*args, dataset_statistics=_OVERRIDE_STATS, **kwargs)


_rlds.make_dataset_from_rlds = _make_aligned
print(f"[probe] action-norm forced to libero_130 from {_OVERRIDE}", flush=True)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor  # noqa: E402
from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # noqa: E402
from prismatic.util.data_utils import PaddedCollatorForActionPrediction  # noqa: E402
from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: E402
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset  # noqa: E402

ACTION_DIM = 7  # (dx, dy, dz, droll, dpitch, dyaw, gripper) -- dim 6 is the gripper


def parse_models(spec):
    out = []
    for item in spec.split():
        tag, rest = item.split("=", 1)
        if ":" in rest:
            path, sr = rest.rsplit(":", 1)
        else:
            path, sr = rest, "?"
        out.append((tag, path, None if sr == "?" else float(sr)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help='"tag=/path:SR tag2=/path2:SR ..." (first = reference)')
    ap.add_argument("--dataset", default="libero_spatial_no_noops")
    ap.add_argument("--data_root", default="/share/fanruochen-local/datasets/rlds")
    ap.add_argument("--batches", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default="/share/fanruochen-local/outputs/ondemo_probe.json")
    ap.add_argument("--seed", type=int, default=0, help="fixes WHICH demo samples are drawn")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples for the paired CIs")
    args = ap.parse_args()

    # The RLDS pipeline shuffles; without a fixed seed each run draws a DIFFERENT subset and the
    # absolute accuracy moves by ~0.06 between runs -- larger than the spread across all our models.
    # Seeding makes runs comparable; the bootstrap below still quantifies the sampling error that
    # remains WITHIN a run (all models see the same samples, so the paired difference is what we CI).
    import tensorflow as tf  # noqa: PLC0415  (pulled in by the RLDS pipeline anyway)

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models = parse_models(args.models)
    print(f"[probe] {len(models)} models on {args.dataset}; reference = {models[0][0]}", flush=True)

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(models[0][1], trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    action_bin_start = action_tokenizer.action_token_begin_idx + 1

    # ---- materialise ONE fixed set of batches; every model sees exactly these ----
    ref_cfg = AutoConfig.from_pretrained(models[0][1], trust_remote_code=True)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    ds = RLDSDataset(
        Path(args.data_root),
        args.dataset,
        batch_transform,
        resize_resolution=tuple(ref_cfg.image_sizes),
        shuffle_buffer_size=1000,
        image_aug=False,  # evaluation: never augment
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=None, collate_fn=collator, num_workers=0)
    fixed = []
    for i, b in enumerate(dl):
        if i >= args.batches:
            break
        fixed.append({k: v for k, v in b.items() if k in ("input_ids", "attention_mask", "pixel_values", "labels")})
    del dl, ds
    n_samples = sum(b["input_ids"].shape[0] for b in fixed)
    print(f"[probe] fixed evaluation set: {len(fixed)} batches / {n_samples} samples", flush=True)

    results = {}
    ref_logp = None  # reference model's per-token log-probs, kept on CPU for the KL

    for tag, path, sr in models:
        print(f"[probe] === {tag} ({path}) ===", flush=True)
        model = AutoModelForVision2Seq.from_pretrained(
            path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        ).to(device)
        model.eval()
        model.requires_grad_(False)
        num_patches = model.vision_backbone.featurizer.patch_embed.num_patches

        # PER-SAMPLE bookkeeping (not just totals): the bootstrap resamples whole demo samples, so
        # the CI reflects "would another draw of demos have flipped this comparison?"
        s_corr, s_tok, s_l1 = [], [], []
        logps, kls, dims = [], [], []
        with torch.no_grad():
            for bi, b in enumerate(fixed):
                ids = b["input_ids"].to(device)
                attn = b["attention_mask"].to(device)
                pix = b["pixel_values"].to(torch.bfloat16).to(device)
                lbl = b["labels"].to(device)
                out = model(input_ids=ids, attention_mask=attn, pixel_values=pix, labels=lbl)

                gt = lbl[:, 1:]
                mask = gt > action_tokenizer.action_token_begin_idx      # [B, T] action positions
                preds = out.logits[:, num_patches:-1].argmax(dim=2)
                ok = (preds == gt) & mask
                for si in range(mask.shape[0]):
                    m = mask[si]
                    ns = int(m.sum().item())
                    if ns == 0:
                        continue
                    s_corr.append(int(ok[si].sum().item()))
                    s_tok.append(ns)
                    p1 = action_tokenizer.decode_token_ids_to_actions(preds[si][m].cpu().numpy())
                    g1 = action_tokenizer.decode_token_ids_to_actions(gt[si][m].cpu().numpy())
                    s_l1.append(float(np.abs(p1 - g1).mean()))

                lp = F.log_softmax(out.logits[:, num_patches:-1, action_bin_start:].float(), dim=-1)
                if ref_logp is None:
                    logps.append(lp.cpu())                                # reference: keep for later
                else:
                    la = ref_logp[bi].to(device)
                    kl = (la.exp() * (la - lp)).sum(dim=-1)               # [B, T] forward-KL(ref||this)
                    for si in range(mask.shape[0]):
                        pos = mask[si].nonzero(as_tuple=True)[0]
                        if pos.numel():
                            kls.append(kl[si, pos].cpu().numpy())
                            dims.append(np.arange(pos.numel()) % ACTION_DIM)
                    del la
                del out
        s_corr = np.array(s_corr, float)
        s_tok = np.array(s_tok, float)
        s_l1 = np.array(s_l1, float)
        acc = float(s_corr.sum() / max(s_tok.sum(), 1))
        rec = {
            "path": path,
            "sr": sr,
            "acc": acc,
            "l1": float(s_l1.mean()),
            "n_action_tokens": int(s_tok.sum()),
            "_s_corr": s_corr,
            "_s_tok": s_tok,
            "_s_l1": s_l1,
        }

        if ref_logp is None:
            ref_logp = logps                                              # this was the reference model
        elif kls:
            k = np.concatenate(kls)
            d = np.concatenate(dims)
            order = np.argsort(-k)
            k5 = max(1, int(0.05 * k.size))
            rec["kl_mean"] = float(k.mean())
            rec["top5pct_mass"] = float(k[order[:k5]].sum() / max(k.sum(), 1e-9))
            rec["top1pct_mass"] = float(k[order[: max(1, int(0.01 * k.size))]].sum() / max(k.sum(), 1e-9))
            rec["top5_dim_share"] = {int(x): float((d[order[:k5]] == x).mean()) for x in range(ACTION_DIM)}
        results[tag] = rec
        print(f"[probe] {tag}: acc={acc:.4f}  l1={rec['l1']:.4f}  SR={sr}", flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ---- verdict ----
    pts = [(r["sr"], r["acc"]) for r in results.values() if r["sr"] is not None]
    corr_txt = "n/a (need >=3 models with measured SR)"
    if len(pts) >= 3:
        srs = np.array([p[0] for p in pts])
        accs = np.array([p[1] for p in pts])
        pear = float(np.corrcoef(srs, accs)[0, 1])
        rs, ra = np.argsort(np.argsort(srs)), np.argsort(np.argsort(accs))
        spear = float(np.corrcoef(rs, ra)[0, 1])
        corr_txt = (
            f"pearson={pear:+.3f}  spearman={spear:+.3f}  over {len(pts)} models; "
            f"SR span {srs.min():.2f}-{srs.max():.2f} ({srs.max()-srs.min():.2f}), "
            f"acc span {accs.min():.4f}-{accs.max():.4f} ({accs.max()-accs.min():.4f})"
        )
    # ---- paired bootstrap: is "model X beats the reference on demo states" real, or a lucky draw? ----
    # Resample whole demo SAMPLES (not tokens) with replacement; every model saw the same samples, so
    # the same resampled index set is applied to both -> this is a paired CI on the DIFFERENCE.
    ref_tag = models[0][0]
    ref = results[ref_tag]
    rng = np.random.default_rng(args.seed)
    n_s = len(ref["_s_tok"])
    idx = rng.integers(0, n_s, size=(args.boot, n_s))
    for tag, r in results.items():
        if tag == ref_tag:
            continue
        d_acc = (r["_s_corr"][idx].sum(1) / r["_s_tok"][idx].sum(1)) - (
            ref["_s_corr"][idx].sum(1) / ref["_s_tok"][idx].sum(1)
        )
        d_l1 = r["_s_l1"][idx].mean(1) - ref["_s_l1"][idx].mean(1)
        r["acc_diff_vs_ref"] = float(d_acc.mean())
        r["acc_diff_ci"] = [float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))]
        r["l1_diff_vs_ref"] = float(d_l1.mean())
        r["l1_diff_ci"] = [float(np.percentile(d_l1, 2.5)), float(np.percentile(d_l1, 97.5))]

    clean = {t: {k: v for k, v in r.items() if not k.startswith("_")} for t, r in results.items()}
    summary = {
        "dataset": args.dataset,
        "n_samples": n_samples,
        "seed": args.seed,
        "reference": ref_tag,
        "correlation": corr_txt,
        "models": clean,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 104)
    print(f"{'model':<16}{'SR':>7}{'acc':>9}{'L1':>9}{'KLvsref':>10}{'Δacc vs ref [95% CI]':>30}{'sig?':>7}")
    for tag, r in results.items():
        sr = f"{r['sr']:.2f}" if r["sr"] is not None else "  ?  "
        kl = f"{r['kl_mean']:.4f}" if "kl_mean" in r else " (ref)"
        if "acc_diff_ci" in r:
            lo, hi = r["acc_diff_ci"]
            dtxt = f"{r['acc_diff_vs_ref']:+.4f} [{lo:+.4f},{hi:+.4f}]"
            sig = "yes" if (lo > 0 or hi < 0) else "no"
        else:
            dtxt, sig = "-", "-"
        print(f"{tag:<16}{sr:>7}{r['acc']:>9.4f}{r['l1']:>9.4f}{kl:>10}{dtxt:>30}{sig:>7}")
    print("-" * 104)
    print(f"  ('sig? = yes' means the 95% CI of the paired difference excludes 0, n={n_s} demo samples)")
    print(f"CORRELATION  {corr_txt}")
    print("  flat acc across a wide SR span -> forgetting is OFF the demo manifold")
    print("  acc tracking SR                -> forgetting is ON the demo manifold (replay suffices)")
    print("=" * 84)
    print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
