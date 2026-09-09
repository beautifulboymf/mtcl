"""build_offdemo_anchor.py -- label OFF-DEMO states with an old checkpoint, offline.

WHY
---
Measured on 2026-09-08: retraining on an old suite's demonstrations until the model fits those
demos SIGNIFICANTLY BETTER than the original (paired bootstrap CI excludes 0) leaves the rollout
success rate 22 points down (0.680 +/- 0.040 vs 0.900 +/- 0.031). The capability being lost does
not live on the demo states, so no amount of demo-side supervision -- replay OR distillation on
demo states, which is what LwF / DER++ / our own anchor all do -- can reach it.

What CAN answer off the demo manifold is the old checkpoint itself: it is a function defined
everywhere, it is task-conditioned (unlike BASE), and for anything acquired by RL there is no data
at all. This script turns that into a concrete, fully OFFLINE dataset:

    states visited by a policy in the simulator   (dumped by RLINF_DUMP_OBS_DIR during any rollout)
      -> ask the old checkpoint what to do there  (this script)
      -> a drop-in anchor batch for finetune_lwf  (labels = the teacher's own action)

No on-policy training: the rollout happens ONCE, up front, and the resulting file is a fixed
dataset thereafter.

Self-consistent labelling: the sample format conditions later action tokens on earlier ones, so a
placeholder action would put the teacher in an off-distribution context. We therefore run the
teacher twice -- once from a zero-action placeholder, then again on its own prediction -- and keep
the second pass's action as the label.

Format parity is guaranteed by reusing RLDSBatchTransform itself (the exact object training uses)
rather than re-implementing the prompt template.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import prismatic  # noqa: E402

_pf = getattr(prismatic, "__file__", "") or ""
if "site-packages" not in _pf:
    raise SystemExit(f"[offdemo] REFUSING: prismatic resolved to {_pf!r} (not the venv OFT prismatic)")

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor  # noqa: E402
from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # noqa: E402
from prismatic.util.data_utils import PaddedCollatorForActionPrediction  # noqa: E402
from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: E402
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK  # noqa: E402
from prismatic.vla.datasets import RLDSBatchTransform  # noqa: E402


def load_states(dump_dir, limit, sigmas=None):
    """Read the sharded npz dumps, KEEPING ONLY states from episodes that SUCCEEDED.

    A state is a usable distillation input only if the teacher actually completed the episode from
    there: in a failed episode the teacher was demonstrably wrong, so its action is a wrong target
    and distilling it teaches the student to fail the same way. Filtering rollouts by success before
    distilling is the standard recipe in the robot-distillation literature. The cost is a known
    selection bias -- we keep the deviations the teacher COULD handle, not the ones it could not --
    and that boundary has to be stated, not hidden.

    `sigmas` optionally restricts to particular injected-noise levels, so the sigma sweep that one
    rollout measured can be turned into a dataset without re-collecting.
    """
    # env_idx/ep_id are numbered PER COLLECTING PROCESS, so a multi-pass collection (several rollouts
    # appended into one dir) would alias pass 1's (env 3, ep 0) onto pass 2's. The writer pid is in
    # every filename -- states_<pid>_<shard>.npz / episodes_<pid>.npz -- so it completes the key.
    def _pid_of(path):
        return os.path.basename(path).split("_")[1].split(".")[0]

    ok = set()
    n_ep = n_ok = 0
    for f in sorted(glob.glob(os.path.join(dump_dir, "episodes_*.npz"))):
        pid = _pid_of(f)
        z = np.load(f, allow_pickle=True)
        for e, p, s in zip(z["env_idx"], z["ep_id"], z["success"]):
            n_ep += 1
            if s:
                n_ok += 1
                ok.add((pid, int(e), int(p)))
    if not n_ep:
        raise SystemExit(f"[offdemo] no episodes_*.npz under {dump_dir} (was RLINF_DUMP_OBS_DIR set?)")
    print(f"[offdemo] episodes: {n_ok}/{n_ep} succeeded -> keeping their states only", flush=True)

    imgs, tasks, kept, seen = [], [], 0, 0
    for f in sorted(glob.glob(os.path.join(dump_dir, "states_*.npz"))):
        pid = _pid_of(f)
        z = np.load(f, allow_pickle=True)
        seen += len(z["full_image"])
        m = np.array(
            [(pid, int(e), int(p)) in ok for e, p in zip(z["env_idx"], z["ep_id"])],
            dtype=bool,
        )
        if sigmas is not None:
            m &= np.isin(np.round(z["sigma"], 4), np.round(np.array(sigmas), 4))
        if m.any():
            imgs.append(z["full_image"][m])
            tasks.append(z["task"][m])
            kept += int(m.sum())
        if kept >= limit:
            break
    if not imgs:
        raise SystemExit("[offdemo] every state came from a FAILED episode -- lower the noise sigma")
    print(f"[offdemo] states: kept {kept} of {seen} scanned", flush=True)
    return np.concatenate(imgs)[:limit], np.concatenate(tasks)[:limit]


def make_sample(transform, img, lang, actions):
    """actions: [NUM_ACTIONS_CHUNK, ACTION_DIM] normalised. Uses the SAME transform as training."""
    return transform(
        {
            "dataset_name": "offdemo",
            "action": actions,
            "observation": {"image_primary": img[None]},
            "task": {"language_instruction": lang.encode() if isinstance(lang, str) else lang},
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True, help="dir with states_*.npz from RLINF_DUMP_OBS_DIR")
    ap.add_argument("--teacher", required=True, help="checkpoint that answers off the demo manifold")
    ap.add_argument("--out", required=True, help=".pt file of collated anchor batches")
    ap.add_argument("--limit", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--passes", type=int, default=2, help="self-consistency passes (2 = recommended)")
    ap.add_argument("--sigmas", default="", help="comma list; keep only states collected at these noise levels")
    args = ap.parse_args()
    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()] or None

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(args.teacher, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    action_bin_start = action_tokenizer.action_token_begin_idx + 1
    transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )

    print(f"[offdemo] loading teacher {args.teacher}", flush=True)
    teacher = AutoModelForVision2Seq.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    num_patches = teacher.vision_backbone.featurizer.patch_embed.num_patches

    imgs, tasks = load_states(args.dump_dir, args.limit, sigmas)
    print(f"[offdemo] {len(imgs)} off-demo states from {args.dump_dir}", flush=True)

    out_batches = []
    zero = np.zeros((NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(imgs), args.batch_size):
            chunk_img = imgs[s : s + args.batch_size]
            chunk_lang = tasks[s : s + args.batch_size]
            acts = [zero] * len(chunk_img)
            for _p in range(args.passes):
                samples = [make_sample(transform, i, l, a) for i, l, a in zip(chunk_img, chunk_lang, acts)]
                batch = collator(samples)
                ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                pix = batch["pixel_values"].to(torch.bfloat16).to(device)
                lbl = batch["labels"].to(device)
                out = teacher(input_ids=ids, attention_mask=attn, pixel_values=pix, labels=lbl)
                gt = lbl[:, 1:]
                mask = gt > action_tokenizer.action_token_begin_idx
                # restrict argmax to the action-bin columns so the teacher can only emit an action
                pred_bin = out.logits[:, num_patches:-1, action_bin_start:].argmax(dim=2) + action_bin_start
                acts = []
                for bi in range(mask.shape[0]):
                    tok = pred_bin[bi][mask[bi]].cpu().numpy()
                    a = action_tokenizer.decode_token_ids_to_actions(tok)
                    acts.append(np.asarray(a, dtype=np.float32).reshape(NUM_ACTIONS_CHUNK, ACTION_DIM))
            # keep the batch built from the teacher's own (self-consistent) action.
            # pixel_values goes out as bf16: training casts to bf16 anyway, and float32 would make
            # this file ~10GB, all of which the training process has to hold in RAM.
            samples = [make_sample(transform, i, l, a) for i, l, a in zip(chunk_img, chunk_lang, acts)]
            b = collator(samples)
            out_batches.append(
                {k: (v.to(torch.bfloat16) if k == "pixel_values" else v) for k, v in b.items()}
            )
            if (s // args.batch_size) % 25 == 0:
                print(f"[offdemo] {s+len(chunk_img)}/{len(imgs)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_batches, args.out)
    meta = {
        "n_states": int(len(imgs)),
        "n_batches": len(out_batches),
        "teacher": args.teacher,
        "dump_dir": args.dump_dir,
        "passes": args.passes,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[offdemo] wrote {args.out}  ({len(out_batches)} batches)  meta={meta}", flush=True)
    print("OFFDEMO_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
