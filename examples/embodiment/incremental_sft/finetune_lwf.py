"""
finetune_lwf.py  —  object-SFT + OFFLINE spatial-distillation (LwF) in ONE training loop.

Modified copy of OpenVLA-OFT vla-scripts/finetune.py. The stock loop does object SFT (CE on
action tokens). We ADD a Learning-without-Forgetting regularizer that, every step, also forwards a
SPATIAL demo batch through the (trainable) student AND a FROZEN spatial teacher, and pulls the
student toward the teacher's action-token distribution via forward-KL:

    L = L_SFT(object, ground-truth)  +  lambda * KL( teacher_spatial || student )  on spatial demo states

Rationale (validated empirically, see project_incremental_sft_opd_weak):
  - object is near-zero -> must be SFT-bootstrapped off zero (OPD/distill on student rollouts can't
    CREATE an absent skill). SFT uses expert demo states -> learns object.
  - spatial is high in the init -> plain object-SFT catastrophically forgets it (0.98 -> 0.00). The
    KL-to-frozen-spatial-teacher on spatial demo states ANCHORS the spatial behavior (LwF). We use a
    spatial-SPECIALIST teacher (our own spatial_opd_student_098, greedy 0.98), NOT the 130 generalist.

Everything is OFFLINE (no env rollout) -> runs at SFT cadence (~1500 steps), unlike joint on-policy
OPD which is rollout-bound to ~15 steps and would starve the SFT. Both the object SFT data and the
spatial distill data are FORCED to the libero_130 action-norm (SFT_NORM_OVERRIDE monkeypatch below)
so the student, the spatial teacher, and eval all share ONE 256-bin action tokenization.

Extra config (draccus flags, in addition to stock finetune.py):
    --spatial_teacher_path   HF dir of the frozen spatial teacher (spatial_opd_student_098)
    --spatial_dataset_name   RLDS name for the spatial distill data (e.g. libero_spatial_no_noops)
    --distill_lambda         weight of the KL anchor (default 1.0)

Run through torchrun from a NEUTRAL cwd (e.g. /tmp) in the rlinf-openvlaoft venv, exactly like
sft_aligned.py, with SFT_NORM_OVERRIDE pointing at the libero_130 stats json.
"""

import itertools
import math
import random
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===================== LwF launcher safety + norm-align monkeypatch =====================
# (folded in from sft_aligned.py so this file is self-contained)
import prismatic  # noqa: E402

_pf = getattr(prismatic, "__file__", "") or ""
if "site-packages" not in _pf:
    raise SystemExit(
        f"[finetune_lwf] REFUSING: `prismatic` resolved to {_pf!r} (not the venv OFT prismatic). "
        f"Run from /tmp with the rlinf-openvlaoft venv; do NOT put the mainline openvla repo on "
        f"PYTHONPATH -- else SFT trains a 1-chunk model that mismatches the 8-chunk teacher."
    )
try:
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK as _NCHUNK  # noqa: E402

    print(f"[finetune_lwf] prismatic={_pf}  NUM_ACTIONS_CHUNK={_NCHUNK}", flush=True)
except Exception as _e:  # pragma: no cover
    print(f"[finetune_lwf] prismatic={_pf}  (NUM_ACTIONS_CHUNK unknown: {_e})", flush=True)

_OVERRIDE = os.environ.get("SFT_NORM_OVERRIDE", "").strip()
if _OVERRIDE:
    import prismatic.vla.datasets.rlds.dataset as _rlds

    with open(_OVERRIDE) as _f:
        _OVERRIDE_STATS = json.load(_f)
    _orig_make = _rlds.make_dataset_from_rlds

    def _make_aligned(*args, dataset_statistics=None, **kwargs):  # noqa: ANN001
        return _orig_make(*args, dataset_statistics=_OVERRIDE_STATS, **kwargs)

    _rlds.make_dataset_from_rlds = _make_aligned
    print(
        f"[finetune_lwf] FORCING action-norm to libero_130 from {_OVERRIDE} "
        f"(action q01[:3]={_OVERRIDE_STATS['action']['q01'][:3]})",
        flush=True,
    )
else:
    print("[finetune_lwf] WARNING: SFT_NORM_OVERRIDE unset -> per-suite norm (NOT aligned to teacher)", flush=True)
# ========================================================================================


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"

    data_root_dir: Path = Path("datasets/open-x-embodiment")
    dataset_name: str = "droid_wipe"                                # OBJECT (new task) RLDS name
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")

    # ---- LwF spatial-preservation (NEW) ----
    spatial_teacher_path: str = ""                                  # frozen spatial specialist (spatial_opd_student_098)
    spatial_dataset_name: str = "libero_spatial_no_noops"           # RLDS name for the distill/anchor data
    distill_lambda: float = 1.0                                     # weight of KL(teacher_spatial || student)

    batch_size: int = 16
    max_steps: int = 200_000
    save_steps: int = 5000
    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 1
    image_aug: bool = True
    shuffle_buffer_size: int = 100_000
    save_latest_checkpoint_only: bool = True

    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False

    wandb_project: str = "openvla"
    wandb_entity: str = "stanford-voltron"
    run_id_note: Optional[str] = None
    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning (LwF) `{cfg.vla_path}` on object=`{cfg.dataset_name}` + spatial-anchor=`{cfg.spatial_dataset_name}` (lambda={cfg.distill_lambda})")
    assert cfg.spatial_teacher_path, "finetune_lwf requires --spatial_teacher_path (the frozen spatial specialist)"

    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    if cfg.use_lora:
        # RESUME (SFT_INIT_ADAPTER=<dir>): continue training an EXISTING LoRA adapter of the same
        # rank instead of a fresh gaussian init -- same mechanism as vla-scripts/finetune.py:174.
        # This is what makes a long run restartable: it can be stopped, and picked back up from any
        # saved `adapters/step_N` (optionally on MORE gpus, since only the adapter carries state).
        # The optimizer state is NOT restored; with the constant lr it re-warms within a few steps.
        _init_adapter = os.environ.get("SFT_INIT_ADAPTER", "")
        if _init_adapter:
            print(f"[resume] init LoRA from existing adapter: {_init_adapter}", flush=True)
            vla = PeftModel.from_pretrained(vla, _init_adapter, is_trainable=True)
        else:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=min(cfg.lora_rank, 16),
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
        # LoRI (SFT_LORI=1): freeze the DOWN-projection A (random, never trained) and train only
        # the UP-projection B. Two independently-drawn A's are near-orthogonal in high dim, so the
        # per-task updates land in different subspaces -> far less interference when adapters are
        # accumulated/merged than plain LoRA (where A also drifts toward the current task).
        # Applied AFTER both branches so it covers a fresh adapter AND an SFT_INIT_ADAPTER resume
        # (the latter is how one LoRI adapter is iteratively carried across the task sequence).
        if os.environ.get("SFT_LORI", "").lower() in ("1", "true", "yes"):
            _n_frozen = 0
            for _n, _p in vla.named_parameters():
                if "lora_A" in _n:
                    _p.requires_grad = False
                    _n_frozen += 1
            print(f"[LoRI] froze {_n_frozen} lora_A tensors -> training B only", flush=True)
        vla.print_trainable_parameters()

    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # ---- LwF: load FROZEN spatial teacher (no LoRA, no DDP, eval, no grad) ----
    print(f"[finetune_lwf] loading frozen spatial teacher from {cfg.spatial_teacher_path}", flush=True)
    teacher = AutoModelForVision2Seq.from_pretrained(
        cfg.spatial_teacher_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)
    teacher.eval()
    teacher.requires_grad_(False)

    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    # action-bin columns of the vocab = token ids > action_token_begin_idx (the last n_bins tokens)
    action_bin_start = action_tokenizer.action_token_begin_idx + 1
    num_patches = vla.module.vision_backbone.featurizer.patch_embed.num_patches

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    # OBJECT dataset (SFT)
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    # ANCHOR dataset(s) (LwF) — same transform/collator, forced to 130 norm by the monkeypatch.
    # spatial_dataset_name accepts a COMMA-SEPARATED list so several old suites can be anchored at
    # once (e.g. "libero_spatial_no_noops,libero_object_no_noops"); anchoring only one of them is
    # what wiped object (0.90 -> 0.02) when goal was learned on top of a spatial+object student.
    _anchor_names = [n.strip() for n in str(cfg.spatial_dataset_name).split(",") if n.strip()]
    spatial_datasets = [
        RLDSDataset(
            cfg.data_root_dir,
            _n,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=cfg.image_aug,
        )
        for _n in _anchor_names
    ]
    spatial_dataset = spatial_datasets[0]

    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0
    )
    spatial_dataloaders = [
        DataLoader(_ds, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)
        for _ds in spatial_datasets
    ]
    spatial_dataloader = spatial_dataloaders[0]
    # one iterator per anchor suite; each optimizer step anchors on ONE of them, round-robin, so
    # over grad_accumulation_steps every old suite gets rehearsed.
    spatial_iters = [iter(_dl) for _dl in spatial_dataloaders]
    spatial_iter = spatial_iters[0]
    _anchor_rr = [0]  # mutable round-robin cursor
    # weight on the ground-truth CE over the anchor (old-suite) batch = plain REPLAY.
    #   0.0  -> distill-only  (arm A: every run before 2026-09-08)
    #   >0   -> replay + distill (arm B, the strong recipe from the CL literature)
    _ANCHOR_CE_W = float(os.environ.get("LWF_ANCHOR_CE", "0") or 0)
    # 1 = anchor on every micro-step (all runs before 2026-09-09). N = pay the anchor 1/N as often,
    # which is the x-axis of the retention-vs-preservation-budget curve.
    _ANCHOR_EVERY = max(1, int(os.environ.get("LWF_ANCHOR_EVERY", "1") or 1))
    print(
        f"[finetune_lwf] anchor replay-CE weight = {_ANCHOR_CE_W}, anchor every {_ANCHOR_EVERY} micro-step(s)",
        flush=True,
    )

    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"lwf+{exp_id}")

    # OFF-DEMO anchor batches (arm C). A .pt written by build_offdemo_anchor.py: states the teacher
    # actually reached in the simulator under injected action noise, kept only from episodes it
    # COMPLETED, with the teacher's own action as the label. Loaded once and cycled -- it is a fixed
    # file, so training stays fully off-policy (no rollout in the loop, unlike OPD).
    # Comma-separated, ONE FILE PER ANCHOR SUITE in the same order as LWF_ANCHOR_DSET: the demo
    # anchor round-robins over the old suites, and the off-demo batch must follow the same cursor,
    # or spatial's teacher distribution would be used to anchor object.
    _offdemo_spec = [p.strip() for p in os.environ.get("LWF_OFFDEMO_FILE", "").split(",") if p.strip()]
    _offdemo = None
    if _offdemo_spec:
        if len(_offdemo_spec) != len(spatial_dataloaders):
            raise SystemExit(
                f"[finetune_lwf] LWF_OFFDEMO_FILE has {len(_offdemo_spec)} files but there are "
                f"{len(spatial_dataloaders)} anchor suites -- they must correspond one-to-one"
            )
        _offdemo = [torch.load(p, map_location="cpu", weights_only=False) for p in _offdemo_spec]
        # The bank was collated at a fixed size by build_offdemo_anchor.py; the run's micro-batch may
        # differ (it is lowered to fit activation memory). Re-chunk so the off-demo batch is the same
        # size as the demo anchor batch -- otherwise the two arms would not see the same number of
        # anchor samples per step, and the off-demo forward could blow the memory budget on its own.
        _keep = ("input_ids", "attention_mask", "pixel_values", "labels", "teacher_logp")
        for _i, _bank in enumerate(_offdemo):
            _bs = _bank[0]["input_ids"].shape[0]
            if _bs == cfg.batch_size:
                continue
            if _bs % cfg.batch_size:
                raise SystemExit(
                    f"[finetune_lwf] off-demo bank batch {_bs} is not a multiple of "
                    f"--batch_size {cfg.batch_size}; rebuild the bank or pick a divisor"
                )
            _split = []
            for _b in _bank:
                for _j in range(0, _bs, cfg.batch_size):
                    # teacher_logp is absent in banks built before augment_bank_logits.py existed
                    _split.append(
                        {k: _b[k][_j : _j + cfg.batch_size] for k in _keep if _b.get(k) is not None}
                    )
            _offdemo[_i] = _split
            print(f"[finetune_lwf] re-chunked off-demo bank {_i}: {_bs} -> {cfg.batch_size} "
                  f"({len(_bank)} -> {len(_split)} batches)", flush=True)
        print(
            "[finetune_lwf] OFF-DEMO anchor: "
            + ", ".join(f"{os.path.basename(p)}={len(b)} batches" for p, b in zip(_offdemo_spec, _offdemo)),
            flush=True,
        )
    _off_cur = [0]

    def _kl_on(batch):
        """forward-KL( frozen teacher || student ) over the action tokens of ONE batch.

        If the batch carries a precomputed "teacher_logp" (augment_bank_logits.py), the teacher
        forward is SKIPPED: on a fixed off-demo state set the teacher's output never changes, so
        re-running a 7B model every step is pure waste. Storing it costs 28 KB/state and also makes
        adding more teachers nearly free (one stored tensor each, not one forward per step).
        """
        ids = batch["input_ids"].to(device_id)
        attn = batch["attention_mask"].to(device_id)
        pix = batch["pixel_values"].to(torch.bfloat16).to(device_id)
        # pass labels so both fwds take the SAME path as the object fwd (the no-labels path computes
        # position_ids via bool attention_mask.cumsum() -> TypeError). Move labels to device: the raw
        # (non-DDP) teacher does NOT auto-move them, so CPU labels vs GPU logits -> device mismatch.
        lbl = batch["labels"].to(device_id)
        s_out = vla(input_ids=ids, attention_mask=attn, pixel_values=pix, labels=lbl)  # student (grad)
        # mirror finetune.py's action slicing, then restrict vocab to the 256 action bins
        s_logits = s_out.logits[:, num_patches:-1, action_bin_start:]
        gt = lbl[:, 1:]
        mask = gt > action_tokenizer.action_token_begin_idx                          # [B, T] action positions

        cached = batch.get("teacher_logp")
        if cached is not None:
            # [B, n_action_pos, bins], aligned with the masked positions in order
            logp_t = cached.to(device_id).float()
            logp_s = F.log_softmax(s_logits[mask].view(logp_t.shape).float(), dim=-1)
            kl_tok = (logp_t.exp() * (logp_t - logp_s)).sum(dim=-1)                    # [B, n_pos]
            return kl_tok.mean(), s_out.loss

        with torch.no_grad():
            t_out = teacher(input_ids=ids, attention_mask=attn, pixel_values=pix, labels=lbl)
        t_logits = t_out.logits[:, num_patches:-1, action_bin_start:]
        logp_t = F.log_softmax(t_logits.float(), dim=-1)
        logp_s = F.log_softmax(s_logits.float(), dim=-1)
        kl_tok = (logp_t.exp() * (logp_t - logp_s)).sum(dim=-1)                        # [B, T] forward-KL
        return (kl_tok * mask).sum() / mask.sum().clamp_min(1), s_out.loss

    # Per-batch priorities for prioritised sampling inside a pool (None = never drawn), a running mean
    # of the scores actually observed, and a lazily-filled cache of each batch's teacher entropy.
    # An undrawn batch is weighted by the OBSERVED MEAN, not by an optimistic constant: a constant
    # (e.g. 1.0) sits ~20x above realistic scores (~0.05), so the sampler would spend the whole run
    # touching each of the 2000 batches once and the drift-based priority would never take effect.
    _bank_prio = [[None] * len(b) for b in _offdemo] if _offdemo is not None else None
    _bank_ent = [[None] * len(b) for b in _offdemo] if _offdemo is not None else None
    _bank_mean = [[0.0, 0] for _ in _offdemo] if _offdemo is not None else None   # [sum, n]

    def _ent_of(j, k):
        """Normalised entropy of the teacher's action distribution on bank j's batch k (cached)."""
        if _bank_ent[j][k] is None:
            lp = _offdemo[j][k].get("teacher_logp")
            if lp is None:
                _bank_ent[j][k] = 0.0
            else:
                lp = lp.float()
                h = -(lp.exp() * lp).sum(-1).mean()
                _bank_ent[j][k] = float(h / math.log(lp.shape[-1]))
        return _bank_ent[j][k]

    def _anchor_one(j):
        """KL anchor for ONE old task, drawing states prioritised by

              score = KL(teacher||student)  x  (1 - normalised teacher entropy)

        Both halves are needed. A confident teacher means the state has one right answer, so a
        student disagreement there really is 'the teacher can, the student no longer can'. Where the
        teacher is diffuse, any action does and the disagreement costs nothing -- spending the
        preservation budget there buys nothing. KL alone would rank those together.
        """
        if _offdemo is None:                       # no pool -> anchor on that suite's demo stream
            try:
                sb = next(spatial_iters[j])
            except StopIteration:
                spatial_iters[j] = iter(spatial_dataloaders[j])
                sb = next(spatial_iters[j])
            return _kl_on(sb)
        bank, prio, mstat = _offdemo[j], _bank_prio[j], _bank_mean[j]
        prior = (mstat[0] / mstat[1]) if mstat[1] else 1.0        # weight for never-drawn batches
        w = [p if p is not None else prior for p in prio]
        r = random.random() * sum(w)
        k = 0
        for k, wk in enumerate(w):
            r -= wk
            if r <= 0:
                break
        kl, ce = _kl_on(bank[k])
        s = max(float(kl.detach()) * (1.0 - _ent_of(j, k)), 1e-6)
        prio[k] = s
        mstat[0] += s
        mstat[1] += 1
        return kl, ce

    def _spatial_distill_loss():
        """Returns (KL anchor, replay CE). WHERE each is computed is the whole experiment:

        arm A  LWF_ANCHOR_CE=0, no off-demo file : KL on demo states           (= LwF)
        arm B  LWF_ANCHOR_CE>0, no off-demo file : KL + ground-truth CE, both on demo states (= DER++)
        arm C  LWF_ANCHOR_CE>0 + off-demo file   : CE on demo states, KL on OFF-demo states

        Arm C's split is the point: on demo states the ground truth is available and strictly better
        than an imperfect old model, so distilling there duplicates the replay signal; off the demo
        manifold the data has no answer at all and only the old checkpoint can supply one.
        """
        nonlocal spatial_iter
        # round-robin over the anchor suites (single suite -> identical to before)
        _k = _anchor_rr[0] % len(spatial_iters)
        _anchor_rr[0] = _k + 1
        try:
            sbatch = next(spatial_iters[_k])
        except StopIteration:
            spatial_iters[_k] = iter(spatial_dataloaders[_k])
            sbatch = next(spatial_iters[_k])
        spatial_iter = spatial_iters[_k]

        if _offdemo is None:
            # one forward serves both terms (arms A and B)
            return _kl_on(sbatch)

        # arm C: KL off the demo manifold, CE on it. The demo forward is skipped entirely when the
        # replay weight is 0, so no compute is spent on a term that cannot affect the loss.
        # _k is the anchor suite chosen above, so the off-demo batch comes from the SAME suite.
        _bank = _offdemo[_k]
        ob = _bank[(_off_cur[0] // len(_offdemo)) % len(_bank)]
        _off_cur[0] += 1
        kl, _ = _kl_on(ob)
        if _ANCHOR_CE_W <= 0:
            return kl, torch.zeros((), device=device_id)
        d_out = vla(
            input_ids=sbatch["input_ids"].to(device_id),
            attention_mask=sbatch["attention_mask"].to(device_id),
            pixel_values=sbatch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=sbatch["labels"].to(device_id),
        )
        return kl, d_out.loss

    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_sft = deque(maxlen=cfg.grad_accumulation_steps)
    recent_distill = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # ---------------- ADAPTIVE mode (LWF_ADAPTIVE=1) ----------------------------------------
    # One rule for everything: at each step train whichever task is currently WORST relative to its
    # own target -- the new task against ground truth (CE), each old task against its own frozen
    # checkpoint (KL). Only ONE task is trained per step, so there is no lambda to trade the losses
    # off against each other (lambda was the least defensible knob in the previous design), and a
    # step costs ONE forward/backward instead of two.
    #
    # Everything that would otherwise be a hand-set constant is removed:
    #   * comparability   -- each task's gap is divided by ITS OWN historical max (self-normalising),
    #                        because CE (floors above 0) and KL (starts exactly at 0) are not
    #                        comparable in raw units
    #   * exploration     -- a UCB bonus sqrt(2 ln t / n_j) makes stale tasks get re-measured on
    #                        their own, instead of a "probe every P steps" constant
    #   * selection       -- sample PROPORTIONAL to the score, not argmax, which would oscillate
    #                        between tasks and then need a smoothing constant to fix
    #   * running stats   -- plain all-history means (weight 1/n), not an EMA with a chosen decay
    _ADAPTIVE = os.environ.get("LWF_ADAPTIVE", "").lower() in ("1", "2", "true", "yes")
    _n_old = len(spatial_dataloaders)
    _gap_sum = [0.0] * (1 + _n_old)     # index 0 = the new task, 1.. = old tasks
    _gap_n = [0] * (1 + _n_old)
    _picks = [0] * (1 + _n_old)
    _last_seen = [0] * _n_old           # micro-step at which each old task was last trained
    _cnt_old = [0] * _n_old             # how many times each old task has been measured (UCB n_j)
    _kl_seen = [0.0, 0]                 # [sum, n] of every old-task KL observed -> self-scaling bonus
    # one preservation slot every _SLOT micro-steps; this is the budget axis we sweep, not a tuned
    # constant (LWF_ANCHOR_EVERY keeps its old meaning in the non-adaptive arms)
    _SLOT = max(2, int(os.environ.get("LWF_SLOT", "4") or 4))
    if _ADAPTIVE:
        print(
            f"[finetune_lwf] ADAPTIVE: 1 new + {_n_old} old task(s), preservation slot every {_SLOT} micro-steps",
            flush=True,
        )

    def _pick_task(t):
        """Which task to train this micro-step.

        The new task is the DEFAULT; old tasks compete only for the preservation slots. Old tasks are
        then chosen by their raw drift KL -- all in the same unit, so the comparison is meaningful and
        fully adaptive.

        Why not one scalar comparison across all three: the new task's gap is a cross-entropy against
        ground truth (floors well above 0) and an old task's gap is a KL against its own checkpoint
        (starts at EXACTLY 0, because the student IS the teacher at init). Raw values would always
        favour the new task; dividing each by its own historical max makes every rising gap read as
        1.0 and always favours the old tasks. Both are wrong. Comparing acquisition against
        preservation needs a conversion into a common outcome unit (predicted success-rate loss),
        which requires a calibration we have not run yet -- so until then the split between the two
        is STRUCTURAL, and it is the preservation-budget axis we sweep and report anyway, not a
        tuned constant.

        At t=0 every old KL is 0, so the first steps necessarily go to the new task.
        """
        if (t % _SLOT) != 0:
            return 0                                    # new task
        best, best_v = 0, -1.0
        n_slots = max(1, t // _SLOT)                        # preservation slots so far
        kl_scale = (_kl_seen[0] / _kl_seen[1]) if _kl_seen[1] else 0.0
        for j in range(_n_old):
            if not _gap_n[1 + j]:                           # never measured -> measure it first
                return 1 + j
            # UCB1 over old tasks: latest drift + exploration bonus. The bonus is scaled by the MEAN
            # of every KL observed so far (self-scaling, no constant), because a task measured once
            # at KL=0 (t~4, student still equal to teacher) would otherwise never be re-measured:
            # a multiplicative form 0 x anything, or a fixed epsilon 4 orders below real KLs, both
            # starved spatial completely in the first real run (picks were 601/1/199).
            score = (_gap_sum[1 + j] / _gap_n[1 + j]) + kl_scale * math.sqrt(
                2.0 * math.log(n_slots + 1) / _cnt_old[j]
            )
            if score > best_v:
                best, best_v = 1 + j, score
        return best

    def _note_gap(i, v, t):
        # keep only the LATEST measurement for old tasks: drift is a moving quantity, and an
        # all-history mean would keep reporting the near-zero KL of the first steps forever.
        if i == 0:
            _gap_sum[i] += v
            _gap_n[i] += 1
        else:
            _gap_sum[i], _gap_n[i] = v, 1
            _last_seen[i - 1] = t
            _cnt_old[i - 1] += 1
            _kl_seen[0] += v
            _kl_seen[1] += 1
            _pending_ref[i - 1] = True         # window mode: next measurement = new post-train baseline
        _picks[i] += 1

    # ---------------- WINDOW mode (LWF_ADAPTIVE=2): adaptive TOTAL preservation budget -------------
    # Once per accumulation window: MEASURE every old task's drift (one no-grad forward each, on a
    # prioritised bank batch), then give training micro-steps to each task that has drifted beyond
    # its own noise level since it was last trained -- more micro-steps for larger drift -- and hand
    # the rest of the window to the new task. Nothing drifting -> the whole window is new-task
    # training and preservation costs zero. No fixed slot fraction, no lambda, no UCB: the budget
    # is an OUTPUT of training. The one convention is "drift = rise above one running-std of that
    # task's own measurements", a statistical criterion rather than a tuned number.
    #
    # Why the previous rule (arm D) over-served the non-drifting task: its UCB bonus equalises
    # measurement COUNTS across arms, which is the wrong objective here -- training an old task
    # lowers its drift, so the freshly-trained task gets fewer picks and the untouched one more,
    # regardless of who is actually drifting (spatial's slot share rose 19% -> 36% over the run).
    # Measuring everyone every window removes the need for any exploration term at all.
    _WINDOW = os.environ.get("LWF_ADAPTIVE", "") == "2"
    _G = cfg.grad_accumulation_steps
    _ref = [0.0] * _n_old              # post-training KL level per old task; drift is measured from here
    _pending_ref = [False] * _n_old    # set when j is trained: its NEXT measurement becomes the new ref
    _wf = [[0, 0.0, 0.0] for _ in range(_n_old)]   # Welford [n, mean, M2] of each task's measurements
    _plan = [0] * _G
    _streak = [0] * _n_old             # consecutive windows in which task j has been drifting
    _win = {"windows": 0, "old_slots": 0, "measure_fwd": 0}

    def _welford_add(j, x):
        n, m, m2 = _wf[j]
        n += 1
        d = x - m
        m += d / n
        m2 += d * (x - m)
        _wf[j] = [n, m, m2]

    def _sigma(j):
        n, _, m2 = _wf[j]
        return math.sqrt(m2 / (n - 1)) if n > 1 else float("inf")

    def _plan_window(t):
        """Measure every old task, decide this window's micro-step plan, broadcast it.

        ALL ranks run the measurement forwards (symmetrically, same order): DDP syncs module buffers
        at the start of every forward even under no_grad, so a rank that skipped the forward would
        deadlock the others. Only rank 0's numbers decide; the plan is broadcast so the graphs match.
        """
        with torch.no_grad():
            kls = [float(_anchor_one(j)[0]) for j in range(_n_old)]
        _win["measure_fwd"] += _n_old
        zs, want = [], []
        for j, k in enumerate(kls):
            if _pending_ref[j]:                     # first reading after training j = its new baseline
                _ref[j] = k
                _pending_ref[j] = False
            _welford_add(j, k)
            s = _sigma(j)
            z = ((k - _ref[j]) / s) if (0.0 < s < float("inf")) else 0.0
            zs.append(z)
            # ESCALATING allocation: 1 micro-step the first window a task drifts, 2 the next if it is
            # still drifting, 3 after that ... and back to 0 once it is not. "Spend more only if the
            # last spend was not enough." No scale constant (floor(z) would hand the whole window to
            # an old task early on, when sigma is still tiny and z is huge), and the new task can
            # never be starved because the total is capped below.
            _streak[j] = (_streak[j] + 1) if z > 1.0 else 0
            want.append(min(_streak[j], _G - 1))
        while sum(want) > _G - 1:                   # the new task keeps >= 1 micro-step per window
            want[max(range(_n_old), key=lambda i: want[i])] -= 1
        plan = []
        for j, w in enumerate(want):
            plan += [1 + j] * w
        plan += [0] * (_G - len(plan))
        if dist.is_initialized() and dist.get_world_size() > 1:
            _pt = torch.tensor(plan, device=device_id, dtype=torch.long)
            dist.broadcast(_pt, src=0)
            plan = [int(x) for x in _pt.tolist()]
        if distributed_state.is_main_process and _win["windows"] % 25 == 0:
            print(
                f"[window {_win['windows']}] micro={t} kl={[round(k, 4) for k in kls]} ref={[round(r, 4) for r in _ref]} "
                f"sigma={[(round(_sigma(j), 4) if _sigma(j) != float('inf') else None) for j in range(_n_old)]} "
                f"z={[round(z, 2) for z in zs]} slots={want} cum_old_slots={_win['old_slots']} "
                f"cum_measure_fwd={_win['measure_fwd']}",
                flush=True,
            )
        _win["windows"] += 1
        _win["old_slots"] += sum(1 for x in plan if x)
        return plan

    if _WINDOW:
        print(f"[finetune_lwf] WINDOW mode: adaptive preservation budget, window={_G} micro-steps", flush=True)

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        # The new-task stream is consumed ONLY on steps that train the new task. Iterating it in the
        # for-loop header would silently discard one batch on every preservation slot, so the
        # adaptive arm would see 1/_SLOT fewer new-task samples than the baselines -- a confound on
        # the new-task result that has nothing to do with the method.
        _sft_it = iter(dataloader)

        def _next_sft():
            nonlocal _sft_it
            try:
                return next(_sft_it)
            except StopIteration:
                _sft_it = iter(dataloader)
                return next(_sft_it)

        for batch_idx in itertools.count():
            batch = None
            if _ADAPTIVE:
                if _WINDOW:
                    if (batch_idx % _G) == 0:
                        _plan = _plan_window(batch_idx + 1)   # already broadcast inside
                    _t = _plan[batch_idx % _G]
                else:
                    _t = _pick_task(batch_idx + 1)
                    # The gap statistics are rank-LOCAL, so two ranks could otherwise pick different
                    # tasks, run different graphs, and desynchronise DDP's gradient reduction (a hang
                    # or silently wrong gradients -- invisible on one GPU, fatal on two). Rank 0 decides.
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        _tt = torch.tensor([_t], device=device_id, dtype=torch.long)
                        dist.broadcast(_tt, src=0)
                        _t = int(_tt.item())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if _t == 0:
                        batch = _next_sft()
                        output = vla(
                            input_ids=batch["input_ids"].to(device_id),
                            attention_mask=batch["attention_mask"].to(device_id),
                            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                            labels=batch["labels"],
                        )
                        loss = output.loss
                        sft_loss, distill_loss = loss, torch.zeros((), device=device_id)
                    else:
                        loss, _ = _anchor_one(_t - 1)
                        output = None                      # no SFT batch this step -> no SFT metrics
                        sft_loss, distill_loss = torch.zeros((), device=device_id), loss
                    _note_gap(_t, float(loss.detach()), batch_idx + 1)
                    anchor_ce = torch.zeros((), device=device_id)
                if distributed_state.is_main_process and (batch_idx % 200) == 0:
                    print(f"[adaptive] micro-step {batch_idx} picks(new,old...)={_picks}", flush=True)
                # fall through to the SHARED tail (backward / optimizer / checkpoint saving) --
                # returning early here would skip adapter saving entirely
            else:
              batch = _next_sft()
              with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                sft_loss = output.loss                                    # object SFT (CE)
                # PRESERVATION BUDGET knob: only pay for the anchor every LWF_ANCHOR_EVERY micro-steps.
                # The anchor (student fwd/bwd on the anchor batch, plus the demo-CE forward in arm C)
                # is 30-40% of a step, so this is the direct lever for the retention-vs-budget curve:
                # every N halves/quarters the preservation cost. Scaled by N so the ACCUMULATED
                # gradient contribution matches an every-step anchor of the same lambda.
                if (batch_idx % _ANCHOR_EVERY) == 0:
                    distill_loss, anchor_ce = _spatial_distill_loss()     # LwF anchor: (forward-KL, replay CE)
                    _w = float(_ANCHOR_EVERY)
                else:
                    distill_loss = anchor_ce = torch.zeros((), device=device_id)
                    _w = 0.0
                loss = sft_loss + _w * (cfg.distill_lambda * distill_loss + _ANCHOR_CE_W * anchor_ce)

            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            # metrics on the OBJECT (SFT) batch, exactly like stock finetune.py.
            # In adaptive mode a step that trained an OLD task has no SFT batch, so carry the last
            # values forward rather than fabricating a number.
            if output is None:
                action_accuracy = recent_action_accuracies[-1] if recent_action_accuracies else 0.0
                action_l1_loss = recent_l1_losses[-1] if recent_l1_losses else 0.0
                action_accuracy = torch.tensor(float(action_accuracy))
                action_l1_loss = torch.tensor(float(action_l1_loss))
            else:
                action_logits = output.logits[:, num_patches:-1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            recent_losses.append(loss.item())
            recent_sft.append(sft_loss.item())
            recent_distill.append(distill_loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_sft = sum(recent_sft) / len(recent_sft)
            smoothened_distill = sum(recent_distill) / len(recent_distill)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "sft_loss": smoothened_sft,
                        "distill_loss": smoothened_distill,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                    },
                    step=gradient_step_idx,
                )
                print(
                    f"[lwf] step {gradient_step_idx}  total={smoothened_loss:.4f}  sft={smoothened_sft:.4f}  "
                    f"distill={smoothened_distill:.4f}  obj_acc={smoothened_action_accuracy:.3f}",
                    flush=True,
                )

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
                    save_dir = adapter_dir if cfg.use_lora else run_dir
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)
                    # ADAPTER-ONLY (SFT_ADAPTER_ONLY=1): snapshot this step's adapter and SKIP the
                    # full-model merge below. The merge reloads a whole 7B on GPU and OOMs whenever a
                    # co-tenant holds most of the card; it also writes 15G per checkpoint.
                    if os.environ.get("SFT_ADAPTER_ONLY", "0") == "1":
                        import shutil as _sh
                        step_adir = run_dir / "adapters" / f"step_{gradient_step_idx}"
                        _sh.rmtree(step_adir, ignore_errors=True)
                        _sh.copytree(save_dir, step_adir)
                        try:
                            save_dataset_statistics(vla_dataset.dataset_statistics, step_adir)
                        except Exception:
                            pass
                        print(f"[adapter-only] per-step adapter -> {step_adir}", flush=True)
                dist.barrier()
                if cfg.use_lora and os.environ.get("SFT_ADAPTER_ONLY", "0") != "1":
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            merged_vla.save_pretrained(run_dir)
                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)
                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")
                dist.barrier()

            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
