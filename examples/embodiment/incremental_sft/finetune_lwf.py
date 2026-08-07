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
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
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
    # SPATIAL dataset (LwF anchor) — same transform/collator, forced to 130 norm by the monkeypatch
    spatial_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.spatial_dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0
    )
    spatial_dataloader = DataLoader(
        spatial_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0
    )
    spatial_iter = iter(spatial_dataloader)

    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"lwf+{exp_id}")

    def _spatial_distill_loss():
        """forward-KL( frozen spatial teacher || student ) on the spatial batch's action tokens."""
        nonlocal spatial_iter
        try:
            sbatch = next(spatial_iter)
        except StopIteration:
            spatial_iter = iter(spatial_dataloader)
            sbatch = next(spatial_iter)
        ids = sbatch["input_ids"].to(device_id)
        attn = sbatch["attention_mask"].to(device_id)
        pix = sbatch["pixel_values"].to(torch.bfloat16).to(device_id)
        s_out = vla(input_ids=ids, attention_mask=attn, pixel_values=pix)            # student (grad)
        with torch.no_grad():
            t_out = teacher(input_ids=ids, attention_mask=attn, pixel_values=pix)     # teacher (frozen)
        # mirror finetune.py's action slicing, then restrict vocab to the 256 action bins
        s_logits = s_out.logits[:, num_patches:-1, action_bin_start:]
        t_logits = t_out.logits[:, num_patches:-1, action_bin_start:]
        gt = sbatch["labels"][:, 1:].to(device_id)
        mask = gt > action_tokenizer.action_token_begin_idx                          # [B, T] action positions
        logp_t = F.log_softmax(t_logits.float(), dim=-1)
        logp_s = F.log_softmax(s_logits.float(), dim=-1)
        kl_tok = (logp_t.exp() * (logp_t - logp_s)).sum(dim=-1)                        # [B, T] forward-KL
        denom = mask.sum().clamp_min(1)
        return (kl_tok * mask).sum() / denom

    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_sft = deque(maxlen=cfg.grad_accumulation_steps)
    recent_distill = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                sft_loss = output.loss                                    # object SFT (CE)
                distill_loss = _spatial_distill_loss()                    # LwF spatial anchor (forward-KL)
                loss = sft_loss + cfg.distill_lambda * distill_loss

            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            # metrics on the OBJECT (SFT) batch, exactly like stock finetune.py
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
                dist.barrier()
                if cfg.use_lora:
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
