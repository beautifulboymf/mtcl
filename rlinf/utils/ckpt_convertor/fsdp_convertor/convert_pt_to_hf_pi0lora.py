# NEW FILE (not modifying upstream convert_pt_to_hf.py).
# pi0-aware LoRA merge: the openpi pi0 model PEFT-wraps ONLY
# model.paligemma_with_expert.paligemma, so the upstream top-level
# model.merge_and_unload() is wrong for pi0. Here we merge that submodule
# specifically, then export a full (non-LoRA) pi0 via sharded safetensors so the
# standard eval pipeline (is_lora=False) can load it.
#
# Usage:
#   python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf_pi0lora \
#     --config-path .../fsdp_convertor/config --config-name fsdp_pi0_convertor \
#     model.model_path=<SFT_release_ckpt> model.is_lora=True model.lora_rank=32 \
#     actor.model.is_lora=True actor.model.lora_rank=32 \
#     convertor.ckpt_path=<full_weights.pt> convertor.save_path=<out_dir>
import os

import hydra
import torch

from rlinf.models import get_model
from rlinf.scheduler.cluster import load_user_extension_module

from .utils import (
    copy_model_config_and_code,
    save_state_dict_sharded_safetensors,
)


@hydra.main(version_base="1.1", config_path="config", config_name="fsdp_pi0_convertor")
def main(cfg) -> None:
    load_user_extension_module()
    assert bool(cfg.model.is_lora), "this converter is only for LoRA pi0 ckpts"

    model = get_model(cfg.model)

    model_dict = torch.load(cfg.convertor.ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(model_dict, strict=False)
    # PEFT-wrapped pi0: expect 0 unexpected; missing should be empty or only
    # non-persistent buffers. Surface anything surprising.
    if unexpected:
        print(f"[pi0lora-merge][WARN] {len(unexpected)} unexpected keys, e.g. {list(unexpected)[:5]}")
    if missing:
        print(f"[pi0lora-merge][WARN] {len(missing)} missing keys, e.g. {list(missing)[:5]}")

    save_path = cfg.convertor.save_path
    copy_model_config_and_code(model_path=cfg.model.model_path, save_path=save_path)

    # Merge depends on HOW get_model wrapped it (must match RLINF_LORA_TARGET used in training):
    #   'vlm'  -> paligemma submodule is a PeftModel  -> merge_and_unload (original path).
    #   'expert'/'both' -> LoRA was INJECTED in-place (inject_adapter_in_model); modules are
    #       peft LoraLayer's, NOT a PeftModel -> merge each LoraLayer then strip the wrapper:
    #       'base_layer.weight' -> 'weight', drop lora_* keys, so a plain pi0 can load it.
    _lora_target = os.environ.get("RLINF_LORA_TARGET", "vlm").lower()
    if _lora_target == "vlm":
        pali = model.paligemma_with_expert.paligemma
        assert hasattr(pali, "merge_and_unload"), (
            "paligemma submodule is not a PeftModel — is_lora/lora_rank mismatch with training?"
        )
        model.paligemma_with_expert.paligemma = pali.merge_and_unload()
        print("[pi0lora-merge] merged LoRA adapters into paligemma submodule")
        model_state_dict = model.state_dict()
    else:
        from peft.tuners.lora import LoraLayer

        n_merged = 0
        for m in model.modules():
            if isinstance(m, LoraLayer):
                m.merge()
                n_merged += 1
        print(f"[pi0lora-merge] target={_lora_target}: merged {n_merged} injected LoRA layers; stripping wrappers")
        raw = model.state_dict()
        model_state_dict = {}
        for k, v in raw.items():
            if "lora_" in k:  # drop lora_A / lora_B / lora_magnitude etc.
                continue
            model_state_dict[k.replace("base_layer.", "")] = v  # unwrap merged base

    # sanity: no PEFT-prefixed keys should remain
    bad = [k for k in model_state_dict if "lora_" in k or "base_model.model" in k or "base_layer" in k]
    if bad:
        print(f"[pi0lora-merge][WARN] {len(bad)} residual PEFT keys after merge, e.g. {bad[:5]}")
    save_state_dict_sharded_safetensors(state_dict=model_state_dict, out_dir=save_path)
    print(f"[pi0lora-merge] DONE -> {save_path}")


if __name__ == "__main__":
    main()
