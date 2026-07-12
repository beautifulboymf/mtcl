# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional

from omegaconf import DictConfig

from rlinf.config import EMBODIED_MODEL, SupportedModel, torch_dtype_from_precision
from rlinf.scheduler import Worker

ModelBuilder = Callable[[DictConfig, Optional[object]], object]
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(
    model_type: str,
    model_builder: ModelBuilder,
    category: str = "embodied",
    force: bool = False,
):
    """Register a model builder for cfg.model_type."""
    if not model_type:
        raise ValueError("model_type must be a non-empty string.")
    if not callable(model_builder):
        raise TypeError("model_builder must be callable.")
    if not force and model_type in _MODEL_REGISTRY:
        raise ValueError(
            f"Model type `{model_type}` is already registered. "
            "Set force=True to override it."
        )
    _MODEL_REGISTRY[model_type] = model_builder
    SupportedModel.register(model_type, force=force)
    if category == "embodied":
        EMBODIED_MODEL.add(SupportedModel(model_type))


def _register_builtin_models():
    def _build_openvla(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openvla import get_model

        return get_model(cfg, torch_dtype)

    def _build_openvla_oft(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openvla_oft import get_model

        return get_model(cfg, torch_dtype)

    def _build_openpi(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openpi import get_model

        return get_model(cfg, torch_dtype)

    def _build_dexbotic_pi(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.dexbotic_pi import get_model

        return get_model(cfg, torch_dtype)

    def _build_dexbotic_dm0(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.dexbotic_dm0 import get_model

        return get_model(cfg, torch_dtype)

    def _build_mlp_policy(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.mlp_policy import get_model

        return get_model(cfg, torch_dtype)

    def _build_gr00t(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.gr00t import get_model

        return get_model(cfg, torch_dtype)

    def _build_cnn_policy(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.cnn_policy import get_model

        return get_model(cfg, torch_dtype)

    def _build_flow_policy(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.flow_policy import get_model

        return get_model(cfg, torch_dtype)

    def _build_lingbotvla(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.lingbotvla import get_model

        return get_model(cfg, torch_dtype)

    def _build_starvla(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.starvla import get_model

        return get_model(cfg, torch_dtype)

    def _build_dreamzero(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.dreamzero import get_model

        return get_model(cfg, torch_dtype)

    def _build_openpi_cfg(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openpi_cfg import get_model

        return get_model(cfg, torch_dtype)

    def _build_value_model(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.value_model import get_model

        return get_model(cfg, torch_dtype)

    register_model(
        SupportedModel.OPENVLA.value,
        _build_openvla,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.OPENVLA_OFT.value,
        _build_openvla_oft,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.OPENPI.value,
        _build_openpi,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.DEXBOTIC_PI.value,
        _build_dexbotic_pi,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.DEXBOTIC_DM0.value,
        _build_dexbotic_dm0,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.MLP_POLICY.value,
        _build_mlp_policy,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.GR00T.value,
        _build_gr00t,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.CNN_POLICY.value,
        _build_cnn_policy,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.FLOW_POLICY.value,
        _build_flow_policy,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.LINGBOTVLA.value,
        _build_lingbotvla,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.STARVLA.value,
        _build_starvla,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.DREAMZERO.value,
        _build_dreamzero,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.CFG_MODEL.value,
        _build_openpi_cfg,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.VALUE_MODEL.value,
        _build_value_model,
        category="embodied",
        force=True,
    )


_register_builtin_models()


def get_model(cfg: DictConfig):
    model_type = str(cfg.model_type)
    model_builder = _MODEL_REGISTRY.get(model_type)
    if model_builder is None:
        return None

    torch_dtype = torch_dtype_from_precision(cfg.precision)
    model = model_builder(cfg, torch_dtype)

    if (
        Worker.torch_platform is not None
        and Worker.torch_platform.is_available()
        and cfg.get("load_to_device", True)
    ):
        model = model.to(Worker.torch_device_type)

    if cfg.is_lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        if not hasattr(cfg, "lora_path") or cfg.lora_path is None:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_rank,
                lora_dropout=0.0,
                target_modules=[
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
                ],
                init_lora_weights="gaussian",
            )
            if SupportedModel(model_type) in (
                SupportedModel.OPENPI,
                SupportedModel.CFG_MODEL,
            ):
                # Flexible LoRA targeting via RLINF_LORA_TARGET (default 'vlm' = ORIGINAL behavior,
                # byte-identical: get_peft_model on the VLM submodule, expert full-FT).
                #   'vlm'    -> (unchanged) LoRA on paligemma submodule.
                #   'expert' -> LoRA on the ACTION EXPERT (gemma_expert) + VLM frozen.
                #   'both'   -> LoRA on BOTH paligemma and gemma_expert.
                # For expert/both we use peft inject_adapter_in_model (IN-PLACE injection) instead of
                # get_peft_model, so the top model stays a pi0 (GAVA shim + pi0's manual layer access
                # `gemma_expert.model.layers` keep working). Any module NOT getting LoRA is FROZEN.
                import os as _os, sys as _sys
                _lt = _os.environ.get("RLINF_LORA_TARGET", "vlm").lower()
                if _lt == "vlm":
                    module_to_lora = model.paligemma_with_expert.paligemma
                    module_to_lora = get_peft_model(module_to_lora, lora_config)
                    tag_vlm_subtree(model, False)
                    tag_vlm_subtree(module_to_lora, True)
                    model.paligemma_with_expert.paligemma = module_to_lora
                else:
                    from peft import inject_adapter_in_model

                    _do_vlm = _lt in ("both",)
                    _do_exp = _lt in ("expert", "both")
                    if _do_vlm:
                        inject_adapter_in_model(
                            lora_config, model.paligemma_with_expert.paligemma
                        )
                    if _do_exp:
                        inject_adapter_in_model(
                            lora_config, model.paligemma_with_expert.gemma_expert
                        )
                    # freeze EVERYTHING, then re-enable ONLY the injected LoRA params -> non-LoRA
                    # modules (e.g. VLM when target=expert) end up frozen, LoRA params trainable.
                    for _p in model.parameters():
                        _p.requires_grad_(False)
                    _ntr = 0
                    for _n, _p in model.named_parameters():
                        if "lora_" in _n:
                            _p.requires_grad_(True)
                            _ntr += _p.numel()
                    # Do NOT separately per-leaf-wrap the LoRA leaves (set _to_lora=False
                    # everywhere): with use_orig_params=True the transformer-layer wrap policy
                    # wraps each decoder layer as ONE flat-param holding both the frozen base and
                    # the trainable LoRA params, and use_orig_params handles the mixed requires_grad.
                    # Separately wrapping the LoRA leaves (as the VLM path does) led to an FSDP
                    # `_writeback_orig_params` shape error on the frozen base of the LoRA'd expert.
                    tag_vlm_subtree(model, False)
                    # BUGFIX (bf16 LoRA rounding): the LoRA'd module was cast to bf16 before
                    # injection, so lora_A/B are bf16 and AdamW updates < bf16 ULP (~6e-5 near 0.01)
                    # get rounded away -> LoRA stalls at ~0.01, model ~= base. Cast the WHOLE LoRA'd
                    # module (frozen base + trainable LoRA) to fp32 so the FSDP flat param is uniform
                    # fp32 and the optimizer steps at full precision (fp32 master). Gated by env
                    # (default off -> byte-identical to prior behavior for other runs).
                    if _os.environ.get("RLINF_EXPERT_FP32", "0") == "1":
                        import torch as _torch
                        # Cast ONLY the action-expert's transformer DECODER LAYERS to fp32 (that is
                        # where all the injected LoRA lives: q/k/v/o/gate/up/down_proj). Each
                        # GemmaDecoderLayer is its own FSDP unit -> becomes a uniform-fp32 flat param;
                        # the VLM layers + all ROOT-level params (embeds, norms, action heads) stay
                        # bf16, so the ROOT flat param stays uniform bf16 (no flatten ValueError) and
                        # we avoid doubling the whole model's memory (only ~300M expert params -> fp32).
                        # Pair with actor.model.precision=bf16 (bf16 compute, fp32 stored = optimizer
                        # master) so tiny updates accumulate at full precision (fixes bf16 rounding).
                        _n_fp32 = 0
                        if _do_exp:
                            for _lyr in model.paligemma_with_expert.gemma_expert.model.layers:
                                _lyr.to(_torch.float32)
                                _n_fp32 += 1
                        if _do_vlm:
                            for _lyr in model.paligemma_with_expert.paligemma.model.language_model.layers:
                                _lyr.to(_torch.float32)
                                _n_fp32 += 1
                        _sys.stderr.write(
                            f"[lora] RLINF_EXPERT_FP32=1 -> cast {_n_fp32} expert/vlm decoder layers to fp32 "
                            "(fp32 master weights on LoRA'd layers; root stays bf16; fixes bf16 update-rounding)\n"
                        )
                    _sys.stderr.write(
                        f"[lora] RLINF_LORA_TARGET={_lt} -> inject LoRA "
                        f"(vlm={_do_vlm} expert={_do_exp}); non-LoRA modules FROZEN; "
                        f"trainable LoRA params={_ntr}; per-leaf LoRA wrap DISABLED (transformer-wrap + use_orig_params)\n"
                    )
            else:
                model = get_peft_model(model, lora_config)
        else:
            model = PeftModel.from_pretrained(model, cfg.lora_path, is_trainable=True)

        if hasattr(model, "value_head"):
            for param in model.value_head.parameters():
                param.requires_grad = True

    import os as _osfp

    if _osfp.environ.get("RLINF_MODEL_FP32", "0") == "1":
        # Cast the ENTIRE pi0 model to fp32 (uniform dtype). to_bfloat16(..."float32") only touched
        # paligemma_with_expert; the action heads (state_proj / action_in_proj / action_out_proj /
        # action_time_mlp) live on the top pi0 model and stayed bf16 -> fp32 activation × bf16 weight
        # mismatch in state_proj_func. model.float() makes every param uniform fp32 (fp32 master;
        # pair with actor.model.precision=fp32 for uniform fp32 compute -> no dtype boundary anywhere).
        model.float()
        import sys as _sysfp

        _sysfp.stderr.write("[lora] RLINF_MODEL_FP32=1 -> whole pi0 model cast to fp32 (uniform)\n")

    return model


def tag_vlm_subtree(model, is_vlm: bool):
    for n, m in model.named_modules():
        setattr(m, "_to_lora", is_vlm)
