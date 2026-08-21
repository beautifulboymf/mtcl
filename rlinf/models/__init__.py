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

# The LoRA target list, hoisted OUT of the `LoraConfig` in `get_model` so the PEFT
# baseline and the slot-LoRI path cannot drift apart. The whole slot-LoRI experiment is
# a comparison against that baseline; if the two arms adapt different modules the
# comparison measures the module list as much as it measures the method, and nothing
# anywhere would say so. One list, two readers -- pass a `list(...)` copy to anything
# that might keep it, never the constant itself.
SLOT_LORA_TARGET_MODULES = [
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
        # `cfg.is_lora` MUST stay True on the slot path, which is why this lives INSIDE
        # this branch instead of replacing it. The per-leaf FSDP wrap policy that gives
        # slot_A/slot_B their own flat params is only registered when is_lora is set
        # (rlinf/hybrid_engines/fsdp/strategy/fsdp.py:163, policy at
        # rlinf/hybrid_engines/fsdp/utils.py:303-311). Without it the trainable slot
        # params land in the enclosing transformer layer's flat param together with the
        # FROZEN base, and FSDP refuses to flatten mixed requires_grad under
        # use_orig_params=False (_flat_param.py:800-808). That failure is loud, but it
        # is avoidable, and it is avoided here.
        _slot_cfg = cfg.get("slot_lora", None)
        if _slot_cfg is not None and _slot_cfg.get("enabled", False):
            # Returns EARLY, replacing the PEFT wrapping below entirely. The only thing
            # skipped past this point is the pi0-only RLINF_MODEL_FP32 tail, which casts
            # "the whole pi0 model" and has no meaning for a slot-LoRI student -- whose
            # dtype policy is the base weights' (the slots inherit it at injection).
            return _apply_slot_lora(model, cfg, _slot_cfg)

        from peft import LoraConfig, PeftModel, get_peft_model

        if not hasattr(cfg, "lora_path") or cfg.lora_path is None:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_rank,
                lora_dropout=0.0,
                # A copy, so nothing downstream can mutate the shared constant.
                target_modules=list(SLOT_LORA_TARGET_MODULES),
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


def _assert_slot_leaves_fsdp_wrappable(model) -> int:
    """Assert nothing stamped ``_to_lora=False`` on a slot leaf; return how many it checked.

    The per-leaf FSDP wrap policy that gives ``slot_A`` / ``slot_B`` their own flat
    params (``rlinf/hybrid_engines/fsdp/utils.py:303-311``) has a fourth condition that
    is easy to miss: ``getattr(module, "_to_lora", True) is True``. And
    :func:`tag_vlm_subtree` ``(model, False)`` stamps ``_to_lora=False`` on EVERY module
    it walks. Only the pi0 branch of :func:`get_model` calls it today, so the slot path
    is clean -- this check passes trivially, which is the point of running it every
    time rather than reasoning about it once.

    If it ever does change, the failure lands a long way from its cause. Under
    ``use_orig_params=False`` (this repo's default) the slot parameters fold into the
    enclosing transformer layer's flat parameter next to the FROZEN base, and FSDP
    refuses to flatten mixed ``requires_grad`` -- loud, but raised from inside FSDP
    initialization with no mention of the tag that caused it. Under
    ``use_orig_params=True`` nothing is raised at all, and what is lost is quieter
    still: the guarantee :class:`SlotProj` is built on (its own flat parameter, hence
    the FULL ``Z`` unsharded whenever the module is touched --
    ``rlinf/models/slot_lora/modules.py:307-314``) degrades to "whatever the enclosing
    unit happens to have gathered", so any read of ``Z`` from outside that unit's own
    forward -- a merge, a diagnostic, a checkpoint conversion -- sees a SHARD and builds
    ``Ā`` from a fraction of its rows, with no exception and no NaN. One assert at
    model-build time names the cause instead.

    Args:
        model: The freshly injected student (not yet FSDP-wrapped).

    Returns:
        How many slot leaves were checked. Zero means the model has no slots.

    Raises:
        AssertionError: if any :class:`SlotProj` or :class:`SlotOut` carries
            ``_to_lora=False``.
    """
    from rlinf.models.slot_lora import SlotOut, SlotProj

    checked = 0
    for name, module in model.named_modules():
        if isinstance(module, (SlotProj, SlotOut)):
            checked += 1
            assert getattr(module, "_to_lora", True) is True, (
                f"{name} ({type(module).__name__}) is tagged _to_lora=False, so the "
                "per-leaf FSDP wrap policy (rlinf/hybrid_engines/fsdp/utils.py:303-311) "
                "skips it and Z/B fold into the enclosing layer's flat parameter: with "
                "use_orig_params=False FSDP then refuses to flatten that unit's mixed "
                "requires_grad, from deep inside its own initialization, and with "
                "use_orig_params=True nothing is raised while Z is no longer guaranteed "
                "unsharded outside that unit's forward, so a merge or a diagnostic "
                "builds Ā from a fraction of its rows. tag_vlm_subtree(model, False) "
                "stamps every module it walks; only the pi0 branch calls it today."
            )
    return checked


def find_slot_gate(model):
    """The model's single :class:`~rlinf.models.slot_lora.modules.SlotGate`, or ``None``.

    THE accessor for the training loop, which installs one routing per MICRO-batch
    through it (``with find_slot_gate(self.model).scoped(ids):``) and runs rollout /
    eval forwards inside ``gate.ungated()``.

    Reads the explicit handle :func:`_apply_slot_lora` stashed on the model, and falls
    back to walking for a :class:`SlotOut` (every layer shares the one instance, so any
    of them answers). The explicit handle is stored rather than only re-derived because
    the walk is O(#modules) -- 200-400 gated linears inside a few thousand modules on a
    7B student -- and this is read once per micro-batch, not once per run. It cannot go
    stale under a ``deepcopy`` of the model either: one ``deepcopy`` call shares one
    memo, so the copied handle and the copied layers' ``gate`` are the same object. The
    walk stays as the fallback for a model that arrived some other way (a checkpoint
    reload that rebuilt modules without re-stashing, a partially copied subtree).

    ``getattr`` resolves through an FSDP root as well: ``FullyShardedDataParallel``
    forwards unknown attributes to ``_fsdp_wrapped_module``, so this works on both the
    raw module and the wrapped one.

    Args:
        model: The student, FSDP-wrapped or not.

    Returns:
        The gate, or ``None`` if this model has no slot-LoRI adapters at all. ``None``
        is not an error here: :func:`get_model` is shared with every non-slot run. A
        caller that REQUIRES a gate must say so itself -- routing through a ``None``
        gate is exactly the ungated forward the strict gate exists to prevent.
    """
    gate = getattr(model, "_slot_gate", None)
    if gate is not None:
        return gate

    from rlinf.models.slot_lora import SlotOut

    for module in model.modules():
        if isinstance(module, SlotOut):
            return module.gate
    return None


def _apply_slot_lora(model, cfg: DictConfig, slot_cfg: DictConfig):
    """Give the student K per-suite LoRA slots on orthogonal input subspaces.

    Replaces PEFT ENTIRELY for this model: K distillation teachers writing into one
    shared LoRA block overwrite each other, and no rank makes that block route. Only the
    STUDENT is built this way -- the teachers and the dual-KL anchor keep using PEFT and
    are untouched, because they are separate models built by separate calls.

    Config, under ``actor.model.slot_lora`` (the rollout worker deep-copies
    ``cfg.actor.model`` and calls this same :func:`get_model`, so putting it there is
    what makes the rollout model structurally identical and keeps weight sync a plain
    state-dict match):

    * ``enabled`` (required to be true to get here) -- the switch.
    * ``slot_ranks`` (REQUIRED) -- ``{suite_name: rank}``.
    * ``slot_order`` (REQUIRED) -- the list that fixes slot INDEX order. The rank list is
      built by indexing ``slot_ranks`` with THIS, never by iterating the mapping: the
      slot index is what routing produces (``match_suite_ids`` returns
      ``suite_order.index(suite)``), so an index order that came from dict iteration
      would silently send suite A's gradient into suite B's slot the moment someone
      reordered the YAML mapping.
    * ``a_scale_mode`` (optional, ``"match_mt4"``) -- see
      :func:`~rlinf.models.slot_lora.inject._module_scale`.
    * ``a_scale_ref_rank`` (optional, ``128``) -- the PEFT baseline rank ``match_mt4``
      matches ``ΔW``'s step-1 magnitude against.
    * ``orth_eps`` (optional, ``1e-6``) -- floor on ``‖Z Zᵀ‖_F``.
    * ``orth_iters`` (optional, ``12``) -- Newton-Schulz iteration count for
      ``Ā = (Z Zᵀ)^(-1/2) Z``. 12 is converged at the production shape. It exists as a
      key because it is the ONE documented remediation for a ``slot/orth_err`` that
      drifts into the 1e-3..5e-2 band -- raise it to 16 and change nothing else
      (measured at cond(Z)=20: 11 -> 6.8e-2, 12 -> 1.07e-3, 13 -> 1.62e-4) -- and a
      remediation with no config path is not a remediation. Bounded above as well: on
      a rank-deficient Z the null-space component grows 1.5x per iteration and is NaN
      by ~50, so 16 is safe and 40+ is not.

    Both required keys fail loudly when absent, and so does every disagreement between
    the two (a name in one and not the other, a repeated name): each of those produces a
    model with the wrong NUMBER of slots or the wrong slot for a suite, and both train
    perfectly happily while learning the wrong thing.

    Args:
        model: The student, already built and moved to its device.
        cfg: The full model config, for the ``lora_path`` conflict check.
        slot_cfg: ``cfg.slot_lora``.

    Returns:
        The same model, edited in place, with the slots injected, everything else
        frozen, and the gate reachable through :func:`find_slot_gate`.

    Raises:
        ValueError: on any config error described above, or if ``lora_path`` is set.
    """
    import sys as _sys

    from rlinf.models.slot_lora import inject_slot_lora
    from rlinf.models.slot_lora.orth import _DEFAULT_NS_ITERS

    if cfg.get("lora_path", None) is not None:
        raise ValueError(
            f"actor.model.slot_lora.enabled=true together with lora_path="
            f"{cfg.lora_path!r}. The slot path replaces PEFT entirely, so that adapter "
            "would never be loaded: the run would silently start from the BASE weights "
            "with fresh zero-initialized slots and look completely normal. Resume a "
            "slot run from a slot checkpoint (runner.resume_dir) instead, or drop one "
            "of the two keys."
        )

    missing = [k for k in ("slot_ranks", "slot_order") if slot_cfg.get(k, None) is None]
    if missing:
        raise ValueError(
            f"actor.model.slot_lora is enabled but required key(s) {missing} are "
            "missing or null. `slot_ranks` is {suite: rank}, `slot_order` is the list "
            "that fixes slot INDEX order; both are required, and neither has a "
            "defensible default -- a guessed order silently routes one suite's "
            "gradient into another suite's slot."
        )

    raw_ranks, raw_order = slot_cfg["slot_ranks"], slot_cfg["slot_order"]
    # Shape guards, so a YAML mistake reads as a YAML mistake. Without the second one a
    # bare string (`slot_order: libero_spatial`, no list) iterates into ONE SLOT PER
    # CHARACTER, and the model builds as far as complaining about a suite named "l".
    if not hasattr(raw_ranks, "keys"):
        raise ValueError(
            f"actor.model.slot_lora.slot_ranks must be a {{suite: rank}} mapping; got "
            f"{type(raw_ranks).__name__}. A bare list of ranks cannot say WHICH suite "
            "each one belongs to, and slot_order is matched against it by name."
        )
    if isinstance(raw_order, str) or hasattr(raw_order, "keys"):
        raise ValueError(
            f"actor.model.slot_lora.slot_order must be a LIST of suite names; got "
            f"{type(raw_order).__name__} ({raw_order!r})."
        )

    ranks_map = {str(k): int(v) for k, v in dict(raw_ranks).items()}
    order = [str(s) for s in raw_order]

    repeated = sorted({s for s in order if order.count(s) > 1})
    if repeated:
        raise ValueError(
            f"actor.model.slot_lora.slot_order repeats {repeated}. Routing resolves a "
            "suite to the FIRST matching index (`suite_order.index`), so every later "
            "copy is a slot no sample can ever reach: it would stay at its zero "
            "initialization for the whole run while the layer still carries its rank."
        )
    unranked = [s for s in order if s not in ranks_map]
    if unranked:
        raise ValueError(
            f"actor.model.slot_lora.slot_order names {unranked}, which slot_ranks "
            f"{sorted(ranks_map)} does not. Every slot needs a rank; there is no "
            "default width to fall back to."
        )
    unordered = sorted(set(ranks_map) - set(order))
    if unordered:
        raise ValueError(
            f"actor.model.slot_lora.slot_ranks gives a rank to {unordered}, which "
            f"slot_order {order} does not list. slot_order alone decides how many slots "
            "the model has, so those suites would get NO slot -- a typo in one of the "
            "two keys produces exactly this, and the run would otherwise train a "
            "model with fewer slots than the config asks for without a word."
        )

    ranks = [ranks_map[s] for s in order]
    scale_mode = str(slot_cfg.get("a_scale_mode", "match_mt4"))
    ref_rank = int(slot_cfg.get("a_scale_ref_rank", 128))
    eps = float(slot_cfg.get("orth_eps", 1e-6))
    # The default is the CONSTANT, not a 12 written here: the number is only
    # defensible together with the measurements in its docstring, and a literal in
    # this file could drift away from them without anything noticing.
    iters = int(slot_cfg.get("orth_iters", _DEFAULT_NS_ITERS))

    injection = inject_slot_lora(
        model,
        ranks,
        list(SLOT_LORA_TARGET_MODULES),
        scale_mode=scale_mode,
        ref_rank=ref_rank,
        eps=eps,
        iters=iters,
    )
    _assert_slot_leaves_fsdp_wrappable(model)

    # THE GATE HAS TO SURVIVE THIS CALL: the actor installs per-micro-batch routing
    # through it and nothing else in the model exposes it. Plain attributes, not
    # buffers or submodules -- a SlotGate is not an nn.Module and a tuple of strings is
    # not a tensor, so neither reaches the state dict, FSDP or the optimizer. Read them
    # back with find_slot_gate(model) / model._slot_order.
    #
    # `_slot_order` rides along because the actor's router needs the SAME order this
    # built the slots from (`match_suite_ids(texts, prompt_to_suite, suite_order)`
    # returns an index into it). Reading it off the model instead of re-reading config
    # makes "the router's order" and "the slots' order" one value rather than two that
    # must agree.
    model._slot_gate = injection.gate
    model._slot_order = tuple(order)

    # Mirrors the PEFT branch, deliberately. inject_slot_lora freezes EVERYTHING that is
    # not a slot parameter (it has to: FSDP cannot flatten mixed requires_grad under
    # use_orig_params=False, and only the slot leaves get their own flat param), and a
    # value head is the one thing a caller legitimately meant to keep training. The
    # attribute only exists when actor.model.add_value_head is set, and OPD runs leave
    # it off (adv_type: opd, no critic; RLINF_CONVERT_VALUE_HEAD=False on conversion),
    # so for them this is a no-op -- but get_model is shared with runs that DO use a
    # value head, and a slot path that froze the critic while
    # the PEFT baseline trained it would differ from the baseline in a second dimension
    # on top of the one being measured. FSDP-safe for the same reason it is on the PEFT
    # path: every ValueHead leaf is a childless nn.Linear, so the per-leaf policy gives
    # each its own uniformly-trainable flat param.
    value_head = "absent"
    if hasattr(model, "value_head"):
        for param in model.value_head.parameters():
            param.requires_grad = True
        value_head = "trainable"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # `skipped` is on this line and not only in inject's own warning because THIS is
    # the line the driver log is read for. A reader comparing the slot arm against the
    # PEFT baseline needs "437 adapted + 2 skipped" in one place; "437" alone invites
    # the conclusion that the two arms adapt the same set, which they do not.
    skipped = (
        "none"
        if not injection.skipped
        else ", ".join(f"{path} ({kind})" for path, kind in injection.skipped)
    )
    _sys.stderr.write(
        f"[slot-lora] injected {len(injection.paths)} SlotLoRALinear "
        f"(targets=SLOT_LORA_TARGET_MODULES); order={order} ranks={ranks} "
        f"R={sum(ranks)}; scale_mode={scale_mode} ref_rank={ref_rank} orth_eps={eps} "
        f"orth_iters={iters}; "
        f"name-matched but not nn.Linear (NOT adapted, PEFT adapts them): {skipped}; "
        f"value_head={value_head}; trainable params={trainable}\n"
    )
    return model


def tag_vlm_subtree(model, is_vlm: bool):
    for n, m in model.named_modules():
        setattr(m, "_to_lora", is_vlm)
