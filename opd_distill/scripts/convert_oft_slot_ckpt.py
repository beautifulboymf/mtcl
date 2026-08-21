# Copyright (c) 2026. Merge an OpenVLA-OFT RLinf slot-LoRI training checkpoint
# (`full_weights.pt`, a plain state dict) into a plain HuggingFace model directory that
# the RLinf eval pipeline can load via `model.model_path`.
#
# This is the slot analogue of `convert_oft_lora_ckpt.py`, and it shares that script's
# base-model construction and auxiliary-file list rather than copying them (see
# `baseline_converter` for why). Only the adapter layer differs: where the single-LoRA
# converter wraps with PEFT and calls `merge_and_unload()`, this one calls
# `inject_slot_lora` and adds each `SlotLoRALinear.delta_weight()` into its base.
#
# THE SCALE IS VERIFIED, NOT RE-DERIVED. `SlotProj` persists the LoRA scaling `s` it was
# trained with through `get_extra_state`, and its `set_extra_state` RAISES when the
# module it is loading into is configured with a different one. This converter derives
# `s` from the CLI flags (`--slot-scale-mode` / `--slot-ref-rank`, exactly as the
# training path derives it from `actor.model.slot_lora`), builds the modules with it,
# and then loads the checkpoint THROUGH those modules -- so the cross-check runs on
# every conversion and a disagreement is an exception, not a silent constant factor on
# every merged delta. `s` leaves no trace in any tensor shape (Ā is invariant to the
# scale of Z), so without that record a converter run with the wrong `--slot-ref-rank`
# would succeed, produce a model that loads and runs, and simply have every ΔW the wrong
# size. The flow below only closes that hole if the injection happens BEFORE the load;
# do not reorder them.
#
# CPU-only by design (CUDA_VISIBLE_DEVICES="" set by the wrapper) so no GPU is touched.
#
# Usage:
#   python convert_oft_slot_ckpt.py \
#       --ckpt   /path/to/full_weights.pt \
#       --base   /share/fanruochen-local/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora \
#       --out    /path/to/output/dir \
#       [--slot-ranks 128,64,48,16] [--slot-scale-mode match_mt4] \
#       [--slot-ref-rank 128] [--slot-eps 1e-6] \
#       [--unnorm-key libero_130_no_noops_trajall]

import argparse
import glob
import importlib.util
import json
import os
import shutil
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASELINE_SCRIPT = os.path.join(_HERE, "convert_oft_lora_ckpt.py")

# Defaults for the R1 slot-LoRI run: four suites, R = 256, matched against the mt4 PEFT
# baseline's rank-128 LoRA. They are DEFAULTS, not a contract -- a run whose config says
# otherwise must pass its own; the scale cross-check catches a wrong scale mode or
# reference rank, and the total-rank check catches a rank list that does not sum right.
DEFAULT_SLOT_RANKS = "128,64,48,16"
DEFAULT_SLOT_SCALE_MODE = "match_mt4"
DEFAULT_SLOT_REF_RANK = 128
DEFAULT_SLOT_EPS = 1e-6

# Substrings that mark a state-dict key as belonging to the slot structure this
# converter built, as opposed to the base model's own weights. Used only to make the
# key-mismatch report readable; the CHECK itself is on the full key sets.
_SLOT_MARKERS = (".slot_A.", ".slot_B.", ".base.")


class SlotCheckpointMismatch(RuntimeError):
    """The checkpoint does not describe the structure this converter built.

    Raised instead of letting the merge proceed on a partial load. A slot checkpoint
    that half-matches is the dangerous case: the base weights load, some slots do not,
    and the merged model is a valid, runnable model that is not the one that was
    trained.
    """


def _load_baseline_converter():
    """Import the sibling single-LoRA converter, once, by path.

    ``opd_distill/scripts`` is a directory of scripts, not a package, so there is no
    import path to reach it by -- and adding one (``sys.path.insert``) would put a
    directory full of short, generic script names ahead of the standard library for the
    whole process. Loading the one file by absolute path costs six lines and touches
    nothing global.
    """
    if not os.path.isfile(_BASELINE_SCRIPT):
        raise ImportError(
            f"{_BASELINE_SCRIPT} not found. This converter shares its base-model "
            "construction with the single-LoRA converter on purpose (see "
            "`baseline_converter`); the two must build the SAME model, and a private "
            "copy here is a copy that can be fixed in one place and left wrong in the "
            "other. Keep the two scripts side by side."
        )
    spec = importlib.util.spec_from_file_location(
        "_convert_oft_lora_ckpt", _BASELINE_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASELINE = _load_baseline_converter()

# THE SHARED SURFACE, imported rather than duplicated.
#
# `build_base_model` encodes how RLinf itself builds an OpenVLAOFTForRLActionPrediction
# (config merge, dataset_statistics, num_images_in_input, eager attention, bf16) and
# `AUX_FILES` is the list that makes the output directory a complete loadable model.
# Both are properties of the MODEL, identical for the PEFT and slot arms, and both are
# exactly the kind of thing that gets fixed in one copy and not the other -- with a
# converted model that quietly differs from the trained one as the only symptom. What is
# NOT shared is anything about the adapter: the target list comes from
# `rlinf.models.SLOT_LORA_TARGET_MODULES` (see `default_target_modules`), which is the
# same constant the training path injects with.
build_base_model = _BASELINE.build_base_model
AUX_FILES = _BASELINE.AUX_FILES


def baseline_converter():
    """The sibling ``convert_oft_lora_ckpt`` module this one shares code with."""
    return _BASELINE


def default_target_modules():
    """The modules to adapt: the SAME list ``get_model`` injects the student with.

    Imported from ``rlinf.models`` rather than restated, for the reason that constant
    was hoisted in the first place: if the converter adapted a different set of modules
    than training did, the checkpoint's keys would not line up -- which the key check
    below does catch -- but a merely SMALLER set would line up for the modules it does
    cover and silently drop the rest.

    It is the same LIST, which is not the same thing as the same adapted SET, and the
    difference is worth naming here because this is the script that produces the model
    the baseline is compared against. ``inject_slot_lora`` adapts a child only when its
    name is in this list AND it is an ``nn.Linear``; on OpenVLA-OFT two name-matched
    children are ``nn.Conv2d`` (``vision_backbone.featurizer.patch_embed.proj`` and
    ``vision_backbone.fused_featurizer.patch_embed.proj``, both matched by ``"proj"``).
    So training and this converter both adapt 437 modules -- consistently, which is all
    the key check needs -- while the PEFT baseline this run is measured against adapted
    439 (measured: 437 two-dimensional ``lora_A`` plus 2 four-dimensional ones). Those
    two convs are frozen in the slot arm and merged from nothing here. See
    ``inject_slot_lora``, which counts and logs them at injection time.
    """
    from rlinf.models import SLOT_LORA_TARGET_MODULES

    return list(SLOT_LORA_TARGET_MODULES)


def parse_slot_ranks(text):
    """``"128,64,48,16"`` -> ``(128, 64, 48, 16)``.

    THE ORDER IS THE TRAINING CONFIG'S ``slot_order``, not the YAML mapping's insertion
    order: the slot index is what routing produced (``match_suite_ids`` returns
    ``slot_order.index(suite)``), so slot k's rank is the rank of ``slot_order[k]``.

    Args:
        text: Comma-separated positive integers.

    Returns:
        The ranks as a tuple, in slot-index order.

    Raises:
        ValueError: if the list is empty, or holds anything but positive integers.
    """
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError(
            f"--slot-ranks needs at least one rank; got {text!r}. Give the per-slot "
            "ranks of the training config, comma separated, in its slot_order."
        )
    try:
        ranks = tuple(int(p) for p in parts)
    except ValueError:
        raise ValueError(
            f"--slot-ranks must be comma-separated integers; got {text!r}."
        ) from None
    if any(r <= 0 for r in ranks):
        raise ValueError(
            f"--slot-ranks must be strictly positive; got {ranks}. A zero-width slot "
            "owns no columns of B."
        )
    return ranks


def wrap_with_slots(
    model,
    slot_ranks,
    scale_mode=DEFAULT_SLOT_SCALE_MODE,
    ref_rank=DEFAULT_SLOT_REF_RANK,
    eps=DEFAULT_SLOT_EPS,
    target_modules=None,
    verbose=False,
):
    """Rebuild the EXACT training-time structure, so the checkpoint's keys line up.

    The same call ``rlinf.models._apply_slot_lora`` makes, with the same target list and
    the same scale policy. It is what turns the CLI's ``--slot-scale-mode`` /
    ``--slot-ref-rank`` into a per-module ``s`` that the subsequent load then CHECKS
    against the one the checkpoint recorded.

    Args:
        model: The freshly built base model. Edited in place.
        slot_ranks: Per-slot rank in slot-index order (see :func:`parse_slot_ranks`).
        scale_mode: ``"match_mt4"`` or ``"unit"``, as in the training config's
            ``a_scale_mode``.
        ref_rank: The PEFT baseline rank ``match_mt4`` matches, as in
            ``a_scale_ref_rank``.
        eps: Floor on ``‖Z Zᵀ‖_F``, as in ``orth_eps``.
        target_modules: Attribute names to adapt; ``None`` uses
            :func:`default_target_modules`, which is what training uses.
        verbose: Whether to print the injection summary.

    Returns:
        The :class:`~rlinf.models.slot_lora.inject.SlotInjection` handle.
    """
    from rlinf.models.slot_lora import inject_slot_lora

    if target_modules is None:
        target_modules = default_target_modules()
    injection = inject_slot_lora(
        model,
        slot_ranks,
        target_modules,
        scale_mode=scale_mode,
        ref_rank=ref_rank,
        eps=eps,
    )
    if verbose:
        print(
            f"    injected {len(injection.paths)} SlotLoRALinear; "
            f"ranks={tuple(slot_ranks)} R={sum(slot_ranks)} "
            f"scale_mode={scale_mode} ref_rank={ref_rank} eps={eps}",
            flush=True,
        )
    return injection


def _check_total_rank(state_dict, slot_ranks):
    """Compare ``sum(slot_ranks)`` against the width the checkpoint's Z and B have.

    A pre-check purely for the error MESSAGE: a wrong ``--slot-ranks`` whose sum differs
    also fails the load, but as ``size mismatch for ...slot_A.weight: copying a param
    with shape torch.Size([256, 4096])`` a few hundred lines long, which does not name
    the flag that caused it.

    IT CANNOT SEE A WRONG ORDER. ``Z`` is ``(R, d_in)`` and ``B`` is ``(d_out, R)``, and
    neither shape depends on where the slot boundaries fall inside ``R`` -- only on the
    SUM. So any permutation of the ranks, and any other list with the same total, passes
    here and loads cleanly. That is benign for the merge itself (``ΔW = Σ_k B_k Ā_k =
    B Ā`` sums over every slot regardless of the partition, so the merged weights are
    bit-identical under a permutation) and it is why the docstring on ``--slot-ranks``
    insists on ``slot_order``: the same mistake in a TRAINING config is not benign at
    all. The production ranks 128/64/48/16 sum to 256 under every permutation, so no
    permutation of them is detectable here or anywhere else in this script.

    Raises:
        SlotCheckpointMismatch: if the totals disagree.
    """
    expected = sum(int(r) for r in slot_ranks)
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if key.endswith(".slot_A.weight"):
            found, axis = value.shape[0], "rows of Z"
        elif key.endswith(".slot_B.weight"):
            found, axis = value.shape[-1], "columns of B"
        else:
            continue
        if found != expected:
            raise SlotCheckpointMismatch(
                f"--slot-ranks {tuple(slot_ranks)} sums to total rank {expected}, but "
                f"the checkpoint's {key} has {found} {axis}. The checkpoint was "
                f"trained with slots summing to {found}; pass that run's slot_ranks, "
                "in its slot_order. (Only the SUM is checkable here -- Z and B are "
                "shaped by R alone -- so a rank list with the right total but the "
                "wrong split will not be caught.)"
            )


def load_slot_checkpoint(model, state_dict, slot_ranks, verbose=False):
    """Load a trained slot checkpoint into the injected model, or refuse.

    Zero missing and zero unexpected keys, both -- the same demand the single-LoRA
    converter makes of its LoRA keys. It is the only thing standing between a structural
    mismatch and a merged model that is quietly wrong: a checkpoint whose slots did not
    load merges ``ΔW = 0`` and produces the BASE model under a new name, which runs, and
    evaluates, and is not what was trained.

    THE KEY SETS ARE DIFFED BEFORE THE LOAD, and the report is checked after it. The
    diff is not there because the report is unreliable -- it is not; ``load_state_dict``
    passes ``strict=True`` down into every ``_load_from_state_dict`` and only gates the
    final RAISE on its own ``strict``, so even at ``strict=False`` it reports an absent
    ``_extra_state`` as a missing key (verified on torch 2.6, which is what this repo
    pins). The diff earns its place by running FIRST: a mismatched checkpoint is refused
    with a message that names the flags to fix, instead of after several hundred tensors
    have already been copied into the model and, on a rank mismatch, after torch has
    raised its own ``size mismatch for ...`` wall of text. The post-load check is then
    pure belt and braces, and should be unreachable.

    Args:
        model: The injected model from :func:`wrap_with_slots`.
        state_dict: The loaded ``full_weights.pt``.
        slot_ranks: The ranks the model was injected with, for the total-rank message.
        verbose: Whether to print the key accounting.

    Returns:
        ``{"ckpt_keys": int, "slot_keys": int}``.

    Raises:
        SlotCheckpointMismatch: on any missing or unexpected key, or a total rank that
            disagrees with the checkpoint.
        ValueError: from ``SlotProj.set_extra_state``, if the scale the CLI derived
            disagrees with the one the checkpoint was trained with.
    """
    _check_total_rank(state_dict, slot_ranks)

    expected = set(model.state_dict().keys())
    present = set(state_dict.keys())
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    slot_keys = sum(1 for k in present if any(m in k for m in _SLOT_MARKERS))
    if verbose:
        print(
            f"    ckpt keys={len(present)} (slot/base={slot_keys}); "
            f"model keys={len(expected)}",
            flush=True,
        )
        print(
            f"    key diff: missing={len(missing)} unexpected={len(unexpected)} "
            "(BOTH MUST BE 0)",
            flush=True,
        )
    if missing or unexpected:
        raise SlotCheckpointMismatch(
            "the checkpoint does not match the module tree this converter built: "
            f"{len(missing)} missing key(s) and {len(unexpected)} unexpected key(s). "
            f"missing (first 20): {missing[:20]}; "
            f"unexpected (first 20): {unexpected[:20]}. "
            "Check --slot-ranks (number of slots), the target module list, and that "
            "this checkpoint really is a slot-LoRI run and not a PEFT one (PEFT keys "
            "carry 'lora_A'/'lora_B'/'base_layer' and belong to "
            "convert_oft_lora_ckpt.py). Merging past this would produce a model that "
            "loads and runs and is not the one that was trained."
        )

    # Only now, with the keys proven to line up, does the load run -- and with them
    # lined up its own report is redundant, so `strict=False` here costs nothing. This
    # is the call that reaches SlotProj.set_extra_state and cross-checks the scale.
    report = model.load_state_dict(state_dict, strict=False)
    if report.missing_keys or report.unexpected_keys:
        raise SlotCheckpointMismatch(
            "load_state_dict disagreed with the key diff above -- "
            f"missing={list(report.missing_keys)[:20]} "
            f"unexpected={list(report.unexpected_keys)[:20]}. This should be "
            "unreachable; do not merge."
        )
    return {"ckpt_keys": len(present), "slot_keys": slot_keys}


def merge_slots(model, verbose=False):
    """``W <- W + s·B Ā`` per layer, then swap each wrapper out for its plain linear.

    ``Ā`` is recomputed from the stored ``Z`` through the same ``orth_weight()`` call
    the training forward made -- ``Z`` is what the checkpoint carries, ``Ā`` is never
    stored and is never a seed to be replayed -- so the merged model reproduces the
    model that was trained rather than an improved version of it.

    THE CAST IS EXPLICIT AND PER LAYER. ``delta_weight()`` returns fp32 or wider while
    the base weight is bf16 in production, and ``W.data.add_(delta)`` does NOT raise:
    ``add_`` downcasts the operand to the destination dtype and lands on the same
    answer, so omitting the cast is lucky rather than correct. Merging the whole model
    in fp32 and casting once at the end would buy nothing -- each base weight is written
    exactly ONCE, so there is no accumulation across the 200-400 layers to avoid -- while
    doubling peak memory on a 7B model and changing the dtype of what gets saved.

    The wrapper is then replaced by its ``base`` so the saved model has no slot
    machinery in it at all: no ``SlotLoRALinear``, no ``slot_``/``_extra_state`` keys,
    and the original parameter names (``q_proj.weight``, not ``q_proj.base.weight``)
    that a plain ``from_pretrained`` expects. The bias is not touched -- ``ΔW`` is
    weight-only -- and rides along on ``base``, exactly once.

    Args:
        model: The injected model, with the checkpoint already loaded.
        verbose: Whether to print the merge count.

    Returns:
        How many layers were merged.
    """
    from rlinf.models.slot_lora import SlotLoRALinear, SlotOut, SlotProj

    merged = 0
    with torch.no_grad():
        for parent in list(model.modules()):
            for child_name, child in list(parent.named_children()):
                if not isinstance(child, SlotLoRALinear):
                    continue
                base = child.base
                base.weight.data.add_(child.delta_weight().to(base.weight.dtype))
                setattr(parent, child_name, base)
                merged += 1
    if verbose:
        print(f"    merged {merged} slot layer(s) into the base weights", flush=True)

    # Cheap, and it is the assertion that the OUTPUT is a plain model rather than a
    # model that merely had its deltas added: a wrapper left in the tree would be saved
    # with slot_A/slot_B/_extra_state keys the eval loader knows nothing about, on top
    # of a base weight that already carries the delta -- i.e. the adapter applied twice.
    leftover = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (SlotLoRALinear, SlotProj, SlotOut))
    ]
    if leftover:
        raise SlotCheckpointMismatch(
            f"{len(leftover)} slot module(s) survived the merge: {leftover[:20]}. The "
            "saved model would carry both a merged delta and the adapter that produced "
            "it."
        )
    return merged


def merge_slot_checkpoint(
    model,
    state_dict,
    slot_ranks,
    scale_mode=DEFAULT_SLOT_SCALE_MODE,
    ref_rank=DEFAULT_SLOT_REF_RANK,
    eps=DEFAULT_SLOT_EPS,
    target_modules=None,
    verbose=False,
):
    """Inject, load, merge -- the whole conversion core, on any ``nn.Module``.

    Kept free of the 7B model, the checkpoint file, the output directory and argparse so
    it can be tested on a two-linear toy, which is where the merge arithmetic and every
    rejection above are actually verifiable.

    THE THREE STEPS ARE ORDERED, and the order is the guarantee: injecting FIRST with
    the CLI-derived scale and loading SECOND is what makes the checkpoint's persisted
    scale a cross-check rather than a decoration. Loading into a bare model and
    injecting afterwards would produce exactly the silent constant-factor error the
    persisted scale exists to close.

    Args:
        model: A freshly built base model. Edited in place.
        state_dict: The trained slot checkpoint.
        slot_ranks: Per-slot ranks in the training config's ``slot_order``.
        scale_mode: ``"match_mt4"`` or ``"unit"``.
        ref_rank: The reference rank for ``match_mt4``.
        eps: Floor on ``‖Z Zᵀ‖_F``.
        target_modules: ``None`` uses :func:`default_target_modules`.
        verbose: Whether to print per-step accounting.

    Returns:
        The same model, now plain and merged.
    """
    injection = wrap_with_slots(
        model,
        slot_ranks,
        scale_mode=scale_mode,
        ref_rank=ref_rank,
        eps=eps,
        target_modules=target_modules,
        verbose=verbose,
    )
    load_slot_checkpoint(model, state_dict, slot_ranks, verbose=verbose)
    merged = merge_slots(model, verbose=verbose)
    if merged != len(injection.paths):
        raise SlotCheckpointMismatch(
            f"injected {len(injection.paths)} slot layer(s) but merged {merged}. Some "
            "adapter was left unmerged; the saved model would be missing its delta."
        )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="slot full_weights.pt state dict")
    ap.add_argument("--base", required=True, help="base student HF model dir")
    ap.add_argument("--out", required=True, help="output dir for merged HF model")
    # `scale` is persisted in the checkpoint via SlotProj.get_extra_state, so these
    # flags are VERIFIED against the trained value rather than merely applied: see the
    # module header. They still have to be right -- the check reports a disagreement,
    # it does not reconcile it.
    ap.add_argument(
        "--slot-ranks",
        default=DEFAULT_SLOT_RANKS,
        help=(
            "comma-separated per-slot ranks, IN THE TRAINING CONFIG'S slot_order "
            "(default 128,64,48,16 = libero_10,goal,spatial,object). The order defines "
            "the slot index routing produced, so it must be that list and not the YAML "
            "mapping's insertion order. Only the SUM is checkable against the "
            "checkpoint."
        ),
    )
    ap.add_argument(
        "--slot-scale-mode",
        default=DEFAULT_SLOT_SCALE_MODE,
        choices=["match_mt4", "unit"],
        help="training config's actor.model.slot_lora.a_scale_mode",
    )
    ap.add_argument(
        "--slot-ref-rank",
        type=int,
        default=DEFAULT_SLOT_REF_RANK,
        help="training config's actor.model.slot_lora.a_scale_ref_rank",
    )
    ap.add_argument(
        "--slot-eps",
        type=float,
        default=DEFAULT_SLOT_EPS,
        help="training config's actor.model.slot_lora.orth_eps",
    )
    ap.add_argument("--action-dim", type=int, default=7)
    ap.add_argument("--num-action-chunks", type=int, default=8)
    ap.add_argument("--max-prompt-length", type=int, default=128)
    ap.add_argument("--num-images-in-input", type=int, default=1)
    ap.add_argument("--center-crop", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument(
        "--unnorm-key",
        default=None,
        help="default: sole key in base/dataset_statistics.json",
    )
    args = ap.parse_args()

    assert os.path.isfile(args.ckpt), f"ckpt not found: {args.ckpt}"
    assert os.path.isdir(args.base), f"base dir not found: {args.base}"

    slot_ranks = parse_slot_ranks(args.slot_ranks)

    if args.unnorm_key is None:
        with open(os.path.join(args.base, "dataset_statistics.json")) as f:
            keys = list(json.load(f).keys())
        assert len(keys) >= 1, "empty dataset_statistics.json"
        args.unnorm_key = keys[0]
    print(
        f"[cfg] slot_ranks={slot_ranks} R={sum(slot_ranks)}  "
        f"scale_mode={args.slot_scale_mode}  ref_rank={args.slot_ref_rank}  "
        f"eps={args.slot_eps}  unnorm_key={args.unnorm_key}",
        flush=True,
    )

    # 1) Build base model (shared with the single-LoRA converter: RLinf-faithful).
    print("[1/5] building base OpenVLAOFTForRLActionPrediction on CPU ...", flush=True)
    model = build_base_model(
        args.base,
        args.unnorm_key,
        args.action_dim,
        args.num_action_chunks,
        args.max_prompt_length,
        args.num_images_in_input,
        args.center_crop,
    )

    # 2) Read the checkpoint off disk.
    print(f"[2/5] loading state dict: {args.ckpt}", flush=True)
    sd = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=True)

    # 3) Inject, load, merge -- through the SAME function the unit tests exercise on a
    #    toy, rather than a second copy of the sequence here. The order is the
    #    guarantee (inject with the CLI scale, then load, so the checkpoint's recorded
    #    scale cross-checks it), and a copy of it in main() is a copy that can drift
    #    from the one under test.
    print(
        "[3/5] inject_slot_lora -> load (verifies scale) -> merge W += s*B@A_bar ...",
        flush=True,
    )
    try:
        merge_slot_checkpoint(
            model,
            sd,
            slot_ranks,
            scale_mode=args.slot_scale_mode,
            ref_rank=args.slot_ref_rank,
            eps=args.slot_eps,
            verbose=True,
        )
    except SlotCheckpointMismatch as exc:
        print(f"FATAL: {exc}", flush=True)
        sys.exit(2)
    except ValueError as exc:
        # SlotProj.set_extra_state, i.e. the scale cross-check this converter exists to
        # perform. Nothing else in this path raises a bare ValueError.
        print(
            f"FATAL: {exc}\n"
            "  -> the scale this converter derived from --slot-scale-mode/"
            "--slot-ref-rank is not the one the checkpoint was trained with. Fix the "
            "flags to match the run's actor.model.slot_lora config.",
            flush=True,
        )
        sys.exit(2)

    # 4) Save merged weights.
    os.makedirs(args.out, exist_ok=True)
    print(f"[4/5] save_pretrained -> {args.out}", flush=True)
    model.save_pretrained(args.out, safe_serialization=True)

    # 5) Copy aux (non-weight) files + all *.py; never overwrite merged safetensors.
    print(
        "[5/5] copying config/tokenizer/*.py from base (not overwriting weights) ...",
        flush=True,
    )
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
