# Copyright 2026 The RLinf Authors.
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

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from rlinf.models.slot_lora.modules import (
    SlotGate,
    SlotLoRALinear,
    SlotOut,
    SlotProj,
)

logger = logging.getLogger(__name__)

_SCALE_MODES = ("match_mt4", "unit")


@dataclass(frozen=True)
class SlotInjection:
    """What one injection pass produced: where the slots went, and their gate.

    A DATACLASS, not a tuple or a NamedTuple, and deliberately neither iterable nor
    sized. Injection used to return a bare list of paths, and every call site that
    still treats it as one -- ``len(replaced)`` in a log line, ``" ".join(replaced)``
    in an assertion, ``paths, gate = inject(...)`` -- has to FAIL, loudly, at the
    first call. A NamedTuple would satisfy all three: ``len()`` would quietly return
    2 (the field count) where the log line means "how many layers did we adapt",
    which is a wrong number in a log that exists to confirm the injection touched the
    modules it was meant to. There is no version of this object that both keeps the
    old shape and reports the new fields honestly, so it keeps neither.

    Attributes:
        paths: Dotted module paths of every replaced linear, in ``named_modules``
            order, for the caller's log line. Resolvable against
            ``dict(model.named_modules())``.
        gate: The model's single :class:`SlotGate`. The training loop installs one
            routing per MICRO-batch through it (``with result.gate.scoped(ids):``),
            so it has to survive the injection call; nothing else in the model
            exposes it, and re-deriving it means walking the module tree.
    """

    paths: tuple[str, ...]
    gate: SlotGate


def _module_scale(in_features: int, scale_mode: str, ref_rank: int) -> float:
    """The LoRA scaling ``s`` for one module under the requested policy.

    ``match_mt4`` reproduces the step-1 ``ΔW`` magnitude of the PEFT baseline this
    experiment is measured against. That baseline used ``init_lora_weights="gaussian"``,
    whose ``A`` has rows of norm ``sqrt(d_in)/r``; here ``Ā``'s rows have norm exactly
    1, so without this factor the same learning rate would move ``ΔW`` by a different
    amount and the comparison would silently be a learning-rate comparison as well.
    ``unit`` leaves the orthonormal rows alone, which is what the unit tests want.
    """
    if scale_mode == "unit":
        return 1.0
    return math.sqrt(in_features) / ref_rank


def inject_slot_lora(
    model: nn.Module,
    slot_ranks: Sequence[int],
    target_modules: Iterable[str],
    scale_mode: str = "match_mt4",
    ref_rank: int = 128,
    eps: float = 1e-6,
    strict_gate: bool = True,
) -> SlotInjection:
    """Replace every targeted ``nn.Linear`` with a :class:`SlotLoRALinear`; freeze the rest.

    The model is edited IN PLACE and is the identity at initialization -- ``B`` is
    zero, so ``ΔW`` is exactly zero and the first forward is the base model's, bit for
    bit -- which is what makes injection safe to switch on without moving a baseline.

    ONE GATE FOR THE WHOLE MODEL. Exactly one :class:`SlotGate` is built here and
    handed to every layer, so the training loop installs a routing with a single
    attribute write rather than a walk over 200-400 modules. It comes back in the
    result because nothing else exposes it.

    WHAT COUNTS AS A TARGET is the child's ATTRIBUTE NAME being in ``target_modules``
    and the child being an ``nn.Linear`` -- the same rule PEFT's ``target_modules``
    uses in this repo, so the slot path adapts exactly the modules the PEFT baseline
    adapted. Nothing already injected is touched again: a ``SlotLoRALinear`` is not an
    ``nn.Linear``, and the walk never descends into one, so the frozen ``base`` inside
    it cannot be wrapped a second time (which would hide a second adapter from
    :meth:`SlotLoRALinear.delta_weight` and silently drop it at merge time).

    Args:
        model: The student. Edited in place.
        slot_ranks: Per-slot rank, one entry per slot, in slot-index order. Their sum
            must not exceed the narrowest targeted ``in_features``.
        target_modules: Attribute names to adapt, e.g. ``["q_proj", "v_proj"]``.
        scale_mode: ``"match_mt4"`` for ``s = sqrt(d_in) / ref_rank`` per module (see
            :func:`_module_scale`), or ``"unit"`` for ``s = 1``.
        ref_rank: The baseline LoRA rank ``match_mt4`` matches against. Ignored by
            ``"unit"``.
        eps: Floor on ``‖Z Zᵀ‖_F``, passed through to :func:`orthogonalize`.
        strict_gate: Whether the gate raises when read with no routing installed.
            Leave it on unless the run genuinely never routes; a strict gate is what
            turns "the routing was not installed" into an error instead of a silently
            ungated forward that trains every slot on every sample.

    Returns:
        A :class:`SlotInjection` carrying the replaced paths and the shared gate.

    Raises:
        ValueError: if ``scale_mode`` is not one of ``("match_mt4", "unit")``, if
            ``ref_rank`` is not positive under ``match_mt4``, if ``slot_ranks`` is
            empty or non-positive, or if ``target_modules`` matched no ``nn.Linear``
            at all. The model is left untouched in every one of those cases -- all of
            them are checked, or raise, before the first replacement.
    """
    if scale_mode not in _SCALE_MODES:
        raise ValueError(
            f"unknown scale_mode {scale_mode!r}; expected one of {_SCALE_MODES}. "
            "'match_mt4' reproduces the PEFT baseline's step-1 ΔW magnitude "
            "(s = sqrt(d_in)/ref_rank); 'unit' leaves Ā's unit-norm rows alone."
        )
    if scale_mode == "match_mt4" and ref_rank <= 0:
        raise ValueError(
            f"match_mt4 needs a positive ref_rank; got {ref_rank}. It is the rank of "
            "the PEFT LoRA whose step-1 ΔW magnitude this reproduces."
        )
    ranks = tuple(int(r) for r in slot_ranks)
    targets = set(target_modules)
    # Built BEFORE anything is replaced, so an empty or non-positive rank list raises
    # here rather than halfway through a 7B model. It is also what every SlotOut
    # validates its own slot count against.
    gate = SlotGate(len(ranks), strict=strict_gate)

    # Two phases: find, then replace. The walk therefore never observes a module it
    # created itself, which is what keeps a freshly wrapped `base` out of the results
    # without relying on when `named_modules` materializes its generator.
    candidates = []
    for parent_name, parent in model.named_modules():
        if isinstance(parent, SlotLoRALinear):
            continue  # its `base` is already adapted and already frozen
        for child_name, child in parent.named_children():
            if child_name in targets and isinstance(child, nn.Linear):
                path = f"{parent_name}.{child_name}" if parent_name else child_name
                candidates.append((parent, child_name, child, path))
    if not candidates:
        raise ValueError(
            f"slot-LoRA target_modules {sorted(targets)} matched no nn.Linear in this "
            "model. That is a config error (a typo, or a target list from a different "
            "architecture), and it is not survivable: the freeze pass below would "
            "leave the model with no trainable parameter at all, so the run would "
            "train nothing while looking entirely normal. Nothing has been modified."
        )

    # requires_grad BEFORE the pass, by identity: parameter NAMES move under the
    # wrapper (`q_proj.weight` becomes `q_proj.base.weight`), the objects do not.
    was_trainable = {id(p) for p in model.parameters() if p.requires_grad}
    adapted_bases: set[int] = set()

    paths = []
    for parent, child_name, child, path in candidates:
        scale = _module_scale(child.in_features, scale_mode, ref_rank)
        wrapper = SlotLoRALinear(child, ranks, scale, gate, eps=eps)
        setattr(parent, child_name, wrapper)
        adapted_bases.update(id(p) for p in wrapper.base.parameters())
        paths.append(path)

    # THE FREEZE. Uniform requires_grad within an FSDP flat parameter is required
    # under use_orig_params=False (this repo's default), and the per-leaf wrap policy
    # only gives slot_A/slot_B their own flat params: everything else folds into the
    # enclosing transformer layer's, which must therefore be uniformly frozen.
    collateral, trainable, frozen = [], 0, 0
    for name, param in model.named_parameters():
        is_slot = ".slot_A." in name or ".slot_B." in name
        param.requires_grad_(is_slot)
        trainable += int(is_slot)
        frozen += int(not is_slot)
        if (
            not is_slot
            and id(param) in was_trainable
            and id(param) not in adapted_bases
        ):
            collateral.append(name)
    # SAID OUT LOUD, ALWAYS, AND IN ONE LINE. Silently freezing is the right
    # BEHAVIOUR -- a trainable non-slot parameter anywhere breaks the flat-parameter
    # uniformity above -- but a caller that had something it meant to keep training (a
    # value head is the real case; Task 7 re-enables one right after this call) must
    # not have to infer the loss from a metric that stops moving. This is INFO and not
    # a warning because it fires on EVERY injection: before injection a freshly loaded
    # HF model has requires_grad=True on everything, so "was trainable" flags the whole
    # backbone and a warning would be noise by construction. The names are capped for
    # the same reason -- an uncapped list is thousands of lines on a 7B model, which is
    # a way of not being read.
    shown = ", ".join(collateral[:8])
    more = "" if len(collateral) <= 8 else f" (+{len(collateral) - 8} more)"
    logger.info(
        "slot-LoRA injection adapted %d linear(s); %d slot parameter(s) trainable, "
        "%d frozen, of which %d were trainable before this call: %s%s. Only "
        ".slot_A./.slot_B. parameters stay trainable, because FSDP cannot flatten "
        "mixed requires_grad under use_orig_params=False -- re-enable anything else "
        "you meant to train (e.g. a value head) AFTER this call.",
        len(paths),
        trainable,
        frozen,
        len(collateral),
        shown,
        more,
    )
    return SlotInjection(paths=tuple(paths), gate=gate)


def _slot_layers(model: nn.Module) -> list[SlotLoRALinear]:
    """Every :class:`SlotLoRALinear` in the model, in ``modules()`` order.

    ``isinstance`` over ``model.modules()`` is correct under FSDP, and this was
    verified rather than assumed: a real ``FullyShardedDataParallel`` wrap (torch
    2.6.0, gloo world of one, this repo's per-leaf LoRA auto-wrap policy,
    ``use_orig_params=False``) registers the wrapped module as the CHILD
    ``_fsdp_wrapped_module``, so ``modules()`` recurses into it and the real
    ``SlotLoRALinear`` / ``SlotProj`` / ``SlotOut`` instances all still appear
    (measured: 2/2/2 for a two-layer toy, and the ``SlotLoRALinear`` itself is never
    wrapped, since the wrap policy only fires on childless modules).
    """
    return [m for m in model.modules() if isinstance(m, SlotLoRALinear)]


def _halves(layer: SlotLoRALinear) -> tuple[SlotProj, SlotOut]:
    """The real ``SlotProj`` and ``SlotOut`` of ONE layer, past any FSDP wrapper.

    NOT ``layer.slot_A`` / ``layer.slot_B``. Under FSDP those attributes are
    ``FullyShardedDataParallel`` wrappers: reading through them works (``__getattr__``
    forwards to ``_fsdp_wrapped_module``) but WRITING through them does not -- measured
    on a real wrap, ``layer.slot_A._diag = None`` lands in the wrapper's own
    ``__dict__`` and shadows the real attribute from that moment on, so every later
    read returns the shadow even though the ``SlotProj`` keeps refreshing its ``_diag``
    in the forward. The diagnostics would report once and then be empty forever, with
    nothing raised.

    Searching within ONE layer keeps the pairing guarantee that
    :meth:`SlotLoRALinear.arm_diag` exists to provide: a ``SlotLoRALinear`` holds
    exactly one of each, so this cannot pair halves across layers.
    """
    return (
        next(m for m in layer.modules() if isinstance(m, SlotProj)),
        next(m for m in layer.modules() if isinstance(m, SlotOut)),
    )


def enable_slot_diag(model: nn.Module) -> bool:
    """Arm the orthogonality/interference diagnostics on exactly ONE layer.

    Armed through :meth:`SlotLoRALinear.arm_diag`, which arms both halves of the SAME
    layer and clears their previous readings. Arming the two halves separately -- "the
    first ``SlotProj``" and "the first ``SlotOut``" found by two independent walks --
    pairs them by module registration order, and in a 7B model full of same-shaped
    projections a mismatched pair produces the right SHAPES and a wrong NUMBER, with
    nothing to raise.

    The numbers are then computed inside that layer's forward, which is what keeps
    them free under FSDP: the module's parameters are already all-gathered there, so
    no extra collective is needed, and ``orthogonalize`` sees the whole ``Z`` rather
    than a shard of it. One layer, not all of them: the extra fp32 orthogonalization
    is ~1.3 ms and it is a sentinel, not a survey.

    Args:
        model: The injected student (FSDP-wrapped or not).

    Returns:
        Whether a layer was found to arm. ``False`` means the model has no slots.
    """
    layers = _slot_layers(model)
    if not layers:
        return False
    layers[0].arm_diag()
    return True


@torch.no_grad()
def collect_slot_diag(model: nn.Module) -> dict[str, float]:
    """Read what the armed forward stashed, as flat ``{metric: float}``.

    Metrics, all from ONE layer:

    * ``slot/orth_err`` -- ``‖Ā Āᵀ − I‖_F`` off the fp32 Gram. Below 1e-3 is healthy;
      1e-3 to 5e-2 means ``Z`` is degrading; above 5e-2 orthogonality has collapsed
      and the slots are no longer isolated.
    * ``slot/dw_norm_{k}`` -- ``‖ΔW_k‖_F = scale · ‖B_k‖_F``, exact because ``Ā_k``
      has orthonormal rows. A slot pinned at 0 has learned nothing.
    * ``slot/cos_{s}_{t}`` -- the cosine between two slots' ``ΔW``, for every ``s < t``.

    THE COSINE, AND WHY IT LOOKS ASYMMETRIC.
    ``⟨ΔW_s, ΔW_t⟩_F = scale² · Σ(B_tᵀB_s ⊙ Ā_tĀ_sᵀ)`` and ``‖ΔW_k‖_F = scale·‖B_k‖_F``,
    so ``scale²`` CANCELS in the cosine and must not be applied here -- while
    ``dw_norm_k`` carries an explicit ``scale``, because there it does not cancel.
    ``cross[(s, t)]`` is ``B_tᵀB_s`` with shape ``(r_t, r_s)``, so it pairs with
    ``gram[t_slice, s_slice]``, NOT ``gram[s_slice, t_slice]``. At the production ranks
    (128/64/48/16) the wrong order raises a shape error, but with EQUAL per-slot ranks
    both slicings have the same shape and the wrong one silently reports a different
    quantity. ``ΔW`` itself is never materialized; it is ``d_out x d_in``.

    PRECISION. No matmul is formed here -- the two that matter (the Gram and
    ``B_tᵀB_s``) were already formed inside the modules under their fp32 floor and
    their no-autocast/no-TF32 guard -- so there is nothing left for an ambient
    autocast or TF32 to demote. The whole read is elementwise products, sums and
    norms over the small ``(R, R)`` and ``(r_t, r_s)`` blocks.

    Args:
        model: The injected student.

    Returns:
        The metrics, or ``{}`` if nothing was armed or no forward reached the armed
        layer. The readings are CLEARED as they are read, so a subsequent call with no
        re-arm returns ``{}`` rather than repeating last step's numbers -- a stale
        constant is exactly the failure these plots must not produce.
    """
    for layer in _slot_layers(model):
        proj, out = _halves(layer)
        # BOTH halves, or neither: a half-populated pair means the forward raised
        # between them, and pairing a fresh Gram with a stale B is worse than silence.
        if proj._diag is None or out._diag is None:
            continue

        gram = proj._diag["gram"]
        scale = proj._diag["scale"]
        b_norms = out._diag["b_norms"]
        eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)

        # Collected as 0-dim tensors and converted in ONE host transfer at the end:
        # per-metric `.item()` would be 1 + K + K(K-1)/2 separate device syncs inside
        # the training step.
        keys = ["slot/orth_err"]
        values = [(gram - eye).norm()]
        for k in range(len(out.slot_ranks)):
            keys.append(f"slot/dw_norm_{k}")
            values.append(scale * b_norms[k])
        for (s, t), cross in out._diag["cross"].items():
            t_slice = slice(out.offsets[t], out.offsets[t] + out.slot_ranks[t])
            s_slice = slice(out.offsets[s], out.offsets[s] + out.slot_ranks[s])
            inner = (cross * gram[t_slice, s_slice].to(cross.dtype)).sum()
            keys.append(f"slot/cos_{s}_{t}")
            values.append(inner / (b_norms[s] * b_norms[t]).clamp_min(1e-12))

        # Cleared on the REAL modules; see _halves for why not through the attribute.
        proj._diag = None
        out._diag = None
        return dict(zip(keys, torch.stack(values).tolist()))
    return {}
