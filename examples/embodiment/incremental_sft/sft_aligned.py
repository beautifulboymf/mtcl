#!/usr/bin/env python
"""Norm-ALIGNED launcher for OpenVLA-OFT SFT (incremental SFT->OPD continual pipeline).

OpenVLA-OFT's stock finetune.py normalizes actions with the PER-SUITE RLDS statistics
(libero_spatial / libero_object q01/q99). For our incremental SFT->OPD pipeline the SFT'd
student MUST share ONE action tokenization with the frozen libero_130 teacher (and with the
evaluation) -- otherwise the 256-bin action grids differ and the OPD forward-KL compares
mismatched distributions, and a libero_130-unnorm eval decodes the student's actions wrongly.

This launcher monkeypatches the RLDS data pipeline to FORCE the action/proprio statistics to a
supplied override (the libero_130 stats), then runs the stock finetune.py unchanged. The openvla
repository is NOT modified -- the patch lives entirely here and is a no-op if SFT_NORM_OVERRIDE
is unset.

Run it exactly like finetune.py (through torchrun), with env SFT_NORM_OVERRIDE pointing at a
single-dataset dataset_statistics json holding the libero_130 action/proprio stats, e.g.
    SFT_NORM_OVERRIDE=/.../norm_override_libero130.json torchrun ... sft_aligned.py <finetune args>
"""
import json
import os
import runpy

FINETUNE = os.environ.get(
    "SFT_FINETUNE_PY", "/home/fanruochen/CL/openvla/vla-scripts/finetune.py"
)
OVERRIDE = os.environ.get("SFT_NORM_OVERRIDE", "").strip()

# SAFETY: the OFT 8-chunk behavior lives in the VENV's prismatic (future_action_window_size=
# NUM_ACTIONS_CHUNK-1). If sys.path accidentally resolves `prismatic` to the mainline openvla
# repo (vanilla, single-action) the SFT would SILENTLY train a 1-chunk model that mismatches the
# 8-chunk teacher/wrapper. Fail LOUD instead: require the venv (site-packages) prismatic and report
# the action-chunk count.
import prismatic  # noqa: E402

_pf = getattr(prismatic, "__file__", "") or ""
if "site-packages" not in _pf:
    raise SystemExit(
        f"[sft_aligned] REFUSING: `prismatic` resolved to {_pf!r} (not the venv OFT prismatic). "
        f"Run from a neutral cwd (e.g. /tmp) with the rlinf-openvlaoft venv and do NOT put the "
        f"mainline openvla repo on PYTHONPATH -- else SFT trains a 1-chunk model."
    )
try:
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK as _NCHUNK  # noqa: E402
    print(f"[sft_aligned] prismatic={_pf}  NUM_ACTIONS_CHUNK={_NCHUNK}", flush=True)
except Exception as _e:  # pragma: no cover
    print(f"[sft_aligned] prismatic={_pf}  (NUM_ACTIONS_CHUNK unknown: {_e})", flush=True)

if OVERRIDE:
    # Patch BEFORE finetune.py builds the dataset. make_interleaved_dataset (same module) calls
    # make_dataset_from_rlds per sub-dataset via the module global, so rebinding the global here
    # takes effect for the RLDSDataset finetune.py constructs. We FORCE the override regardless of
    # any per-dataset stats make_interleaved_dataset computed/passed, so single-suite RLDS data is
    # normalized with libero_130 bounds.
    import prismatic.vla.datasets.rlds.dataset as _rlds

    with open(OVERRIDE) as _f:
        _OVERRIDE_STATS = json.load(_f)
    _orig_make = _rlds.make_dataset_from_rlds

    def _make_aligned(*args, dataset_statistics=None, **kwargs):  # noqa: ANN001
        # ignore the incoming (per-suite) stats; force libero_130
        return _orig_make(*args, dataset_statistics=_OVERRIDE_STATS, **kwargs)

    _rlds.make_dataset_from_rlds = _make_aligned
    print(
        f"[sft_aligned] FORCING action-norm to libero_130 from {OVERRIDE} "
        f"(action q01[:3]={_OVERRIDE_STATS['action']['q01'][:3]})",
        flush=True,
    )
else:
    print(
        "[sft_aligned] WARNING: SFT_NORM_OVERRIDE unset -> per-suite RLDS norm (NOT aligned to teacher)",
        flush=True,
    )

# Hand argv straight to finetune.py's draccus entrypoint (runpy preserves sys.argv[1:]).
runpy.run_path(FINETUNE, run_name="__main__")
