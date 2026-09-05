#!/usr/bin/env python
"""Export and plot the slot-LoRI / OPD training curves from one or more TensorBoard runs.

WHY THE PRIMARY CURVE IS NOT ``env/success_once``. That scalar is the share of ROLLOUT
TRAJECTORIES that succeeded, and the suites are NOT sampled in a fixed proportion: across four
consecutive steps of one run the libero_10 share alternated 40/24/40/24 trajectories out of 144.
libero_10 is the hardest suite, so a step that happens to draw fewer of them reads higher for a
reason that has nothing to do with the model -- measured on this project, a third of one step's
apparent +0.111 gain was that draw. The macro average over the four per-suite rates
(``succ_num_<suite> / succ_den_<suite>``) weights every suite equally whatever the draw was, so
it is mix-independent by construction. Both are exported; the macro average is what to read.

AND EVEN THAT IS AN IN-TRAINING NUMBER. It is measured on the student's own on-policy rollouts at
``temperature_train`` (1.0 here), not on a fixed evaluation set, and on this project in-training
per-suite rates have been off by +-0.33 in BOTH directions against a post-hoc greedy eval. Read
these curves for TRENDS and for liveness. The verdict comes from a post-hoc evaluation of the
final weights, never from here.

Usage
-----
    python plot_slot_curves.py                          # the default run set
    python plot_slot_curves.py --runs A=/path/to/run    # explicit, repeatable
    python plot_slot_curves.py --out /tmp/curves        # where the csv/png go

Reads TensorBoard event files only; safe to run against a LIVE training run.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: this box has no display and importing pyplot would fail on one
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)

SUITES = ["libero_10", "libero_goal", "libero_spatial", "libero_object"]

# Runs worth putting on one figure by default: the two mt4 controls this method has to beat, the
# first slot attempt (constant BBA, and its metrics are a MIXTURE of two configurations because a
# restart reused the step indices -- see the note it prints), and the clean annealed run.
DEFAULT_RUNS = [
    ("mt4 (control)", "/share/fanruochen-local/outputs/seqcl_mt4/tensorboard"),
    ("mt4w2 (control)", "/share/fanruochen-local/outputs/seqcl_mt4w2/tensorboard"),
    ("slot BBA (mixed cfg)", "/share/fanruochen-local/outputs/seqcl_mt4slot/tensorboard"),
    ("slot anneal", "/share/fanruochen-local/outputs/seqcl_mt4slotA/tensorboard"),
]


def load(path: str) -> Dict[str, Dict[int, float]]:
    """Every scalar in one run as ``{tag: {step: value}}``; ``{}`` when there is nothing yet."""
    if not os.path.isdir(path):
        return {}
    acc = EventAccumulator(path, size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    return {t: {s.step: s.value for s in acc.Scalars(t)} for t in tags}


def per_suite(series: Dict[str, Dict[int, float]], suite: str, step: int) -> Optional[float]:
    """That suite's success RATE at ``step``: successes over trajectories drawn from it.

    ``succ_den`` is the suite's SHARE of the rollout, so a suite absent from a step has a
    denominator of zero and no rate -- ``None``, never 0.0, which would read as "it failed
    everything" and drag the macro average down for a step that simply never tested it.
    """
    num = series.get(f"env/succ_num_{suite}", {}).get(step)
    den = series.get(f"env/succ_den_{suite}", {}).get(step)
    if num is None or den is None or den <= 0:
        return None
    return num / den


def macro_sr(series: Dict[str, Dict[int, float]], step: int) -> Optional[float]:
    """Mean of the per-suite rates: the mix-independent number. ``None`` unless ALL four exist.

    Requiring all four is the point. Averaging whichever suites happen to be present would make
    the curve jump whenever a suite drops out, which is exactly the artefact this metric exists
    to remove.
    """
    rates = [per_suite(series, s, step) for s in SUITES]
    if any(r is None for r in rates):
        return None
    return sum(rates) / len(rates)


def steps_of(series: Dict[str, Dict[int, float]]) -> List[int]:
    return sorted(series.get("env/success_once", {}))


def collect(runs: List[Tuple[str, str]]) -> Dict[str, Dict[str, Dict[int, float]]]:
    out = {}
    for name, path in runs:
        s = load(path)
        if not s:
            print(f"  {name:24s} no scalars yet at {path}")
            continue
        n = len(steps_of(s))
        has_slot = any(t.startswith("train/slot/") for t in s)
        print(f"  {name:24s} {n:3d} steps  slot_metrics={'yes' if has_slot else 'no'}")
        out[name] = s
    return out


def write_csv(data: Dict[str, Dict[str, Dict[int, float]]], path: str) -> int:
    """Long format -- one row per (run, step, metric). Wide would need a union of tag sets across
    runs that do not share one, and would silently pad the difference with blanks."""
    rows = 0
    with open(path, "w") as fh:
        fh.write("run,step,metric,value\n")
        for name, series in data.items():
            for step in steps_of(series):
                derived = {"macro_sr": macro_sr(series, step)}
                for suite in SUITES:
                    derived[f"sr_{suite}"] = per_suite(series, suite, step)
                for tag, value in derived.items():
                    if value is not None:
                        fh.write(f"{name},{step},{tag},{value:.6g}\n")
                        rows += 1
                for tag, by_step in series.items():
                    if step in by_step and not math.isnan(by_step[step]):
                        fh.write(f"{name},{step},{tag},{by_step[step]:.6g}\n")
                        rows += 1
    return rows


def _plot(ax, data, getter, title, ylabel):
    drew = False
    for name, series in data.items():
        xs, ys = [], []
        for step in steps_of(series):
            v = getter(series, step)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                xs.append(step)
                ys.append(v)
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, lw=1.4, label=name)
            drew = True
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return drew


def plot(data, out_png: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    tag = lambda t: (lambda s, k: s.get(t, {}).get(k))  # noqa: E731

    _plot(axes[0][0], data, lambda s, k: macro_sr(s, k),
          "macro-avg SR over 4 suites (mix-independent)", "SR")
    _plot(axes[0][1], data, tag("env/success_once"),
          "env/success_once (RAW -- confounded by suite mix)", "SR")
    _plot(axes[0][2], data, tag("train/actor/opd_kl_stu_tea"),
          "KL(teacher || student)", "opd_kl")
    _plot(axes[1][0], data, tag("train/actor/distill_loss"), "distill loss", "loss")

    # Per-slot ΔW growth, annealed run only: four lines from one run would be unreadable stacked
    # against four lines from another.
    ax = axes[1][1]
    slot_run = next((n for n in ("slot anneal", "slot BBA (mixed cfg)") if n in data), None)
    if slot_run:
        for i, suite in enumerate(SUITES):
            xs, ys = [], []
            for step in steps_of(data[slot_run]):
                v = data[slot_run].get(f"train/slot/dw_norm_{i}", {}).get(step)
                if v is not None:
                    xs.append(step)
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, marker="o", ms=3, lw=1.4, label=f"slot{i} {suite}")
        ax.set_title(f"‖ΔW_k‖_F per slot — {slot_run}", fontsize=10)
        ax.set_xlabel("training step")
        ax.set_ylabel("dw_norm")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    # phase_is_A against the CONFIGURED share: 0.0 is healthy under a pure-B anneal stage and a
    # dead mechanism under any other, and the two are indistinguishable without both lines.
    ax = axes[1][2]
    if slot_run:
        for t, lbl, st in [("train/slot/phase_is_A", "phase_is_A (realized)", "-"),
                           ("train/slot/alt_a_frac_cfg", "alt_a_frac_cfg (configured)", "--")]:
            xs = [k for k in steps_of(data[slot_run]) if k in data[slot_run].get(t, {})]
            if xs:
                ax.plot(xs, [data[slot_run][t][k] for k in xs], st, marker="o", ms=3,
                        lw=1.4, label=lbl)
        ax.set_title(f"A-phase share — {slot_run}", fontsize=10)
        ax.set_xlabel("training step")
        ax.set_ylabel("share of updates")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    for a in (axes[0][0], axes[0][1], axes[0][2], axes[1][0]):
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)

    fig.suptitle(
        "slot-LoRI OPD — IN-TRAINING metrics (on-policy rollouts at temperature 1.0). "
        "Trends only; the verdict comes from a post-hoc eval of the final weights.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=140)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=None, metavar="NAME=TB_DIR")
    p.add_argument("--out", default="/share/fanruochen-local/outputs/slot_curves")
    args = p.parse_args()

    runs = DEFAULT_RUNS
    if args.runs:
        runs = []
        for spec in args.runs:
            if "=" not in spec:
                raise SystemExit(f"--runs takes NAME=TB_DIR; got {spec!r}")
            name, path = spec.split("=", 1)
            runs.append((name, path))

    os.makedirs(args.out, exist_ok=True)
    print("runs:")
    data = collect(runs)
    if not data:
        raise SystemExit("no run had any scalars -- nothing to plot")

    csv_path = os.path.join(args.out, "curves.csv")
    png_path = os.path.join(args.out, "curves.png")
    rows = write_csv(data, csv_path)
    plot(data, png_path)
    print(f"\n  {csv_path}  ({rows} rows)")
    print(f"  {png_path}")


if __name__ == "__main__":
    main()
