#!/usr/bin/env python3
"""Paper figures from the result files (CPU only, a few seconds).

Run:  python scripts/make_figures.py
Writes PDF (for LaTeX) and PNG to OUT_DIR:

  fig_perclass_gain   per-class IoU change of FiLM over the no-action model on
                      Cityscapes, mid term, executed action, mean over seeds;
                      fig_perclass_gain.csv also lists the BDD100K values.
  fig_dependence      mid-term mIoU change of concat and FiLM relative to the
                      no-action model under each intervention, one panel per
                      dataset (trained and evaluated on the same dataset),
                      mean +/- std over seeds.

Only established metrics are plotted (mIoU and per-class IoU, %).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wamad import io, paths as P  # noqa: E402

# ============================================================ SETTINGS
FORECAST_DIR = P.RUNS_DIR / "forecast"
OUT_DIR = FORECAST_DIR / "figures"
METRIC = "mIoU"                  # or "MO-mIoU"
COLUMN_IN = 3.3                  # one column of a two-column paper
# ======================================================================
CLASS_NAMES = ("road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
               "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
               "motorcycle", "bicycle")
DS_SHORT = {"cityscapes": "Cityscapes", "bdd100k": "BDD100K"}
LABELS = {"film": "FiLM", "concat": "Concat", "none": "No action"}
COLORS = {"film": "#2a78d6", "concat": "#eb6834"}
IV_ORDER = ("true", "shuffle", "zero_yaw", "stop", "mean")
IV_LABELS = {"true": "true", "shuffle": "shuffled", "zero_yaw": "zero yaw", "stop": "stop", "mean": "mean"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white",
        "axes.facecolor": "white", "hatch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.dpi": 300})
    return plt


def load_rows():
    """All result rows; if an evaluation was repeated, the last row wins."""
    latest = {}
    for f in sorted(Path(FORECAST_DIR).glob("*/results.jsonl")):
        if f.parent.name in ("summary", "figures"):
            continue
        for r in io.load_jsonl(f):
            latest[(r["run_id"], r.get("seed"), r["train_set"], r["eval_set"], r["horizon"], r["model"], r["intervention"])] = r
    return list(latest.values())


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    print("wrote", OUT_DIR / f"{name}.pdf")


def fig_perclass(plt, horizon="mid", dataset="cityscapes"):
    """Per-class IoU change FiLM - no action (executed action, in-domain), mean over seeds."""
    rows = load_rows()
    per = defaultdict(list)          # (dataset, model) -> list of {class: IoU}
    for r in rows:
        if (r["intervention"] == "true" and r["horizon"] == horizon and r["train_set"] == r["eval_set"]
                and r["model"] in ("film", "none") and r.get("per_class_IoU")):
            per[(r["eval_set"], r["model"])].append(r["per_class_IoU"])
    sets = [d for d in ("cityscapes", "bdd100k") if per.get((d, "film")) and per.get((d, "none"))]
    if dataset not in sets:
        print("no per-class results, skipping fig_perclass_gain")
        return

    def mean_iou(ds, m, c):
        v = [x[c] for x in per[(ds, m)] if x.get(c) is not None]
        return 100 * float(np.mean(v)) if v else np.nan

    gain = {ds: np.array([mean_iou(ds, "film", c) - mean_iou(ds, "none", c) for c in CLASS_NAMES]) for ds in sets}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["class," + ",".join(f"gain_{ds}" for ds in sets)]
    lines += [c + "," + ",".join(f"{gain[ds][i]:.3f}" for ds in sets) for i, c in enumerate(CLASS_NAMES)]
    (OUT_DIR / "fig_perclass_gain.csv").write_text("\n".join(lines) + "\n")

    g = gain[dataset]
    with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.grid": False,
                         "savefig.bbox": None, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.35))
        y = np.arange(len(CLASS_NAMES))
        ax.barh(y, g, 0.72, color=COLORS["film"], edgecolor="white", linewidth=0.5)
        for i, v in enumerate(g):
            txt = "0.0" if abs(v) < 0.05 else f"{v:+.1f}".replace("-", "\u2212")
            ax.text(v + (0.15 if v >= 0 else -0.15), i, txt, va="center",
                    ha="left" if v >= 0 else "right", fontsize=5.5, color="#333333")
        ax.set_yticks(y, CLASS_NAMES)
        ax.invert_yaxis()
        ax.axvline(0, color="#555555", lw=0.6)
        ax.set_xlim(min(-1.8, g.min() - 1.0), g.max() + 1.7)
        ax.set_xlabel(f"IoU change, FiLM $-$ no action ({DS_SHORT[dataset]}, {horizon} term)")
        ax.grid(axis="x", color="#e5e5e5", lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        fig.tight_layout(pad=0.2)
        save(fig, "fig_perclass_gain")
        plt.close(fig)


def fig_dependence(plt, horizon="mid"):
    """Per-seed (model - mean no-action) under each intervention, in-domain, two stacked panels."""
    rows = load_rows()
    vals = defaultdict(list)          # (dataset, model, iv) -> per-seed mIoU (%)
    for r in rows:
        if r["horizon"] == horizon and r["train_set"] == r["eval_set"]:
            vals[(r["eval_set"], r["model"], r["intervention"])].append(100 * r[METRIC])
    sets = [d for d in ("cityscapes", "bdd100k")
            if vals.get((d, "none", "true")) and (vals.get((d, "film", "true")) or vals.get((d, "concat", "true")))]
    if not sets:
        print("no in-domain results yet, skipping fig_dependence")
        return
    models = ("concat", "film")
    fig, axes = plt.subplots(len(sets), 1, figsize=(COLUMN_IN, 1.55 * len(sets) + 0.55), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(IV_ORDER))
    w = 0.38
    lo = 0.0
    for ax, ds in zip(axes, sets):
        ref = float(np.mean(vals[(ds, "none", "true")]))
        seeds = {"none": len(vals[(ds, "none", "true")])}
        for j, m in enumerate(models):
            d = [np.array(vals.get((ds, m, iv), [])) - ref for iv in IV_ORDER]
            if not len(d[0]):
                continue
            seeds[m] = len(d[0])
            mean = np.array([v.mean() if len(v) else np.nan for v in d])
            sd = np.array([v.std(ddof=1) if len(v) > 1 else 0.0 for v in d])
            pos = x + (j - 0.5) * w
            ax.bar(pos, mean, w * 0.92, yerr=sd, color=COLORS[m], edgecolor=SURFACE, linewidth=0.6,
                   label=LABELS[m].split(" (")[0],
                   error_kw={"elinewidth": 0.6, "capsize": 1.2, "ecolor": INK})
            for i in (0, 1):            # label the true and shuffled bars only
                if np.isnan(mean[i]):
                    continue
                y = mean[i] + (sd[i] if mean[i] >= 0 else -sd[i])
                ax.text(pos[i], y, f"{mean[i]:+.1f}".replace("-", "\u2212"), ha="center",
                        va="bottom" if mean[i] >= 0 else "top", fontsize=6, color=INK)
            lo = min(lo, np.nanmin(mean - sd))
        ax.axhline(0, color=INK2, lw=0.8, zorder=0.5)
        ax.set_title(DS_SHORT[ds], loc="left", color=INK, pad=2)
        ax.set_ylabel("\u0394mIoU vs. no action")
        print(f"fig_dependence {DS_SHORT[ds]}: seeds per model {seeds}")
    axes[0].set_ylim(lo * 1.18, 6)
    axes[-1].set_xticks(x, [IV_LABELS[i] for i in IV_ORDER])
    axes[0].legend(frameon=False, ncol=2, loc="lower right", fontsize=7)
    fig.tight_layout(h_pad=0.4)
    save(fig, "fig_dependence")
    plt.close(fig)


def main() -> int:
    plt = style()
    fig_perclass(plt)
    fig_dependence(plt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
