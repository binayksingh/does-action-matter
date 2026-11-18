#!/usr/bin/env python3
"""Collect every result and write the paper tables (LaTeX) plus check files.

Run any time (CPU, seconds):  python scripts/summarize_results.py

Reads
  RUNS_DIR/forecast/*/results.jsonl     trained models (one folder per run) and baselines
  RUNS_DIR/forecast/efficiency.json     from scripts/efficiency.py        (optional)
  RUNS_DIR/synthetic/report.json     from scripts/synthetic.py        (optional)
Writes RUNS_DIR/forecast/summary/
  completeness.txt        which of the 18 runs x 20 evaluation conditions are present
  all_results.csv         one row per group: mean, std, n over seeds (mIoU, MO-mIoU, pixel acc.)
  summary.json            the same groups + every derived number quoted in the paper text
  check.md                all tables in plain text, for reading before using the .tex files
  tab_main.tex            baselines, published results, none / concat / FiLM
  tab_interventions.tex   mid-term mIoU under each action intervention (in-domain)
  tab_interventions_short.tex  the same at short term
  tab_cross.tex           train on one dataset, evaluate on each (mid-term)
  tab_synthetic.tex       synthetic validation
  tab_efficiency.tex      model size and inference cost

Numbers are never typed by hand: every cell comes from the result files, except the
published Cityscapes results in PUBLISHED (copied from the original papers and checked).
Means and standard deviations are over seeds (sample std, ddof = 1). Changes under an
intervention are computed per seed (intervention minus true action of the same model)
and then averaged. Only established metrics appear: mIoU and MO-mIoU in %.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wamad import io, paths as P  # noqa: E402

# ============================================================ SETTINGS
FORECAST_DIR = P.RUNS_DIR / "forecast"
OUT_DIR = FORECAST_DIR / "summary"
EFFICIENCY_JSON = FORECAST_DIR / "efficiency.json"
SYNTHETIC_REPORT = P.RUNS_DIR / "synthetic" / "report.json"
TRAIN_SETS = ("cityscapes", "bdd100k")
MODES = ("none", "concat", "film")
SEEDS = (0, 1, 2)
HORIZONS = ("short", "mid")
INTERVENTIONS = ("true", "shuffle", "zero_yaw", "stop", "mean")
# Published Cityscapes val results on the same protocol, taken from the original papers
# (label, cite key, short mIoU, short MO, mid mIoU, mid MO); None = not reported.
PUBLISHED = [
    ("S2S-dil", "luc2017predicting", 59.4, 55.3, 47.8, 40.8),
    ("Bayes-WD-SL", "bhattacharyya2019bayesian", 64.8, None, 45.9, None),
    ("F2MF-DN121", "saric2021dense", 69.6, 67.7, 57.9, 54.6),
]
# ======================================================================

NAMES = {"copy_last": "Copy-last", "flow_warp": "Flow warping", "none": "No action",
         "concat": "Concat", "film": "FiLM", "film_blind": "FiLM (shuffled training)"}
DS = {"cityscapes": "Cityscapes", "bdd100k": "BDD100K"}
IV_TEX = {"shuffle": "Shuffled", "zero_yaw": "Zero yaw", "stop": "Stop", "mean": "Mean"}


# ------------------------------------------------------------------ loading
def load_rows() -> list[dict]:
    """All result rows; if an evaluation was repeated, the last row wins."""
    latest = {}
    for f in sorted(FORECAST_DIR.glob("*/results.jsonl")):
        if f.parent.name in ("summary", "figures"):
            continue
        for r in io.load_jsonl(f):
            latest[(r["run_id"], r.get("seed"), r["train_set"], r["eval_set"], r["horizon"], r["model"], r["intervention"])] = r
    return list(latest.values())


def completeness(rows) -> list[str]:
    have = defaultdict(set)
    for r in rows:
        have[r["run_id"]].add((r["eval_set"], r["horizon"], r["intervention"]))
    need = {(e, h, iv) for e in TRAIN_SETS for h in HORIZONS for iv in INTERVENTIONS}
    lines, n_done = [], 0
    for ts in TRAIN_SETS:
        for m in MODES:
            for s in SEEDS:
                run = f"{ts}_{m}_s{s}"
                missing = need - have.get(run, set())
                n_done += not missing
                lines.append(f"{'OK     ' if not missing else 'MISSING'} {run:22s} "
                             f"{len(need) - len(missing):2d}/{len(need)} conditions")
    lines.insert(0, f"{n_done} of {len(TRAIN_SETS) * len(MODES) * len(SEEDS)} runs complete\n")
    return lines


# ------------------------------------------------------------------ statistics
def per_seed(rows, key="mIoU"):
    """{(train, eval, horizon, model, intervention): {seed: value in %}}"""
    d = defaultdict(dict)
    for r in rows:
        d[(r["train_set"], r["eval_set"], r["horizon"], r["model"], r["intervention"])][r.get("seed")] = 100 * r[key]
    return d


def ms(values):
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return None
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}


def stat(ps, ts, es, h, m, iv="true"):
    if m in ("copy_last", "flow_warp"):
        ts = "-"
    return ms(ps.get((ts, es, h, m, iv), {}).values())


def delta(ps, ts, es, h, m, iv):
    """Per-seed change iv - true of the same model, then mean / std over seeds."""
    a, b = ps.get((ts, es, h, m, iv), {}), ps.get((ts, es, h, m, "true"), {})
    return ms(a[s] - b[s] for s in a if s in b)


# ------------------------------------------------------------------ formatting
def num(x, d=2, sign=False):
    if x is None:
        return "--"
    if sign and abs(x) < 0.5 * 10 ** -d:
        x, sign = 0.0, False                      # an exact zero change is printed as 0.00
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def pm(st, d=2, bold=False):
    if st is None:
        return "--"
    v = num(st["mean"], d)
    v = rf"\textbf{{{v}}}" if bold else v
    return v + (rf"{{\scriptsize$\pm${st['std']:.{d}f}}}" if st["n"] > 1 else "")


def best(stats):
    vals = [s["mean"] for s in stats if s is not None]
    return max(vals) if vals else None


def is_best(st, b):
    return st is not None and b is not None and abs(st["mean"] - b) < 1e-9


# ------------------------------------------------------------------ tables
def tab_main(ps, pmo, out):
    """Main table: in-domain results, true action; baselines, published (Cityscapes only), ours."""
    cols = [(es, h, k) for es in TRAIN_SETS for h in HORIZONS for k in ("mIoU", "MO")]
    rows_ours = ["copy_last", "flow_warp", "none", "concat", "film"]
    cell = {}
    for m in rows_ours:
        for es, h, k in cols:
            cell[(m, es, h, k)] = stat(ps if k == "mIoU" else pmo, es, es, h, m)
    bests = {c: best(cell[(m, *c)] for m in rows_ours) for c in cols}
    L = [r"\begin{table*}[t]", r"  \centering", r"  \small", r"  \setlength{\tabcolsep}{4pt}",
         r"  \begin{tabular}{@{}l cccc cccc@{}}", r"    \toprule",
         r"    & \multicolumn{4}{c}{Cityscapes (gtFine)} & \multicolumn{4}{c}{BDD100K (pseudo-labels)} \\",
         r"    \cmidrule(lr){2-5}\cmidrule(l){6-9}",
         r"    & \multicolumn{2}{c}{Short term} & \multicolumn{2}{c}{Mid term} & \multicolumn{2}{c}{Short term} & \multicolumn{2}{c}{Mid term} \\",
         r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
         r"    Method & mIoU$\uparrow$ & MO$\uparrow$ & mIoU$\uparrow$ & MO$\uparrow$ & mIoU$\uparrow$ & MO$\uparrow$ & mIoU$\uparrow$ & MO$\uparrow$ \\",
         r"    \midrule"]

    def row(label, m):
        return "    " + label + " & " + " & ".join(pm(cell[(m, *c)], bold=is_best(cell[(m, *c)], bests[c])) for c in cols) + r" \\"

    L += [row(NAMES[m], m) for m in ("copy_last", "flow_warp")]
    L.append(r"    \midrule")
    for label, key, s, smo, mi, mmo in PUBLISHED:
        L.append(f"    {label}~\\cite{{{key}}}$^\\dagger$ & {num(s, 1)} & {num(smo, 1)} & {num(mi, 1)} & {num(mmo, 1)} & -- & -- & -- & -- \\\\")
    L.append(r"    \midrule")
    L += [row(NAMES[m], m) for m in ("none", "concat", "film")]
    n = max((st["n"] for st in cell.values() if st and st["n"] > 1), default=1)
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \caption{Semantic forecasting results with the executed action (mIoU and MO-mIoU in \%; "
          rf"mean $\pm$ std over {n} seeds). Models are trained and evaluated on the same dataset. "
          r"$^\dagger$Published results use weaker single-frame segmenters than ours and are given for "
          r"reference only. Best value per column among methods run in our pipeline in bold.}",
          r"  \label{tab:main}", r"\end{table*}"]
    out.write_text("\n".join(L) + "\n")


def tab_interventions(ps, out, horizon="mid"):
    """Interventions table (mid or short term): in-domain, true-action mIoU and the per-seed change under each intervention."""
    L = [r"\begin{table}[t]", r"  \centering", r"  \footnotesize", r"  \setlength{\tabcolsep}{2.1pt}",
         r"  \begin{tabular}{@{}ll c cccc@{}}", r"    \toprule",
         r"    & & True & \multicolumn{4}{c}{Change under intervention} \\", r"    \cmidrule(l){4-7}",
         r"    Train / eval & Model & mIoU$\uparrow$ & " + " & ".join(IV_TEX[i] for i in INTERVENTIONS[1:]) + r" \\",
         r"    \midrule"]
    max_sd = 0.0
    for k, ts in enumerate(TRAIN_SETS):
        for j, m in enumerate(MODES):
            t = stat(ps, ts, ts, horizon, m)
            ds = [delta(ps, ts, ts, horizon, m, iv) for iv in INTERVENTIONS[1:]]
            max_sd = max([max_sd] + [d["std"] for d in ds if d])
            lab = rf"\multirow{{3}}{{*}}{{{DS[ts]}}}" if j == 0 else ""
            L.append(f"    {lab} & {NAMES[m]} & {pm(t)} & " + " & ".join(num(d['mean'] if d else None, 2, True) for d in ds) + r" \\")
        if k < len(TRAIN_SETS) - 1:
            L.append(r"    \midrule")
    L += [r"    \bottomrule", r"  \end{tabular}"]
    if horizon == "mid":
        L += [rf"  \caption{{Mid-term mIoU (\%) with the executed action and its change when the future "
              rf"action is replaced (shuffled: another sequence's action; zero yaw; stop; training-set mean). "
              rf"Changes are averaged over seeds; their standard deviation is at most {max_sd:.2f}.}}",
              r"  \label{tab:interventions}", r"\end{table}"]
    else:
        L += [r"  \caption{Short-term mIoU (\%) with the executed action and its change under the",
              r"    interventions of \cref{tab:interventions}, averaged over seeds.}",
              r"  \label{tab:interventions_short}", r"\end{table}"]
    out.write_text("\n".join(L) + "\n")


def tab_cross(ps, out, horizon="mid"):
    """Cross-platform table: every train -> eval pair, true-action mIoU and change under the shuffled action."""
    L = [r"\begin{table}[t]", r"  \centering", r"  \footnotesize", r"  \setlength{\tabcolsep}{4pt}",
         r"  \begin{tabular}{@{}ll cc cc@{}}", r"    \toprule",
         r"    & & \multicolumn{2}{c}{Eval: Cityscapes} & \multicolumn{2}{c}{Eval: BDD100K} \\",
         r"    \cmidrule(lr){3-4}\cmidrule(l){5-6}",
         r"    Train & Model & True & Shuffled & True & Shuffled \\", r"    \midrule"]
    for k, ts in enumerate(TRAIN_SETS):
        for j, m in enumerate(MODES):
            lab = rf"\multirow{{3}}{{*}}{{{DS[ts]}}}" if j == 0 else ""
            cells = []
            for es in TRAIN_SETS:
                d = delta(ps, ts, es, horizon, m, "shuffle")
                cells += [pm(stat(ps, ts, es, horizon, m)), num(d["mean"] if d else None, 2, True)]
            L.append(f"    {lab} & {NAMES[m]} & " + " & ".join(cells) + r" \\")
        if k < len(TRAIN_SETS) - 1:
            L.append(r"    \midrule")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \caption{Transfer between platforms: mid-term mIoU (\%) with the executed action and its "
          r"change under the shuffled action, for every training and evaluation dataset.}",
          r"  \label{tab:cross}", r"\end{table}"]
    out.write_text("\n".join(L) + "\n")


def tab_synthetic(out) -> dict | None:
    """Synthetic world, mid-term, single run."""
    if not SYNTHETIC_REPORT.exists():
        print(f"(no {SYNTHETIC_REPORT}; synthetic table skipped)")
        return None
    tab = json.loads(SYNTHETIC_REPORT.read_text())["table"]
    get = lambda m, iv: 100 * tab[f"{m}|mid|{iv}"]["mIoU"] if f"{m}|mid|{iv}" in tab else None  # noqa: E731
    L = [r"\begin{table}[t]", r"  \centering", r"  \footnotesize", r"  \setlength{\tabcolsep}{2.6pt}",
         r"  \begin{tabular}{@{}l c cccc@{}}", r"    \toprule",
         r"    & True & \multicolumn{4}{c}{Change under intervention} \\", r"    \cmidrule(l){3-6}",
         r"    Model & mIoU$\uparrow$ & " + " & ".join(IV_TEX[i] for i in INTERVENTIONS[1:]) + r" \\", r"    \midrule"]
    out_vals = {}
    for m in ("copy_last", "flow_warp", "none", "concat", "film", "film_blind"):
        t = get(m, "true")
        if t is None:
            continue
        ds = [(get(m, iv) - t) if get(m, iv) is not None else None for iv in INTERVENTIONS[1:]]
        out_vals[m] = {"true": t, **{iv: d for iv, d in zip(INTERVENTIONS[1:], ds)}}
        L.append(f"    {NAMES[m]} & {num(t)} & " + " & ".join(num(d, 2, True) for d in ds) + r" \\")
        if m == "flow_warp":
            L.append(r"    \midrule")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \caption{Synthetic validation environment: mid-term mIoU (\%) with the executed action and "
          r"its change under each intervention (one run per model). Baselines do not take an action.}",
          r"  \label{tab:synthetic}", r"\end{table}"]
    out.write_text("\n".join(L) + "\n")
    return out_vals


def tab_efficiency(ps, out) -> dict | None:
    """Efficiency table: efficiency.json + Cityscapes mid-term true-action mIoU and shuffled change."""
    if not EFFICIENCY_JSON.exists():
        print(f"(no {EFFICIENCY_JSON}; efficiency table skipped)")
        return None
    eff = json.loads(EFFICIENCY_JSON.read_text())
    mods = eff["models"]
    t = {m: stat(ps, "cityscapes", "cityscapes", "mid", m) for m in MODES}
    d = {m: delta(ps, "cityscapes", "cityscapes", "mid", m, "shuffle") for m in MODES}
    def fmt(m, key, d):
        v = mods[m][key]
        s = f"{v:.{d}f}"
        lowest = min(mods[x][key] for x in MODES)
        return r"\textbf{" + s + "}" if abs(v - lowest) < 1e-9 else s

    L = [r"\begin{table}[t]", r"  \centering", r"  \footnotesize", r"  \setlength{\tabcolsep}{2.3pt}",
         r"  \begin{tabular}{@{}lcccccc@{}}", r"    \toprule",
         r"    Method & Params & $\Delta$Params & GFLOPs & Latency (ms)$\downarrow$ & mIoU$\uparrow$ & $\Delta_\text{shuf}$ \\",
         r"           & (M)$\downarrow$ & (M) & $\downarrow$ & 1 / 3 steps & mid & mid \\", r"    \midrule"]
    bt = best(t[m] for m in MODES)
    for m in MODES:
        dparams = "--" if m == "none" else "+%.2f" % mods[m]["dparams_M"]
        dshuf = "0.00" if m == "none" else num(d[m]["mean"] if d[m] else None, 2, True)
        L.append("    " + " & ".join([NAMES[m], fmt(m, "params_M", 2), dparams, fmt(m, "gflops_step", 1),
                                      fmt(m, "latency_step_ms", 1) + " / " + fmt(m, "latency_rollout3_ms", 1),
                                      (r"\textbf{%.2f}" if is_best(t[m], bt) else "%.2f") % t[m]["mean"], dshuf]) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}",
          rf"  \caption{{Model size and inference cost at ${eff['size_hw'][0]}{{\times}}{eff['size_hw'][1]}$ "
          rf"({eff['precision']}, batch {eff['batch']}, one {eff['gpu'].replace('Tesla ', 'NVIDIA ')}). "
          r"$\Delta$Params is relative to the no-action model; GFLOPs count convolutions and matrix "
          r"products of one forecasting step. mIoU and $\Delta_\text{shuf}$ (change under the shuffled "
          r"action) are mid-term values on Cityscapes validation.}",
          r"  \label{tab:efficiency}", r"\end{table}"]
    out.write_text("\n".join(L) + "\n")
    return {m: {**mods[m]} for m in MODES}


# ------------------------------------------------------------------ main
def main() -> int:
    rows = load_rows()
    if not rows:
        print(f"no results under {FORECAST_DIR}")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comp = completeness(rows)
    (OUT_DIR / "completeness.txt").write_text("\n".join(comp) + "\n")
    print("\n".join(comp[:1]))

    ps, pmo, ppa = per_seed(rows, "mIoU"), per_seed(rows, "MO-mIoU"), per_seed(rows, "pixel_acc")
    groups = {}
    with (OUT_DIR / "all_results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_set", "eval_set", "horizon", "model", "intervention", "n_seeds", "seeds",
                    "mIoU", "mIoU_std", "MO-mIoU", "MO-mIoU_std", "pixel_acc", "pixel_acc_std"])
        for k in sorted(ps, key=lambda x: tuple(str(v) for v in x)):
            a, b, c = ms(ps[k].values()), ms(pmo[k].values()), ms(ppa[k].values())
            seeds = sorted(s for s in ps[k] if s is not None)
            groups["|".join(map(str, k))] = {"mIoU": a, "MO-mIoU": b, "pixel_acc": c, "seeds": seeds}
            w.writerow([*k, a["n"], " ".join(map(str, seeds)), f"{a['mean']:.3f}", f"{a['std']:.3f}",
                        f"{b['mean']:.3f}", f"{b['std']:.3f}", f"{c['mean']:.3f}", f"{c['std']:.3f}"])

    tab_main(ps, pmo, OUT_DIR / "tab_main.tex")
    tab_interventions(ps, OUT_DIR / "tab_interventions.tex")
    tab_interventions(ps, OUT_DIR / "tab_interventions_short.tex", horizon="short")
    tab_cross(ps, OUT_DIR / "tab_cross.tex")
    syn = tab_synthetic(OUT_DIR / "tab_synthetic.tex")
    eff = tab_efficiency(ps, OUT_DIR / "tab_efficiency.tex")

    # every derived number the text may quote, so it can be checked against the tables
    text = {}
    for ts in TRAIN_SETS:
        for h in HORIZONS:
            no = stat(ps, ts, ts, h, "none")
            for m in ("concat", "film"):
                a = stat(ps, ts, ts, h, m)
                if a and no:
                    text[f"gain_{m}_vs_none|{ts}|{h}"] = a["mean"] - no["mean"]
                for iv in INTERVENTIONS[1:]:
                    d = delta(ps, ts, ts, h, m, iv)
                    if d:
                        text[f"change_{iv}|{m}|{ts}|{h}"] = d
            fl = stat(ps, "-", ts, h, "flow_warp")
            for m in MODES:
                a = stat(ps, ts, ts, h, m)
                if a and fl:
                    text[f"over_flow|{m}|{ts}|{h}"] = a["mean"] - fl["mean"]
    json.dump({"groups": groups, "text_numbers": text, "synthetic": syn, "efficiency": eff},
              (OUT_DIR / "summary.json").open("w"), indent=1)

    md = ["# Check file: all paper tables in plain text", ""]
    for f in ("tab_main", "tab_interventions", "tab_interventions_short", "tab_cross", "tab_synthetic", "tab_efficiency"):
        p = OUT_DIR / f"{f}.tex"
        if p.exists():
            md += [f"## {f}", "```", p.read_text(), "```", ""]
    md += ["## Derived numbers for the text", ""]
    for k, v in sorted(text.items()):
        md.append(f"- {k}: " + (f"{v:+.2f}" if isinstance(v, float) else f"{v['mean']:+.2f} +/- {v['std']:.2f} (n={v['n']})"))
    (OUT_DIR / "check.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} result rows -> {OUT_DIR}")
    print("tables:", ", ".join(sorted(p.name for p in OUT_DIR.glob("tab_*.tex"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())