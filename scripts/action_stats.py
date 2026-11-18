#!/usr/bin/env python3
"""Distribution of the ego actions on each validation set (CPU only, reads no images or labels).

Purpose: interpret the intervention results. For example, the zero-yaw condition can only
change a prediction where the true yaw rate is not already close to zero, and the shuffled
condition only where another sequence's action differs from the true one.

For Cityscapes and BDD100K val (and train, for reference), mid-term protocol, it reports:
  speed  v [m/s]          : mean, std, percentiles, fraction stopped (v < STOP_SPEED)
  |yaw rate| [rad/s]      : mean, percentiles, fraction near zero (|w| < STRAIGHT_YAW)
  |true - shuffled| action: mean absolute change in v and w under the shuffle condition
  |true - zero_yaw|       : = |w|; |true - stop| : = (v, |w|)

Edit SETTINGS, then:  python scripts/action_stats.py
Output: printed table and RUNS_DIR/forecast/action_stats.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wamad import paths as P  # noqa: E402
from wamad.data.forecast_data import ForecastSet  # noqa: E402

# ============================================================ SETTINGS
CITYSCAPES_ROOT = P.CITYSCAPES_ROOT
BDD_ROOT = P.BDD_ROOT
PSEUDO_ROOT = P.PSEUDO_ROOT
BDD_INDEX_DIR = P.BDD_INDEX_DIR
OUT_FILE = P.RUNS_DIR / "forecast" / "action_stats.json"
SETS = [("cityscapes", "val"), ("bdd100k", "val"), ("cityscapes", "train"), ("bdd100k", "train")]
HORIZON = "mid"
STOP_SPEED = 0.5      # m/s
STRAIGHT_YAW = 0.05   # rad/s  (~2.9 deg/s)
SHUFFLE_SEED = 0      # same seed as the shuffle evaluation in train.py
# ======================================================================


def stats(dataset: str, split: str) -> dict:
    ds = ForecastSet(dataset, split, HORIZON, train=False, cityscapes_root=CITYSCAPES_ROOT,
                     bdd_root=BDD_ROOT, pseudo_root=PSEUDO_ROOT, bdd_index_dir=BDD_INDEX_DIR,
                     shuffle_actions=True, shuffle_seed=SHUFFLE_SEED)
    fut = np.stack([ds.future_actions(i) for i in range(len(ds))])      # (N, steps, 2)
    shf = fut[ds.perm]                                                  # shuffled future actions
    v, w = fut[..., 0].ravel(), np.abs(fut[..., 1]).ravel()
    pct = lambda x: {f"p{q}": float(np.percentile(x, q)) for q in (10, 50, 90, 99)}  # noqa: E731
    return {
        "n_samples": int(len(ds)), "n_future_steps": int(fut.shape[1]),
        "speed": {"mean": float(v.mean()), "std": float(v.std()), **pct(v),
                  "frac_stopped": float((v < STOP_SPEED).mean())},
        "abs_yaw": {"mean": float(w.mean()), **pct(w),
                    "frac_straight": float((w < STRAIGHT_YAW).mean())},
        "shuffle_abs_change": {"speed": float(np.abs(fut[..., 0] - shf[..., 0]).mean()),
                               "yaw": float(np.abs(fut[..., 1] - shf[..., 1]).mean())},
    }


def main() -> int:
    out = {}
    for dataset, split in SETS:
        try:
            out[f"{dataset}_{split}"] = s = stats(dataset, split)
        except FileNotFoundError as e:
            print(f"skip {dataset} {split}: {e}")
            continue
        sp, yw, sh = s["speed"], s["abs_yaw"], s["shuffle_abs_change"]
        print(f"\n{dataset} {split} ({s['n_samples']} samples x {s['n_future_steps']} future steps)")
        print(f"  speed  mean {sp['mean']:.2f} m/s  std {sp['std']:.2f}  p50 {sp['p50']:.2f}  p90 {sp['p90']:.2f}"
              f"  stopped (<{STOP_SPEED}) {100 * sp['frac_stopped']:.1f}%")
        print(f"  |yaw|  mean {yw['mean']:.3f} rad/s  p50 {yw['p50']:.3f}  p90 {yw['p90']:.3f}  p99 {yw['p99']:.3f}"
              f"  straight (<{STRAIGHT_YAW}) {100 * yw['frac_straight']:.1f}%")
        print(f"  shuffle changes the action by  |dv| {sh['speed']:.2f} m/s  |dw| {sh['yaw']:.3f} rad/s on average")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
