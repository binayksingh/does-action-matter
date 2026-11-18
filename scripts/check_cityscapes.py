#!/usr/bin/env python3
"""Data and ego-signal checks for Cityscapes (no segmenter or GPU needed).

Run:  python scripts/check_cityscapes.py

Checks, written to OUT_DIR/check_cityscapes/:
  1. Completeness: every annotated train/val snippet has its gtFine label,
     the 7 protocol RGB frames, and vehicle JSON for every frame used.
  2. Ego-signal ranges: speed and yaw-rate statistics and histograms.
  3. UNIT check against GPS (the key one). Over each snippet:
       distance from GPS lat/lon   vs  integral of `speed` dt
       heading change from GPS     vs  integral of `yawRate` dt
     Fitted slope ~1 means the assumed units (m/s, rad/s) are right;
     ~3.6 would mean km/h, ~57 would mean deg/s; a negative heading slope
     means yawRate's sign is opposite to the left-positive convention.
  4. Visual check: for a few snippets with the strongest turns, frames
     2/11/20 next to the speed and yaw-rate curves. A left turn should
     show positive yaw rate if the sign convention holds.
  5. report.json with all numbers, plus a PASS/CHECK verdict per item.
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wamad import paths as P, protocol  # noqa: E402
from wamad.data import cityscapes as cs  # noqa: E402

# ============================================================ SETTINGS
CITYSCAPES_ROOT = str(P.CITYSCAPES_ROOT)   # set in wamad/paths.py
OUT_DIR = str(P.RUNS_DIR)                 # report and figures go to OUT_DIR/check_cityscapes/
SPLITS = ("train", "val")
N_VISUAL = 6                                          # snippets shown in the turn figure
MIN_SPEED_FOR_HEADING = 3.0                           # m/s; GPS heading is meaningless when nearly stopped
SEED = 0
# ======================================================================

FPS = protocol.CITYSCAPES_FPS
EARTH_R = 6_371_000.0


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(min(1.0, a)))


def read_snippet_vehicle(paths, s):
    """All 30 frames of a snippet that exist on disk -> dict of arrays."""
    rows = []
    for pos in range(1, protocol.CITYSCAPES_SNIPPET_LEN + 1):
        p = paths.vehicle(s, protocol.cityscapes_frame_id(s.annotated_frame, pos))
        if p.exists():
            d = json.loads(p.read_text())
            rows.append((pos, d))
    if len(rows) < 2:
        return None
    keys = rows[0][1].keys()
    out = {"pos": np.array([r[0] for r in rows])}
    for k in keys:
        try:
            out[k] = np.array([float(r[1][k]) for r in rows])
        except (TypeError, ValueError, KeyError):
            pass
    return out


def fit_slope(x, y):
    """Least-squares slope through the origin, and correlation."""
    x, y = np.asarray(x), np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10 or np.dot(x, x) == 0:
        return float("nan"), float("nan"), len(x)
    return float(np.dot(x, y) / np.dot(x, x)), float(np.corrcoef(x, y)[0, 1]), len(x)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, out = Path(CITYSCAPES_ROOT), Path(OUT_DIR) / "check_cityscapes"
    out.mkdir(parents=True, exist_ok=True)
    paths = cs.CityscapesPaths(root)
    report = {"root": str(root), "fps_assumed": FPS}

    # ---------------------------------------------------- 1. completeness
    samples = {sp: cs.build_index(root, sp) for sp in SPLITS}
    missing = defaultdict(list)
    for sp, ss in samples.items():
        for s in ss:
            for h in protocol.HORIZONS.values():
                for p in cs.check_sample_files(paths, s, h, need_rgb=True):
                    missing[sp].append(str(p.relative_to(root)))
    report["completeness"] = {
        sp: {"snippets": len(samples[sp]), "missing_files": len(set(missing[sp])),
             "examples": sorted(set(missing[sp]))[:10]} for sp in SPLITS}
    print("1. completeness:", {sp: (v["snippets"], v["missing_files"]) for sp, v in report["completeness"].items()})

    # -------------------------------------- 2-3. signal ranges + unit check
    all_speed, all_yaw, keys_seen = [], [], set()
    gps_dist, int_speed, gps_dhead, int_yaw = [], [], [], []
    per_snippet = []
    for sp, ss in samples.items():
        for s in ss:
            v = read_snippet_vehicle(paths, s)
            if v is None or "speed" not in v or "yawRate" not in v:
                continue
            keys_seen.update(v.keys())
            t = (v["pos"] - v["pos"][0]) / FPS
            all_speed.append(v["speed"])
            all_yaw.append(v["yawRate"])
            iy = float(np.trapezoid(v["yawRate"], t)) if hasattr(np, "trapezoid") else float(np.trapz(v["yawRate"], t))
            isp = float(np.trapezoid(v["speed"], t)) if hasattr(np, "trapezoid") else float(np.trapz(v["speed"], t))
            per_snippet.append((sp, s, iy, float(v["speed"].mean())))
            if {"gpsLatitude", "gpsLongitude"} <= v.keys():
                gps_dist.append(haversine(v["gpsLatitude"][0], v["gpsLongitude"][0],
                                          v["gpsLatitude"][-1], v["gpsLongitude"][-1]))
                int_speed.append(isp)
            if "gpsHeading" in v and v["speed"].min() >= MIN_SPEED_FOR_HEADING:
                dh = math.radians((v["gpsHeading"][-1] - v["gpsHeading"][0] + 180) % 360 - 180)
                # compass heading is clockwise; left turn = heading decreases
                gps_dhead.append(-dh)
                int_yaw.append(iy)

    sp_all, yw_all = np.concatenate(all_speed), np.concatenate(all_yaw)
    q = lambda a: {k: float(np.percentile(a, p)) for k, p in  # noqa: E731
                   (("p1", 1), ("p50", 50), ("p99", 99))} | {"min": float(a.min()), "max": float(a.max())}
    report["vehicle_keys"] = sorted(keys_seen)
    report["speed_stats"], report["yaw_rate_stats"] = q(sp_all), q(yw_all)
    print("2. speed:", {k: round(v, 3) for k, v in report["speed_stats"].items()})
    print("   yawRate:", {k: round(v, 4) for k, v in report["yaw_rate_stats"].items()})

    s_slope, s_r, s_n = fit_slope(int_speed, gps_dist)
    h_slope, h_r, h_n = fit_slope(int_yaw, gps_dhead)
    report["unit_check"] = {
        "speed_vs_gps_distance": {"slope": s_slope, "corr": s_r, "n": s_n,
                                  "verdict": "PASS (m/s)" if 0.8 < s_slope < 1.25 and s_r > 0.9 else "CHECK"},
        "yaw_vs_gps_heading": {"slope": h_slope, "corr": h_r, "n": h_n,
                               "verdict": ("PASS (rad/s, left-positive)" if 0.7 < h_slope < 1.4 and h_r > 0.7
                                           else "SIGN FLIPPED (use yaw_scale=-1)" if -1.4 < h_slope < -0.7 and h_r < -0.7
                                           else "CHECK")}}
    print("3. units:", json.dumps(report["unit_check"], indent=1))

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].hist(sp_all, bins=60); ax[0, 0].set_title("speed (all frames)"); ax[0, 0].set_xlabel("raw value")
    ax[0, 1].hist(yw_all, bins=80); ax[0, 1].set_title("yawRate (all frames)"); ax[0, 1].set_xlabel("raw value")
    ax[1, 0].scatter(int_speed, gps_dist, s=3, alpha=.4)
    m = max(max(int_speed, default=1), max(gps_dist, default=1))
    ax[1, 0].plot([0, m], [0, m], "k--", lw=.8)
    ax[1, 0].set_xlabel("\u222b speed dt"); ax[1, 0].set_ylabel("GPS distance [m]")
    ax[1, 0].set_title(f"slope {s_slope:.2f}, r {s_r:.2f}")
    ax[1, 1].scatter(int_yaw, gps_dhead, s=3, alpha=.4)
    lim = max(np.abs(int_yaw).max(initial=.1), np.abs(gps_dhead).max(initial=.1))
    ax[1, 1].plot([-lim, lim], [-lim, lim], "k--", lw=.8)
    ax[1, 1].set_xlabel("\u222b yawRate dt"); ax[1, 1].set_ylabel("GPS heading change [rad, left +]")
    ax[1, 1].set_title(f"slope {h_slope:.2f}, r {h_r:.2f}")
    fig.tight_layout(); fig.savefig(out / "units_check.png", dpi=130); plt.close(fig)

    # ------------------------------------------------------ 4. turn figure
    from PIL import Image
    rng = random.Random(SEED)
    moving = [x for x in per_snippet if x[3] > MIN_SPEED_FOR_HEADING]
    lefts = sorted(moving, key=lambda x: -x[2])[: max(1, N_VISUAL // 2) * 5]
    rights = sorted(moving, key=lambda x: x[2])[: max(1, N_VISUAL // 2) * 5]
    picks = rng.sample(lefts, min(len(lefts), N_VISUAL // 2)) + rng.sample(rights, min(len(rights), N_VISUAL - N_VISUAL // 2))
    if picks:
        fig, axes = plt.subplots(len(picks), 4, figsize=(16, 2.6 * len(picks)), squeeze=False)
        for r, (sp, s, iy, vm) in enumerate(picks):
            for c, pos in enumerate((2, 11, 20)):
                p = paths.rgb(s, protocol.cityscapes_frame_id(s.annotated_frame, pos))
                a = axes[r, c]; a.axis("off")
                if p.exists():
                    a.imshow(Image.open(p).convert("RGB").resize((512, 256)))
                a.set_title(f"{s.city}_{s.seq} pos {pos}", fontsize=7)
            v = read_snippet_vehicle(paths, s)
            a = axes[r, 3]; t = (v["pos"] - 1) / FPS
            a.plot(t, v["yawRate"], label="yawRate"); a.axhline(0, color=".6", lw=.6)
            a2 = a.twinx(); a2.plot(t, v["speed"], color="tab:orange", label="speed")
            a.set_title(f"\u222byaw = {iy:+.3f}  ({'LEFT?' if iy > 0 else 'RIGHT?'} if left-positive)", fontsize=8)
            a.set_xlabel("t [s]", fontsize=7)
        fig.tight_layout(); fig.savefig(out / "turn_check.png", dpi=110); plt.close(fig)

    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out / 'report.json'}, units_check.png, turn_check.png")
    print("Look at turn_check.png: rows labelled LEFT? should show the road curving left.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
