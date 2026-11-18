#!/usr/bin/env python3
"""Build the BDD100K sample index and check frames and ego signals (no segmenter or GPU).

Run:  python scripts/prepare_bdd100k.py

Writes to BDD_INDEX_DIR (default RUNS_DIR/bdd100k_index/):
  1. Frames: per-sequence frame count, gaps in numbering, and the frame rate
     implied by (frame count / GPS duration). Expect ~5 fps.
  2. GPS quality: sequences rejected by bdd_signal_quality and why.
  3. Units: integral of GPS `speed` dt vs distance from GPS lat/lon
     (slope ~1 => m/s). Also correlation of the course-derived yaw rate
     with each gyro axis (shows which axis is yaw; informational only).
  4. TIME ALIGNMENT (key one): for a sample of val sequences, image motion
     (mean absolute difference of consecutive downsampled frames) is
     cross-correlated with GPS speed over lags -3..+3 s. A peak near 0 s
     means frame i is at (i-1)/5 s after startTime. A consistent peak at
     L s means frames are L s later on the GPS clock: pass
     time_offset_s=L to bt.build_index.
  5. Sample index for both horizons and both splits, saved as JSONL
     (used by all later scripts), plus a turn figure.
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wamad import io, paths as P, protocol  # noqa: E402
from wamad.data import bdd100k_track as bt  # noqa: E402
from wamad.ego import bdd_signal_quality, load_bdd_info  # noqa: E402

# ============================================================ SETTINGS
BDD_ROOT = str(P.BDD_ROOT)          # set in wamad/paths.py
OUT_DIR = str(P.BDD_INDEX_DIR)
SPLITS = ("train", "val")
STRIDE = 10                 # one scored frame every 10 frames (2 s) per sequence
N_ALIGN = 40                # val sequences used for the alignment check
N_VISUAL = 6
TIME_OFFSET_S = 0.0         # set from the alignment check if it reports a lag, then rerun
SEED = 0
# ======================================================================

FPS = protocol.BDD_TRACK_FPS


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6_371_000.0 * math.asin(math.sqrt(min(1.0, a)))


def trapz(y, x):
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def motion_signal(paths, split, seq, frames, w=96, h=54):
    """Mean |frame_i - frame_{i-1}| on small grayscale frames; time of each value."""
    from PIL import Image
    prev, vals, ts = None, [], []
    for i in frames:
        g = np.asarray(Image.open(paths.frame(split, seq, i)).convert("L").resize((w, h)), dtype=np.float32)
        # ignore the bottom 20% (dashboard / hood reflections)
        g = g[: int(h * 0.8)]
        if prev is not None:
            vals.append(float(np.abs(g - prev).mean()))
            ts.append(bt.frame_time(i, FPS, TIME_OFFSET_S) - 0.5 / FPS)
        prev = g
    return np.array(ts), np.array(vals)


def best_lag(ts, motion, sig, lags):
    """Correlation of motion with speed shifted by each lag (speed evaluated at t + lag)."""
    out = []
    for L in lags:
        sp = sig.at(ts + L)[:, 0]
        ok = np.isfinite(sp)
        if ok.sum() < 30 or np.std(sp[ok]) < 0.3 or np.std(motion[ok]) == 0:
            out.append(np.nan)
            continue
        out.append(float(np.corrcoef(motion[ok], sp[ok])[0, 1]))
    return np.array(out)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    paths = bt.TrackPaths(BDD_ROOT)
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    report = {"root": BDD_ROOT, "fps_assumed": FPS}

    # ----------------------------------------- 1-3. frames, GPS, units
    frames_by, sig_by, reasons = {}, {}, Counter()
    fps_est, gaps, int_speed, gps_dist = [], 0, [], []
    gyro_corr = {"x": [], "y": [], "z": []}
    for sp in SPLITS:
        for seq in bt.list_sequences(paths, sp):
            fr = bt.frame_indices(paths, sp, seq)
            frames_by[(sp, seq)] = fr
            gaps += int(fr and (fr[-1] - fr[0] + 1 != len(fr)))
            ip = paths.info(seq)
            if ip is None:
                reasons["no_info_file"] += 1
                continue
            d = json.loads(Path(ip).read_text())
            try:
                sig = load_bdd_info(ip)
            except (ValueError, KeyError):
                reasons["info_error"] += 1
                continue
            q = bdd_signal_quality(sig)
            for r in q["reasons"]:
                reasons[r] += 1
            if not q["ok"]:
                continue
            sig_by[(sp, seq)] = sig
            dur = (d["endTime"] - d["startTime"]) / 1000 if "endTime" in d else sig.duration
            if dur > 0:
                fps_est.append(len(fr) / dur)
            locs = [l for l in d["locations"] if float(l.get("speed", -1)) >= 0]
            if len(locs) >= 2:
                dist = sum(haversine(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
                           for a, b in zip(locs[:-1], locs[1:]))
                gps_dist.append(dist)
                int_speed.append(trapz(sig.speed, sig.t))
            g = d.get("gyro") or []
            if len(g) > 20 and np.std(sig.yaw_rate) > 0.02:
                tg = (np.array([s["timestamp"] for s in g], float) - d["startTime"]) / 1000
                yaw_at_g = sig.at(tg)[:, 1]
                ok = np.isfinite(yaw_at_g)
                for ax in "xyz":
                    a = np.array([s[ax] for s in g], float)[ok]
                    if np.std(a) > 0:
                        gyro_corr[ax].append(float(np.corrcoef(a, yaw_at_g[ok])[0, 1]))

    fps_est = np.array(fps_est)
    slope = float(np.dot(int_speed, gps_dist) / np.dot(int_speed, int_speed)) if int_speed else float("nan")
    corr = float(np.corrcoef(int_speed, gps_dist)[0, 1]) if len(int_speed) > 2 else float("nan")
    report["frames"] = {"sequences": len(frames_by), "with_numbering_gaps": gaps,
                        "frames_per_seq_median": float(np.median([len(v) for v in frames_by.values()])),
                        "implied_fps_p5_p50_p95": [float(np.percentile(fps_est, p)) for p in (5, 50, 95)]}
    report["gps"] = {"usable_sequences": len(sig_by), "issues": dict(reasons)}
    report["units"] = {"speed_vs_gps_distance": {"slope": slope, "corr": corr, "n": len(int_speed),
                                                 "verdict": "PASS (m/s)" if 0.8 < slope < 1.25 and corr > 0.9 else "CHECK"},
                       "course_yaw_vs_gyro_axis_median_corr":
                           {ax: (float(np.median(v)) if v else None) for ax, v in gyro_corr.items()}}
    print("1. frames:", report["frames"])
    print("2. gps:", report["gps"]["usable_sequences"], "usable;", dict(reasons))
    print("3. units:", json.dumps(report["units"], indent=1))

    # ------------------------------------------------- 4. time alignment
    lags = np.round(np.arange(-3.0, 3.01, 0.2), 2)
    val_seqs = [k for k in sig_by if k[0] == "val"] or list(sig_by)
    pick = rng.sample(val_seqs, min(N_ALIGN, len(val_seqs)))
    curves, peaks = [], []
    for sp, seq in pick:
        ts, mo = motion_signal(paths, sp, seq, frames_by[(sp, seq)])
        c = best_lag(ts, mo, sig_by[(sp, seq)], lags)
        if np.isfinite(c).sum() > len(lags) // 2:
            curves.append(c)
            peaks.append(float(lags[np.nanargmax(c)]))
    if curves:
        mean_curve = np.nanmean(np.stack(curves), 0)
        peak = float(lags[np.nanargmax(mean_curve)])
        report["alignment"] = {"sequences_used": len(curves), "mean_curve_peak_lag_s": peak,
                               "per_sequence_peak_median_s": float(np.median(peaks)),
                               "corr_at_peak": float(np.nanmax(mean_curve)),
                               "corr_at_zero": float(mean_curve[np.argmin(np.abs(lags))]),
                               "verdict": f"PASS (time_offset_s = {TIME_OFFSET_S:+.1f} is right)" if abs(peak) <= 0.4 else
                                          f"SET TIME_OFFSET_S = {TIME_OFFSET_S + peak:+.1f} and rerun"}
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for c in curves:
            ax.plot(lags, c, color=".8", lw=.6)
        ax.plot(lags, mean_curve, lw=2)
        ax.axvline(0, color="k", ls="--", lw=.8)
        ax.set_xlabel("lag [s] (speed sampled at frame time + lag)")
        ax.set_ylabel("corr(image motion, speed)")
        ax.set_title(f"peak at {peak:+.1f} s")
        fig.tight_layout(); fig.savefig(out / "alignment_check.png", dpi=130); plt.close(fig)
    else:
        report["alignment"] = {"verdict": "CHECK: not enough moving sequences"}
    print("4. alignment:", report["alignment"])

    # -------------------------------------------- 5. sample index + figure
    report["index"] = {}
    for sp in SPLITS:
        for hn in protocol.HORIZONS:
            samples, rep = bt.build_index(paths, sp, hn, stride=STRIDE, time_offset_s=TIME_OFFSET_S)
            f = out / f"index_{sp}_{hn}.jsonl"
            io.save_jsonl(f, [s.to_dict() for s in samples])
            report["index"][f"{sp}_{hn}"] = {"samples": rep["samples"], "rejected_sequences": rep["rejected"],
                                             "file": str(f)}
    print("5. index:", {k: v["samples"] for k, v in report["index"].items()})

    turns = sorted(((trapz(s.yaw_rate, s.t), k) for k, s in sig_by.items() if s.speed.mean() > 3), reverse=True)
    picks = turns[:N_VISUAL // 2] + turns[-(N_VISUAL - N_VISUAL // 2):] if turns else []
    if picks:
        fig, axes = plt.subplots(len(picks), 4, figsize=(16, 2.6 * len(picks)), squeeze=False)
        for r, (iy, (sp, seq)) in enumerate(picks):
            sig, fr = sig_by[(sp, seq)], frames_by[(sp, seq)]
            k = int(np.argmax(np.abs(sig.yaw_rate)))
            tc = sig.t[k]
            for c, dt in enumerate((-2.0, 0.0, 2.0)):
                i = int(np.clip(round((tc + dt) * FPS) + 1, fr[0], fr[-1]))
                a = axes[r, c]; a.axis("off")
                a.imshow(Image.open(paths.frame(sp, seq, i)).convert("RGB").resize((512, 288)))
                a.set_title(f"{seq} frame {i} (t={(i - 1) / FPS:.1f}s)", fontsize=7)
            a = axes[r, 3]
            a.plot(sig.t, sig.yaw_rate); a.axhline(0, color=".6", lw=.6); a.axvline(tc, color="r", lw=.6)
            a2 = a.twinx(); a2.plot(sig.t, sig.speed, color="tab:orange")
            a.set_title(f"yaw at red line {sig.yaw_rate[k]:+.2f} rad/s "
                        f"({'LEFT' if sig.yaw_rate[k] > 0 else 'RIGHT'} if left-positive)", fontsize=8)
        fig.tight_layout(); fig.savefig(out / "turn_check.png", dpi=110); plt.close(fig)

    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}/report.json, alignment_check.png, turn_check.png, index_*.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
