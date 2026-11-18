#!/usr/bin/env python3
"""Baselines that need no training: copy-last and optical-flow warping.

Edit SETTINGS, then run (CPU job is enough):  python scripts/baselines.py

Both work at full resolution on the cached pseudo-labels of the context
frames (as in Luc et al., ICCV 2017):
  copy_last  the last context segmentation is the prediction
  flow_warp  the last context segmentation is warped by optical flow between
             the last two context frames (Farneback, computed at 512 px width),
             extrapolated linearly for 1 (short) or 3 (mid) steps
Scored with mIoU / MO-mIoU on Cityscapes val (gtFine) and BDD100K val
(pseudo-labels), both horizons. Output: RUNS_DIR/forecast/baselines/results.*
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wamad import flow, io, metrics, paths as P  # noqa: E402
from wamad.data import bdd100k_track as bt, cityscapes as cs  # noqa: E402
from wamad.protocol import HORIZONS  # noqa: E402

# ============================================================ SETTINGS
CITYSCAPES_ROOT = P.CITYSCAPES_ROOT
BDD_ROOT = P.BDD_ROOT
PSEUDO_ROOT = P.PSEUDO_ROOT
BDD_INDEX_DIR = P.BDD_INDEX_DIR
OUT_DIR = P.RUNS_DIR / "forecast" / "baselines"
EVAL_SETS = ("cityscapes", "bdd100k")
NUM_PROCS = 8
LIMIT = None          # e.g. 20 for a quick test
# ======================================================================


def _job(args):
    dataset, h, s = args
    steps = 3 if h == "mid" else 1
    if dataset == "cityscapes":
        paths = cs.CityscapesPaths(CITYSCAPES_ROOT, PSEUDO_ROOT)
        d = cs.load_sample(paths, s, h, load_rgb=True, load_pseudo_targets=False)
    else:
        paths = bt.TrackPaths(BDD_ROOT, PSEUDO_ROOT)
        d = bt.load_sample(paths, s, load_rgb=True)
    gt, ctx, rgb = d["eval_gt"], d["ctx_labels"], d["ctx_rgb"]
    a, b = metrics.SegMetricAccumulator(), metrics.SegMetricAccumulator()
    a.update(flow.copy_last_baseline(ctx), gt)
    b.update(flow.flow_baseline(rgb, ctx, steps, work_width=512), gt)
    return a, b


def samples_for(dataset, h):
    if dataset == "cityscapes":
        return cs.build_index(CITYSCAPES_ROOT, "val")
    return [bt.TrackSample.from_dict(r) for r in io.load_jsonl(BDD_INDEX_DIR / f"index_val_{h}.jsonl")]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = io.ResultsLogger(OUT_DIR, fresh=True)
    for es in EVAL_SETS:
        for h in HORIZONS:
            ss = samples_for(es, h)[:LIMIT]
            t0 = time.time()
            acc_c, acc_f = metrics.SegMetricAccumulator(), metrics.SegMetricAccumulator()
            with Pool(NUM_PROCS) as pool:
                for k, (a, b) in enumerate(pool.imap(_job, [(es, h, s) for s in ss], chunksize=4)):
                    acc_c.merge(a); acc_f.merge(b)
                    if (k + 1) % 500 == 0:
                        print(f"  {es} {h}: {k + 1}/{len(ss)} ({time.time() - t0:.0f}s)", flush=True)
            for name, acc in (("copy_last", acc_c), ("flow_warp", acc_f)):
                summ = acc.summary()
                logger.log(run_id="baselines", train_set="-", eval_set=es, horizon=h, model=name, summary=summ,
                           notes="gtFine" if es == "cityscapes" else "pseudo-label targets")
                print(f"[{name:9s}] {es:10s} {h:5s} {metrics.format_summary(summ)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
