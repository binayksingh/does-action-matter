#!/usr/bin/env python3
"""Check the segmenter, then cache pseudo-labels for both datasets.

Run on a GPU:  python scripts/make_pseudolabels.py

Steps (each can be switched off in SETTINGS):
  EVAL             Segment Cityscapes val frame 20 (= the gtFine image) and score
                   against gtFine with wamad.metrics. Compare with the reference
                   number (82.25 mIoU for SegFormer-B5). Saved to
                   RUNS_DIR/segmenter/eval.json and results.csv.
  CACHE_CITYSCAPES Pseudo-labels for every protocol frame (7 per snippet, train+val).
  CACHE_BDD        Pseudo-labels for every frame in the BDD100K sample index
                   (written by scripts/prepare_bdd100k.py).

Existing label files are skipped. To split the work over a SLURM job array,
set NUM_SHARDS; the shard index is read from SLURM_ARRAY_TASK_ID.

The segmenter checkpoint is downloaded from the Hugging Face Hub on first use.
On machines without internet access, download it beforehand:
    python -c "from transformers import SegformerForSemanticSegmentation as S; \\
               S.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')"
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wamad import io, metrics, paths as P, protocol, viz  # noqa: E402
from wamad.data import bdd100k_track as bt, cityscapes as cs, pseudo  # noqa: E402
from wamad.labels import labelids_to_trainids  # noqa: E402

# ============================================================ SETTINGS
CITYSCAPES_ROOT = P.CITYSCAPES_ROOT
BDD_ROOT = P.BDD_ROOT
PSEUDO_ROOT = P.PSEUDO_ROOT
RUNS_DIR = P.RUNS_DIR
BDD_INDEX_DIR = P.BDD_INDEX_DIR

MODEL_ID = os.environ.get("SEGFORMER_MODEL", "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")  # hub id or local folder
MODE = "slide"            # "slide" (matches the reference number) or "whole" (faster, slightly lower)
BATCH_SIZE = 2            # images loaded per step (slide mode cuts each into 2-3 crops of 1024x1024)
MAX_WINDOWS = 2           # crops per forward pass: 2 fits a 16 GB GPU; raise to 8+ on an 80 GB GPU
NUM_WORKERS = 8           # image-reading processes
FP16 = True

EVAL = True
CACHE_CITYSCAPES = True
CACHE_BDD = True
EVAL_LIMIT = None         # e.g. 50 for a quick test; None = all 500 val images
SMOKE_TEST = False        # True: only 16 images per step, to check the pipeline end to end

NUM_SHARDS = 1            # >1 to split caching over a SLURM job array
SHARD_INDEX = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
# ======================================================================


class _Images:
    """torch Dataset over (rgb_path, extra) pairs; returns uint8 HWC + index."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return np.ascontiguousarray(io.read_rgb(self.items[i][0])), i


def _loader(items):
    import torch
    from torch.utils.data import DataLoader

    def collate(batch):
        sizes = {b[0].shape for b in batch}
        if len(sizes) != 1:  # mixed sizes: caller handles one-by-one
            return [torch.from_numpy(b[0]).permute(2, 0, 1)[None] for b in batch], [b[1] for b in batch]
        return [torch.from_numpy(np.stack([b[0] for b in batch])).permute(0, 3, 1, 2)], [b[1] for b in batch]

    return DataLoader(_Images(items), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                      collate_fn=collate, pin_memory=True, persistent_workers=False)


def _predict(seg, tensors):
    return np.concatenate([seg.predict_logits(t).argmax(1).to("cpu").numpy().astype(np.uint8) for t in tensors])


def run_eval(seg) -> dict:
    paths = cs.CityscapesPaths(CITYSCAPES_ROOT)
    val = cs.build_index(CITYSCAPES_ROOT, "val")
    if SMOKE_TEST:
        val = val[:16]
    elif EVAL_LIMIT:
        val = val[:EVAL_LIMIT]
    items = [(paths.rgb(s, s.annotated_frame), s) for s in val]
    acc = metrics.SegMetricAccumulator()
    out = RUNS_DIR / "segmenter"
    t0 = time.time()
    for tensors, idx in _loader(items):
        preds = _predict(seg, tensors)
        for p, i in zip(preds, idx):
            s = items[i][1]
            gt = labelids_to_trainids(io.read_label(paths.gt(s)))
            acc.update(p, gt)
            if i < 3:
                viz.save_forecast_panel(out / f"example_{i}.png", p[None], {"segmenter": p}, gt,
                                        title=f"{s.city}_{s.seq}_{s.annotated_frame:06d}")
    summ = acc.summary()
    ref = 82.25 if "segformer-b5" in MODEL_ID and MODE == "slide" else None
    summ.update({"segmenter_id": seg.segmenter_id, "reference_mIoU": ref, "seconds": time.time() - t0,
                 "verdict": ("n/a" if ref is None or SMOKE_TEST or EVAL_LIMIT else
                             "PASS" if abs(summ["mIoU"] * 100 - ref) <= 1.0 else
                             "CHECK: more than 1 point from the reference")})
    io.save_json(out / "eval.json", summ)
    io.ResultsLogger(RUNS_DIR).log(run_id="segmenter", train_set="cityscapes", eval_set="cityscapes_val",
                                   horizon="frame20", model=seg.segmenter_id, summary=summ,
                                   notes="single-frame segmentation (upper bound)")
    print("EVAL:", metrics.format_summary(summ), "| reference", ref, "|", summ["verdict"])
    return summ


def run_cache(seg, jobs, name) -> dict:
    jobs = sorted(jobs)
    jobs = jobs[SHARD_INDEX::NUM_SHARDS]
    todo = [(r, o) for r, o in jobs if not Path(o).exists()]
    if SMOKE_TEST:
        todo = todo[:16]
    print(f"[{name}] shard {SHARD_INDEX}/{NUM_SHARDS}: {len(jobs)} jobs, {len(todo)} to do")
    if not todo:
        return {"jobs": len(jobs), "written": 0}
    t0, done = time.time(), 0
    with ThreadPoolExecutor(8) as writer:
        futs = []
        for tensors, idx in _loader(todo):
            preds = _predict(seg, tensors)
            for p, i in zip(preds, idx):
                futs.append(writer.submit(io.write_label, todo[i][1], p, False))
            done += len(idx)
            if done % 2000 < len(idx) or done == len(todo):
                rate = done / (time.time() - t0)
                print(f"  [{name}] {done}/{len(todo)}  {rate:.1f} img/s  "
                      f"ETA {(len(todo) - done) / max(rate, 1e-9) / 3600:.1f} h", flush=True)
            futs = [f for f in futs if not f.done() or f.result() is not None]
        for f in futs:
            f.result()
    return {"jobs": len(jobs), "written": done, "seconds": time.time() - t0}


def main() -> int:
    import torch
    from wamad.segmenter import Segmenter
    if not torch.cuda.is_available():
        print("WARNING: no GPU visible; this will be very slow.")
    seg = Segmenter(MODEL_ID, device="cuda" if torch.cuda.is_available() else "cpu", fp16=FP16, mode=MODE,
                    max_windows=MAX_WINDOWS)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GiB")
    print("segmenter:", seg.segmenter_id)
    report = {}
    if EVAL:
        report["eval"] = run_eval(seg)
    if CACHE_CITYSCAPES or CACHE_BDD:
        pseudo.claim_manifest(PSEUDO_ROOT, seg.segmenter_id)
    if CACHE_CITYSCAPES:
        paths = cs.CityscapesPaths(CITYSCAPES_ROOT, PSEUDO_ROOT)
        samples = cs.build_index(CITYSCAPES_ROOT, "train") + cs.build_index(CITYSCAPES_ROOT, "val")
        report["cityscapes"] = run_cache(seg, cs.pseudo_label_jobs(paths, samples), "cityscapes")
    if CACHE_BDD:
        paths = bt.TrackPaths(BDD_ROOT, PSEUDO_ROOT)
        samples = []
        for sp in ("train", "val"):
            for hn in protocol.HORIZONS:
                f = BDD_INDEX_DIR / f"index_{sp}_{hn}.jsonl"
                if not f.exists():
                    raise FileNotFoundError(f"{f} missing: run scripts/prepare_bdd100k.py first")
                samples += [bt.TrackSample.from_dict(r) for r in io.load_jsonl(f)]
        report["bdd100k"] = run_cache(seg, bt.pseudo_label_jobs(paths, samples), "bdd100k")
    io.save_json(RUNS_DIR / "segmenter" / f"cache_report_shard{SHARD_INDEX}.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "eval"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
