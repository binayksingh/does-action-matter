"""Cache segmenter outputs (pseudo-labels) for every frame the protocol needs.

The same segmenter must be used for all runs.
This module is model-agnostic: pass any `segment_fn(list_of_rgb_uint8) ->
list_of_trainId_uint8_maps`. Already-cached files are skipped.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .. import io

SegmentFn = Callable[[Sequence[np.ndarray]], Sequence[np.ndarray]]


def claim_manifest(pseudo_root: Path, segmenter_id: str) -> None:
    """Record which segmenter owns `pseudo_root`; refuse a different one.

    Stored in <pseudo_root>/pseudo_manifest.json so one cache can never mix
    outputs of two segmenters.
    """
    manifest = Path(pseudo_root) / "pseudo_manifest.json"
    if manifest.exists():
        prev = json.loads(manifest.read_text()).get("segmenter_id")
        if prev != segmenter_id:
            raise RuntimeError(f"pseudo-label cache at {pseudo_root} was made with {prev!r}, "
                               f"not {segmenter_id!r}. Use a new pseudo_root.")
    else:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"segmenter_id": segmenter_id}, indent=2))


def cache_pseudo_labels(jobs: list[tuple[Path, Path]], segment_fn: SegmentFn,
                        pseudo_root: Path, segmenter_id: str, batch_size: int = 8,
                        log_every: int = 200) -> dict:
    """Run `segment_fn` on each (rgb_path, out_path) job whose output is missing.

    `segmenter_id` (e.g. "segformer-b2-cityscapes@<checkpoint hash>") is stored in
    <pseudo_root>/pseudo_manifest.json; a different id on the same root is refused,
    so one cache can never mix two segmenters.
    """
    claim_manifest(pseudo_root, segmenter_id)
    todo = [(r, o) for r, o in jobs if not Path(o).exists()]
    done = 0
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        imgs = [io.read_rgb(r) for r, _ in chunk]
        preds = segment_fn(imgs)
        if len(preds) != len(chunk):
            raise RuntimeError("segment_fn returned a different number of maps than images")
        for (r, o), img, p in zip(chunk, imgs, preds):
            p = np.asarray(p)
            if p.shape != img.shape[:2]:
                raise ValueError(f"{r}: prediction {p.shape} != image {img.shape[:2]}; "
                                 "upsample to full resolution before returning")
            io.write_label(o, p)
        done += len(chunk)
        if log_every and (done % log_every < len(chunk) or done == len(todo)):
            print(f"[pseudo] {done}/{len(todo)}")
    return {"jobs": len(jobs), "already_cached": len(jobs) - len(todo), "written": done,
            "segmenter_id": segmenter_id}
