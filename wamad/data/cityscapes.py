"""Cityscapes semantic-forecasting index and loader.

Expected layout (official downloads, unzipped into one root):

    <root>/gtFine/{split}/{city}/{city}_{seq}_{frame}_gtFine_labelIds.png
    <root>/leftImg8bit_sequence/{split}/{city}/{city}_{seq}_{frame}_leftImg8bit.png
    <root>/vehicle_sequence/{split}/{city}/{city}_{seq}_{frame}_vehicle.json

Pseudo-labels (segmenter outputs, train IDs) are cached separately:

    <pseudo_root>/cityscapes/{split}/{city}/{city}_{seq}_{frame}_trainIds.png

One sample = one annotated snippet. Model inputs are the segmentations of the
context frames (pseudo-labels), as in Luc et al. 2017. Intermediate
autoregressive targets (mid-term frames 14, 17) have no human labels, so
their training targets are pseudo-labels; the scored frame 20 uses gtFine.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .. import io
from ..ego import EgoSignal, load_cityscapes_vehicle
from ..labels import labelids_to_trainids
from ..protocol import CITYSCAPES_FPS, HORIZONS, Horizon, cityscapes_frame_id, cityscapes_frame_ids

_GT_RE = re.compile(r"^(?P<city>[a-z_]+)_(?P<seq>\d{6})_(?P<frame>\d{6})_gtFine_labelIds\.png$")


@dataclass
class CityscapesSample:
    split: str
    city: str
    seq: str
    annotated_frame: int

    def stem(self, frame: int) -> str:
        return f"{self.city}_{self.seq}_{frame:06d}"

    def to_dict(self) -> dict:
        return asdict(self)


class CityscapesPaths:
    def __init__(self, root: Path, pseudo_root: Path | None = None):
        self.root = Path(root)
        self.pseudo_root = Path(pseudo_root) if pseudo_root else None

    def gt(self, s: CityscapesSample) -> Path:
        return self.root / "gtFine" / s.split / s.city / f"{s.stem(s.annotated_frame)}_gtFine_labelIds.png"

    def rgb(self, s: CityscapesSample, frame: int) -> Path:
        return self.root / "leftImg8bit_sequence" / s.split / s.city / f"{s.stem(frame)}_leftImg8bit.png"

    def vehicle(self, s: CityscapesSample, frame: int) -> Path:
        return self.root / "vehicle_sequence" / s.split / s.city / f"{s.stem(frame)}_vehicle.json"

    def pseudo(self, s: CityscapesSample, frame: int) -> Path:
        if self.pseudo_root is None:
            raise ValueError("pseudo_root not set")
        return self.pseudo_root / "cityscapes" / s.split / s.city / f"{s.stem(frame)}_trainIds.png"


def build_index(root: Path, split: str) -> list[CityscapesSample]:
    """All annotated snippets of a split, sorted for reproducibility."""
    gt_dir = Path(root) / "gtFine" / split
    if not gt_dir.is_dir():
        raise FileNotFoundError(gt_dir)
    out = []
    for p in sorted(gt_dir.glob("*/*_gtFine_labelIds.png")):
        m = _GT_RE.match(p.name)
        if m:
            out.append(CityscapesSample(split, m["city"], m["seq"], int(m["frame"])))
    return out


def check_sample_files(paths: CityscapesPaths, s: CityscapesSample, horizon: Horizon,
                       need_rgb: bool = True, need_pseudo: bool = False) -> list[Path]:
    """Return the list of MISSING files for one sample (empty list = complete)."""
    ids = cityscapes_frame_ids(s.annotated_frame, horizon)
    want = [paths.gt(s)]
    first, last = min(ids["context"]), ids["eval"]
    want += [paths.vehicle(s, f) for f in range(first, last + 1)]
    if need_rgb:
        want += [paths.rgb(s, f) for f in ids["context"]]
    if need_pseudo:
        want += [paths.pseudo(s, f) for f in ids["context"] + ids["targets"]]
    return [p for p in want if not p.exists()]


def ego_signal(paths: CityscapesPaths, s: CityscapesSample, horizon: Horizon, **unit_kw) -> EgoSignal:
    ids = cityscapes_frame_ids(s.annotated_frame, horizon)
    frames = range(min(ids["context"]), ids["eval"] + 1)
    return load_cityscapes_vehicle({f: paths.vehicle(s, f) for f in frames}, CITYSCAPES_FPS, **unit_kw)


def sample_actions(sig: EgoSignal, s: CityscapesSample, horizon: Horizon) -> dict:
    """past: instantaneous action at each context frame, shape (T_ctx, 2).
    future: mean action over each autoregressive interval, shape (n_steps, 2)."""
    ids = cityscapes_frame_ids(s.annotated_frame, horizon)
    t = lambda f: f / CITYSCAPES_FPS  # noqa: E731
    past = sig.at([t(f) for f in ids["context"]])
    edges = [ids["context"][-1]] + ids["targets"]
    future = np.stack([sig.interval_mean(t(a), t(b)) for a, b in zip(edges[:-1], edges[1:])])
    return {"past": past, "future": future}


def load_sample(paths: CityscapesPaths, s: CityscapesSample, horizon: Horizon | str,
                size_hw: tuple[int, int] | None = None, load_rgb: bool = False,
                load_pseudo_targets: bool = True, load_eval_gt: bool = True, **unit_kw) -> dict:
    """Load everything one forecasting example needs, as numpy arrays.

    Returns keys: ctx_labels (T,H,W) uint8, targets (n_steps,H,W) uint8 pseudo-labels
    [optional], eval_gt (H,W) uint8 at FULL resolution (never resized, for scoring),
    actions_past (T,2), actions_future (n_steps,2), ctx_rgb (T,H,W,3) [optional], meta.
    """
    h = HORIZONS[horizon] if isinstance(horizon, str) else horizon
    ids = cityscapes_frame_ids(s.annotated_frame, h)
    out = {
        "ctx_labels": np.stack([io.read_label(paths.pseudo(s, f), size_hw) for f in ids["context"]]),
        "meta": {**s.to_dict(), "horizon": h.name, "frame_ids": ids, "dataset": "cityscapes"},
    }
    if load_eval_gt:
        out["eval_gt"] = labelids_to_trainids(io.read_label(paths.gt(s)))
    if load_pseudo_targets:
        out["targets"] = np.stack([io.read_label(paths.pseudo(s, f), size_hw) for f in ids["targets"]])
    if load_rgb:
        out["ctx_rgb"] = np.stack([io.read_rgb(paths.rgb(s, f), size_hw) for f in ids["context"]])
    acts = sample_actions(ego_signal(paths, s, h, **unit_kw), s, h)
    out["actions_past"], out["actions_future"] = acts["past"], acts["future"]
    return out


def pseudo_label_jobs(paths: CityscapesPaths, samples: list[CityscapesSample],
                      horizons=("short", "mid")) -> list[tuple[Path, Path]]:
    """(rgb_path, pseudo_out_path) for every frame any horizon needs; deduplicated."""
    jobs = {}
    for s in samples:
        for hn in horizons:
            ids = cityscapes_frame_ids(s.annotated_frame, HORIZONS[hn])
            for f in ids["context"] + ids["targets"]:
                jobs[paths.pseudo(s, f)] = paths.rgb(s, f)
    return [(rgb, out) for out, rgb in sorted(jobs.items())]


__all__ = ["CityscapesSample", "CityscapesPaths", "build_index", "check_sample_files",
           "ego_signal", "sample_actions", "load_sample", "pseudo_label_jobs", "cityscapes_frame_id"]
