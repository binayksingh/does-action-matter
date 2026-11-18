"""BDD100K tracking-frame (MOT 2020, 5 fps) index and loader.

Layout (as unzipped):

    <root>/images/track/{train,val}/<seq>/<seq>-0000001.jpg ...   (1-based, 5 fps)
    <root>/info/100k/{train,val}/<seq>.json                         (GPS/IMU)

The info file of a tracking sequence may sit in either info split, so both
are searched. Derived data:

    <pseudo_root>/bdd100k/{split}/<seq>/<idx:07d>_trainIds.png

Timing: frame i (1-based) is at (i - 1) / 5 s + time_offset_s after the info
`startTime`. `time_offset_s` defaults to 0; scripts/prepare_bdd100k.py measures the
alignment and reports any consistent lag.

These frames have no 19-class human labels: every target, including the
scored one, is a pseudo-label from the fixed segmenter.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import io
from ..ego import EgoSignal, bdd_signal_quality, load_bdd_info
from ..protocol import BDD_TRACK_FPS, HORIZONS, Horizon, bdd_track_frame_ids


@dataclass
class TrackSample:
    split: str
    seq: str
    horizon: str
    frame_ids: dict
    actions_past: list = field(default_factory=list)
    actions_future: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrackSample":
        return cls(**d)


class TrackPaths:
    def __init__(self, root: Path, pseudo_root: Path | None = None):
        self.root = Path(root)
        self.pseudo_root = Path(pseudo_root) if pseudo_root else None
        self._info: dict[str, Path] | None = None

    def seq_dir(self, split: str, seq: str) -> Path:
        return self.root / "images" / "track" / split / seq

    def frame(self, split: str, seq: str, idx: int) -> Path:
        return self.seq_dir(split, seq) / f"{seq}-{idx:07d}.jpg"

    def info(self, seq: str) -> Path | None:
        if self._info is None:
            self._info = {}
            for sp in ("train", "val"):
                d = self.root / "info" / "100k" / sp
                if d.is_dir():
                    for f in os.listdir(d):
                        if f.endswith(".json"):
                            self._info.setdefault(f[:-5], d / f)
        return self._info.get(seq)

    def pseudo(self, split: str, seq: str, idx: int) -> Path:
        if self.pseudo_root is None:
            raise ValueError("pseudo_root not set")
        return self.pseudo_root / "bdd100k" / split / seq / f"{idx:07d}_trainIds.png"


def list_sequences(paths: TrackPaths, split: str) -> list[str]:
    return sorted(os.listdir(paths.root / "images" / "track" / split))


def frame_indices(paths: TrackPaths, split: str, seq: str) -> list[int]:
    pre = f"{seq}-"
    out = []
    for f in os.listdir(paths.seq_dir(split, seq)):
        if f.startswith(pre) and f.endswith(".jpg"):
            out.append(int(f[len(pre):-4]))
    return sorted(out)


def frame_time(idx: int, fps: float = BDD_TRACK_FPS, time_offset_s: float = 0.0) -> float:
    return (idx - 1) / fps + time_offset_s


def samples_for_sequence(sig: EgoSignal, split: str, seq: str, frames: list[int], horizon: Horizon,
                         stride: int = 10, fps: float = BDD_TRACK_FPS,
                         time_offset_s: float = 0.0) -> list[TrackSample]:
    """Scored frames every `stride` frames (default 10 = 2 s) where all frames exist
    and the GPS signal covers the whole window."""
    have = set(frames)
    out = []
    for ev in range(frames[0], frames[-1] + 1, stride):
        ids = bdd_track_frame_ids(ev, horizon)
        if not all(i in have for i in ids["context"] + ids["targets"]):
            continue
        t = lambda i: frame_time(i, fps, time_offset_s)  # noqa: E731
        past = sig.at([t(i) for i in ids["context"]])
        edges = [ids["context"][-1]] + ids["targets"]
        future = np.stack([sig.interval_mean(t(a), t(b)) for a, b in zip(edges[:-1], edges[1:])])
        if np.isnan(past).any() or np.isnan(future).any():
            continue
        out.append(TrackSample(split, seq, horizon.name, ids, past.tolist(), future.tolist()))
    return out


def build_index(paths: TrackPaths, split: str, horizon: Horizon | str, stride: int = 10,
                yaw_source: str = "course", time_offset_s: float = 0.0,
                max_sequences: int | None = None) -> tuple[list[TrackSample], dict]:
    """All samples of a split (saved with io.save_jsonl by scripts/prepare_bdd100k.py)."""
    h = HORIZONS[horizon] if isinstance(horizon, str) else horizon
    seqs = list_sequences(paths, split)[:max_sequences]
    samples, rejected = [], {}
    for seq in seqs:
        ip = paths.info(seq)
        if ip is None:
            rejected[seq] = ["no_info_file"]
            continue
        try:
            sig = load_bdd_info(ip, yaw_source=yaw_source)
        except (ValueError, KeyError) as e:
            rejected[seq] = [f"info_error: {e}"]
            continue
        q = bdd_signal_quality(sig)
        if not q["ok"]:
            rejected[seq] = q["reasons"]
            continue
        frames = frame_indices(paths, split, seq)
        if len(frames) < 10:
            rejected[seq] = ["too_few_frames"]
            continue
        samples += samples_for_sequence(sig, split, seq, frames, h, stride, time_offset_s=time_offset_s)
    report = {"split": split, "horizon": h.name, "sequences": len(seqs), "rejected": len(rejected),
              "rejected_reasons": rejected, "samples": len(samples)}
    return samples, report


def frames_needed(samples: list[TrackSample]) -> dict[tuple[str, str], set[int]]:
    need: dict[tuple[str, str], set[int]] = {}
    for s in samples:
        need.setdefault((s.split, s.seq), set()).update(s.frame_ids["context"] + s.frame_ids["targets"])
    return need


def pseudo_label_jobs(paths: TrackPaths, samples: list[TrackSample]) -> list[tuple[Path, Path]]:
    jobs = {}
    for (sp, seq), idxs in frames_needed(samples).items():
        for i in idxs:
            jobs[paths.pseudo(sp, seq, i)] = paths.frame(sp, seq, i)
    return [(rgb, out) for out, rgb in sorted(jobs.items())]


def load_sample(paths: TrackPaths, s: TrackSample, size_hw: tuple[int, int] | None = None,
                load_rgb: bool = False, eval_size_hw: tuple[int, int] | None = None,
                load_eval_gt: bool = True) -> dict:
    """Same keys as the Cityscapes loader. `eval_gt` here is a PSEUDO-label."""
    ids = s.frame_ids
    out = {
        "ctx_labels": np.stack([io.read_label(paths.pseudo(s.split, s.seq, i), size_hw) for i in ids["context"]]),
        "targets": np.stack([io.read_label(paths.pseudo(s.split, s.seq, i), size_hw) for i in ids["targets"]]),
        "actions_past": np.asarray(s.actions_past, dtype=np.float64),
        "actions_future": np.asarray(s.actions_future, dtype=np.float64),
        "meta": {**s.to_dict(), "dataset": "bdd100k_track", "eval_target": "pseudo_label"},
    }
    if load_eval_gt:
        out["eval_gt"] = io.read_label(paths.pseudo(s.split, s.seq, ids["eval"]), eval_size_hw)
    if load_rgb:
        out["ctx_rgb"] = np.stack([io.read_rgb(paths.frame(s.split, s.seq, i), size_hw) for i in ids["context"]])
    return out


def ego_signal(paths: TrackPaths, seq: str, yaw_source: str = "course") -> EgoSignal:
    return load_bdd_info(paths.info(seq), yaw_source=yaw_source)


__all__ = ["TrackSample", "TrackPaths", "list_sequences", "frame_indices", "frame_time",
           "build_index", "frames_needed", "pseudo_label_jobs", "load_sample", "ego_signal",
           "bdd_signal_quality"]
