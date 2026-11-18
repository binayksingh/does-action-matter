"""Real-data forecasting sets (numpy only; works as a torch Dataset).

One class covers both datasets and both uses:
  * training: model-resolution labels, optional horizontal flip (yaw negated),
    no full-resolution target read;
  * evaluation: also returns `eval_gt` at full resolution (gtFine for
    Cityscapes, pseudo-label for BDD100K), and can swap in the future actions
    of a sample from a DIFFERENT sequence (the "shuffle" intervention).

Model input sizes keep each camera's aspect ratio and are divisible by 8:
  Cityscapes 1024x2048 -> 256x512, BDD100K 720x1280 -> 288x512.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .. import io, protocol
from . import bdd100k_track as bt
from . import cityscapes as cs

MODEL_SIZE = {"cityscapes": (256, 512), "bdd100k": (288, 512)}
DATASETS = tuple(MODEL_SIZE)


def flip_sample(d: dict) -> dict:
    """Mirror left-right: flip every map and negate yaw rate (left turn <-> right turn)."""
    out = dict(d)
    for k in ("ctx_labels", "targets", "eval_gt", "ctx_rgb"):
        if k in out:
            out[k] = np.ascontiguousarray(np.flip(out[k], axis=2 if k != "eval_gt" else 1))
    for k in ("actions_past", "actions_future"):
        a = np.array(out[k], dtype=np.float64, copy=True)
        a[..., 1] *= -1
        out[k] = a
    return out


def cross_sequence_permutation(keys: list, seed: int = 0) -> np.ndarray:
    """Index map p with keys[p[i]] != keys[i]: each sample gets another sequence's actions.

    Within one sequence neighbouring samples have nearly the same actions, so a
    plain shuffle could hand a sample its own neighbour's actions. Starts from a
    random permutation (keeps the action distribution) and re-draws only the
    entries that landed in their own sequence."""
    n = len(keys)
    if len(set(keys)) < 2:
        raise ValueError("need at least two sequences for a cross-sequence shuffle")
    rng = np.random.default_rng(seed)
    keys_arr = np.array([hash(k) for k in keys])
    p = rng.permutation(n)
    for i in np.flatnonzero(keys_arr[p] == keys_arr):
        others = np.flatnonzero(keys_arr != keys_arr[i])
        p[i] = int(rng.choice(others))
    return p


class ForecastSet:
    def __init__(self, dataset: str, split: str, horizon: str, *, train: bool,
                 cityscapes_root: Path | None = None, bdd_root: Path | None = None,
                 pseudo_root: Path | None = None, bdd_index_dir: Path | None = None,
                 size_hw: tuple[int, int] | None = None, flip: bool = False,
                 shuffle_actions: bool = False, shuffle_seed: int = 0, limit: int | None = None):
        if dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}")
        self.dataset, self.split, self.horizon, self.train = dataset, split, horizon, train
        self.size = size_hw or MODEL_SIZE[dataset]
        self.flip = flip and train
        self.h = protocol.HORIZONS[horizon]
        if dataset == "cityscapes":
            self.paths = cs.CityscapesPaths(cityscapes_root, pseudo_root)
            self.samples = cs.build_index(cityscapes_root, split)
            self._key = lambda s: (s.split, s.city, s.seq)  # noqa: E731
        else:
            self.paths = bt.TrackPaths(bdd_root, pseudo_root)
            f = Path(bdd_index_dir) / f"index_{split}_{horizon}.jsonl"
            if not f.exists():
                raise FileNotFoundError(f"{f} (run scripts/prepare_bdd100k.py first)")
            self.samples = [bt.TrackSample.from_dict(r) for r in io.load_jsonl(f)]
            self._key = lambda s: (s.split, s.seq)  # noqa: E731
        if limit:
            self.samples = self.samples[:limit]
        self.perm = (cross_sequence_permutation([self._key(s) for s in self.samples], shuffle_seed)
                     if shuffle_actions else None)

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, s) -> dict:
        if self.dataset == "cityscapes":
            return cs.load_sample(self.paths, s, self.h, size_hw=self.size, load_eval_gt=not self.train)
        return bt.load_sample(self.paths, s, size_hw=self.size, load_eval_gt=not self.train)

    def future_actions(self, i: int) -> np.ndarray:
        """True future actions of sample i (physical units), without reading any label."""
        s = self.samples[i]
        if self.dataset == "cityscapes":
            return cs.sample_actions(cs.ego_signal(self.paths, s, self.h), s, self.h)["future"]
        return np.asarray(s.actions_future, dtype=np.float64)

    def all_actions(self) -> np.ndarray:
        """Past and future actions of every sample, (N*(T+S), 2): for the normaliser."""
        out = []
        for i, s in enumerate(self.samples):
            if self.dataset == "cityscapes":
                a = cs.sample_actions(cs.ego_signal(self.paths, s, self.h), s, self.h)
                out += [a["past"], a["future"]]
            else:
                out += [np.asarray(s.actions_past), np.asarray(s.actions_future)]
        return np.concatenate(out)

    def __getitem__(self, i: int) -> dict:
        d = self._load(self.samples[i])
        if self.perm is not None:
            d["actions_future"] = self.future_actions(int(self.perm[i]))
        if self.flip and random.random() < 0.5:
            d = flip_sample(d)
        d.pop("meta", None)
        return d
