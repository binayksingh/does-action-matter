"""PyTorch Dataset wrapper around the numpy loaders. Requires torch."""
from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:  # pragma: no cover
    raise ImportError("wamad.data.torch_dataset needs PyTorch") from e

from ..ego import ActionNormalizer, apply_intervention
from ..labels import IGNORE_INDEX, NUM_CLASSES


def one_hot_labels(labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """(..., H, W) long with 255 ignore -> (..., C, H, W) float; ignore pixels are all-zero."""
    valid = labels != IGNORE_INDEX
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    oh = torch.nn.functional.one_hot(safe, num_classes).to(torch.float32)
    oh = oh * valid.unsqueeze(-1)
    return oh.movedim(-1, -3)


class ForecastDataset(Dataset):
    """Generic wrapper: `samples` is any list, `load_fn(sample) -> dict` is a dataset loader
    (e.g. functools.partial(cityscapes.load_sample, paths, horizon='short', size_hw=(256, 512))).

    Output tensors:
      ctx      (T, C, H, W) float one-hot of context segmentations
      targets  (n_steps, H, W) long   [if loaded]
      eval_gt  (H_full, W_full) long  (full resolution, for scoring only)
      a_past   (T, 2) float, normalised
      a_future (n_steps, 2) float, normalised (after the optional intervention)
    """

    def __init__(self, samples: list, load_fn: Callable[[object], dict],
                 normalizer: ActionNormalizer | None = None, intervention: str = "true",
                 intervention_seed: int = 0, train_mean: np.ndarray | None = None):
        self.samples, self.load_fn, self.normalizer = samples, load_fn, normalizer
        self.intervention = intervention
        self._rng = np.random.default_rng(intervention_seed)
        self._train_mean = train_mean
        self._shuffle_perm = (self._rng.permutation(len(samples))
                              if intervention == "shuffle" else None)

    def __len__(self) -> int:
        return len(self.samples)

    def _future_actions(self, idx: int, d: dict) -> np.ndarray:
        a = d["actions_future"]
        if self.intervention == "shuffle":
            # deterministic: take the TRUE future actions of another sample
            return self.load_fn(self.samples[int(self._shuffle_perm[idx])])["actions_future"]
        return apply_intervention(self.intervention, a, mean=self._train_mean)

    def __getitem__(self, idx: int) -> dict:
        d = self.load_fn(self.samples[idx])
        a_past = np.asarray(d["actions_past"], dtype=np.float64)
        a_fut = np.asarray(self._future_actions(idx, d), dtype=np.float64)
        if self.normalizer is not None:
            a_past, a_fut = self.normalizer(a_past), self.normalizer(a_fut)
        out = {
            "ctx": one_hot_labels(torch.from_numpy(d["ctx_labels"].astype(np.int64))),
            "eval_gt": torch.from_numpy(d["eval_gt"].astype(np.int64)),
            "a_past": torch.from_numpy(a_past).float(),
            "a_future": torch.from_numpy(a_fut).float(),
            "index": idx,
        }
        if "targets" in d:
            out["targets"] = torch.from_numpy(d["targets"].astype(np.int64))
        if "ctx_rgb" in d:
            out["ctx_rgb"] = torch.from_numpy(d["ctx_rgb"]).permute(0, 3, 1, 2).float() / 255.0
        return out
