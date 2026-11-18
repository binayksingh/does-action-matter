"""Evaluation metrics. Only established, previously published metrics are here.

* mIoU over the 19 Cityscapes classes (Cordts et al., CVPR 2016; the
  standard for semantic segmentation and for semantic forecasting,
  Luc et al., ICCV 2017).
* MO-mIoU: mIoU restricted to the 8 movable-object classes, as reported in
  the semantic-forecasting literature (Luc et al., ICCV 2017 and follow-ups).
* Per-class IoU and global pixel accuracy (standard companions to mIoU).

IoU is computed from a confusion matrix accumulated over the WHOLE evaluation
set (not averaged per image), matching the official Cityscapes evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .labels import CLASS_NAMES, IGNORE_INDEX, MOVING_CLASS_IDS, NUM_CLASSES


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int = NUM_CLASSES,
                     ignore_index: int = IGNORE_INDEX) -> tuple[np.ndarray, np.ndarray]:
    """Returns (C, missed): confusion matrix C[gt, pred] ignoring `ignore_index` in gt,
    and per-class counts of valid gt pixels whose prediction was out of range.

    Predictions equal to `ignore_index` (or otherwise out of range) on valid gt
    pixels are counted as errors: they add to that gt class's false negatives.
    """
    pred = np.asarray(pred).astype(np.int64).ravel()
    gt = np.asarray(gt).astype(np.int64).ravel()
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    valid = gt != ignore_index
    if np.any((gt[valid] < 0) | (gt[valid] >= num_classes)):
        raise ValueError("gt contains values outside [0, num_classes) other than ignore_index")
    gt = gt[valid]
    pred = pred[valid]
    in_range = (pred >= 0) & (pred < num_classes)
    cm = np.bincount(num_classes * gt[in_range] + pred[in_range],
                     minlength=num_classes ** 2).reshape(num_classes, num_classes)
    # out-of-range predictions: count as missed pixels (false negatives only)
    missed = np.bincount(gt[~in_range], minlength=num_classes)
    return cm, missed


def iou_from_confusion(cm: np.ndarray, missed: np.ndarray | None = None) -> np.ndarray:
    """Per-class IoU = TP / (TP + FP + FN). Classes absent from gt and pred -> NaN."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    if missed is not None:
        fn = fn + missed
    denom = tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, tp / denom, np.nan)


@dataclass
class SegMetricAccumulator:
    """Accumulates a dataset-level confusion matrix and reports standard metrics."""
    num_classes: int = NUM_CLASSES
    ignore_index: int = IGNORE_INDEX
    cm: np.ndarray = field(init=False)
    missed: np.ndarray = field(init=False)
    n_images: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.missed = np.zeros(self.num_classes, dtype=np.int64)
        self.n_images = 0

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """Add one image, or a batch with matching leading dims."""
        cm, missed = confusion_matrix(pred, gt, self.num_classes, self.ignore_index)
        self.cm += cm
        self.missed += missed
        p = np.asarray(pred)
        self.n_images += p.shape[0] if p.ndim == 3 else 1

    def merge(self, other: "SegMetricAccumulator") -> None:
        self.cm += other.cm
        self.missed += other.missed
        self.n_images += other.n_images

    def per_class_iou(self) -> np.ndarray:
        return iou_from_confusion(self.cm, self.missed)

    def miou(self) -> float:
        """mIoU over all 19 classes (classes absent from gt and pred are skipped)."""
        iou = self.per_class_iou()
        return float(np.nanmean(iou)) if np.isfinite(iou).any() else float("nan")

    def mo_miou(self) -> float:
        """MO-mIoU: mIoU over the 8 movable-object classes."""
        mo = self.per_class_iou()[list(MOVING_CLASS_IDS)]
        return float(np.nanmean(mo)) if np.isfinite(mo).any() else float("nan")

    def pixel_accuracy(self) -> float:
        total = self.cm.sum() + self.missed.sum()
        return float(np.diag(self.cm).sum() / total) if total else float("nan")

    def summary(self) -> dict:
        ious = self.per_class_iou()
        return {
            "mIoU": self.miou(),
            "MO-mIoU": self.mo_miou(),
            "pixel_acc": self.pixel_accuracy(),
            "n_images": self.n_images,
            "per_class_IoU": {n: (None if np.isnan(v) else float(v)) for n, v in zip(CLASS_NAMES, ious)},
        }


def format_summary(summary: dict, percent: bool = True) -> str:
    """One-line string, e.g. 'mIoU 62.31 | MO-mIoU 55.02 | pixAcc 93.10 (n=500)'."""
    s = 100.0 if percent else 1.0
    return (f"mIoU {summary['mIoU'] * s:.2f} | MO-mIoU {summary['MO-mIoU'] * s:.2f} | "
            f"pixAcc {summary['pixel_acc'] * s:.2f} (n={summary['n_images']})")
