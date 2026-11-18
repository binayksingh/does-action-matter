"""Visualisation helpers for sanity checks and paper figures."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .labels import CLASS_NAMES, IGNORE_COLOR, IGNORE_INDEX, NUM_CLASSES, PALETTE


def colorize(label: np.ndarray) -> np.ndarray:
    """Train-ID map -> RGB uint8 with the official Cityscapes colours (ignore = black)."""
    label = np.asarray(label)
    lut = np.tile(IGNORE_COLOR, (256, 1))
    lut[:NUM_CLASSES] = PALETTE
    return lut[label.astype(np.uint8)]


def overlay(rgb: np.ndarray, label: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    col = colorize(label).astype(np.float32)
    mask = (np.asarray(label) != IGNORE_INDEX)[..., None]
    out = np.where(mask, (1 - alpha) * rgb.astype(np.float32) + alpha * col, rgb.astype(np.float32))
    return out.clip(0, 255).astype(np.uint8)


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """White = wrong, black = right, grey = ignored in gt."""
    out = np.zeros(gt.shape + (3,), dtype=np.uint8)
    out[(pred != gt) & (gt != IGNORE_INDEX)] = 255
    out[gt == IGNORE_INDEX] = 90
    return out


def save_forecast_panel(path: Path, ctx_labels: np.ndarray, preds: dict[str, np.ndarray],
                        gt: np.ndarray, ctx_rgb: np.ndarray | None = None, title: str = "",
                        show_errors: bool = True) -> None:
    """Row 1: context frames. Then one row per method: prediction | error. Last row: ground truth.

    `preds` maps method name -> (H, W) prediction at the same size as `gt`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(ctx_labels)
    ncols = max(T, 2 if show_errors else 1)
    nrows = 1 + len(preds) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 1.8 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i in range(T):
        img = overlay(ctx_rgb[i], ctx_labels[i]) if ctx_rgb is not None else colorize(ctx_labels[i])
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"context t-{T - 1 - i}", fontsize=8)
    for r, (name, p) in enumerate(preds.items(), start=1):
        axes[r, 0].imshow(colorize(p))
        axes[r, 0].set_title(name, fontsize=8)
        if show_errors:
            axes[r, 1].imshow(error_map(p, gt))
            axes[r, 1].set_title(f"{name}: errors", fontsize=8)
    axes[-1, 0].imshow(colorize(gt))
    axes[-1, 0].set_title("ground truth", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_legend(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.axis("off")
    ax.legend(handles=[Patch(color=PALETTE[i] / 255, label=n) for i, n in enumerate(CLASS_NAMES)],
              ncol=7, fontsize=7, loc="center", frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ego_signal(path: Path, t: np.ndarray, speed: np.ndarray, yaw_rate: np.ndarray,
                    marks: dict[str, list[float]] | None = None, title: str = "") -> None:
    """Speed and yaw rate over time, with vertical lines at sample times.

    Compare against the video: stops should show speed ~0, left turns positive yaw rate.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 3.6), sharex=True)
    a1.plot(t, speed, lw=1.5)
    a1.set_ylabel("speed [m/s]")
    a2.plot(t, yaw_rate, lw=1.5)
    a2.axhline(0, color="0.6", lw=0.8)
    a2.set_ylabel("yaw rate [rad/s]\n(+ = left)")
    a2.set_xlabel("time [s]")
    for label, ts in (marks or {}).items():
        for i, x in enumerate(ts):
            for ax in (a1, a2):
                ax.axvline(x, ls="--", lw=0.8, color="0.4", label=label if (i == 0 and ax is a1) else None)
    if marks:
        a1.legend(fontsize=7, frameon=False)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
