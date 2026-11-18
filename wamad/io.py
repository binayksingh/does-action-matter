"""File I/O: images, label maps, JSON, and results logging."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .labels import validate_trainids


def read_rgb(path: Path, size_hw: tuple[int, int] | None = None) -> np.ndarray:
    """RGB uint8 (H, W, 3). Optional resize with bilinear interpolation."""
    img = Image.open(path).convert("RGB")
    if size_hw is not None:
        img = img.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    return np.asarray(img)


def read_label(path: Path, size_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Single-channel label map uint8 (H, W). Resize uses NEAREST (never interpolate labels)."""
    img = Image.open(path)
    if img.mode not in ("L", "P", "I", "I;16"):
        raise ValueError(f"{path}: expected a single-channel label image, got mode {img.mode}")
    if size_hw is not None:
        img = img.resize((size_hw[1], size_hw[0]), Image.NEAREST)
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def write_label(path: Path, label: np.ndarray, check: bool = True) -> None:
    """Save a train-ID map (0..18, 255) as a lossless 8-bit PNG."""
    label = np.asarray(label, dtype=np.uint8)
    if check:
        validate_trainids(label)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label, mode="L").save(path)


def resize_label(label: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a label map, e.g. upsampling predictions to 1024x2048."""
    return np.asarray(Image.fromarray(np.asarray(label, dtype=np.uint8), mode="L")
                      .resize((size_hw[1], size_hw[0]), Image.NEAREST))


def save_json(path: Path, obj, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, default=_json_default))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def save_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


class ResultsLogger:
    """Append-only results log: one JSON line per evaluation, plus a CSV table.

    Every row records run id, dataset, horizon, model, intervention and the
    metric values, so all tables can be rebuilt from this file alone.
    """
    CSV_FIELDS = ("time", "run_id", "train_set", "eval_set", "horizon", "model",
                  "intervention", "seed", "mIoU", "MO-mIoU", "pixel_acc", "n_images", "notes")

    def __init__(self, out_dir: Path, fresh: bool = False):
        """`fresh=True` removes earlier results in `out_dir` before logging."""
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.out_dir / "results.jsonl"
        self.csv = self.out_dir / "results.csv"
        if fresh:
            self.jsonl.unlink(missing_ok=True)
            self.csv.unlink(missing_ok=True)

    def log(self, *, run_id: str, train_set: str, eval_set: str, horizon: str, model: str,
            summary: dict, intervention: str = "true", seed: int | None = None,
            notes: str = "", extra: dict | None = None) -> dict:
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run_id": run_id, "train_set": train_set,
               "eval_set": eval_set, "horizon": horizon, "model": model,
               "intervention": intervention, "seed": seed, "mIoU": summary["mIoU"],
               "MO-mIoU": summary["MO-mIoU"], "pixel_acc": summary["pixel_acc"],
               "n_images": summary["n_images"], "notes": notes}
        with self.jsonl.open("a") as f:
            f.write(json.dumps({**row, "per_class_IoU": summary.get("per_class_IoU"),
                                **(extra or {})}, default=_json_default) + "\n")
        new = not self.csv.exists()
        with self.csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        return row

    def rows(self) -> list[dict]:
        return load_jsonl(self.jsonl) if self.jsonl.exists() else []
