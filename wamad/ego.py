"""Ego-action signals: parsing, alignment, normalisation and input interventions.

The action used everywhere is a 2-vector  a = (speed [m/s], yaw_rate [rad/s]),
taken as the mean over the interval being predicted. Both datasets are mapped
to these units so a model trained on one can be run on the other.

Units are verified with scripts/check_cityscapes.py and scripts/prepare_bdd100k.py:
* Cityscapes `vehicle_sequence/*_vehicle.json`: keys `speed`, `yawRate`
  (assumed m/s and rad/s; override with `speed_scale` / `yaw_scale`).
* BDD100K `info/*.json`: `locations` list with `timestamp` [ms], `speed`
  [m/s], `course` [deg]; `gyro` list with x, y, z [rad/s]. Yaw rate is
  derived from the GPS course by default because the phone's mounting,
  and therefore the gyro axes, is unknown.

The intervention helpers at the bottom change the INPUT action only. Their
effect is measured with the standard metrics (mIoU / MO-mIoU) - they are
an input ablation, not a new metric.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTION_DIM = 2
ACTION_NAMES = ("speed_mps", "yaw_rate_radps")


@dataclass
class EgoSignal:
    """Time series of ego speed and yaw rate, time in seconds from clip start."""
    t: np.ndarray
    speed: np.ndarray
    yaw_rate: np.ndarray
    source: str = ""

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=np.float64)
        self.speed = np.asarray(self.speed, dtype=np.float64)
        self.yaw_rate = np.asarray(self.yaw_rate, dtype=np.float64)
        if not (self.t.shape == self.speed.shape == self.yaw_rate.shape):
            raise ValueError("t, speed and yaw_rate must have the same shape")
        order = np.argsort(self.t)
        self.t, self.speed, self.yaw_rate = self.t[order], self.speed[order], self.yaw_rate[order]

    def __len__(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self) else 0.0

    def at(self, times) -> np.ndarray:
        """Linearly interpolated actions at `times`, shape (N, 2). Outside the range -> NaN."""
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        out = np.stack([np.interp(times, self.t, self.speed),
                        np.interp(times, self.t, self.yaw_rate)], axis=-1)
        outside = (times < self.t[0]) | (times > self.t[-1])
        out[outside] = np.nan
        return out

    def interval_mean(self, t0: float, t1: float, n: int = 16) -> np.ndarray:
        """Mean action over [t0, t1] (trapezoid on n samples), shape (2,)."""
        ts = np.linspace(t0, t1, n)
        return np.mean(self.at(ts), axis=0)

    def smoothed(self, window_s: float) -> "EgoSignal":
        """Moving-average smoothing (centred), for noisy phone GPS."""
        if len(self) < 3 or window_s <= 0:
            return self
        dt = np.median(np.diff(self.t))
        k = max(1, int(round(window_s / dt)))
        if k <= 1:
            return self
        ker = np.ones(k) / k
        sm = lambda x: np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), ker, mode="valid")  # noqa: E731
        return EgoSignal(self.t, sm(self.speed), sm(self.yaw_rate), self.source + "+smooth")


# ----------------------------------------------------------- Cityscapes
def load_cityscapes_vehicle(paths_by_frame: dict[int, Path], fps: float,
                            speed_scale: float = 1.0, yaw_scale: float = 1.0) -> EgoSignal:
    """Build an EgoSignal from per-frame Cityscapes vehicle JSON files.

    `paths_by_frame` maps filename frame number -> json path. Time is
    frame_number / fps (only differences matter).
    """
    frames, speed, yaw = [], [], []
    for fr, p in sorted(paths_by_frame.items()):
        d = json.loads(Path(p).read_text())
        if "speed" not in d or "yawRate" not in d:
            raise KeyError(f"{p}: expected keys 'speed' and 'yawRate', got {sorted(d)}")
        frames.append(fr)
        speed.append(float(d["speed"]) * speed_scale)
        yaw.append(float(d["yawRate"]) * yaw_scale)
    return EgoSignal(np.asarray(frames) / fps, speed, yaw, source="cityscapes_vehicle")


# --------------------------------------------------------------- BDD100K
def load_bdd_info(path: Path, yaw_source: str = "course", min_speed_for_course: float = 1.0,
                  smooth_s: float = 1.0) -> EgoSignal:
    """Build an EgoSignal from a BDD100K `info/*.json`.

    yaw_source:
      "course" - d(course)/dt from GPS (robust to phone orientation; default).
                 Course is unreliable when nearly stopped, so yaw rate is set
                 to 0 below `min_speed_for_course` m/s.
      "gyro_z" - raw gyro z axis; only valid if the phone was mounted level.
    Timestamps are converted to seconds relative to `startTime` (or to the
    first sample if absent) so they align with video time.
    """
    d = json.loads(Path(path).read_text())
    # BDD100K marks unavailable GPS readings with negative speed/course; drop them
    # rather than treating them as "stopped".
    locs = [l for l in d.get("locations", [])
            if l.get("speed") is not None and float(l["speed"]) >= 0 and "timestamp" in l]
    if len(locs) < 2:
        raise ValueError(f"{path}: fewer than 2 GPS samples")
    t0_ms = float(d.get("startTime", locs[0]["timestamp"]))
    t = (np.array([l["timestamp"] for l in locs], dtype=np.float64) - t0_ms) / 1000.0
    speed = np.array([max(float(l["speed"]), 0.0) for l in locs])

    if yaw_source == "course":
        course = np.deg2rad(np.array([float(l.get("course", 0.0)) for l in locs]))
        course = np.unwrap(course)
        yaw = np.gradient(course, t)
        # GPS course is compass heading (clockwise); convert to the vehicle
        # convention used here: positive yaw rate = turning LEFT (counter-clockwise).
        yaw = -yaw
        yaw[speed < min_speed_for_course] = 0.0
    elif yaw_source == "gyro_z":
        g = d.get("gyro", [])
        if len(g) < 2:
            raise ValueError(f"{path}: no gyro samples")
        tg = (np.array([s["timestamp"] for s in g], dtype=np.float64) - t0_ms) / 1000.0
        yaw = np.interp(t, tg, np.array([float(s["z"]) for s in g]))
    else:
        raise ValueError(f"unknown yaw_source {yaw_source!r}")

    sig = EgoSignal(t, speed, yaw, source=f"bdd_{yaw_source}")
    return sig.smoothed(smooth_s) if smooth_s else sig


def bdd_signal_quality(sig: EgoSignal, max_gap_s: float = 2.0, max_speed: float = 45.0,
                       max_abs_yaw: float = 1.0) -> dict:
    """Simple checks used to filter clips; returns flags and the reason for rejection."""
    gaps = np.diff(sig.t)
    reasons = []
    if len(sig) < 5:
        reasons.append("too_few_samples")
    if gaps.size and gaps.max() > max_gap_s:
        reasons.append("gps_gap")
    if np.nanmax(sig.speed) > max_speed:
        reasons.append("speed_outlier")
    if np.nanmax(np.abs(sig.yaw_rate)) > max_abs_yaw:
        reasons.append("yaw_outlier")
    return {"ok": not reasons, "reasons": reasons,
            "max_gap_s": float(gaps.max()) if gaps.size else None}


# ---------------------------------------------------------- normalisation
@dataclass
class ActionNormalizer:
    """Per-dimension standardisation fitted on the TRAINING split only."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, actions: np.ndarray, eps: float = 1e-6) -> "ActionNormalizer":
        a = np.asarray(actions, dtype=np.float64).reshape(-1, ACTION_DIM)
        a = a[~np.isnan(a).any(axis=1)]
        return cls(a.mean(axis=0), np.maximum(a.std(axis=0), eps))

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        return (np.asarray(actions) - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z) * self.std + self.mean

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"mean": self.mean.tolist(), "std": self.std.tolist(),
                                          "names": list(ACTION_NAMES)}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ActionNormalizer":
        d = json.loads(Path(path).read_text())
        return cls(np.asarray(d["mean"]), np.asarray(d["std"]))


# ------------------------------------------------- input interventions
# Each returns a new action array of the same shape (..., 2) in PHYSICAL
# units; apply normalisation afterwards. Evaluate with mIoU / MO-mIoU.
def intervene_shuffle(actions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace each sample's actions with another sample's (permutation along axis 0)."""
    actions = np.asarray(actions)
    return actions[rng.permutation(len(actions))]


def intervene_zero_yaw(actions: np.ndarray) -> np.ndarray:
    """Straight driving at the true speed."""
    out = np.array(actions, dtype=np.float64, copy=True)
    out[..., 1] = 0.0
    return out


def intervene_stop(actions: np.ndarray) -> np.ndarray:
    """Ego stationary (speed 0, yaw 0)."""
    return np.zeros_like(np.asarray(actions, dtype=np.float64))


def intervene_mean(actions: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Replace every action with the training-set mean action."""
    return np.broadcast_to(np.asarray(mean, dtype=np.float64), np.shape(actions)).copy()


INTERVENTIONS = ("true", "shuffle", "zero_yaw", "stop", "mean")


def apply_intervention(name: str, actions: np.ndarray, rng: np.random.Generator | None = None,
                       mean: np.ndarray | None = None) -> np.ndarray:
    if name == "true":
        return np.asarray(actions, dtype=np.float64)
    if name == "shuffle":
        return intervene_shuffle(actions, rng or np.random.default_rng(0))
    if name == "zero_yaw":
        return intervene_zero_yaw(actions)
    if name == "stop":
        return intervene_stop(actions)
    if name == "mean":
        if mean is None:
            raise ValueError("'mean' intervention needs the training mean")
        return intervene_mean(actions, mean)
    raise ValueError(f"unknown intervention {name!r}; choose from {INTERVENTIONS}")
