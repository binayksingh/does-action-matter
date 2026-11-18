"""Synthetic driving world (numpy only).

A camera looks at a wide label panorama (sky, buildings with gaps, poles,
vegetation, sidewalk, road) with moving cars and pedestrians. Ego motion
changes the view exactly as the action says:
  * yaw rate  -> horizontal pan  (turning LEFT shifts the scene RIGHT)
  * speed     -> zoom about the vanishing point (forward motion)
Frames are 0.2 s apart (the BDD100K step). The FUTURE action is drawn
independently of the past one 70 % of the time, so past frames alone cannot
predict where the view goes next - only the action can. A model that uses
the action should beat one that does not, and should get worse when given
a wrong action. scripts/synthetic.py tests this.

Samples have the same keys as the real-data loaders:
  ctx_labels (4,H,W), targets (3,H,W), eval_gt, actions_past (4,2),
  actions_future (3,2) in physical units (m/s, rad/s), ctx_rgb (4,H,W,3).
"""
from __future__ import annotations

import math

import numpy as np

# Cityscapes train IDs used in the toy world
ROAD, SIDEWALK, BUILDING, POLE, VEGETATION, SKY, PERSON, CAR = 0, 1, 2, 5, 8, 10, 11, 13

DT = 0.2                      # seconds between frames
CONTEXT, FUTURE = 4, 3


class ToyWorld:
    def __init__(self, H: int = 64, W: int = 128, pano_w: int = 1024, hfov_rad: float = math.pi / 2,
                 zoom_dist_m: float = 30.0, p_same_action: float = 0.3):
        self.H, self.W, self.PW = H, W, pano_w
        self.horizon = int(0.45 * H)
        self.px_per_rad = W / hfov_rad
        self.zoom_dist = zoom_dist_m
        self.p_same = p_same_action
        from .labels import PALETTE
        self.palette = PALETTE.astype(np.float32)

    # ---------------------------------------------------------- world
    def _panorama(self, rng) -> np.ndarray:
        H, PW, hz = self.H, self.PW, self.horizon
        pano = np.full((H, PW), SKY, np.uint8)
        side_top = hz + int(0.12 * H)
        pano[hz:, :] = SIDEWALK
        pano[side_top:, :] = ROAD
        x = 0
        while x < PW:                                     # buildings / vegetation / gaps
            w = int(rng.integers(20, 90))
            kind = rng.choice([BUILDING, BUILDING, VEGETATION, SKY])
            if kind != SKY:
                top = int(rng.integers(2, hz - 2))
                pano[top:hz, x:x + w] = kind
            x += w
        for px in rng.integers(0, PW, size=PW // 60):     # poles
            pano[int(hz * 0.3):side_top, px:px + 2] = POLE
        return pano

    def _objects(self, rng) -> list[dict]:
        objs = []
        for _ in range(int(rng.integers(14, 26))):
            is_car = rng.random() < 0.6
            h = int(rng.integers(6, 14)) if is_car else int(rng.integers(8, 14))
            w = int(h * rng.uniform(1.4, 2.2)) if is_car else max(2, h // 3)
            y_bottom = int(rng.integers(self.horizon + int(0.15 * self.H), self.H)) if is_car \
                else self.horizon + int(0.12 * self.H) + int(rng.integers(-1, 3))
            objs.append({"cls": CAR if is_car else PERSON, "x": float(rng.uniform(0, self.PW)),
                         "vx": float(rng.uniform(-3, 3) if is_car else rng.uniform(-1, 1)),
                         "y0": y_bottom - h, "y1": y_bottom, "w": w})
        return objs

    @staticmethod
    def _draw(pano, objs, k):
        out = pano.copy()
        PW = pano.shape[1]
        for o in objs:
            x0 = int(round(o["x"] + o["vx"] * k)) % PW
            xs = np.arange(x0, x0 + o["w"]) % PW
            out[o["y0"]:o["y1"], xs] = o["cls"]
        return out

    def _project(self, pano, cx, z, fill_top=SKY, fill_bottom=ROAD):
        H, W, hz = self.H, self.W, self.horizon
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        px = np.round(cx + (u - W / 2) / z).astype(np.int64) % self.PW
        py = np.round(hz + (v - hz) / z).astype(np.int64)
        out = np.empty((H, W), pano.dtype)
        inside = (py >= 0) & (py < H)
        out[inside] = pano[py[inside]][:, px]
        out[py < 0] = fill_top
        out[py >= H] = fill_bottom
        return out

    # -------------------------------------------------------- actions
    def _random_action(self, rng) -> np.ndarray:
        speed = rng.uniform(0, 15)
        yaw = rng.normal(0, 0.03) if rng.random() < 0.5 else rng.uniform(-0.3, 0.3)
        return np.array([speed, yaw])

    # --------------------------------------------------------- sample
    def sample(self, seed: int, with_rgb: bool = False) -> dict:
        rng = np.random.default_rng(seed)
        pano, objs = self._panorama(rng), self._objects(rng)
        tex = rng.random((self.H, self.PW)).astype(np.float32) if with_rgb else None
        a_p = self._random_action(rng)
        a_f = a_p.copy() if rng.random() < self.p_same else self._random_action(rng)
        n = CONTEXT + FUTURE
        # action over interval k -> k+1 (small jitter so it is not perfectly constant)
        acts = np.array([(a_p if k < CONTEXT - 1 else a_f) + rng.normal(0, [0.2, 0.005]) for k in range(n - 1)])
        acts[:, 0] = np.clip(acts[:, 0], 0, None)
        cx, z = float(rng.uniform(0, self.PW)), 1.0
        frames, rgbs = [], []
        for k in range(n):
            frames.append(self._project(self._draw(pano, objs, k), cx, z))
            if with_rgb:
                t = self._project((tex * 255).astype(np.uint8), cx, z, 128, 128).astype(np.float32) / 255
                rgbs.append(np.clip(self.palette[frames[-1]] * (0.6 + 0.4 * t[..., None]), 0, 255).astype(np.uint8))
            if k < n - 1:
                speed, yaw = acts[k]
                cx -= self.px_per_rad * yaw * DT
                z *= math.exp(speed * DT / self.zoom_dist)
        frames = np.stack(frames)
        out = {
            "ctx_labels": frames[:CONTEXT],
            "targets": frames[CONTEXT:],
            # instantaneous action at each context frame (interval starting there; last uses the
            # previous interval, as the real loaders interpolate from the past)
            "actions_past": np.stack([acts[min(k, CONTEXT - 2)] for k in range(CONTEXT)]),
            "actions_future": acts[CONTEXT - 1:],
            "meta": {"seed": seed, "same_action": bool(np.allclose(a_p, a_f))},
        }
        if with_rgb:
            out["ctx_rgb"] = np.stack(rgbs[:CONTEXT])
        return out


def eval_target(sample: dict, horizon: str) -> np.ndarray:
    """Scored frame: first future frame for 'short', last for 'mid'."""
    return sample["targets"][0] if horizon == "short" else sample["targets"][-1]


def steps(horizon: str) -> int:
    return 1 if horizon == "short" else FUTURE
