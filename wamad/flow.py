"""Optical-flow utilities for the flow-warping baseline.

Warping the last observed segmentation with extrapolated optical flow is the
standard baseline in semantic forecasting (Luc et al., ICCV 2017). Here flow
is OpenCV Farneback. A different flow field (e.g. RAFT) can be passed to
`extrapolate_warp` directly.
"""
from __future__ import annotations

import numpy as np


def farneback_flow(prev_rgb: np.ndarray, next_rgb: np.ndarray, work_width: int = 512) -> np.ndarray:
    """Dense flow (H, W, 2) in pixels mapping next_rgb -> prev_rgb (backward flow):
    pixel p in `next` came from p + flow[p] in `prev`. Computed at `work_width` and
    rescaled to full resolution."""
    import cv2
    h, w = prev_rgb.shape[:2]
    s = min(1.0, work_width / w)
    size = (int(round(w * s)), int(round(h * s)))
    g0 = cv2.cvtColor(cv2.resize(prev_rgb, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(cv2.resize(next_rgb, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g0, None, pyr_scale=0.5, levels=4, winsize=21,
                                        iterations=3, poly_n=7, poly_sigma=1.5, flags=0)
    flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / s
    return flow.astype(np.float32)


def warp_labels(label: np.ndarray, backward_flow: np.ndarray, fill: int = 255) -> np.ndarray:
    """out[p] = label[p + backward_flow[p]] with nearest-neighbour sampling."""
    import cv2
    h, w = label.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xx + backward_flow[..., 0]
    map_y = yy + backward_flow[..., 1]
    return cv2.remap(np.asarray(label, dtype=np.uint8), map_x, map_y, interpolation=cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def extrapolate_warp(last_label: np.ndarray, backward_flow: np.ndarray, steps_ahead: float,
                     fill: int = 255) -> np.ndarray:
    """Constant-velocity extrapolation: the observed per-step motion is assumed to
    continue for `steps_ahead` steps (e.g. 1 for short-term, 3 for mid-term at a
    3-frame step). Out-of-view pixels get `fill` (counted as errors in scoring)."""
    return warp_labels(last_label, backward_flow * float(steps_ahead), fill=fill)


def flow_baseline(ctx_rgb: np.ndarray, ctx_labels: np.ndarray, steps_ahead: float,
                  work_width: int = 512, fill: int = 255) -> np.ndarray:
    """Flow-warping baseline from the last two context frames.

    ctx_rgb: (T, H, W, 3) uint8; ctx_labels: (T, H, W) uint8 at the same resolution.
    """
    flow = farneback_flow(ctx_rgb[-2], ctx_rgb[-1], work_width)
    return extrapolate_warp(ctx_labels[-1], flow, steps_ahead, fill)


def copy_last_baseline(ctx_labels: np.ndarray) -> np.ndarray:
    """Predict that the future equals the last observed segmentation."""
    return np.array(ctx_labels[-1], copy=True)
