"""Evaluation protocol: frame indices and horizons for both datasets.

Cityscapes follows the standard semantic-forecasting protocol of
Luc et al., ICCV 2017 ("Predicting Deeper into the Future of Semantic
Segmentation"), used by later work:

* Each sequence snippet has 30 frames; frame 20 (1-based) carries the
  gtFine annotation. Frames are ~17 Hz, so a 3-frame step is ~0.18 s.
* Short-term: context frames 8, 11, 14, 17 -> predict frame 20 (0.18 s ahead).
* Mid-term:   context frames 2, 5, 8, 11 -> predict 14, 17, 20
  autoregressively; evaluated on frame 20 (0.54 s ahead).
* Metrics: mIoU over 19 classes and MO-mIoU over the 8 movable classes,
  on the Cityscapes val split (500 snippets), full resolution 1024x2048.

BDD100K has no standard semantic-forecasting protocol, so we transfer the
Cityscapes one in TIME: the same 4-frame context, the same step, the same
horizons, mapped to the nearest BDD100K video frame.
"""
from __future__ import annotations

from dataclasses import dataclass

CITYSCAPES_FPS = 17.0          # approximate camera rate of the sequence snippets
CITYSCAPES_SNIPPET_LEN = 30
CITYSCAPES_ANNOTATED_POS = 20  # 1-based position of the gtFine frame in a snippet
CITYSCAPES_EVAL_RES = (1024, 2048)  # (H, W)


@dataclass(frozen=True)
class Horizon:
    name: str
    context_pos: tuple[int, ...]   # 1-based snippet positions of input frames
    target_pos: tuple[int, ...]    # positions predicted (autoregressive chain)
    eval_pos: int                  # position that is scored

    @property
    def step(self) -> int:
        return self.context_pos[1] - self.context_pos[0]

    @property
    def seconds_ahead(self) -> float:
        return (self.eval_pos - self.context_pos[-1]) / CITYSCAPES_FPS

    @property
    def n_autoregressive_steps(self) -> int:
        return len(self.target_pos)


SHORT_TERM = Horizon("short", (8, 11, 14, 17), (20,), 20)
MID_TERM = Horizon("mid", (2, 5, 8, 11), (14, 17, 20), 20)
HORIZONS = {h.name: h for h in (SHORT_TERM, MID_TERM)}


def cityscapes_frame_id(annotated_frame_id: int, pos: int) -> int:
    """Frame number in the filename for 1-based snippet position `pos`.

    Example: annotated frame 000019 -> position 20; position 8 -> 000007.
    """
    if not 1 <= pos <= CITYSCAPES_SNIPPET_LEN:
        raise ValueError(f"position {pos} outside 1..{CITYSCAPES_SNIPPET_LEN}")
    return annotated_frame_id - CITYSCAPES_ANNOTATED_POS + pos


def cityscapes_frame_ids(annotated_frame_id: int, horizon: Horizon) -> dict:
    """Filename frame numbers for context, autoregressive targets and the scored frame."""
    f = lambda p: cityscapes_frame_id(annotated_frame_id, p)  # noqa: E731
    return {
        "context": [f(p) for p in horizon.context_pos],
        "targets": [f(p) for p in horizon.target_pos],
        "eval": f(horizon.eval_pos),
    }


# ---------------------------------------------------------------- BDD100K
def pos_to_relative_seconds(pos: int, eval_pos: int = CITYSCAPES_ANNOTATED_POS) -> float:
    """Time of snippet position `pos` relative to the scored frame (negative = past)."""
    return (pos - eval_pos) / CITYSCAPES_FPS


def bdd_frame_indices(target_time_s: float, horizon: Horizon, video_fps: float) -> dict:
    """Nearest BDD100K video-frame indices reproducing the Cityscapes timing.

    `target_time_s` is the time (from the video start) of the scored frame.
    Returns frame indices plus the actual time offsets so the deviation from
    the Cityscapes timing can be reported.
    """
    def idx(pos: int) -> int:
        return int(round((target_time_s + pos_to_relative_seconds(pos, horizon.eval_pos)) * video_fps))

    ctx = [idx(p) for p in horizon.context_pos]
    tgt = [idx(p) for p in horizon.target_pos]
    ev = idx(horizon.eval_pos)
    if min(ctx) < 0:
        raise ValueError("context starts before the video; choose a later target time")
    return {
        "context": ctx, "targets": tgt, "eval": ev,
        "actual_offsets_s": [(i - ev) / video_fps for i in ctx + tgt],
    }


def describe() -> str:
    lines = ["Evaluation protocol:"]
    for h in HORIZONS.values():
        lines.append(f"  {h.name:5s}: context {h.context_pos} -> targets {h.target_pos}, "
                     f"scored {h.eval_pos} (~{h.seconds_ahead:.2f} s ahead, "
                     f"{h.n_autoregressive_steps} AR step(s))")
    lines.append("  metrics: mIoU (19 cls), MO-mIoU (8 movable cls), pixel acc; Cityscapes val @ 1024x2048")
    return "\n".join(lines)


# --------------------------------------------- BDD100K tracking frames (5 fps)
# The tracking subset ships frames at 5 fps (0.2 s apart). The Cityscapes
# protocol uses a 3-frame step at ~17 Hz (~0.18 s), so ONE BDD frame is the
# nearest equivalent step. Expressed in steps relative to the scored frame:
#   short: context -4,-3,-2,-1 -> predict 0             (0.2 s ahead)
#   mid:   context -6,-5,-4,-3 -> predict -2,-1,0 (AR)  (0.6 s ahead)
BDD_TRACK_FPS = 5.0


def step_offsets(horizon: Horizon) -> dict:
    """Context/target offsets in protocol steps relative to the scored frame."""
    rel = lambda p: (p - horizon.eval_pos) // horizon.step  # noqa: E731
    for p in horizon.context_pos + horizon.target_pos:
        if (p - horizon.eval_pos) % horizon.step:
            raise ValueError("positions are not on the step grid")
    return {"context": [rel(p) for p in horizon.context_pos],
            "targets": [rel(p) for p in horizon.target_pos]}


def bdd_track_frame_ids(eval_idx: int, horizon: Horizon, frames_per_step: int = 1) -> dict:
    """1-based BDD tracking frame indices for a scored frame `eval_idx`."""
    off = step_offsets(horizon)
    return {"context": [eval_idx + frames_per_step * o for o in off["context"]],
            "targets": [eval_idx + frames_per_step * o for o in off["targets"]],
            "eval": eval_idx}


def bdd_track_seconds_ahead(horizon: Horizon, fps: float = BDD_TRACK_FPS, frames_per_step: int = 1) -> float:
    return -step_offsets(horizon)["context"][-1] * frames_per_step / fps
