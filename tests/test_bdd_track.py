"""Tests for the BDD100K tracking-frame loader."""
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from wamad import io, protocol
from wamad.data import bdd100k_track as bt, pseudo


def _fake(root: Path, seq="abcd1234-ef567890", split="val", n=60, info_split="train"):
    d = root / "images/track" / split / seq; d.mkdir(parents=True)
    for i in range(1, n + 1):
        Image.fromarray(np.full((18, 32, 3), i, np.uint8)).save(d / f"{seq}-{i:07d}.jpg")
    t0 = 1_000_000
    locs = [{"timestamp": t0 + 1000 * s, "speed": 2.0 * s if s < 5 else -1 if s == 5 else 10.0,
             "course": 90.0, "latitude": 0, "longitude": 0} for s in range(n // 5 + 1)]
    di = root / "info/100k" / info_split; di.mkdir(parents=True)
    (di / f"{seq}.json").write_text(json.dumps({"startTime": t0, "endTime": t0 + 12000, "locations": locs}))
    return seq


def test_track_index_load_and_timing():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        seq = _fake(root)
        paths = bt.TrackPaths(root, root / "pseudo")
        assert paths.info(seq) is not None                    # found in the *other* info split
        assert bt.frame_indices(paths, "val", seq) == list(range(1, 61))
        assert bt.frame_time(6) == 1.0 and bt.frame_time(6, time_offset_s=0.5) == 1.5
        samples, rep = bt.build_index(paths, "val", "mid", stride=10)
        assert rep["rejected"] == 0 and len(samples) >= 4
        s = samples[0]
        assert s.frame_ids["targets"][-1] == s.frame_ids["eval"]
        assert s.frame_ids["eval"] - s.frame_ids["context"][0] == 6
        assert protocol.bdd_track_seconds_ahead(protocol.MID_TERM) == 0.6
        # the GPS reading with speed -1 at t=5 s is dropped, not read as "stopped"
        sig = bt.ego_signal(paths, seq)
        assert 5.0 not in sig.t.tolist() and sig.speed.min() >= 0
        io.save_jsonl(root / "i.jsonl", [x.to_dict() for x in samples])
        assert [bt.TrackSample.from_dict(r) for r in io.load_jsonl(root / "i.jsonl")] == samples
        seg = lambda imgs: [np.full(im.shape[:2], 13, np.uint8) for im in imgs]  # noqa: E731
        pseudo.cache_pseudo_labels(bt.pseudo_label_jobs(paths, samples), seg, root / "pseudo", "fake", log_every=0)
        smp = bt.load_sample(paths, s, load_rgb=True)
        assert smp["ctx_labels"].shape == (4, 18, 32) and smp["targets"].shape == (3, 18, 32)
        assert smp["actions_future"].shape == (3, 2) and smp["meta"]["eval_target"] == "pseudo_label"
