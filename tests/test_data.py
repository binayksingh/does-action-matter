"""End-to-end tests of the dataset utilities on tiny synthetic folder trees."""
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from wamad import io, metrics, protocol
from wamad.data import cityscapes, pseudo

H, W = 32, 64


def _fake_cityscapes(root: Path, split="val", city="frankfurt", seq="000000", ann=19):
    gt_dir = root / "gtFine" / split / city
    gt_dir.mkdir(parents=True)
    lid = np.full((H, W), 7, np.uint8); lid[:, W // 2:] = 26          # road | car
    Image.fromarray(lid).save(gt_dir / f"{city}_{seq}_{ann:06d}_gtFine_labelIds.png")
    for pos in range(1, 31):
        f = protocol.cityscapes_frame_id(ann, pos)
        stem = f"{city}_{seq}_{f:06d}"
        d = root / "leftImg8bit_sequence" / split / city; d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((H, W, 3), pos * 8, np.uint8)).save(d / f"{stem}_leftImg8bit.png")
        v = root / "vehicle_sequence" / split / city; v.mkdir(parents=True, exist_ok=True)
        (v / f"{stem}_vehicle.json").write_text(json.dumps({"speed": float(pos), "yawRate": 0.02}))
    return cityscapes.CityscapesSample(split, city, seq, ann)


def _seg_fn(imgs):
    out = []
    for im in imgs:
        lab = np.zeros(im.shape[:2], np.uint8); lab[:, im.shape[1] // 2:] = 13
        out.append(lab)
    return out


def test_cityscapes_index_pseudo_cache_and_load():
    with tempfile.TemporaryDirectory() as d:
        root, proot = Path(d) / "cs", Path(d) / "pseudo"
        s = _fake_cityscapes(root)
        idx = cityscapes.build_index(root, "val")
        assert idx == [s]
        paths = cityscapes.CityscapesPaths(root, proot)
        assert cityscapes.check_sample_files(paths, s, protocol.MID_TERM) == []
        jobs = cityscapes.pseudo_label_jobs(paths, idx)
        # union of short {8,11,14,17,20} and mid {2,5,8,11,14,17,20} -> 7 unique frames
        assert len(jobs) == len({p for _, p in jobs}) == 7
        rep = pseudo.cache_pseudo_labels(jobs, _seg_fn, proot, "fake-seg-v1", batch_size=3, log_every=0)
        assert rep["written"] == 7
        assert pseudo.cache_pseudo_labels(jobs, _seg_fn, proot, "fake-seg-v1")["written"] == 0
        try:
            pseudo.cache_pseudo_labels(jobs, _seg_fn, proot, "other-seg")
        except RuntimeError:
            pass
        else:
            raise AssertionError("mixing segmenters must be refused")

        smp = cityscapes.load_sample(paths, s, "mid", size_hw=(16, 32), load_rgb=True)
        assert smp["ctx_labels"].shape == (4, 16, 32) and smp["targets"].shape == (3, 16, 32)
        assert smp["eval_gt"].shape == (H, W)                      # full res for scoring
        assert set(np.unique(smp["eval_gt"])) == {0, 13}
        assert smp["ctx_rgb"].shape == (4, 16, 32, 3)
        # speed == snippet position; mid-term future intervals 11->14, 14->17, 17->20
        assert np.allclose(smp["actions_future"][:, 0], [12.5, 15.5, 18.5], atol=1e-6)
        assert np.allclose(smp["actions_past"][:, 0], [2, 5, 8, 11])

        acc = metrics.SegMetricAccumulator()
        pred = io.resize_label(smp["ctx_labels"][-1], smp["eval_gt"].shape)
        acc.update(pred, smp["eval_gt"])
        assert acc.miou() == 1.0
