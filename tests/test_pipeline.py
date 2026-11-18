"""Tests for the real-data wrapper, baselines and results summary (numpy only)."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from wamad import io, protocol
from wamad.data import bdd100k_track as bt, cityscapes as cs, pseudo
from wamad.data.forecast_data import ForecastSet, cross_sequence_permutation, flip_sample

H, W = 32, 64


def _seg(imgs):
    out = []
    for im in imgs:
        lab = np.zeros(im.shape[:2], np.uint8); lab[:, : im.shape[1] // 3] = 13
        out.append(lab)
    return out


def fake_cityscapes(root: Path, pseudo_root: Path):
    snippets = [("val", "frankfurt", "000000", 19), ("val", "frankfurt", "000000", 294),
                ("val", "lindau", "000001", 19), ("train", "aachen", "000002", 19), ("train", "aachen", "000003", 19)]
    for k, (sp, city, seq, ann) in enumerate(snippets):
        g = root / "gtFine" / sp / city; g.mkdir(parents=True, exist_ok=True)
        lid = np.full((H, W), 7, np.uint8); lid[:, : W // 3] = 26
        Image.fromarray(lid).save(g / f"{city}_{seq}_{ann:06d}_gtFine_labelIds.png")
        for pos in range(1, 31):
            f = protocol.cityscapes_frame_id(ann, pos); stem = f"{city}_{seq}_{f:06d}"
            d = root / "leftImg8bit_sequence" / sp / city; d.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(pos)
            Image.fromarray((rng.random((H, W, 3)) * 255).astype(np.uint8)).save(d / f"{stem}_leftImg8bit.png")
            v = root / "vehicle_sequence" / sp / city; v.mkdir(parents=True, exist_ok=True)
            (v / f"{stem}_vehicle.json").write_text(json.dumps({"speed": 5.0 + k, "yawRate": 0.05 * (k - 2)}))
    paths = cs.CityscapesPaths(root, pseudo_root)
    ss = cs.build_index(root, "train") + cs.build_index(root, "val")
    pseudo.cache_pseudo_labels(cs.pseudo_label_jobs(paths, ss), _seg, pseudo_root, "fake", log_every=0)


def fake_bdd(root: Path, pseudo_root: Path, index_dir: Path):
    t0 = 1_000_000
    for k, split in enumerate(["val", "val", "train"]):
        seq = f"{k:08x}-{k:08x}"
        d = root / "images/track" / split / seq; d.mkdir(parents=True)
        for i in range(1, 41):
            Image.fromarray(np.full((18, 32, 3), i * 5, np.uint8)).save(d / f"{seq}-{i:07d}.jpg")
        di = root / "info/100k" / split; di.mkdir(parents=True, exist_ok=True)
        locs = [{"timestamp": t0 + 1000 * s, "speed": 6.0 + k, "course": 90.0 + 5 * s * (k - 1),
                 "latitude": 0, "longitude": 0} for s in range(10)]
        (di / f"{seq}.json").write_text(json.dumps({"startTime": t0, "endTime": t0 + 9000, "locations": locs}))
    paths = bt.TrackPaths(root, pseudo_root)
    index_dir.mkdir(parents=True)
    allss = []
    for sp in ("train", "val"):
        for hn in protocol.HORIZONS:
            ss, _ = bt.build_index(paths, sp, hn, stride=5)
            io.save_jsonl(index_dir / f"index_{sp}_{hn}.jsonl", [s.to_dict() for s in ss])
            allss += ss
    pseudo.cache_pseudo_labels(bt.pseudo_label_jobs(paths, allss), _seg, pseudo_root, "fake", log_every=0)


def test_flip_negates_yaw_and_mirrors():
    d = {"ctx_labels": np.arange(2 * 3 * 4).reshape(2, 3, 4), "eval_gt": np.arange(12).reshape(3, 4),
         "actions_past": np.array([[5.0, 0.2]]), "actions_future": np.array([[5.0, -0.1]])}
    f = flip_sample(d)
    assert (f["ctx_labels"][:, :, 0] == d["ctx_labels"][:, :, -1]).all()
    assert (f["eval_gt"][:, 0] == d["eval_gt"][:, -1]).all()
    assert f["actions_past"][0, 1] == -0.2 and f["actions_future"][0, 0] == 5.0
    assert d["actions_past"][0, 1] == 0.2          # input untouched


def test_cross_sequence_permutation():
    keys = ["a"] * 5 + ["b"] * 3 + ["c"] * 2
    p = cross_sequence_permutation(keys, seed=1)
    assert all(keys[p[i]] != keys[i] for i in range(10))
    # with many balanced sequences it stays (almost) a permutation
    big = [i // 10 for i in range(1000)]
    q = cross_sequence_permutation(big, seed=0)
    assert all(big[q[i]] != big[i] for i in range(1000)) and len(set(q.tolist())) > 950


def test_forecast_sets_both_datasets():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fake_cityscapes(d / "cs", d / "pseudo")
        fake_bdd(d / "bdd", d / "pseudo", d / "idx")
        kw = dict(cityscapes_root=d / "cs", bdd_root=d / "bdd", pseudo_root=d / "pseudo", bdd_index_dir=d / "idx")
        tr = ForecastSet("cityscapes", "train", "mid", train=True, flip=True, size_hw=(16, 32), **kw)
        x = tr[0]
        assert x["ctx_labels"].shape == (4, 16, 32) and x["targets"].shape == (3, 16, 32) and "eval_gt" not in x
        assert tr.all_actions().shape == (2 * 7, 2)
        va = ForecastSet("cityscapes", "val", "short", train=False, size_hw=(16, 32), **kw)
        y = va[0]
        assert y["eval_gt"].shape == (H, W) and y["targets"].shape == (1, 16, 32)
        sh = ForecastSet("cityscapes", "val", "short", train=False, size_hw=(16, 32), shuffle_actions=True, **kw)
        for i in range(len(sh)):
            assert sh._key(sh.samples[sh.perm[i]]) != sh._key(sh.samples[i])
            assert np.allclose(sh[i]["actions_future"], va.future_actions(int(sh.perm[i])))
        b = ForecastSet("bdd100k", "val", "mid", train=False, **kw)
        z = b[0]
        assert z["ctx_labels"].shape == (4, 288, 512) and z["eval_gt"].shape == (18, 32)
        bs = ForecastSet("bdd100k", "val", "mid", train=False, shuffle_actions=True, **kw)
        assert all(bs._key(bs.samples[bs.perm[i]]) != bs._key(bs.samples[i]) for i in range(len(bs)))


def test_baselines_and_summary():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import baselines as pb
    import summarize_results as sr
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fake_cityscapes(d / "cs", d / "pseudo")
        fake_bdd(d / "bdd", d / "pseudo", d / "idx")
        pb.CITYSCAPES_ROOT, pb.BDD_ROOT, pb.PSEUDO_ROOT, pb.BDD_INDEX_DIR = d / "cs", d / "bdd", d / "pseudo", d / "idx"
        pb.OUT_DIR, pb.NUM_PROCS = d / "forecast" / "baselines", 2
        assert pb.main() == 0
        rows = io.load_jsonl(d / "forecast" / "baselines" / "results.jsonl")
        assert len(rows) == 2 * 2 * 2
        cl = [r for r in rows if r["model"] == "copy_last" and r["eval_set"] == "cityscapes"]
        assert all(abs(r["mIoU"] - 1.0) < 1e-9 for r in cl)   # fake gt == fake pseudo-labels
        assert pb.main() == 0 and len(io.load_jsonl(d / "forecast" / "baselines" / "results.jsonl")) == 8  # a rerun replaces the results
        # two fake model runs with 2 seeds
        for seed, val in ((0, 0.60), (1, 0.62)):
            lg = io.ResultsLogger(d / "forecast" / f"cityscapes_film_s{seed}")
            for iv, off in (("true", 0.0), ("shuffle", -0.05)):
                lg.log(run_id="x", train_set="cityscapes", eval_set="cityscapes", horizon="mid", model="film",
                       intervention=iv, seed=seed, summary={"mIoU": val + off, "MO-mIoU": val - 0.1 + off,
                                                            "pixel_acc": 0.9, "n_images": 3})
        sr.FORECAST_DIR, sr.OUT_DIR = d / "forecast", d / "forecast" / "summary"
        sr.SYNTHETIC_REPORT, sr.EFFICIENCY_JSON = d / "none.json", d / "none.json"
        assert sr.main() == 0
        out = d / "forecast" / "summary"
        txt = (out / "tab_main.tex").read_text()
        assert "61.00" in txt and "1.41" in txt and "Copy-last" in txt
        assert (out / "tab_interventions.tex").exists() and (out / "summary.json").exists()
