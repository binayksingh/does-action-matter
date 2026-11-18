"""Tests for labels, metrics, protocol, ego signals, flow, io and viz.

Run with `pytest tests/`.
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from wamad import ego, flow, io, labels, metrics, protocol, repro, viz


# ------------------------------------------------------------------ labels
def test_labelid_mapping_matches_official_table():
    lid = np.array([[7, 8, 26, 33], [0, 9, 29, 255]], dtype=np.uint8)
    tid = labels.labelids_to_trainids(lid)
    assert tid.tolist() == [[0, 1, 13, 18], [255, 255, 255, 255]]
    assert len(labels.CLASS_NAMES) == 19 and labels.PALETTE.shape == (19, 3)
    assert labels.MOVING_CLASS_IDS == (11, 12, 13, 14, 15, 16, 17, 18)


def test_validate_trainids_rejects_bad_values():
    labels.validate_trainids(np.array([0, 18, 255], dtype=np.uint8))
    try:
        labels.validate_trainids(np.array([19], dtype=np.uint8))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ----------------------------------------------------------------- metrics
def test_perfect_prediction_gives_miou_one():
    gt = np.random.default_rng(0).integers(0, 19, (64, 64)).astype(np.uint8)
    acc = metrics.SegMetricAccumulator()
    acc.update(gt, gt)
    assert abs(acc.miou() - 1.0) < 1e-12 and abs(acc.mo_miou() - 1.0) < 1e-12
    assert abs(acc.pixel_accuracy() - 1.0) < 1e-12


def test_iou_hand_computed():
    # class 0: tp=2, fp=1, fn=1 -> 0.5 ; class 1: tp=1, fp=1, fn=1 -> 1/3
    gt = np.array([0, 0, 0, 1, 1, 255])
    pr = np.array([0, 0, 1, 0, 1, 1])
    acc = metrics.SegMetricAccumulator(num_classes=2)
    acc.update(pr, gt)
    iou = acc.per_class_iou()
    assert np.allclose(iou, [0.5, 1 / 3])
    assert abs(acc.pixel_accuracy() - 3 / 5) < 1e-12  # ignore pixel excluded


def test_ignore_prediction_counts_as_error():
    gt = np.array([0, 0, 1, 1])
    pr = np.array([0, 255, 1, 1])
    acc = metrics.SegMetricAccumulator(num_classes=2)
    acc.update(pr, gt)
    assert np.allclose(acc.per_class_iou(), [0.5, 1.0])
    assert abs(acc.pixel_accuracy() - 0.75) < 1e-12


def test_dataset_level_accumulation_not_per_image_mean():
    # image A: class 0 perfect on 100 px. image B: class 0 IoU 0 on 1 px.
    a_gt = np.zeros(100, int); a_pr = np.zeros(100, int)
    b_gt = np.zeros(1, int); b_pr = np.ones(1, int)
    acc = metrics.SegMetricAccumulator(num_classes=2)
    acc.update(a_pr, a_gt); acc.update(b_pr, b_gt)
    # dataset-level: class0 tp=100, fn=1 -> 100/101; class1 tp=0, fp=1 -> 0
    assert np.allclose(acc.per_class_iou(), [100 / 101, 0.0])


def test_absent_classes_skipped_and_merge():
    acc1 = metrics.SegMetricAccumulator(); acc2 = metrics.SegMetricAccumulator()
    acc1.update(np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8))
    acc2.update(np.full((4, 4), 13, np.uint8), np.full((4, 4), 13, np.uint8))
    acc1.merge(acc2)
    s = acc1.summary()
    assert s["mIoU"] == 1.0 and s["MO-mIoU"] == 1.0 and s["n_images"] == 2
    assert s["per_class_IoU"]["sky"] is None
    assert "mIoU 100.00" in metrics.format_summary(s)


def test_batch_update_counts_images():
    acc = metrics.SegMetricAccumulator()
    acc.update(np.zeros((3, 8, 8), np.uint8), np.zeros((3, 8, 8), np.uint8))
    assert acc.n_images == 3


# ---------------------------------------------------------------- protocol
def test_protocol_frames_match_luc2017():
    assert protocol.SHORT_TERM.context_pos == (8, 11, 14, 17)
    assert protocol.MID_TERM.context_pos == (2, 5, 8, 11)
    assert protocol.MID_TERM.target_pos == (14, 17, 20)
    assert abs(protocol.SHORT_TERM.seconds_ahead - 3 / 17) < 1e-9
    assert abs(protocol.MID_TERM.seconds_ahead - 9 / 17) < 1e-9
    ids = protocol.cityscapes_frame_ids(19, protocol.SHORT_TERM)
    assert ids == {"context": [7, 10, 13, 16], "targets": [19], "eval": 19}


def test_bdd_frame_timing_close_to_cityscapes():
    ids = protocol.bdd_frame_indices(10.0, protocol.MID_TERM, video_fps=30.0)
    assert ids["eval"] == 300
    want = [protocol.pos_to_relative_seconds(p) for p in protocol.MID_TERM.context_pos + protocol.MID_TERM.target_pos]
    assert np.max(np.abs(np.array(ids["actual_offsets_s"]) - np.array(want))) <= 0.5 / 30 + 1e-9
    try:
        protocol.bdd_frame_indices(0.5, protocol.MID_TERM, 30.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for context before video start")


# --------------------------------------------------------------------- ego
def test_ego_interp_and_interval_mean():
    sig = ego.EgoSignal(t=[0, 1, 2], speed=[0, 10, 20], yaw_rate=[0, 0.1, 0.2])
    assert np.allclose(sig.at([0.5]), [[5.0, 0.05]])
    assert np.isnan(sig.at([3.0])).all()
    assert np.allclose(sig.interval_mean(0, 2), [10.0, 0.1])


def test_cityscapes_vehicle_parser():
    with tempfile.TemporaryDirectory() as d:
        paths = {}
        for f in range(5):
            p = Path(d) / f"{f}.json"
            p.write_text(json.dumps({"speed": 2.0 * f, "yawRate": 0.01, "gpsHeading": 0}))
            paths[f] = p
        sig = ego.load_cityscapes_vehicle(paths, fps=17.0)
        assert np.allclose(sig.t, np.arange(5) / 17.0) and sig.speed[-1] == 8.0


def _bdd_info(turn_left: bool) -> dict:
    # 20 s at 1 Hz, 10 m/s, heading changing 5 deg/s (compass, clockwise positive)
    sign = -1 if turn_left else 1
    return {"startTime": 1_000_000, "locations": [
        {"timestamp": 1_000_000 + 1000 * i, "speed": 10.0, "course": (90 + sign * 5 * i) % 360,
         "latitude": 0, "longitude": 0, "accuracy": 5} for i in range(20)]}


def test_bdd_info_parser_sign_and_wraparound():
    with tempfile.TemporaryDirectory() as d:
        for left in (True, False):
            p = Path(d) / "x.json"
            p.write_text(json.dumps(_bdd_info(left)))
            sig = ego.load_bdd_info(p, smooth_s=0)
            expected = np.deg2rad(5) * (1 if left else -1)
            assert np.allclose(sig.yaw_rate, expected, atol=1e-6), sig.yaw_rate
            assert sig.t[0] == 0.0 and np.allclose(sig.speed, 10.0)
            assert ego.bdd_signal_quality(sig)["ok"]


def test_normalizer_roundtrip_and_interventions():
    rng = np.random.default_rng(0)
    a = np.stack([rng.uniform(0, 20, 50), rng.normal(0, 0.1, 50)], -1)
    n = ego.ActionNormalizer.fit(a)
    assert np.allclose(n.inverse(n(a)), a)
    with tempfile.TemporaryDirectory() as d:
        n.save(Path(d) / "n.json")
        n2 = ego.ActionNormalizer.load(Path(d) / "n.json")
        assert np.allclose(n2.mean, n.mean)
    assert (ego.apply_intervention("zero_yaw", a)[:, 1] == 0).all()
    assert (ego.apply_intervention("stop", a) == 0).all()
    sh = ego.apply_intervention("shuffle", a, rng=np.random.default_rng(1))
    assert sorted(sh[:, 0].tolist()) == sorted(a[:, 0].tolist())
    assert np.allclose(ego.apply_intervention("mean", a, mean=n.mean), n.mean)


# -------------------------------------------------------------------- flow
def test_flow_warp_recovers_known_shift():
    # textured image shifted right by 4 px; extrapolating one step should shift labels right by 4
    rng = np.random.default_rng(0)
    base = (rng.random((96, 160)) * 255).astype(np.uint8)
    import cv2
    base = cv2.GaussianBlur(base, (0, 0), 2)
    f0 = np.stack([base] * 3, -1)
    f1 = np.roll(f0, 4, axis=1)
    lab = np.zeros((96, 160), np.uint8); lab[:, 60:80] = 13
    lab1 = np.roll(lab, 4, axis=1)
    fl = flow.farneback_flow(f0, f1, work_width=160)
    assert abs(np.median(fl[10:-10, 10:-10, 0]) + 4) < 0.5   # backward flow points left
    pred = flow.extrapolate_warp(lab1, fl, 1)
    assert (pred[20:-20, 68:88] == 13).mean() > 0.9   # 64..83 moved 4 px further
    assert (flow.copy_last_baseline(np.stack([lab, lab1])) == lab1).all()


# --------------------------------------------------------------- io / viz
def test_label_io_roundtrip_and_nearest_resize():
    lab = np.random.default_rng(0).integers(0, 19, (32, 64)).astype(np.uint8)
    lab[0, 0] = 255
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a" / "x.png"
        io.write_label(p, lab)
        assert (io.read_label(p) == lab).all()
        small = io.read_label(p, (16, 32))
        assert set(np.unique(small)) <= set(np.unique(lab))
        up = io.resize_label(small, (32, 64))
        assert up.shape == (32, 64)


def test_results_logger():
    with tempfile.TemporaryDirectory() as d:
        acc = metrics.SegMetricAccumulator()
        acc.update(np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8))
        lg = io.ResultsLogger(Path(d))
        lg.log(run_id="r1", train_set="cityscapes", eval_set="cityscapes", horizon="short",
               model="copy_last", summary=acc.summary())
        lg.log(run_id="r1", train_set="cityscapes", eval_set="cityscapes", horizon="short",
               model="ours", summary=acc.summary(), intervention="shuffle", seed=0)
        assert len(lg.rows()) == 2 and (Path(d) / "results.csv").read_text().count("\n") == 3


def test_viz_outputs():
    lab = np.zeros((20, 40), np.uint8); lab[:, :10] = 255
    col = viz.colorize(lab)
    assert col.shape == (20, 40, 3) and (col[0, 0] == 0).all() and tuple(col[0, 20]) == (128, 64, 128)
    with tempfile.TemporaryDirectory() as d:
        viz.save_forecast_panel(Path(d) / "p.png", np.stack([lab] * 4), {"copy": lab}, lab)
        viz.save_legend(Path(d) / "l.png")
        viz.plot_ego_signal(Path(d) / "e.png", np.arange(5), np.arange(5), np.zeros(5), {"samples": [1, 3]})
        assert all((Path(d) / f).stat().st_size > 1000 for f in ("p.png", "l.png", "e.png"))


def test_set_seed_and_env():
    repro.set_seed(3)
    a = np.random.rand()
    repro.set_seed(3)
    assert a == np.random.rand()
    assert "python" in repro.env_info()
