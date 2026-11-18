"""Tests for the PyTorch parts (model, training loop, segmenter wrapper).

They return early (count as passed) when torch is not installed.
"""
from functools import partial

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def test_forecastnet_shapes_and_modes():
    if not HAVE_TORCH:
        return
    from wamad.models.forecaster import ForecastNet
    from wamad.data.torch_dataset import one_hot_labels
    ctx = one_hot_labels(torch.randint(0, 19, (2, 4, 32, 64)))
    ctx[0, 0, :, 0, 0] = 0  # an ignore pixel
    a_past, a_fut = torch.randn(2, 4, 2), torch.randn(2, 3, 2)
    outs = {}
    for mode in ("film", "concat", "none"):
        torch.manual_seed(0)
        m = ForecastNet(context=4, action_mode=mode, base=8, depth=3)
        o = m.rollout(ctx, a_past, a_fut)
        assert len(o) == 3 and o[-1].shape == (2, 19, 32, 64)
        sum(x.mean() for x in o).backward()
        outs[mode] = m
    # 'none' ignores the action completely
    m = outs["none"]
    with torch.no_grad():
        assert torch.allclose(m(ctx, a_past, a_fut[:, 0]), m(ctx, a_past * 5, -a_fut[:, 0]))
    # untrained model starts as copy-last (residual prior)
    with torch.no_grad():
        pred = ForecastNet(context=4, action_mode="film", base=8, depth=3)(ctx, a_past, a_fut[:, 0]).argmax(1)
    assert (pred == ctx[:, -1].argmax(1)).float().mean() > 0.9


def test_interventions():
    if not HAVE_TORCH:
        return
    from wamad.train import intervene
    a = torch.tensor([[[5.0, 0.1]], [[10.0, -0.2]], [[0.0, 0.0]], [[7.0, 0.3]]])
    tm = torch.tensor([6.0, 0.0])
    assert torch.equal(intervene("true", a, tm), a)
    assert not torch.equal(intervene("shuffle", a, tm), a)
    assert (intervene("zero_yaw", a, tm)[..., 1] == 0).all() and torch.equal(intervene("zero_yaw", a, tm)[..., 0], a[..., 0])
    assert (intervene("stop", a, tm) == 0).all()
    assert torch.equal(intervene("mean", a, tm)[1, 0], tm)


def test_train_and_evaluate_on_toy():
    if not HAVE_TORCH:
        return
    from torch.utils.data import DataLoader
    from wamad import toy
    from wamad.ego import ActionNormalizer
    from wamad.models.forecaster import ForecastNet
    from wamad.train import TorchNormalizer, TrainConfig, collate, evaluate, train_model

    w = toy.ToyWorld(H=32, W=64)

    class DS:
        def __init__(self, seeds, h): self.s, self.h = list(seeds), h
        def __len__(self): return len(self.s)
        def __getitem__(self, i):
            x = w.sample(self.s[i]); x["eval_gt"] = toy.eval_target(x, self.h); return x

    acts = np.concatenate([np.concatenate([w.sample(i)["actions_past"], w.sample(i)["actions_future"]]) for i in range(50)])
    norm = TorchNormalizer(ActionNormalizer.fit(acts), "cpu")
    dl = lambda ds, sh: DataLoader(ds, batch_size=8, shuffle=sh, collate_fn=partial(collate, horizon_steps=3))  # noqa: E731
    model = ForecastNet(context=4, action_mode="film", base=8, depth=3)
    hist = train_model(model, dl(DS(range(32), "mid"), True), TrainConfig(epochs=1, amp=False, log_every=0),
                       norm, "cpu", log=lambda *a: None)
    assert len(hist) == 1
    for iv in ("true", "shuffle", "zero_yaw", "stop", "mean"):
        s = evaluate(model, dl(DS(range(100, 116), "mid"), False), norm, "cpu", 3, iv, [6.0, 0.0])
        assert 0 <= s["mIoU"] <= 1 and s["n_images"] == 16


def test_segmenter_sliding_window():
    if not HAVE_TORCH:
        return
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError:
        return
    from wamad.segmenter import Segmenter, window_starts
    assert window_starts(2048, 1024, 768) == [0, 768, 1024]
    assert window_starts(720, 1024, 768) == [0]
    cfg = SegformerConfig(num_labels=19, depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 32, 64],
                          decoder_hidden_size=32, num_attention_heads=[1, 1, 1, 1])
    net = SegformerForSemanticSegmentation(cfg)
    imgs = [np.random.default_rng(i).integers(0, 255, (72, 200, 3), dtype=np.uint8) for i in range(2)]
    slide = Segmenter("tiny", device="cpu", mode="slide", crop=64, stride=48, model=net)
    whole = Segmenter("tiny", device="cpu", mode="whole", model=net)
    p1, p2 = slide(imgs), whole(imgs)
    assert p1[0].shape == (72, 200) and p1[0].dtype == np.uint8 and p2[0].shape == (72, 200)
    assert p1[0].max() < 19
    # an image smaller than the crop is padded, still returns full size
    small = Segmenter("tiny", device="cpu", mode="slide", crop=128, stride=96, model=net)(imgs)
    assert small[0].shape == (72, 200)
    assert "tiny|slide|crop64|stride48" == slide.segmenter_id
    # processing crops one at a time gives the same result as all at once
    one = Segmenter("tiny", device="cpu", mode="slide", crop=64, stride=48, model=net, max_windows=1)(imgs)
    many = Segmenter("tiny", device="cpu", mode="slide", crop=64, stride=48, model=net, max_windows=64)(imgs)
    assert all((a == b).all() for a, b in zip(one, many))


def test_run_end_to_end():
    """One full run (train + 20 evaluations + figures) on tiny fake datasets."""
    if not HAVE_TORCH:
        return
    import sys
    import tempfile
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train as pt
    from test_pipeline import fake_bdd, fake_cityscapes
    from wamad import io
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fake_cityscapes(d / "cs", d / "pseudo")
        fake_bdd(d / "bdd", d / "pseudo", d / "idx")
        pt.CITYSCAPES_ROOT, pt.BDD_ROOT, pt.PSEUDO_ROOT, pt.BDD_INDEX_DIR = d / "cs", d / "bdd", d / "pseudo", d / "idx"
        pt.OUT_ROOT = d / "forecast"
        pt.EPOCHS = {"cityscapes": 2, "bdd100k": 1}
        pt.BATCH_SIZE, pt.EVAL_BATCH_SIZE, pt.BASE_CHANNELS, pt.DEPTH, pt.NUM_WORKERS = 2, 2, 8, 3, 0
        for r in ({"train_set": "cityscapes", "mode": "film", "seed": 0},
                  {"train_set": "bdd100k", "mode": "none", "seed": 0}):
            pt.one_run(r)
            out = d / "forecast" / pt.run_name(r)
            rows = io.load_jsonl(out / "results.jsonl")
            assert len(rows) == 2 * 2 * 5 and (out / "final.pt").exists()
        none_rows = io.load_jsonl(d / "forecast" / "bdd100k_none_s0" / "results.jsonl")
        by_iv = {}
        for x in none_rows:
            by_iv.setdefault((x["eval_set"], x["horizon"]), set()).add(round(x["mIoU"], 9))
        assert all(len(v) == 1 for v in by_iv.values())   # no-action model: interventions change nothing


def test_amp_dtype_matches_hardware():
    if not HAVE_TORCH or not torch.cuda.is_available():
        return
    from wamad.train import amp_dtype
    major = torch.cuda.get_device_capability(0)[0]
    assert amp_dtype("cuda") == (torch.bfloat16 if major >= 8 else torch.float16)


def test_gradient_accumulation_matches_large_batch():
    """The gradient used for the update with 2 x batch 4 (accumulated) equals that of one batch of 8."""
    if not HAVE_TORCH:
        return
    from wamad.ego import ActionNormalizer
    from wamad.models.forecaster import ForecastNet
    from wamad.train import TorchNormalizer, TrainConfig, train_model
    torch.manual_seed(0)
    data = [{"ctx": torch.randint(0, 19, (8, 4, 16, 32)), "targets": torch.randint(0, 19, (8, 3, 16, 32)),
             "a_past": torch.randn(8, 4, 2), "a_future": torch.randn(8, 3, 2), "eval_gt": torch.zeros(8, 16, 32).long()}]
    halves = [{k: v[:4] for k, v in data[0].items()}, {k: v[4:] for k, v in data[0].items()}]
    norm = TorchNormalizer(ActionNormalizer(np.zeros(2), np.ones(2)), "cpu")
    grads, orig_step = [], torch.optim.AdamW.step

    def recording_step(self, *a, **k):
        grads.append(torch.cat([p.grad.detach().flatten().clone() for g in self.param_groups
                                for p in g["params"] if p.grad is not None]))
        return orig_step(self, *a, **k)

    torch.optim.AdamW.step = recording_step
    try:
        for loader, acc in ((data, 1), (halves, 2)):
            torch.manual_seed(1)
            m = ForecastNet(context=4, action_mode="film", base=8, depth=3)
            train_model(m, loader, TrainConfig(epochs=1, amp=False, grad_clip=0, log_every=0, accum_steps=acc),
                        norm, "cpu", log=lambda *a: None)
    finally:
        torch.optim.AdamW.step = orig_step
    assert len(grads) == 2, len(grads)          # exactly one optimizer update per configuration
    assert torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-6), (grads[0] - grads[1]).abs().max()