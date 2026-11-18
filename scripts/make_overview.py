#!/usr/bin/env python3
"""Image assets for the method overview figure (Figure 1 of the paper).

Picks one Cityscapes val scene with a clear turn (largest |future yaw rate|), runs the
trained FiLM model, and writes colourised label maps to OUT_DIR:

  ctx_0.png .. ctx_3.png          the four context segmentations S_{t-3D} .. S_t (model input)
  pred_true_1..3.png              rollout with the executed action: t+D, t+2D, t+3D
  pred_shuffle.png, pred_zero_yaw.png, pred_stop.png, pred_mean.png
                                  mid-term (t+3D) prediction under each intervention
  gt.png                          gtFine of the target frame (t+3D)
  overview_meta.json              scene id and the action values used (for the caption)

All maps are written at the model resolution (512 x 256) with the Cityscapes palette.

Edit SETTINGS, then:  python scripts/make_overview.py   (GPU optional; one scene)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wamad import paths as P, viz  # noqa: E402
from wamad.data import cityscapes as cs  # noqa: E402
from wamad.data.forecast_data import ForecastSet  # noqa: E402

# ============================================================ SETTINGS
RUN = "cityscapes_film_s0"            # trained FiLM run used for all predictions
SAMPLE_RANK = 0                       # 0 = strongest turn, 1 = second strongest, ...
OUT_DIR = P.RUNS_DIR / "forecast" / "figures" / "overview"
SIZE = (512, 256)                     # (width, height) of every written image
BDD_INDEX_DIR = P.BDD_INDEX_DIR
FORECAST_DIR = P.RUNS_DIR / "forecast"


def load_model(run: str, device):
    import torch
    from wamad.ego import ActionNormalizer
    from wamad.models.forecaster import ForecastNet
    d = FORECAST_DIR / run
    cfg = json.loads((d / "config.json").read_text())
    ck = torch.load(d / "final.pt", map_location=device, weights_only=False)
    m = ForecastNet(num_classes=19, context=4, action_mode=cfg["mode"], base=cfg["base"], depth=cfg["depth"])
    m.load_state_dict(ck["state_dict"])
    norm = ActionNormalizer(np.asarray(ck["normalizer"]["mean"]), np.asarray(ck["normalizer"]["std"]))
    return m.to(device).eval(), norm


def turn_samples(ds: ForecastSet) -> list[int]:
    """Moving samples (> 3 m/s) sorted by |mean future yaw rate|, at most one per driving sequence."""
    scores = []
    for i, s in enumerate(ds.samples):
        a = cs.sample_actions(cs.ego_signal(ds.paths, s, ds.h), s, ds.h)
        if a["past"][-1, 0] > 3:
            scores.append((abs(a["future"][:, 1].mean()), i))
    out, seen = [], set()
    for _, i in sorted(scores, reverse=True):
        key = ds._key(ds.samples[i])
        if key not in seen:
            out.append(i); seen.add(key)
    return out


def save(label: np.ndarray, name: str) -> None:
    from PIL import Image
    img = Image.fromarray(label.astype(np.uint8)).resize(SIZE, Image.NEAREST)
    Image.fromarray(viz.colorize(np.asarray(img))).save(OUT_DIR / f"{name}.png")


def main() -> int:
    import torch
    from wamad.train import TorchNormalizer, _prep, collate, intervene

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, norm = load_model(RUN, device)
    kw = dict(train=False, cityscapes_root=P.CITYSCAPES_ROOT, bdd_root=P.BDD_ROOT,
              pseudo_root=P.PSEUDO_ROOT, bdd_index_dir=BDD_INDEX_DIR)
    ds = ForecastSet("cityscapes", "val", "mid", **kw)
    ds_shuf = ForecastSet("cityscapes", "val", "mid", shuffle_actions=True, **kw)  # same perm as evaluation
    i = turn_samples(ds)[SAMPLE_RANK]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    batch = collate([ds[i]], 3)
    for k in range(4):
        save(batch["ctx"][0, k].numpy(), f"ctx_{k}")
    save(batch["eval_gt"][0].numpy(), "gt")

    tn = TorchNormalizer(norm, device)
    mean = torch.tensor(norm.mean, dtype=torch.float32, device=device)

    def rollout(b, iv):
        with torch.no_grad():
            ctx, a_past, a_fut, _ = _prep(b, device, tn)
            a = intervene(iv, a_fut[:, :3], mean)
            return [lg.float().argmax(1)[0].cpu().numpy() for lg in model.rollout(ctx, a_past, tn(a))]

    steps = rollout(batch, "true")
    for k, p in enumerate(steps, 1):
        save(p, f"pred_true_{k}")
    for iv in ("zero_yaw", "stop", "mean"):
        save(rollout(batch, iv)[-1], f"pred_{iv}")
    save(rollout(collate([ds_shuf[i]], 3), "true")[-1], "pred_shuffle")

    a_fut = batch["a_future"][0].numpy()
    meta = {"run": RUN, "sample": ds.samples[i].to_dict(),
            "future_action_true": a_fut.round(3).tolist(),
            "future_action_shuffled": ds_shuf.future_actions(int(ds_shuf.perm[i])).round(3).tolist(),
            "train_mean_action": np.asarray(norm.mean).round(3).tolist()}
    (OUT_DIR / "overview_meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())