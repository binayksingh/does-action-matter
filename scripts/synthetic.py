#!/usr/bin/env python3
"""Validate the forecaster and the action interventions in a synthetic environment.

Run (GPU recommended):  python scripts/synthetic.py
SMOKE_TEST = True runs a tiny version in about a minute, also on CPU.

Uses the synthetic world in wamad/toy.py, where the future view depends on the
future ego action in a known way. Trains the same ForecastNet as scripts/train.py:
  film        action injected with FiLM            (main model)
  concat      action as extra input planes          (injection ablation)
  none        no action                             (key baseline)
  film_blind  FiLM, but trained with shuffled actions, so the action carries
              no information -> must learn to ignore it (negative control)
plus copy-last and optical-flow-warping baselines.

Every model is scored with mIoU / MO-mIoU at short (1 step) and mid (3 steps)
horizons, under the input interventions true / shuffle / zero_yaw / stop / mean.

Validation checks (written to report.json):
  C1  action helps:          film beats none at mid-term by >= 3 mIoU
  C2  wrong action hurts:    film loses >= 3 mIoU at mid-term with shuffled actions
  C3  test is not fooled:    none and film_blind change by < 1 mIoU under shuffle
  C4  beats baselines:       film beats copy-last and flow-warping at both horizons
"""
from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wamad import flow, io, metrics, paths as P, repro, toy, viz  # noqa: E402
from wamad.ego import ActionNormalizer  # noqa: E402

# ============================================================ SETTINGS
OUT_DIR = P.RUNS_DIR / "synthetic"
N_TRAIN = 20000
N_VAL = 2000
EPOCHS = 15
BATCH_SIZE = 64
LR = 1e-3
BASE_CHANNELS = 32
NUM_WORKERS = 8
SEED = 0
MODELS = ("film", "concat", "none", "film_blind")
SMOKE_TEST = False
# ======================================================================

if SMOKE_TEST:
    N_TRAIN, N_VAL, EPOCHS, BATCH_SIZE, NUM_WORKERS = 256, 128, 1, 16, 0
    OUT_DIR = OUT_DIR.parent / "synthetic_smoke"

VAL_OFFSET = 10_000_000


class ToyDataset:
    def __init__(self, world, seeds, horizon, with_rgb=False):
        self.w, self.seeds, self.h, self.rgb = world, list(seeds), horizon, with_rgb

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, i):
        s = self.w.sample(self.seeds[i], with_rgb=self.rgb)
        s["eval_gt"] = toy.eval_target(s, self.h)
        return s


def main() -> int:
    import torch
    from torch.utils.data import DataLoader
    from wamad.models.forecaster import ForecastNet
    from wamad.train import INTERVENTIONS, TorchNormalizer, TrainConfig, collate, evaluate, predict, train_model

    repro.set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logf = open(OUT_DIR / "log.txt", "a")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    log(f"=== synthetic environment {time.strftime('%Y-%m-%d %H:%M')} on {device} ===")
    world = toy.ToyWorld()
    train_seeds = range(N_TRAIN)
    val_seeds = range(VAL_OFFSET, VAL_OFFSET + N_VAL)

    # action normaliser from training actions only
    acts = []
    for i in train_seeds[: min(5000, N_TRAIN)]:
        s = world.sample(i)
        acts += [s["actions_past"], s["actions_future"]]
    norm_np = ActionNormalizer.fit(np.concatenate(acts))
    norm_np.save(OUT_DIR / "action_normalizer.json")
    norm = TorchNormalizer(norm_np, device)
    train_mean = norm_np.mean.astype(np.float32)

    mk = lambda ds, shuffle: DataLoader(  # noqa: E731
        ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
        collate_fn=partial(collate, horizon_steps=toy.FUTURE), pin_memory=device == "cuda",
        drop_last=shuffle, persistent_workers=NUM_WORKERS > 0)
    train_loader = mk(ToyDataset(world, train_seeds, "mid"), True)
    val_loaders = {h: mk(ToyDataset(world, val_seeds, h), False) for h in ("short", "mid")}
    results = io.ResultsLogger(OUT_DIR)
    table = {}

    # ------------------------------------------------------- baselines
    for h in ("short", "mid"):
        a_copy, a_flow = metrics.SegMetricAccumulator(), metrics.SegMetricAccumulator()
        for i in val_seeds:
            s = world.sample(i, with_rgb=True)
            gt = toy.eval_target(s, h)
            a_copy.update(flow.copy_last_baseline(s["ctx_labels"]), gt)
            a_flow.update(flow.flow_baseline(s["ctx_rgb"], s["ctx_labels"], toy.steps(h), work_width=128), gt)
        for name, acc in (("copy_last", a_copy), ("flow_warp", a_flow)):
            summ = acc.summary()
            table[(name, h, "true")] = summ
            results.log(run_id="synthetic", train_set="toy", eval_set="toy_val", horizon=h, model=name, summary=summ)
            log(f"[baseline] {name:10s} {h:5s} {metrics.format_summary(summ)}")

    # ---------------------------------------------------------- models
    models = {}
    for name in MODELS:
        repro.set_seed(SEED)
        mode = "film" if name.startswith("film") else name
        model = ForecastNet(num_classes=19, context=toy.CONTEXT, action_mode=mode, base=BASE_CHANNELS).to(device)
        cfg = TrainConfig(epochs=EPOCHS, lr=LR, train_steps=toy.FUTURE,
                          shuffle_train_actions=(name == "film_blind"), log_every=200)
        log(f"[train] {name}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params")
        t0 = time.time()
        hist = train_model(model, train_loader, cfg, norm, device,
                           val_fn=lambda m: {"val_mid_mIoU": evaluate(m, val_loaders["mid"], norm, device, 3)["mIoU"]},
                           log=log)
        log(f"[train] {name} done in {time.time() - t0:.0f}s")
        torch.save({"state_dict": model.state_dict(), "action_mode": mode, "base": BASE_CHANNELS,
                    "history": hist}, OUT_DIR / f"{name}.pt")
        models[name] = model
        for h in ("short", "mid"):
            for iv in INTERVENTIONS:
                summ = evaluate(model, val_loaders[h], norm, device, toy.steps(h), iv, train_mean)
                table[(name, h, iv)] = summ
                results.log(run_id="synthetic", train_set="toy", eval_set="toy_val", horizon=h, model=name,
                            intervention=iv, seed=SEED, summary=summ)
                log(f"[eval] {name:10s} {h:5s} {iv:8s} {metrics.format_summary(summ)}")

    # ------------------------------------------------------- criteria
    m = lambda name, h, iv="true": 100 * table[(name, h, iv)]["mIoU"]  # noqa: E731
    crit = {}
    if {"film", "none"} <= set(MODELS):
        d = m("film", "mid") - m("none", "mid")
        crit["C1_action_helps"] = {"film_minus_none_mid": d, "pass": d >= 3}
    if "film" in MODELS:
        d = m("film", "mid") - m("film", "mid", "shuffle")
        crit["C2_wrong_action_hurts"] = {"film_true_minus_shuffle_mid": d, "pass": d >= 3}
    ctl = {n: abs(m(n, "mid") - m(n, "mid", "shuffle")) for n in ("none", "film_blind") if n in MODELS}
    if ctl:
        crit["C3_test_not_fooled"] = {"abs_change_under_shuffle_mid": ctl, "pass": all(v < 1 for v in ctl.values())}
    if "film" in MODELS:
        gaps = {h: min(m("film", h) - m("copy_last", h), m("film", h) - m("flow_warp", h)) for h in ("short", "mid")}
        crit["C4_beats_baselines"] = {"min_margin": gaps, "pass": all(v > 0 for v in gaps.values())}
    passed = all(c["pass"] for c in crit.values())
    log("CRITERIA:", json.dumps(crit, indent=1))
    log("RESULT:", "PASS" if passed else "FAIL", "(smoke test - numbers not meaningful)" if SMOKE_TEST else "")

    # ---------------------------------------------------------- figure
    if "film" in models:
        picks = []
        for i in val_seeds:
            s = world.sample(i)
            if not s["meta"]["same_action"] and abs(s["actions_future"][0, 1]) > 0.15:
                s["eval_gt"] = toy.eval_target(s, "mid")
                picks.append(s)
            if len(picks) == 4:
                break
        if picks:
            batch = collate(picks, toy.FUTURE)
            preds = {f"film|{iv}": predict(models["film"], batch, norm, device, 3, iv, train_mean)
                     for iv in ("true", "zero_yaw", "stop", "shuffle")}
            if "none" in models:
                preds["none"] = predict(models["none"], batch, norm, device, 3)
            for j, s in enumerate(picks):
                viz.save_forecast_panel(OUT_DIR / f"example_{j}.png", s["ctx_labels"],
                                        {k: v[j] for k, v in preds.items()}, s["eval_gt"],
                                        title=f"future action (speed, yaw) = {np.round(s['actions_future'][0], 2).tolist()}")

    report = {"criteria": crit, "passed": passed, "smoke_test": SMOKE_TEST,
              "settings": {"N_TRAIN": N_TRAIN, "N_VAL": N_VAL, "EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE,
                           "LR": LR, "BASE_CHANNELS": BASE_CHANNELS, "SEED": SEED},
              "table": {"|".join(k): {kk: v[kk] for kk in ("mIoU", "MO-mIoU", "pixel_acc")} for k, v in table.items()},
              "env": repro.env_info()}
    io.save_json(OUT_DIR / "report.json", report)
    log(f"wrote {OUT_DIR}/report.json, results.csv, example_*.png")
    return 0 if passed or SMOKE_TEST else 1


if __name__ == "__main__":
    sys.exit(main())
