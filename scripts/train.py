#!/usr/bin/env python3
"""Train action-conditioned forecasters on real data and evaluate them.

A run = (training set, action mode, seed); 2 datasets x 3 action modes x 3 seeds
gives 18 runs. List them, then run one or all:

    python scripts/train.py --list
    python scripts/train.py --run 4        # one run
    python scripts/train.py                # all runs, one after another

On a SLURM cluster, one run per array task (SLURM_ARRAY_TASK_ID selects the run):

    sbatch --array=0-17 slurm/train.sh

Each run:
  1. trains ForecastNet on the training set (mid-term samples, 3-step unrolled
     loss, horizontal-flip augmentation with yaw negated);
  2. evaluates the final model (no model selection on val, because the
     Cityscapes val split is the test set of the standard protocol) on
     Cityscapes val and BDD100K val, short and mid horizons, under the
     interventions true / shuffle / zero_yaw / stop / mean;
  3. writes RUNS_DIR/forecast/<run>/results.jsonl (+ .csv) and final.pt.

Metrics: mIoU and MO-mIoU only (wamad.metrics). Cityscapes val is scored
against gtFine at 1024x2048; BDD100K val against pseudo-labels at 720x1280.
"""
from __future__ import annotations

import itertools
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wamad import io, metrics, paths as P, repro, viz  # noqa: E402
from wamad.data.forecast_data import MODEL_SIZE, ForecastSet  # noqa: E402
from wamad.ego import ActionNormalizer  # noqa: E402

# ============================================================ SETTINGS
CITYSCAPES_ROOT = P.CITYSCAPES_ROOT
BDD_ROOT = P.BDD_ROOT
PSEUDO_ROOT = P.PSEUDO_ROOT
BDD_INDEX_DIR = P.BDD_INDEX_DIR
OUT_ROOT = P.RUNS_DIR / "forecast"

TRAIN_SETS = ("cityscapes", "bdd100k")
ACTION_MODES = ("film", "concat", "none")
SEEDS = (0, 1, 2)
EVAL_SETS = ("cityscapes", "bdd100k")
HORIZONS = ("short", "mid")
INTERVENTIONS = ("true", "shuffle", "zero_yaw", "stop", "mean")

EPOCHS = {"cityscapes": 60, "bdd100k": 12}     # ~11k and ~19k optimizer updates at effective batch 16
BATCH_SIZE = 8                # samples per forward pass (fits a 32 GB GPU with the 3-step unroll)
ACCUM_STEPS = 2               # gradient accumulation -> effective batch 16
EVAL_BATCH_SIZE = 8
LR = 1e-3
BASE_CHANNELS = 64
DEPTH = 4
FLIP_AUG = True
NUM_WORKERS = 8
VAL_EVERY = 5                 # epochs between (logging-only) val checks on a 100-sample subset
SMOKE_TEST = False            # tiny run (1 epoch, 64 train / 16 val samples) to test the pipeline
# ======================================================================

if SMOKE_TEST:
    EPOCHS = {k: 1 for k in EPOCHS}
    OUT_ROOT = OUT_ROOT.parent / "forecast_smoke"

RUNS = [dict(train_set=d, mode=m, seed=s) for d, m, s in itertools.product(TRAIN_SETS, ACTION_MODES, SEEDS)]


def run_name(r: dict) -> str:
    return f"{r['train_set']}_{r['mode']}_s{r['seed']}"


def make_set(dataset, split, horizon, **kw):
    return ForecastSet(dataset, split, horizon, cityscapes_root=CITYSCAPES_ROOT, bdd_root=BDD_ROOT,
                       pseudo_root=PSEUDO_ROOT, bdd_index_dir=BDD_INDEX_DIR, **kw)


def get_normalizer(train_set: str) -> ActionNormalizer:
    """Fitted once per training set on its TRAIN split; shared by all runs on that set."""
    f = OUT_ROOT / f"action_normalizer_{train_set}.json"
    if f.exists():
        return ActionNormalizer.load(f)
    ds = make_set(train_set, "train", "mid", train=True)
    n = ActionNormalizer.fit(ds.all_actions())
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f"{f.name}.{os.getpid()}.tmp")   # array jobs may race: write atomically
    n.save(tmp)
    os.replace(tmp, f)
    return n


def one_run(r: dict) -> None:
    import torch
    from torch.utils.data import DataLoader
    from wamad.models.forecaster import ForecastNet
    from wamad.train import TorchNormalizer, TrainConfig, collate, evaluate, predict, train_model

    out = OUT_ROOT / run_name(r)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "w")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    repro.set_seed(r["seed"])
    log(f"=== {run_name(r)} {time.strftime('%Y-%m-%d %H:%M')} on "
        f"{torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'} ===")
    io.save_json(out / "config.json", {**r, "epochs": EPOCHS[r["train_set"]], "batch_size": BATCH_SIZE,
                                       "accum_steps": ACCUM_STEPS, "effective_batch": BATCH_SIZE * ACCUM_STEPS, "lr": LR,
                                       "base": BASE_CHANNELS, "depth": DEPTH, "flip": FLIP_AUG,
                                       "model_size": MODEL_SIZE[r["train_set"]], "env": repro.env_info()})

    norm_np = get_normalizer(r["train_set"])
    norm = TorchNormalizer(norm_np, device)
    train_mean = norm_np.mean.astype(np.float32)

    lim_tr = 64 if SMOKE_TEST else None
    lim_va = 16 if SMOKE_TEST else None
    train_set = make_set(r["train_set"], "train", "mid", train=True, flip=FLIP_AUG, limit=lim_tr)
    loader = lambda ds, bs, shuffle: DataLoader(  # noqa: E731
        ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle, num_workers=NUM_WORKERS,
        collate_fn=partial(collate, horizon_steps=3 if ds.horizon == "mid" else 1,
                           eval_key="targets" if ds.train else "eval_gt"),
        pin_memory=device == "cuda", persistent_workers=NUM_WORKERS > 0)
    train_loader = loader(train_set, BATCH_SIZE, True)
    log(f"train samples: {len(train_set)}  mini-batches/epoch: {len(train_loader)}  "
        f"effective batch: {BATCH_SIZE * ACCUM_STEPS}")

    # logging-only validation on a fixed subset of the in-domain val split
    val_small = loader(make_set(r["train_set"], "val", "mid", train=False, limit=lim_va or 100), EVAL_BATCH_SIZE, False)
    val_fn = lambda m: {"val100_mid_mIoU": evaluate(m, val_small, norm, device, 3)["mIoU"]}  # noqa: E731

    model = ForecastNet(num_classes=19, context=4, action_mode=r["mode"], base=BASE_CHANNELS, depth=DEPTH).to(device)
    log(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    cfg = TrainConfig(epochs=EPOCHS[r["train_set"]], lr=LR, train_steps=3, log_every=100,
                      accum_steps=ACCUM_STEPS)
    t0 = time.time()
    hist = train_model(model, train_loader, cfg, norm, device, val_fn=val_fn, log=log,
                       val_every=VAL_EVERY)
    log(f"training done in {(time.time() - t0) / 60:.1f} min")
    torch.save({"state_dict": model.state_dict(), "run": r, "history": hist,
                "normalizer": {"mean": norm_np.mean.tolist(), "std": norm_np.std.tolist()}}, out / "final.pt")

    # ------------------------------------------------------------ evaluation
    results = io.ResultsLogger(out, fresh=True)
    for es in EVAL_SETS:
        for h in HORIZONS:
            steps = 3 if h == "mid" else 1
            plain = loader(make_set(es, "val", h, train=False, limit=lim_va), EVAL_BATCH_SIZE, False)
            for iv in INTERVENTIONS:
                if iv == "shuffle":   # future actions of a sample from another sequence
                    dl = loader(make_set(es, "val", h, train=False, limit=lim_va, shuffle_actions=True),
                                EVAL_BATCH_SIZE, False)
                    summ = evaluate(model, dl, norm, device, steps, "true", train_mean)
                else:
                    summ = evaluate(model, plain, norm, device, steps, iv, train_mean)
                results.log(run_id=run_name(r), train_set=r["train_set"], eval_set=es, horizon=h,
                            model=r["mode"], intervention=iv, seed=r["seed"], summary=summ,
                            notes="gtFine" if es == "cityscapes" else "pseudo-label targets")
                log(f"[eval] {es:10s} {h:5s} {iv:8s} {metrics.format_summary(summ)}")

    # a few qualitative examples on Cityscapes val, mid-term, true vs. altered actions
    try:
        ds = make_set("cityscapes", "val", "mid", train=False, limit=4)
        batch = collate([ds[i] for i in range(len(ds))], 3)
        preds = {iv: predict(model, batch, norm, device, 3, iv, train_mean) for iv in ("true", "zero_yaw", "stop")}
        for j in range(len(ds)):
            gt = batch["eval_gt"][j].numpy()
            small = {k: io.resize_label(v[j].astype(np.uint8), gt.shape) for k, v in preds.items()}
            viz.save_forecast_panel(out / f"example_{j}.png", batch["ctx"][j].numpy().astype(np.uint8),
                                    small, gt, title=f"{run_name(r)} | future action "
                                    f"{np.round(batch['a_future'][j, 0].numpy(), 2).tolist()}")
    except Exception as e:  # figures must never fail a run
        log(f"(example figures skipped: {e})")
    log(f"=== {run_name(r)} finished ===")


def main() -> int:
    if "--list" in sys.argv:
        for i, r in enumerate(RUNS):
            print(i, run_name(r))
        return 0
    if "--run" in sys.argv:
        idx = sys.argv[sys.argv.index("--run") + 1]
    else:
        idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    selected = RUNS if idx is None else [RUNS[int(idx)]]
    for r in selected:
        one_run(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())