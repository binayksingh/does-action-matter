"""Training and evaluation loops for the synthetic and real-data experiments. Requires torch.

Batches are dicts with (after `collate`):
  ctx       (B,T,H,W) long  context segmentations (255 = ignore)
  targets   (B,S,H,W) long  future segmentations at model resolution
  eval_gt   (B,He,We) long  scored frame at evaluation resolution
  a_past    (B,T,2)   float physical units
  a_future  (B,S,2)   float physical units
Actions are normalised here (ActionNormalizer fitted on the training split),
and test-time interventions are applied to the PHYSICAL future actions
before normalisation. Scoring uses wamad.metrics (mIoU, MO-mIoU) only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .data.torch_dataset import one_hot_labels
from .ego import ActionNormalizer
from .labels import IGNORE_INDEX, NUM_CLASSES
from .metrics import SegMetricAccumulator


def collate(samples: list[dict], horizon_steps: int, eval_key: str = "eval_gt") -> dict:
    """List of numpy sample dicts -> batch of tensors (first `horizon_steps` targets)."""
    t = lambda k, dt: torch.from_numpy(np.stack([np.asarray(s[k]) for s in samples]).astype(dt))  # noqa: E731
    return {
        "ctx": t("ctx_labels", np.int64),
        "targets": t("targets", np.int64)[:, :horizon_steps],
        "eval_gt": t(eval_key, np.int64),
        "a_past": t("actions_past", np.float32),
        "a_future": t("actions_future", np.float32)[:, :horizon_steps],
    }


class TorchNormalizer:
    def __init__(self, n: ActionNormalizer, device):
        self.mean = torch.tensor(n.mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(n.std, dtype=torch.float32, device=device)

    def __call__(self, a):
        return (a - self.mean) / self.std


def intervene(name: str, a: torch.Tensor, train_mean: torch.Tensor) -> torch.Tensor:
    """Replace the future actions (B,S,2) in physical units.

    true: unchanged | shuffle: another sample's actions (batch rolled by half)
    zero_yaw: same speed, yaw 0 | stop: speed 0, yaw 0 | mean: training-mean action
    """
    if name == "true":
        return a
    if name == "shuffle":
        return torch.roll(a, shifts=max(1, a.shape[0] // 2), dims=0)
    if name == "zero_yaw":
        b = a.clone(); b[..., 1] = 0; return b
    if name == "stop":
        return torch.zeros_like(a)
    if name == "mean":
        return train_mean.expand_as(a).clone()
    raise ValueError(name)


INTERVENTIONS = ("true", "shuffle", "zero_yaw", "stop", "mean")


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_steps: int = 3          # autoregressive steps unrolled in the loss (1 = single-step)
    amp: bool = True
    grad_clip: float = 1.0
    log_every: int = 100
    shuffle_train_actions: bool = False   # control: actions uncorrelated with targets
    accum_steps: int = 1          # gradient accumulation: effective batch = loader batch x accum_steps


def _prep(batch, device, norm, num_classes=NUM_CLASSES):
    ctx = one_hot_labels(batch["ctx"].to(device, non_blocking=True), num_classes)
    return (ctx, norm(batch["a_past"].to(device)), batch["a_future"].to(device),
            batch["targets"].to(device, non_blocking=True))


def amp_dtype(device) -> torch.dtype:
    """bfloat16 on GPUs with native support (compute capability >= 8), else float16.

    torch.cuda.is_bf16_supported() is not used on its own because it also returns
    True on older GPUs where bfloat16 is only emulated."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(device)
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        if torch.cuda.get_device_capability(idx)[0] >= 8:
            return torch.bfloat16
    return torch.float16


def train_model(model, train_loader, cfg: TrainConfig, norm: TorchNormalizer, device, val_fn=None, log=print,
                val_every: int = 1):
    """Train with AdamW and a one-cycle schedule; returns the per-epoch history."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    accum = max(1, int(cfg.accum_steps))
    updates_per_epoch = max(1, len(train_loader) // accum)
    total = cfg.epochs * updates_per_epoch          # optimizer updates, not mini-batches
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total, pct_start=0.1)
    use_amp = cfg.amp and str(device).startswith("cuda")
    dtype = amp_dtype(device)
    # fp16 needs loss scaling to avoid underflow; bf16 does not
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    if use_amp:
        log(f"  mixed precision: {dtype}")
    history, step, t0 = [], 0, time.time()
    for ep in range(cfg.epochs):
        model.train()
        run, n_run = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            if i >= updates_per_epoch * accum:      # drop a trailing partial accumulation group
                break
            ctx, a_past, a_fut, tgt = _prep(batch, device, norm)
            if cfg.shuffle_train_actions:
                a_fut = intervene("shuffle", a_fut, None)
            S = min(cfg.train_steps, tgt.shape[1])
            with torch.autocast("cuda", dtype=dtype, enabled=use_amp):
                outs = model.rollout(ctx, a_past, norm(a_fut[:, :S]))
                loss = sum(F.cross_entropy(o.float(), tgt[:, k], ignore_index=IGNORE_INDEX)
                           for k, o in enumerate(outs)) / S
            scaler.scale(loss / accum).backward()
            run += loss.item(); n_run += 1
            if (i + 1) % accum:
                continue
            if cfg.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if cfg.log_every and step % cfg.log_every == 0:
                log(f"  ep {ep + 1}/{cfg.epochs} step {step}/{total} loss {run / n_run:.4f} "
                    f"({time.time() - t0:.0f}s)")
                run, n_run = 0.0, 0
        rec = {"epoch": ep + 1}
        if val_fn is not None and ((ep + 1) % val_every == 0 or ep + 1 == cfg.epochs):
            rec.update(val_fn(model))
            log(f"  ep {ep + 1}: {rec}")
        history.append(rec)
    return history


@torch.no_grad()
def evaluate(model, loader, norm: TorchNormalizer, device, steps: int, intervention: str = "true",
             train_mean=None, upsample_mode: str = "bilinear") -> dict:
    """Score the prediction at the last of `steps` autoregressive steps against eval_gt."""
    model.eval()
    acc = SegMetricAccumulator()
    tm = torch.tensor(train_mean if train_mean is not None else [0.0, 0.0], dtype=torch.float32, device=device)
    for batch in loader:
        ctx, a_past, a_fut, _ = _prep(batch, device, norm)
        a_fut = intervene(intervention, a_fut[:, :steps], tm)
        logits = model.rollout(ctx, a_past, norm(a_fut))[-1].float()
        gt = batch["eval_gt"]
        if logits.shape[-2:] != gt.shape[-2:]:
            logits = F.interpolate(logits, size=gt.shape[-2:], mode=upsample_mode,
                                   align_corners=False if upsample_mode == "bilinear" else None)
        acc.update(logits.argmax(1).cpu().numpy(), gt.numpy())
    return acc.summary()


@torch.no_grad()
def predict(model, batch, norm, device, steps, intervention="true", train_mean=None):
    model.eval()
    ctx, a_past, a_fut, _ = _prep(batch, device, norm)
    tm = torch.tensor(train_mean if train_mean is not None else [0.0, 0.0], dtype=torch.float32, device=device)
    a_fut = intervene(intervention, a_fut[:, :steps], tm)
    return model.rollout(ctx, a_past, norm(a_fut))[-1].argmax(1).cpu().numpy()