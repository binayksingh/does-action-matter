"""Action-conditioned semantic forecaster. Requires torch.

Input : T context segmentations as one-hot maps (B,T,C,H,W),
        past ego actions a_past (B,T,2), next-step action a_next (B,2)
Output: logits of the next segmentation (B,C,H,W)

Backbone: a small U-Net over the stacked context (T*C input channels), the
S2S-style design of Luc et al. (ICCV 2017) with a U-Net encoder-decoder.

Action injection (`action_mode`):
  "film"   - an MLP maps [a_past, a_next] to per-channel scale/shift applied
             after every encoder and decoder block (FiLM, Perez et al., AAAI 2018)
  "concat" - [a_past, a_next] broadcast to constant input planes
  "none"   - the no-action baseline; same network, action ignored

Multi-step (mid-term) prediction is autoregressive: `rollout` feeds the
softmax of each prediction back as the newest context frame.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_MODES = ("film", "concat", "none")


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)          # starts as identity

    def forward(self, x, cond):
        g, b = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + g[..., None, None]) + b[..., None, None]


class Block(nn.Module):
    def __init__(self, cin, cout, cond_dim=None):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(min(8, cout), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.film = FiLM(cond_dim, cout) if cond_dim else None

    def forward(self, x, cond=None):
        x = F.gelu(self.n1(self.c1(x)))
        x = self.n2(self.c2(x))
        if self.film is not None:
            x = self.film(x, cond)
        return F.gelu(x)


class ForecastNet(nn.Module):
    def __init__(self, num_classes: int = 19, context: int = 4, action_dim: int = 2,
                 action_mode: str = "film", base: int = 64, depth: int = 4):
        super().__init__()
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}")
        self.C, self.T, self.A, self.mode = num_classes, context, action_dim, action_mode
        cond_dim = (context + 1) * action_dim
        film_dim = 128 if action_mode == "film" else None
        self.cond_mlp = (nn.Sequential(nn.Linear(cond_dim, 128), nn.GELU(), nn.Linear(128, 128))
                         if action_mode == "film" else None)
        cin = context * num_classes + (cond_dim if action_mode == "concat" else 0)
        chs = [base * 2 ** i for i in range(depth)]
        self.down = nn.ModuleList()
        prev = cin
        for c in chs:
            self.down.append(Block(prev, c, film_dim))
            prev = c
        self.mid = Block(chs[-1], chs[-1], film_dim)
        self.up = nn.ModuleList()
        for c in reversed(chs[:-1]):
            self.up.append(Block(prev + c, c, film_dim))
            prev = c
        self.head = nn.Conv2d(prev, num_classes, 1)
        self.copy_weight = nn.Parameter(torch.tensor(4.0))

    def forward(self, ctx: torch.Tensor, a_past: torch.Tensor, a_next: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = ctx.shape
        x = ctx.reshape(B, T * C, H, W)
        cond_vec = torch.cat([a_past.reshape(B, -1), a_next.reshape(B, -1)], -1)
        if self.mode == "none":
            cond_vec = torch.zeros_like(cond_vec)
        cond = self.cond_mlp(cond_vec) if self.cond_mlp is not None else None
        if self.mode == "concat":
            x = torch.cat([x, cond_vec[..., None, None].expand(-1, -1, H, W)], 1)
        skips = []
        for i, blk in enumerate(self.down):
            if i > 0:
                x = F.max_pool2d(x, 2)
            x = blk(x, cond)
            skips.append(x)
        x = self.mid(x, cond)
        for blk, skip in zip(self.up, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, skip], 1), cond)
        # residual on the last observed frame (learned weight, starts as a strong
        # copy-last prior); the network then only has to predict the change
        return self.head(x) + self.copy_weight * ctx[:, -1]

    def rollout(self, ctx: torch.Tensor, a_past: torch.Tensor, a_future: torch.Tensor) -> list[torch.Tensor]:
        """Autoregressive prediction for a_future.shape[1] steps; returns list of logits."""
        outs = []
        for k in range(a_future.shape[1]):
            a_next = a_future[:, k]
            logits = self(ctx, a_past, a_next)
            outs.append(logits)
            ctx = torch.cat([ctx[:, 1:], logits.softmax(1)[:, None]], 1)
            a_past = torch.cat([a_past[:, 1:], a_next[:, None]], 1)
        return outs
