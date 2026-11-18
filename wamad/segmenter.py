"""Frozen single-frame segmenter used to make pseudo-labels. Requires torch + transformers.

Default: SegFormer-B5 fine-tuned on Cityscapes (Xie et al., NeurIPS 2021),
Hugging Face checkpoint `nvidia/segformer-b5-finetuned-cityscapes-1024-1024`.
Reference accuracy on Cityscapes val, single scale, 1024x1024 sliding window:
82.25 mIoU (mmsegmentation model zoo) / 82.4 (paper). scripts/make_pseudolabels.py
re-measures it with the metric code of this repository.

Inference follows the mmsegmentation "slide" test mode: 1024x1024 crops,
stride 768, overlapping logits averaged; images smaller than the crop are
zero-padded. Outputs are the 19 Cityscapes train IDs at full input resolution.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_MODEL_ID = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
REFERENCE_MIOU = {"nvidia/segformer-b5-finetuned-cityscapes-1024-1024": 82.25}


def window_starts(size: int, crop: int, stride: int) -> list[int]:
    """Start offsets covering [0, size) with crops of `crop`, mmseg-style."""
    if size <= crop:
        return [0]
    n = math.ceil((size - crop) / stride) + 1
    return [min(i * stride, size - crop) for i in range(n)]


class Segmenter:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda", fp16: bool = True,
                 mode: str = "slide", crop: int = 1024, stride: int = 768, model=None,
                 max_windows: int = 2):
        """`model` may be passed directly (tests); otherwise loaded from Hugging Face."""
        if model is None:
            from transformers import SegformerForSemanticSegmentation
            model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model = model.to(device).eval()
        self.model_id, self.device, self.mode = model_id, device, mode
        self.crop, self.stride = crop, stride
        self.max_windows = max(1, int(max_windows))   # crops per forward pass; lower = less GPU memory
        self.dtype = torch.float16 if (fp16 and str(device).startswith("cuda")) else torch.float32
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.num_classes = int(getattr(self.model.config, "num_labels", 19))

    @property
    def segmenter_id(self) -> str:
        return f"{self.model_id}|{self.mode}|crop{self.crop}|stride{self.stride}"

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """(N,3,h,w) normalised -> (N,C,h,w) float32 logits, `max_windows` crops at a time."""
        outs = []
        for i in range(0, x.shape[0], self.max_windows):
            xi = x[i:i + self.max_windows]
            with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.dtype == torch.float16):
                out = self.model(pixel_values=xi).logits
            outs.append(F.interpolate(out.float(), size=xi.shape[-2:], mode="bilinear", align_corners=False))
        return torch.cat(outs, 0)

    @torch.inference_mode()
    def predict_logits(self, batch: torch.Tensor) -> torch.Tensor:
        """batch: (B,3,H,W) uint8 or float in [0,255] on any device -> (B,C,H,W) float32 logits."""
        x = batch.to(self.device, non_blocking=True).float().div_(255.0)
        x = (x - self.mean) / self.std
        B, _, H, W = x.shape
        if self.mode == "whole":
            return self._logits(x)
        ph, pw = max(self.crop - H, 0), max(self.crop - W, 0)
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        Hp, Wp = x.shape[-2:]
        ys, xs = window_starts(Hp, self.crop, self.stride), window_starts(Wp, self.crop, self.stride)
        acc = torch.zeros(B, self.num_classes, Hp, Wp, device=self.device)
        cnt = torch.zeros(1, 1, Hp, Wp, device=self.device)
        ch, cw = min(self.crop, Hp), min(self.crop, Wp)
        wins = [(y, xx) for y in ys for xx in xs]
        crops = torch.cat([x[:, :, y:y + ch, xx:xx + cw] for y, xx in wins], 0)  # (len(wins)*B,...)
        logits = self._logits(crops)
        for k, (y, xx) in enumerate(wins):
            acc[:, :, y:y + ch, xx:xx + cw] += logits[k * B:(k + 1) * B]
            cnt[:, :, y:y + ch, xx:xx + cw] += 1
        return (acc / cnt)[:, :, :H, :W]

    def __call__(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        """List of (H,W,3) uint8 RGB (same size) -> list of (H,W) uint8 train-ID maps."""
        batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
        pred = self.predict_logits(batch).argmax(1).to(torch.uint8).cpu().numpy()
        return list(pred)
