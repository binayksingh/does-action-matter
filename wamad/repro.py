"""Reproducibility: seeding and environment capture."""
from __future__ import annotations

import os
import platform
import random
import subprocess
import sys

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and (if installed) torch. `deterministic` also forces cuDNN determinism."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def env_info() -> dict:
    """Versions and hardware, saved alongside every run."""
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # not a git repo / git missing
        info["git_commit"] = None
    return info

