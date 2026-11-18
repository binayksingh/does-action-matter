"""Data and output locations used by all scripts.

Defaults are relative to the repository root. Override any of them with an
environment variable of the same name, e.g.

    export CITYSCAPES_ROOT=/data/cityscapes
    export RUNS_DIR=/scratch/runs
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# folder containing gtFine/, leftImg8bit_sequence/, vehicle_sequence/
CITYSCAPES_ROOT = Path(os.environ.get("CITYSCAPES_ROOT", _REPO / "data" / "cityscapes"))

# folder containing images/track/{train,val}/ and info/100k/{train,val}/
BDD_ROOT = Path(os.environ.get("BDD_ROOT", _REPO / "data" / "bdd100k"))

# cached segmenter outputs (pseudo-labels); created on first use
PSEUDO_ROOT = Path(os.environ.get("PSEUDO_ROOT", _REPO / "data" / "pseudo_labels"))

# outputs: reports, checkpoints, results tables and figures
RUNS_DIR = Path(os.environ.get("RUNS_DIR", _REPO / "runs"))

# BDD100K sample index written by scripts/prepare_bdd100k.py
BDD_INDEX_DIR = Path(os.environ.get("BDD_INDEX_DIR", RUNS_DIR / "bdd100k_index"))


def check(verbose: bool = True) -> dict:
    """Report which roots exist. Run: python -m wamad.paths"""
    status = {
        "CITYSCAPES_ROOT": all((CITYSCAPES_ROOT / d).is_dir() for d in ("gtFine", "leftImg8bit_sequence", "vehicle_sequence")),
        "BDD_ROOT": all((BDD_ROOT / d).is_dir() for d in ("images/track", "info/100k")),
    }
    if verbose:
        for k, ok in status.items():
            print(f"{k:16s} {'OK     ' if ok else 'MISSING'} {globals()[k]}")
        for k in ("PSEUDO_ROOT", "RUNS_DIR", "BDD_INDEX_DIR"):
            p = globals()[k]
            print(f"{k:16s} {'(exists)' if p.exists() else '(created later)'} {p}")
    return status


if __name__ == "__main__":
    check()
