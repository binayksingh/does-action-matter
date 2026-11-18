#!/bin/bash
# One training run per array task; list the runs with: python scripts/train.py --list
#   0-8  : trained on Cityscapes (film / concat / none x seeds 0, 1, 2)
#   9-17 : trained on BDD100K
# Every run is evaluated on both validation sets, so the pseudo-labels of both
# datasets must exist first (scripts/make_pseudolabels.py).
# Usage: sbatch --array=0-17 slurm/train.sh
# Adjust the partition, GPU type and paths to your cluster.
#SBATCH --job-name=train
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

mkdir -p logs
python scripts/train.py
