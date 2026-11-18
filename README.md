# Does the Action Matter? Probing Action-Conditioned Semantic Forecasting as a World Model for Autonomous Driving

Code for the WAM-AD Workshop paper at WACV 2027.

Action-conditioned world models are expected to predict how a driving scene evolves
under the ego vehicle's action. This repository measures how strongly such a model
depends on its action: a semantic forecaster is trained with feature-wise (FiLM),
input-level (concat) or no action conditioning, and evaluated with test-time action
interventions (shuffled, zero yaw, stop, mean action) scored by standard
segmentation metrics (mIoU, MO-mIoU). The interventions are validated in a
synthetic environment and applied to Cityscapes and BDD100K.

## Installation

```bash
git clone https://github.com/binayksingh/does-action-matter.git
cd does-action-matter
pip install -r requirements.txt
pytest tests/
```

The experiments were run with PyTorch 2.5.1 on single NVIDIA GPUs.

## Data

Set the locations with environment variables (defaults in `wamad/paths.py`):

| Variable | Default | Contents |
|---|---|---|
| `CITYSCAPES_ROOT` | `data/cityscapes` | `gtFine/`, `leftImg8bit_sequence/`, `vehicle_sequence/` |
| `BDD_ROOT` | `data/bdd100k` | `images/track/{train,val}/`, `info/100k/{train,val}/` |
| `PSEUDO_ROOT` | `data/pseudo_labels` | segmenter outputs (created) |
| `RUNS_DIR` | `runs` | indices, results, tables, figures (created) |

**Cityscapes** ([cityscapes-dataset.com](https://www.cityscapes-dataset.com)):
`gtFine_trainvaltest`, `leftImg8bit_sequence_trainvaltest` and `vehicle_sequence`.
The protocol uses frames 2, 5, 8, 11, 14, 17 and 20 of each 30-frame snippet.

**BDD100K** ([bdd-data.berkeley.edu](https://bdd-data.berkeley.edu)): the
multi-object-tracking frames (`images/track`, 5 fps) and the GPS/IMU `info` files.

Check the paths with `python -m wamad.paths`.

## Reproducing the paper

All scripts read their settings from a `SETTINGS` block at the top of the file.

| Step | Command | GPU | Output (under `RUNS_DIR`) |
|---|---|---|---|
| 1 | `python scripts/check_cityscapes.py` | no | `check_cityscapes/`: data completeness, ego-signal units |
| 2 | `python scripts/prepare_bdd100k.py` | no | `bdd100k_index/`: sample index, GPS units, time alignment |
| 3 | `python scripts/make_pseudolabels.py` | yes | segmenter accuracy, pseudo-labels in `PSEUDO_ROOT` |
| 4 | `python scripts/synthetic.py` | yes | `synthetic/`: synthetic validation |
| 5 | `python scripts/baselines.py` | no | `forecast/baselines/`: copy-last and flow warping |
| 6 | `python scripts/train.py --run <i>` for i = 0..17 | yes | `forecast/<run>/`: model and 20 evaluations per run |
| 7 | `python scripts/efficiency.py` | yes | `forecast/efficiency.json` |
| 8 | `python scripts/action_stats.py` | no | `forecast/action_stats.json` |
| 9 | `python scripts/summarize_results.py` | no | `forecast/summary/`: LaTeX tables and `summary.json` |
| 10 | `python scripts/make_figures.py` | no | `forecast/figures/`: per-class and action-dependence figures |
| 11 | `python scripts/make_overview.py` | optional | `forecast/figures/overview/`: images for Figure 1 |

`python scripts/train.py --list` prints the 18 runs (2 training sets x 3 action
modes x 3 seeds). On a SLURM cluster, `sbatch --array=0-17 slurm/train.sh` runs
one per array task. Every run is evaluated on both datasets, so step 3 must be
complete for both before step 6.

`summarize_results.py` writes one LaTeX table per paper table (`tab_main`,
`tab_interventions`, `tab_interventions_short`, `tab_cross`, `tab_synthetic`,
`tab_efficiency`). Means and standard deviations are over seeds; changes under an
intervention are computed per seed and then averaged.

### Segmenter

Pseudo-labels come from SegFormer-B5 fine-tuned on Cityscapes
(`nvidia/segformer-b5-finetuned-cityscapes-1024-1024`) with sliding-window
inference (1024 x 1024 crops, stride 768). Set `SEGFORMER_MODEL` to use a local
copy. With PyTorch < 2.6, recent versions of `transformers` refuse to load
`pytorch_model.bin` files; convert the checkpoint once to `model.safetensors`:

```python
import torch
from safetensors.torch import save_file
sd = torch.load("pytorch_model.bin", map_location="cpu", weights_only=True)
save_file({k: v.contiguous() for k, v in sd.items()}, "model.safetensors", metadata={"format": "pt"})
```

## Repository structure

```
wamad/
  models/forecaster.py   U-Net forecaster with FiLM, concat or no action input
  train.py               training loop, evaluation, action interventions
  data/                  Cityscapes and BDD100K loaders, pseudo-label cache
  ego.py                 speed and yaw-rate signals, normalisation
  protocol.py            frame indices and horizons
  metrics.py             mIoU, MO-mIoU, per-class IoU, pixel accuracy
  flow.py                copy-last and optical-flow warping baselines
  toy.py                 synthetic driving environment
  segmenter.py           SegFormer wrapper for pseudo-labels
scripts/                 the steps above
tests/                   unit and end-to-end tests on synthetic data
```

## Citation

```bibtex
@inproceedings{singh2027action,
  title     = {Does the Action Matter? Probing Action-Conditioned Semantic Forecasting as a World Model for Autonomous Driving},
  author    = {Singh, Binay Kumar and Lobo, Niels Da Vitoria},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
  year      = {2027}
}
```

## License

MIT, see [LICENSE](LICENSE). Cityscapes and BDD100K are subject to their own licenses.
