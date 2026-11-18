"""Cityscapes 19-class label definitions, shared by Cityscapes and BDD100K.

BDD100K's semantic-segmentation labels use the same 19 train classes and the
same train-ID ordering as Cityscapes, so one definition serves both datasets.

Source of the mapping and colours: the official cityscapesScripts
`labels.py` (Cordts et al., CVPR 2016).
"""
from __future__ import annotations

import numpy as np

IGNORE_INDEX = 255
NUM_CLASSES = 19

# train-ID order (0..18)
CLASS_NAMES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
)

# Movable-object classes used for MO-mIoU in the semantic-forecasting
# literature (Luc et al., ICCV 2017 and follow-ups):
# person, rider, car, truck, bus, train, motorcycle, bicycle.
MOVING_CLASS_NAMES = ("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
MOVING_CLASS_IDS = tuple(CLASS_NAMES.index(n) for n in MOVING_CLASS_NAMES)  # (11..18)

# Cityscapes labelId -> trainId (official table; unlisted ids -> ignore).
_LABELID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}

LABELID_TO_TRAINID_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _lid, _tid in _LABELID_TO_TRAINID.items():
    LABELID_TO_TRAINID_LUT[_lid] = _tid

# Official Cityscapes colours, train-ID order.
PALETTE = np.array([
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
], dtype=np.uint8)
IGNORE_COLOR = np.array((0, 0, 0), dtype=np.uint8)


def labelids_to_trainids(label_ids: np.ndarray) -> np.ndarray:
    """Map a Cityscapes `*_gtFine_labelIds.png` array to train IDs (0..18, 255)."""
    label_ids = np.asarray(label_ids)
    if label_ids.dtype != np.uint8:
        if label_ids.min() < 0 or label_ids.max() > 255:
            raise ValueError("labelIds outside 0..255")
        label_ids = label_ids.astype(np.uint8)
    return LABELID_TO_TRAINID_LUT[label_ids]


def validate_trainids(label: np.ndarray) -> None:
    """Raise if a train-ID map has values other than 0..18 or 255."""
    vals = np.unique(label)
    bad = vals[(vals >= NUM_CLASSES) & (vals != IGNORE_INDEX)]
    if bad.size:
        raise ValueError(f"invalid train IDs present: {bad.tolist()}")
