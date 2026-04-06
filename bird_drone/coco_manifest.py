"""Build image-level labels from COCO annotations (majority vote per image)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

import pandas as pd

from bird_drone.config import (
    COCO_CAT_BIRD,
    COCO_CAT_DRONE,
    DATASET_DIR,
    LABEL_BIRD,
    LABEL_DRONE,
    OUTPUT_DIR,
)

Split = Literal["train", "valid", "test"]


def _coco_cat_to_label(cat_id: int) -> int | None:
    if cat_id == COCO_CAT_BIRD:
        return LABEL_BIRD
    if cat_id == COCO_CAT_DRONE:
        return LABEL_DRONE
    return None


def image_label_from_annotations(category_ids: list[int]) -> int | None:
    """Majority vote; on tie prefer Bird (0)."""
    mapped = [_coco_cat_to_label(c) for c in category_ids]
    mapped = [m for m in mapped if m is not None]
    if not mapped:
        return None
    counts = Counter(mapped)
    return max(counts.items(), key=lambda x: (x[1], -x[0]))[0]


def build_split_manifest(split: Split) -> pd.DataFrame:
    coco_path = DATASET_DIR / split / "_annotations.coco.json"
    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    id_to_name = {im["id"]: im["file_name"] for im in coco["images"]}
    img_to_cats: dict[int, list[int]] = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        img_to_cats.setdefault(iid, []).append(ann["category_id"])

    rows: list[dict] = []
    for image_id, file_name in id_to_name.items():
        cats = img_to_cats.get(image_id, [])
        label = image_label_from_annotations(cats)
        if label is None:
            continue
        abs_path = str((DATASET_DIR / split / file_name).resolve())
        rows.append(
            {
                "path": abs_path,
                "split": split,
                "label": label,
                "class_name": "Bird" if label == LABEL_BIRD else "Drone",
            }
        )

    return pd.DataFrame(rows)


def build_all_manifests() -> pd.DataFrame:
    parts = [build_split_manifest("train"), build_split_manifest("valid"), build_split_manifest("test")]
    return pd.concat(parts, ignore_index=True)


def save_manifest_csv(path: Path | None = None) -> Path:
    path = path or OUTPUT_DIR / "classification_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = build_all_manifests()
    df.to_csv(path, index=False)
    return path
