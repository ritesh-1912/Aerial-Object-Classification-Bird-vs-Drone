#!/usr/bin/env python3
"""
Convert COCO `_annotations.coco.json` to YOLO `.txt` labels and a standard layout:
  yolo_dataset/{train,valid,test}/images/  (symlinks to original jpgs)
  yolo_dataset/{train,valid,test}/labels/

Run once before YOLO training. Idempotent: overwrites labels, recreates symlinks.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from bird_drone.config import COCO_CAT_BIRD, COCO_CAT_DRONE, DATASET_DIR, PROJECT_ROOT

# YOLO class index: 0 = Bird, 1 = drone (matches data.yaml)
COCO_TO_YOLO_CLASS = {COCO_CAT_BIRD: 0, COCO_CAT_DRONE: 1}


def coco_bbox_to_yolo_line(
    bbox: list[float], img_w: int, img_h: int, coco_category_id: int
) -> str | None:
    """COCO: [x,y,w,h] absolute pixels -> YOLO: cls cx cy w h normalized."""
    if coco_category_id not in COCO_TO_YOLO_CLASS:
        return None
    cls = COCO_TO_YOLO_CLASS[coco_category_id]
    x, y, w, h = bbox
    x_c = (x + w / 2) / img_w
    y_c = (y + h / 2) / img_h
    w_n = w / img_w
    h_n = h / img_h
    x_c = min(max(x_c, 0), 1)
    y_c = min(max(y_c, 0), 1)
    w_n = min(max(w_n, 0), 1)
    h_n = min(max(h_n, 0), 1)
    return f"{cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"


def process_split(split: str, yolo_root: Path, use_symlinks: bool = True) -> None:
    src_dir = DATASET_DIR / split
    coco_path = src_dir / "_annotations.coco.json"
    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    id_to_file = {im["id"]: im for im in coco["images"]}
    img_dir = yolo_root / split / "images"
    lbl_dir = yolo_root / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Group annotations by image
    anns_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    for image_id, im in id_to_file.items():
        fname = im["file_name"]
        stem = Path(fname).stem
        src_img = src_dir / fname
        dst_img = img_dir / fname
        if src_img.exists():
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            if use_symlinks:
                os.symlink(src_img.resolve(), dst_img)
            else:
                shutil.copy2(src_img, dst_img)

        label_lines: list[str] = []
        for ann in anns_by_img.get(image_id, []):
            line = coco_bbox_to_yolo_line(
                ann["bbox"], im["width"], im["height"], ann["category_id"]
            )
            if line:
                label_lines.append(line)

        lbl_path = lbl_dir / f"{stem}.txt"
        if label_lines:
            lbl_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        else:
            # Empty file = image with no objects (YOLO still trains on background)
            lbl_path.write_text("", encoding="utf-8")


def main() -> None:
    yolo_root = PROJECT_ROOT / "drones and birds" / "yolo_dataset"
    for split in ("train", "valid", "test"):
        process_split(split, yolo_root)
    print(f"YOLO layout written under: {yolo_root}")


if __name__ == "__main__":
    main()
