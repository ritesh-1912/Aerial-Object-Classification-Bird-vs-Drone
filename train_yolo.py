#!/usr/bin/env python3
"""Train YOLOv8 on the converted dataset. Requires: python convert_coco_to_yolo.py"""

from __future__ import annotations

import argparse
from pathlib import Path

from bird_drone.config import PROJECT_ROOT, YOLO_OUT, ensure_dirs


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics model name or .pt path")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "data.yaml"))
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"Missing {data_yaml}. Run convert_coco_to_yolo.py first.")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(YOLO_OUT),
        name="train",
        exist_ok=True,
    )
    print("Training finished. Best weights typically at:", YOLO_OUT / "train" / "weights" / "best.pt")


if __name__ == "__main__":
    main()
