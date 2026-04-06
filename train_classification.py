#!/usr/bin/env python3
"""CLI: train one classification model (PyTorch). Example:
  python train_classification.py --model resnet50 --epochs 25
"""

from __future__ import annotations

import argparse

import pandas as pd

from bird_drone.coco_manifest import save_manifest_csv
from bird_drone.config import CLASSIFICATION_OUT, ensure_dirs
from bird_drone.trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Bird vs Drone classifier")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["custom_cnn", "resnet50", "mobilenet_v3_small", "efficientnet_b0"],
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    ensure_dirs()
    manifest_path = save_manifest_csv()
    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "valid"].reset_index(drop=True)

    out = CLASSIFICATION_OUT / args.model
    meta = train_model(
        args.model,
        train_df,
        val_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=out,
        num_workers=args.num_workers,
        patience=args.patience,
    )
    print("Done:", meta)


if __name__ == "__main__":
    main()
