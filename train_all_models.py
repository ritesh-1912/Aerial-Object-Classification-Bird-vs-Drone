#!/usr/bin/env python3
"""Train all four classification models sequentially (for full capstone run)."""

from __future__ import annotations

import argparse

import pandas as pd

from bird_drone.coco_manifest import save_manifest_csv
from bird_drone.config import CLASSIFICATION_OUT, ensure_dirs
from bird_drone.trainer import train_model


MODELS = ["custom_cnn", "resnet50", "mobilenet_v3_small", "efficientnet_b0"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    ensure_dirs()
    save_manifest_csv()
    df = pd.read_csv("outputs/classification_manifest.csv")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "valid"].reset_index(drop=True)

    for name in MODELS:
        print(f"\n========== Training {name} ==========\n")
        train_model(
            name,
            train_df,
            val_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_dir=CLASSIFICATION_OUT / name,
            num_workers=args.num_workers,
            patience=args.patience,
        )


if __name__ == "__main__":
    main()
