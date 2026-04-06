#!/usr/bin/env python3
"""Exploratory analysis: class balance, sample grid — saves figures to outputs/figures/."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

from bird_drone.coco_manifest import save_manifest_csv
from bird_drone.config import FIGURES_DIR, ensure_dirs


def main() -> None:
    ensure_dirs()
    save_manifest_csv()
    df = pd.read_csv("outputs/classification_manifest.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Images per split and class (annotated images only)")
    ax.set_xlabel("Split")
    ax.legend(title="Class")
    fig.tight_layout()
    p1 = FIGURES_DIR / "eda_class_counts.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(p1)

    fig, ax = plt.subplots(figsize=(5, 4))
    df["label_name"] = df["class_name"]
    sns.histplot(data=df, x="split", hue="label_name", multiple="dodge", shrink=0.9, ax=ax)
    ax.set_title("Distribution by split")
    fig.tight_layout()
    p2 = FIGURES_DIR / "eda_split_distribution.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(p2)

    sample_parts = [g.sample(min(8, len(g)), random_state=42) for _, g in df.groupby("class_name")]
    sample = pd.concat(sample_parts)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for ax, (_, row) in zip(axes.flat, sample.iterrows()):
        im = Image.open(row["path"]).convert("RGB")
        ax.imshow(im)
        ax.set_title(row["class_name"], fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(sample) :]:
        ax.axis("off")
    fig.suptitle("Sample images (8 per class when available)")
    fig.tight_layout()
    p3 = FIGURES_DIR / "eda_sample_grid.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(p3)


if __name__ == "__main__":
    main()
