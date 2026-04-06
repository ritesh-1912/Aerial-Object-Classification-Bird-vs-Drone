#!/usr/bin/env python3
"""Plot loss/accuracy/F1 curves from outputs/classification/*/training_meta.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from bird_drone.config import CLASSIFICATION_OUT, FIGURES_DIR, ensure_dirs


def main() -> None:
    ensure_dirs()
    for sub in sorted(CLASSIFICATION_OUT.iterdir()):
        meta_path = sub / "training_meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        h = data.get("history", {})
        if not h.get("train_loss"):
            continue
        epochs = range(1, len(h["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, h["train_loss"], label="train")
        axes[0].plot(epochs, h["val_loss"], label="val")
        axes[0].set_title(f"{sub.name} — loss")
        axes[0].legend()
        axes[0].set_xlabel("epoch")
        axes[1].plot(epochs, h["val_acc"], label="val acc")
        axes[1].plot(epochs, h["val_f1"], label="val F1 (drone pos)")
        axes[1].set_title(f"{sub.name} — metrics")
        axes[1].legend()
        axes[1].set_xlabel("epoch")
        fig.tight_layout()
        out = FIGURES_DIR / f"training_curves_{sub.name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
