"""Evaluate a saved checkpoint on a DataFrame split; plots + sklearn report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from bird_drone.config import CLASS_NAMES, FIGURES_DIR, IMG_SIZE, ensure_dirs
from bird_drone.datasets import ManifestImageDataset, get_transforms
from bird_drone.models import build_model


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    test_df,
    batch_size: int = 32,
    num_workers: int = 0,
    prefix: str = "eval",
) -> dict:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt["model_name"]
    model = build_model(model_name, num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = ManifestImageDataset(test_df, get_transforms(False, IMG_SIZE))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_y: list[int] = []
    all_p: list[int] = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        all_y.extend(y)
        all_p.extend(pred.cpu().tolist())

    cm = confusion_matrix(all_y, all_p, labels=[0, 1])
    report = classification_report(
        all_y, all_p, target_names=list(CLASS_NAMES), digits=4, zero_division=0
    )
    acc = float(np.mean(np.array(all_y) == np.array(all_p)))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — confusion ({prefix})")
    fig.tight_layout()
    fig_path = FIGURES_DIR / f"confusion_{model_name}_{prefix}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "figure_path": str(fig_path),
        "n_samples": len(all_y),
    }
