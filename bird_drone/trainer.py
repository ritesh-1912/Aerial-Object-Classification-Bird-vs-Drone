"""Train a single classification model; save checkpoint + metrics history."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from bird_drone.config import CLASS_NAMES, IMG_SIZE, ensure_dirs
from bird_drone.datasets import ManifestImageDataset, get_transforms
from bird_drone.models import build_model


def _class_weights(train_df) -> torch.Tensor:
    counts = train_df["label"].value_counts().sort_index()
    n = len(train_df)
    w = n / (2 * counts.reindex([0, 1], fill_value=1).values.astype(np.float32))
    return torch.tensor(w, dtype=torch.float32)


def train_model(
    model_name: str,
    train_df,
    val_df,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-4,
    output_dir: Path | None = None,
    num_workers: int = 0,
    patience: int = 7,
) -> dict:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = output_dir or Path("outputs/classification") / model_name
    out.mkdir(parents=True, exist_ok=True)

    train_ds = ManifestImageDataset(train_df, get_transforms(True, IMG_SIZE))
    val_ds = ManifestImageDataset(val_df, get_transforms(False, IMG_SIZE))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(model_name, num_classes=2).to(device)
    weights = _class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
    }
    best_f1 = -1.0
    best_path = out / "best_model.pt"
    stale = 0
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"{model_name} ep{epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_ds)

        model.eval()
        val_running = 0.0
        all_y: list[int] = []
        all_p: list[int] = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_running += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                all_y.extend(y.cpu().tolist())
                all_p.extend(pred.cpu().tolist())

        val_loss = val_running / len(val_ds)
        val_acc = accuracy_score(all_y, all_p)
        val_f1 = f1_score(all_y, all_p, average="binary", pos_label=1, zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(float(val_acc))
        history["val_f1"].append(float(val_f1))
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "class_names": list(CLASS_NAMES),
                    "epoch": epoch,
                    "val_f1": val_f1,
                },
                best_path,
            )
        else:
            stale += 1

        if stale >= patience:
            break

    elapsed = time.perf_counter() - t0
    epochs_ran = len(history["train_loss"])
    meta = {
        "model_name": model_name,
        "epochs_ran": epochs_ran,
        "best_val_f1": best_f1,
        "seconds": elapsed,
        "device": str(device),
        "checkpoint": str(best_path),
    }
    with open(out / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "history": history}, f, indent=2)

    return meta
