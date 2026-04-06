#!/usr/bin/env python3
"""Evaluate all saved classification checkpoints on the test split."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bird_drone.config import CLASSIFICATION_OUT, REPORTS_DIR, ensure_dirs
from bird_drone.evaluator import evaluate_checkpoint


def main() -> None:
    ensure_dirs()
    manifest = Path("outputs/classification_manifest.csv")
    if not manifest.exists():
        raise SystemExit("Run train_classification.py first (builds manifest).")
    df = pd.read_csv(manifest)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    rows = []
    for sub in sorted(CLASSIFICATION_OUT.iterdir()):
        if not sub.is_dir():
            continue
        ckpt = sub / "best_model.pt"
        if not ckpt.exists():
            continue
        r = evaluate_checkpoint(ckpt, test_df, prefix="test")
        rows.append(
            {
                "model": sub.name,
                "accuracy": r["accuracy"],
                "n_test": r["n_samples"],
                "confusion_figure": r["figure_path"],
            }
        )
        report_path = REPORTS_DIR / f"classification_report_{sub.name}.txt"
        report_path.write_text(r["classification_report"], encoding="utf-8")

    if not rows:
        raise SystemExit("No checkpoints found under outputs/classification/*/best_model.pt")

    comparison = REPORTS_DIR / "model_comparison_classification.md"
    lines = [
        "# Classification model comparison (test set)\n",
        "\n",
        "| Model | Accuracy | Test samples |\n",
        "|-------|----------|-------------|\n",
    ]
    for row in sorted(rows, key=lambda x: -x["accuracy"]):
        lines.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['n_test']} |\n")
    lines.append("\n## Notes\n")
    lines.append(
        "Metric: **accuracy** on held-out test split; "
        "per-class reports saved as `classification_report_<model>.txt`.\n"
    )
    comparison.write_text("".join(lines), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"Wrote {comparison}")


if __name__ == "__main__":
    main()
