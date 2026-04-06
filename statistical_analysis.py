#!/usr/bin/env python3
"""
EDA statistics for the capstone write-up: class balance per split (chi-square test of
homogeneity), and optional McNemar placeholder when two model predictions are saved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from bird_drone.coco_manifest import save_manifest_csv
from bird_drone.config import REPORTS_DIR, ensure_dirs


def main() -> None:
    ensure_dirs()
    save_manifest_csv()
    df = pd.read_csv("outputs/classification_manifest.csv")

    lines = [
        "# Statistical notes (for capstone report)\n\n",
        "## Class counts per split\n\n",
    ]
    for split in ["train", "valid", "test"]:
        sub = df[df["split"] == split]
        c = sub["label"].value_counts().reindex([0, 1], fill_value=0)
        lines.append(f"### {split}\n")
        lines.append(f"- Bird (0): {int(c[0])}\n")
        lines.append(f"- Drone (1): {int(c[1])}\n\n")

    # Chi-square test: homogeneity of class proportions across splits
    table = []
    for split in ["train", "valid", "test"]:
        sub = df[df["split"] == split]
        c = sub["label"].value_counts().reindex([0, 1], fill_value=0)
        table.append([int(c[0]), int(c[1])])
    chi2, p, dof, _ = chi2_contingency(np.array(table))
    lines.append("## Chi-square test of homogeneity (class proportions across splits)\n\n")
    lines.append(
        f"- χ² = {chi2:.4f}, df = {dof}, p-value = {p:.6f}\n"
    )
    if p < 0.05:
        lines.append(
            "- Interpretation: proportions differ significantly across splits; "
            "report metrics on validation and held-out test, not training only.\n\n"
        )
    else:
        lines.append(
            "- Interpretation: no strong evidence that class mix differs across splits.\n\n"
        )

    lines.append("## Imbalance handling in training\n\n")
    lines.append(
        "- Weighted cross-entropy (`CrossEntropyLoss(weight=...)`) uses inverse-frequency "
        "weights computed on the training split.\n\n"
    )

    out = REPORTS_DIR / "statistical_analysis.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
