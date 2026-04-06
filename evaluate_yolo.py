#!/usr/bin/env python3
"""Run YOLO validation on the test split (see data.yaml)."""

from __future__ import annotations

import argparse
from pathlib import Path

from bird_drone.config import PROJECT_ROOT, REPORTS_DIR, YOLO_OUT, ensure_dirs


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default=str(YOLO_OUT / "train" / "weights" / "best.pt"),
    )
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "data.yaml"))
    args = parser.parse_args()

    w = Path(args.weights)
    if not w.exists():
        raise SystemExit(f"No weights at {w}. Train with train_yolo.py first.")

    from ultralytics import YOLO

    model = YOLO(str(w))
    metrics = model.val(data=args.data, split="test")

    rd = getattr(metrics, "results_dict", {}) or {}
    lines = [
        "# YOLOv8 test-split validation\n\n",
        f"- Weights: `{w}`\n",
        f"- Data: `{args.data}`\n\n",
        "## Metrics\n\n",
    ]
    for k, v in sorted(rd.items()):
        lines.append(f"- **{k}**: {v}\n")

    report = REPORTS_DIR / "yolo_validation_summary.md"
    report.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {report}")
    print(rd)


if __name__ == "__main__":
    main()
