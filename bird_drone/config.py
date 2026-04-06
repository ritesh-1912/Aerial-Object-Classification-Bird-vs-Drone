"""Paths and shared constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "drones and birds"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CLASSIFICATION_OUT = OUTPUT_DIR / "classification"
YOLO_OUT = OUTPUT_DIR / "yolo"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

CLASS_NAMES = ("Bird", "Drone")
# COCO category_id from Roboflow export
COCO_CAT_BIRD = 1
COCO_CAT_DRONE = 2
# Model label index 0 = Bird, 1 = Drone
LABEL_BIRD = 0
LABEL_DRONE = 1

IMG_SIZE = 224


def ensure_dirs() -> None:
    for d in (
        CLASSIFICATION_OUT,
        YOLO_OUT,
        FIGURES_DIR,
        REPORTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
