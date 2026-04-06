# Submission checklist

This repository is designed to be lightweight in Git:

- `outputs/` and model weights (`*.pt`) are typically ignored in `.gitignore`.

If your evaluator/grader expects trained artifacts, submit a zip that includes them.

## Include these paths in your submission zip

- `README.md`
- `app.py`
- `bird_drone/`
- `run_eda.py`, `statistical_analysis.py`, `train_*.py`, `evaluate_*.py`, `plot_training_curves.py`, `convert_coco_to_yolo.py`
- `data.yaml`
- `requirements.txt`
- `notebooks/01_eda.ipynb` (optional but recommended)

### Include outputs if required

- `outputs/INDEX.txt`
- `outputs/reports/`
- `outputs/figures/`
- `outputs/classification/*/best_model.pt`
- `outputs/yolo/train/weights/best.pt`

## “Where are the final results?”

Open:

- `outputs/reports/model_comparison_classification.md`
- `outputs/reports/statistical_analysis.md`
- `outputs/reports/yolo_validation_summary.md`

