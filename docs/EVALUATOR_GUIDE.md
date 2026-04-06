# Evaluator Guide (what to run, what to open)

This project is structured so you can (1) run a few scripts, then (2) open the generated reports/figures.

## Quick verify (recommended)

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH=.
python run_eda.py
python statistical_analysis.py
python evaluate_classification.py
python plot_training_curves.py
python evaluate_yolo.py
```

Then open:

- `outputs/reports/model_comparison_classification.md`
- `outputs/reports/statistical_analysis.md`
- `outputs/reports/yolo_validation_summary.md`
- `outputs/figures/` (EDA plots, confusion matrices, training curves)

## Interactive demo (Streamlit)

```bash
export PYTHONPATH=.
streamlit run app.py
```

Open `http://localhost:8501`.

## Notes on expectations

- If trained weights are present, the Streamlit app will auto-detect them under `outputs/classification/*/best_model.pt`.
- YOLO weights default to `outputs/yolo/train/weights/best.pt` when present.

