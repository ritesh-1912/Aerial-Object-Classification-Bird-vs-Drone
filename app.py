#!/usr/bin/env python3
"""
Streamlit UI: Bird vs Drone classification (+ optional YOLO detection).
Run from project root:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from bird_drone.config import CLASSIFICATION_OUT, IMG_SIZE, PROJECT_ROOT, YOLO_OUT
from bird_drone.datasets import get_transforms
from bird_drone.models import build_model


def _default_classifier() -> Path | None:
    out = CLASSIFICATION_OUT
    if not out.exists():
        return None
    candidates = []
    for sub in out.iterdir():
        p = sub / "best_model.pt"
        if p.exists():
            candidates.append(p)
    if not candidates:
        return None
    # Prefer ResNet50 if present (usually strongest baseline)
    for c in candidates:
        if "resnet50" in str(c):
            return c
    return candidates[0]


@st.cache_resource
def load_classifier(path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    names = ckpt.get("class_names", ["Bird", "Drone"])
    return model, names, device


@st.cache_resource
def load_yolo(weights_path: str):
    from ultralytics import YOLO

    return YOLO(weights_path)


def main() -> None:
    st.set_page_config(page_title="Bird vs Drone", layout="wide")
    st.title("Aerial object classification (Bird / Drone)")
    st.caption(
        "Capstone: PyTorch classifiers + optional YOLOv8 detection. "
        "TensorFlow was not used (Python 3.14); training uses PyTorch as allowed by the brief."
    )

    default_ckpt = _default_classifier()
    yolo_default = YOLO_OUT / "train" / "weights" / "best.pt"

    with st.sidebar:
        st.header("Models")
        ckpt_path = st.text_input(
            "Classifier checkpoint (.pt)",
            value=str(default_ckpt) if default_ckpt else "",
        )
        use_yolo = st.checkbox("Run YOLO detection", value=yolo_default.exists())
        yolo_path = st.text_input(
            "YOLO weights",
            value=str(yolo_default) if yolo_default.exists() else "",
        )

    if not ckpt_path or not Path(ckpt_path).exists():
        st.warning(
            "Train classifiers first: `python train_all_models.py` then refresh. "
            f"Expected files under `{CLASSIFICATION_OUT}`."
        )
        return

    model, class_names, device = load_classifier(ckpt_path)
    tfm = get_transforms(False, IMG_SIZE)

    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if up is None:
        st.info("Upload an aerial image to see Bird vs Drone prediction.")
        return

    pil = Image.open(up).convert("RGB")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Input")
        st.image(pil, use_container_width=True)

    x = tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].cpu()
    pred_i = int(prob.argmax())
    conf = float(prob[pred_i])

    with c2:
        st.subheader("Classification")
        st.metric("Prediction", class_names[pred_i])
        st.metric("Confidence", f"{conf:.2%}")
        for i, name in enumerate(class_names):
            st.write(f"{name}: **{float(prob[i]):.2%}**")
            st.progress(float(prob[i]))

    if use_yolo and yolo_path and Path(yolo_path).exists():
        st.subheader("YOLO detection (optional)")
        yolo = load_yolo(yolo_path)
        import numpy as np

        arr = np.array(pil)
        res = yolo(arr, verbose=False)[0]
        im_bgr = res.plot()
        st.image(im_bgr[:, :, ::-1], caption="Bounding boxes (YOLO)", use_container_width=True)
    elif use_yolo:
        st.info("YOLO weights not found. Train with `python train_yolo.py` after `python convert_coco_to_yolo.py`.")


if __name__ == "__main__":
    main()
