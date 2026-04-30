import random
from collections import Counter
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import EXPERIMENTS, get_model

# ------------------------
# Config
# ------------------------
DATA_DIR = Path("PokemonData")
CHECKPOINT_DIR = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ------------------------
# Load class names
# ------------------------
@st.cache_data
def load_classes():
    return sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])


# ------------------------
# Load all models
# ------------------------
@st.cache_resource
def load_models(classes):
    loaded = {}
    for name in EXPERIMENTS:
        ckpt = CHECKPOINT_DIR / f"{name}_best.pth"
        if not ckpt.exists():
            st.warning(f"Checkpoint not found: {ckpt}")
            continue
        model = get_model(name).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        loaded[name] = model
    return loaded


# ------------------------
# Predict
# ------------------------
@torch.no_grad()
def predict(model, img: Image.Image, classes: list, top_k: int = 3):
    tensor = EVAL_TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0]
    top_probs, top_idxs = probs.topk(top_k)
    return [(classes[i], p.item()) for i, p in zip(top_idxs, top_probs)]


# ------------------------
# Reference Image from Dataset
# ------------------------
def get_reference_image(label: str):
    """Get a random image from the dataset matching the predicted label"""
    label_dir = DATA_DIR / label
    if not label_dir.exists():
        return None
    all_imgs = list(label_dir.glob("*.png")) + list(label_dir.glob("*.jpg"))
    if not all_imgs:
        return None
    path = random.choice(all_imgs)
    return Image.open(path).convert("RGB")


# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="Pokemon Classifier", layout="wide")
st.title("Pokemon Classifier")
st.caption("Transfer Learning with ResNet50 / VGG16 / ConvNeXt")

classes = load_classes()
models = load_models(classes)

if not models:
    st.error("No checkpoints found. Run train.py first.")
    st.stop()

# ------------------------
# Image Input
# ------------------------
st.subheader("Upload Image")
uploaded = st.file_uploader("Upload a Pokemon image", type=["jpg", "jpeg", "png"])
input_img = None
if uploaded:
    input_img = Image.open(uploaded).convert("RGB")

# ------------------------
# Prediction
# ------------------------
majority_label = None
majority_votes = 0
model_top1 = {}
model_top3 = {}

if input_img is not None:
    with st.spinner("Running models..."):
        for name, model in models.items():
            results = predict(model, input_img, classes, top_k=3)
            model_top1[name] = results[0]
            model_top3[name] = results

    votes = [label for label, _ in model_top1.values()]
    vote_count = Counter(votes)
    majority_label, majority_votes = vote_count.most_common(1)[0]

# ------------------------
# Image Display
# ------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Uploaded Image")
    if input_img:
        st.image(input_img, width="stretch")
    else:
        st.info("No image uploaded yet.")

with col_right:
    st.subheader("Reference from Dataset")
    if majority_label is not None:
        ref_img = get_reference_image(majority_label)
        if ref_img:
            st.image(ref_img, caption=f"Label: {majority_label}", width="stretch")
        else:
            st.info(f"No reference image found for: {majority_label}")
    else:
        st.info("Upload an image to see a matching reference.")

if majority_label is not None:
    # Top: Majority vote result ──────────────────────────────────────
    st.markdown(
        f"""
        <div style='text-align:center; padding: 20px;
                    background: #1e1e2e; border-radius: 12px; margin-bottom: 20px;'>
            <div style='font-size: 18px; color: #aaa;'>Prediction</div>
            <div style='font-size: 48px; font-weight: bold; color: #f5c542;'>
                {majority_label}
            </div>
            <div style='font-size: 16px; color: #aaa;'>
                {majority_votes} / {len(models)} models agree
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Bottom: Per-model results ──────────────────────────────────────
    st.subheader("Model Results")

    MODEL_LABELS = {
        "resnet50_frozen": "ResNet50 — Frozen",
        "resnet50_finetune": "ResNet50 — Fine-tuned",
        "vgg16_frozen": "VGG16 — Frozen",
        "convnext_finetune": "ConvNeXt — Fine-tuned",
    }

    cols = st.columns(len(models))
    for col, (name, (top1_label, top1_prob)) in zip(cols, model_top1.items()):
        with col:
            is_correct = top1_label == majority_label
            badge = "✅" if is_correct else "❌"
            st.markdown(f"**{MODEL_LABELS.get(name, name)}**")
            st.markdown(
                f"<div style='font-size:20px; font-weight:bold;'>"
                f"{badge} {top1_label}</div>"
                f"<div style='color:#aaa;'>{top1_prob*100:.1f}%</div>",
                unsafe_allow_html=True,
            )

            # Top-3 breakdown
            st.markdown("**Top-3:**")
            for rank, (label, prob) in enumerate(model_top3[name], 1):
                bar_color = "#f5c542" if rank == 1 else "#555"
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:space-between;
                                margin-bottom:4px;'>
                        <span>{rank}. {label}</span>
                        <span style='color:#aaa;'>{prob*100:.1f}%</span>
                    </div>
                    <div style='background:#333; border-radius:4px; height:6px; margin-bottom:8px;'>
                        <div style='background:{bar_color}; width:{prob*100:.1f}%;
                                    height:6px; border-radius:4px;'></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
