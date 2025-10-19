# streamlit_app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from pathlib import Path
from PIL import Image

# -----------------------------
# Paths
# -----------------------------
models_dir = Path("models")
model_path = models_dir / "model.keras"
class_mapping_path = models_dir / "class_mapping.json"

# -----------------------------
# Load Model and Class Mapping
# -----------------------------
@st.cache_resource
def load_model_and_mapping():
    model = tf.keras.models.load_model(model_path)
    with open(class_mapping_path, "r", encoding="utf8") as f:
        class_mapping = json.load(f)
    return model, class_mapping

model, class_mapping = load_model_and_mapping()

# -----------------------------
# App Title
# -----------------------------
st.title("🌱 Plant Disease Detection")
st.write("Upload a plant leaf image or take a photo to detect its disease.")

# -----------------------------
# Image Upload / Camera
# -----------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Prediction
    preds = model.predict(x)
    idx = int(np.argmax(preds))
    prob = float(preds[0, idx])
    label = class_mapping[str(idx)]

    # Show Result
    st.success(f"Predicted: **{label}** ({prob*100:.2f}% confidence)")

# Optional: take photo from camera
if st.button("Take a Photo"):
    picture = st.camera_input("Capture your leaf")
    if picture is not None:
        img = Image.open(picture).convert("RGB")
        st.image(img, caption='Captured Image', use_column_width=True)

        # Preprocess and predict
        img_resized = img.resize((224, 224))
        x = image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        preds = model.predict(x)
        idx = int(np.argmax(preds))
        prob = float(preds[0, idx])
        label = class_mapping[str(idx)]
        st.success(f"Predicted: **{label}** ({prob*100:.2f}% confidence)")
