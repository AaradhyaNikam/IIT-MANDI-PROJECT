import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
import json

# ------------------------------
# Load Model and Class Mapping
# ------------------------------
models_dir = Path("models")
model_path = models_dir / "model.keras"
model = tf.keras.models.load_model(str(model_path))

mapping_path = models_dir / "class_mapping.json"
if mapping_path.exists():
    with open(mapping_path, "r", encoding="utf8") as f:
        class_mapping = json.load(f)
else:
    class_mapping = None


# ------------------------------
# Image Preprocessing Function
# ------------------------------
def preprocess(image, image_size=(224, 224)):  # Adjusted for transfer learning models
    img = image.convert("RGB").resize(image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ------------------------------
# Streamlit UI Setup
# ------------------------------
st.set_page_config(page_title="AgriVision AI", page_icon="🌿", layout="centered")

st.title("🌾 AgriVision AI – Crop Disease Detection")
st.write("Upload a plant leaf image below and click **Analyze Image** to detect crop diseases using AI.")

# ------------------------------
# Upload Section
# ------------------------------
uploaded_file = st.file_uploader("📤 Upload a crop image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Add Analyze Button
    analyze_button = st.button("🔍 Analyze Image")

    if analyze_button:
        with st.spinner("Analyzing... please wait ⏳"):
            # Preprocess and predict
            x = preprocess(image, image_size=(224, 224))
            preds = model.predict(x)
            idx = int(preds.argmax(axis=-1)[0])
            prob = float(preds[0, idx])
            percent = prob * 100.0
            label = class_mapping.get(str(idx), f"Class {idx}") if class_mapping else f"Class {idx}"

        # Display result
        st.success(f"✅ **Prediction:** {label}")
        st.info(f"🎯 **Confidence:** {percent:.2f}%")

        # Optional detailed view
        if st.checkbox("Show detailed probabilities"):
            st.subheader("🔢 Class Probabilities")
            for i, p in enumerate(preds[0]):
                cls = class_mapping.get(str(i), f"Class {i}") if class_mapping else f"Class {i}"
                st.write(f"{cls}: {p*100:.2f}%")

else:
    st.info("👆 Please upload an image to begin analysis.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Developed by Aaradhya Aashish Nikam | Team Mavericks | B.Tech Data Science & AI")

