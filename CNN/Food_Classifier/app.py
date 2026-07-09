import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Food Classifier", page_icon="🍽️")

MODEL_PATH = "model/food_model.keras"
IMG_SIZE   = (224, 224)
CLASSES    = {
    "Apple":    ("🍎", "Fruit"),
    "Banana":   ("🍌", "Fruit"),
    "Biryani":  ("🍛", "Rice Dish"),
    "Burger":   ("🍔", "Fast Food"),
    "Pizza":    ("🍕", "Fast Food"),
    "Sandwich": ("🥪", "Snack")
}

st.title("🍽️ Food Item Classifier")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

model = load_model()
if not model:
    st.warning("Model not found. Run `python train.py` first.")
    st.stop()

uploaded = st.file_uploader("Upload a food image", type=["jpg","jpeg","png","webp"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", width="stretch")

    arr = np.array(image.convert("RGB").resize(IMG_SIZE), dtype=np.float32)
    arr = preprocess_input(arr)
    try:
        preds = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
        st.write("Prediction array:", preds)
        st.write("Predicted index:", int(np.argmax(preds)))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    top   = np.argsort(preds)[::-1]

    # Build a label list matching model output length. If the hardcoded
    # `CLASSES` mapping length doesn't match the model output, fall back to
    # numeric class names for missing entries and avoid IndexError.
    names = list(CLASSES.keys())
    if len(preds) != len(names):
        names = names.copy()
        for k in range(len(names), len(preds)):
            names.append(f"Class_{k}")

    # Top prediction (safe lookup)
    top0 = int(top[0])
    name = names[top0]
    conf = float(preds[top0]) * 100
    emoji, category = CLASSES.get(name, ("", ""))

    st.subheader("Prediction")
    st.write(f"{emoji} **{name}** — {category}")
    st.metric("Confidence", f"{conf:.1f}%")
    st.progress(conf / 100)

    st.subheader("Other Possibilities")
    for i in top[1:]:
        idx = int(i)
        n = names[idx]
        emoji = CLASSES.get(n, ("", ""))[0]
        st.write(f"{emoji} {n} — {float(preds[idx])*100:.1f}%")
