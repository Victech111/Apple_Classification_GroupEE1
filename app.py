import streamlit as st
from PIL import Image
from tensorflow import keras
import numpy as np

st.set_page_config(page_title="Apple Freshness Classifier EE9", page_icon="🍎", layout="centered")
st.title("🍎 Apple Freshness Classifier")
st.write("Upload a photo of an apple and the model will predict whether it's fresh or rotten.")

IMG_SIZE = (128, 128)
CLASS_NAMES = ["Fresh", "Rotten"]
MODEL_PATH = "my_mobilenet_model.keras"


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


model = load_model()

uploaded_file = st.file_uploader("Upload an image of an apple", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Force RGB — handles PNGs with alpha channel and grayscale images safely
    image = Image.open(uploaded_file).convert("RGB")
    display_image = image.copy()

    # Preprocess to match training pipeline exactly
    image = image.resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(image_array, verbose=0)

    # Model has a single sigmoid output (binary classification)
    raw_score = float(prediction[0][0])
    predicted_class = CLASS_NAMES[1] if raw_score > 0.5 else CLASS_NAMES[0]
    confidence = raw_score if predicted_class == CLASS_NAMES[1] else 1 - raw_score

    st.subheader("Prediction Result")
    st.image(display_image, caption="Uploaded Image", use_container_width=True)

    if predicted_class == "Fresh":
        st.success(f"Predicted Class: **{predicted_class}**")
    else:
        st.error(f"Predicted Class: **{predicted_class}**")

    st.write(f"Confidence: **{confidence:.2%}**")
    st.progress(confidence)

    with st.expander("Raw model output"):
        st.write(f"Sigmoid output (probability of Rotten): {raw_score:.4f}")