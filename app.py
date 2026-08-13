import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image

# Guard heavy/non-critical imports so UI can run without them
try:
    import joblib
except Exception:
    joblib = None

try:
    import tensorflow as tf
except Exception:
    tf = None



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Defect Classifier",
    page_icon="",
    layout="centered"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

LOGISTIC_MODEL_PATH = (
    MODEL_DIR / "logistic_regression_model.pkl"
)

EFFICIENTNET_MODEL_PATH = (
    MODEL_DIR / "efficientnet_b0_feature_extractor.keras"
)


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = {
    0: "Crack",
    1: "Honeycomb",
    2: "Spalling"
}


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    # Return (None, None) when required libraries or model files are not available
    if joblib is None or tf is None:
        return None, None

    if not LOGISTIC_MODEL_PATH.exists() or not EFFICIENTNET_MODEL_PATH.exists():
        return None, None

    classifier = joblib.load(LOGISTIC_MODEL_PATH)
    feature_extractor = tf.keras.models.load_model(EFFICIENTNET_MODEL_PATH)

    return classifier, feature_extractor


classifier, feature_extractor = load_models()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image):
    if feature_extractor is None:
        return None

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    features = feature_extractor.predict(image_array, verbose=0)
    return features


# ============================================================
# INTERFACE
# ============================================================

st.title("Concrete Defect Detection")

st.write(
    "Upload an image to identify the type of concrete defect."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if classifier is None or feature_extractor is None:
        st.error("Model not loaded — predictions are unavailable. Install required packages or ensure model files exist in the `models/` folder.")
    else:
        with st.spinner("Analyzing image..."):
            features = extract_features(image)

            prediction = classifier.predict(features)[0]
            probabilities = classifier.predict_proba(features)[0]

            predicted_class = CLASS_NAMES[int(prediction)]
            confidence = probabilities[int(prediction)]

        # RESULT
        st.divider()
        st.subheader("Result")
        st.write(f"**Defect:** {predicted_class}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
        st.progress(float(confidence))

        # PROBABILITIES
        with st.expander("View class probabilities"):
            st.write(f"Crack: {probabilities[0] * 100:.2f}%")
            st.write(f"Honeycomb: {probabilities[1] * 100:.2f}%")
            st.write(f"Spalling: {probabilities[2] * 100:.2f}%")