import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image

# ============================================================
# OPTIONAL / HEAVY IMPORTS
# ============================================================

try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception as e:
    joblib = None
    JOBLIB_AVAILABLE = False
    JOBLIB_ERROR = str(e)

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    tf = None
    TENSORFLOW_AVAILABLE = False
    TENSORFLOW_ERROR = str(e)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Concrete Defect Detection",
    page_icon="🏗️",
    layout="centered"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"
EFFICIENTNET_MODEL_PATH = MODEL_DIR / "efficientnet_b0_feature_extractor.keras"


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = {
    0: "Crack",
    1: "Honeycomb",
    2: "Spalling"
}


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    # --------------------------------------------------------
    # Check required libraries
    # --------------------------------------------------------

    if not JOBLIB_AVAILABLE:
        raise ImportError(
            f"Joblib could not be imported.\n\n{JOBLIB_ERROR}"
        )

    if not TENSORFLOW_AVAILABLE:
        raise ImportError(
            f"TensorFlow could not be imported.\n\n{TENSORFLOW_ERROR}"
        )

    # --------------------------------------------------------
    # Check model directory
    # --------------------------------------------------------

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory does not exist:\n{MODEL_DIR}"
        )

    # --------------------------------------------------------
    # Check Logistic Regression model
    # --------------------------------------------------------

    if not LOGISTIC_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Logistic Regression model not found:\n"
            f"{LOGISTIC_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Check EfficientNet model
    # --------------------------------------------------------

    if not EFFICIENTNET_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"EfficientNet model not found:\n"
            f"{EFFICIENTNET_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load Logistic Regression classifier
    # --------------------------------------------------------

    classifier = joblib.load(LOGISTIC_MODEL_PATH)

    # --------------------------------------------------------
    # Load EfficientNet feature extractor
    # --------------------------------------------------------

    feature_extractor = tf.keras.models.load_model(
        EFFICIENTNET_MODEL_PATH
    )

    return classifier, feature_extractor


# ============================================================
# LOAD MODELS
# ============================================================

model_loading_error = None

try:
    classifier, feature_extractor = load_models()

except Exception as e:
    classifier = None
    feature_extractor = None
    model_loading_error = e


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image):

    if feature_extractor is None:
        raise RuntimeError(
            "EfficientNet feature extractor is not loaded."
        )

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to EfficientNet input size
    image = image.resize((224, 224))

    # Convert to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Extract EfficientNet features
    features = feature_extractor.predict(
        image_array,
        verbose=0
    )

    return features


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🏗️ Concrete Defect Detection")

st.write(
    "Upload an image to identify the type of concrete wall defect."
)

st.info(
    "Supported defects: Crack, Honeycomb and Spalling"
)


# ============================================================
# MODEL STATUS
# ============================================================

if model_loading_error is not None:

    st.error(
        "⚠️ Model loading failed."
    )

    with st.expander("View model loading details"):

        st.code(
            str(model_loading_error)
        )

else:

    st.success(
        "✅ AI model loaded successfully."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Concrete Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # Open uploaded image
        # ----------------------------------------------------

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # ----------------------------------------------------
        # Check model availability
        # ----------------------------------------------------

        if classifier is None or feature_extractor is None:

            st.error(
                "Predictions are unavailable because the model "
                "could not be loaded."
            )

            st.stop()

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with st.spinner(
            "Analyzing concrete image..."
        ):

            # Extract EfficientNet features
            features = extract_features(image)

            # Predict class
            prediction = classifier.predict(
                features
            )[0]

            # Get probabilities
            probabilities = classifier.predict_proba(
                features
            )[0]

            # Map prediction to class name
            predicted_class = CLASS_NAMES.get(
                int(prediction),
                "Unknown"
            )

            # Confidence
            confidence = probabilities[
                int(prediction)
            ]

        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.subheader("🔍 Prediction Result")

        st.write(
            f"**Defect:** {predicted_class}"
        )

        st.write(
            f"**Confidence:** {confidence * 100:.2f}%"
        )

        st.progress(
            float(confidence)
        )

        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        st.subheader("📊 Class Probabilities")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Crack",
                f"{probabilities[0] * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Honeycomb",
                f"{probabilities[1] * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Spalling",
                f"{probabilities[2] * 100:.2f}%"
            )

        # ====================================================
        # DETAILED PROBABILITIES
        # ====================================================

        with st.expander(
            "View detailed class probabilities"
        ):

            st.write(
                f"Crack: "
                f"{probabilities[0] * 100:.2f}%"
            )

            st.write(
                f"Honeycomb: "
                f"{probabilities[1] * 100:.2f}%"
            )

            st.write(
                f"Spalling: "
                f"{probabilities[2] * 100:.2f}%"
            )

        # ====================================================
        # INTERPRETATION
        # ====================================================

        st.divider()

        st.subheader("📋 Interpretation")

        if predicted_class == "Crack":

            st.write(
                "The model identifies the uploaded image "
                "as showing a concrete crack."
            )

        elif predicted_class == "Honeycomb":

            st.write(
                "The model identifies the uploaded image "
                "as showing honeycombing in the concrete."
            )

        elif predicted_class == "Spalling":

            st.write(
                "The model identifies the uploaded image "
                "as showing concrete spalling."
            )

    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        with st.expander(
            "View prediction error"
        ):

            st.code(
                f"{type(e).__name__}: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Concrete Defect Detection • "
    "EfficientNet-B0 Feature Extraction + "
    "Logistic Regression"
)