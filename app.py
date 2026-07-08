# ==============================================================================
# SkinCareAI - Dermatological Screening Assistant
# Streamlit Web Application
# ==============================================================================

# ==============================================================================
# Imports
# ==============================================================================
import os
import sys
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import CLASS_NAMES, CLASS_FULL_NAMES, TRIAGE_MAP
from src.data_loader import preprocess_single_image


# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="SkinCareAI — Dermatological Screening",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# Custom CSS
# ==============================================================================
st.markdown("""
<style>
/* ---- Google Fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- Global Styles ---- */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 95rem;
}

/* ---- Dark Glassmorphism Cards ---- */
.glass-card {
    background: rgba(17, 25, 40, 0.65);
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
}

.glass-header {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #8892b0;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    padding-bottom: 0.5rem;
}

/* ---- Animated Triage Cards ---- */
.triage-urgent {
    background: linear-gradient(135deg, rgba(231, 76, 60, 0.15) 0%, rgba(192, 57, 43, 0.25) 100%);
    border: 1px solid rgba(231, 76, 60, 0.4);
    color: #ff6b6b;
    padding: 1.2rem;
    border-radius: 12px;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-align: center;
    box-shadow: 0 4px 20px rgba(231, 76, 60, 0.15);
    animation: pulseRed 2.5s infinite alternate;
}

.triage-monitor {
    background: linear-gradient(135deg, rgba(243, 156, 18, 0.15) 0%, rgba(211, 84, 0, 0.25) 100%);
    border: 1px solid rgba(243, 156, 18, 0.4);
    color: #f39c12;
    padding: 1.2rem;
    border-radius: 12px;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-align: center;
    box-shadow: 0 4px 20px rgba(243, 156, 18, 0.1);
}

.triage-low {
    background: linear-gradient(135deg, rgba(39, 174, 96, 0.12) 0%, rgba(30, 130, 76, 0.22) 100%);
    border: 1px solid rgba(39, 174, 96, 0.4);
    color: #2ecc71;
    padding: 1.2rem;
    border-radius: 12px;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-align: center;
    box-shadow: 0 4px 20px rgba(39, 174, 96, 0.1);
}

.triage-message {
    background: rgba(255, 255, 255, 0.02);
    border-left: 4px solid #8892b0;
    padding: 1rem;
    border-radius: 4px 12px 12px 4px;
    font-size: 0.92rem;
    color: #cbd5e1;
    line-height: 1.5;
    margin-top: 0.8rem;
    border: 1px solid rgba(255, 255, 255, 0.03);
    border-left-width: 4px;
}

/* ---- Result Details ---- */
.result-class {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #64ffda 0%, #00b4d8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}

.result-full {
    font-size: 1rem;
    color: #a8b2d1;
    margin-bottom: 1.2rem;
    font-weight: 500;
}

/* ---- Probability Row Layout ---- */
.prob-label-container {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    margin-bottom: 0.2rem;
    color: #cbd5e1;
}
.prob-label-top {
    font-weight: 700;
    color: #64ffda;
}
.prob-pct-top {
    font-weight: 700;
    color: #64ffda;
}

/* ---- Pulse Animation ---- */
@keyframes pulseRed {
    0% {
        box-shadow: 0 4px 20px rgba(231, 76, 60, 0.1);
        border-color: rgba(231, 76, 60, 0.3);
    }
    100% {
        box-shadow: 0 4px 30px rgba(231, 76, 60, 0.35);
        border-color: rgba(231, 76, 60, 0.6);
    }
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Load Model
# ==============================================================================
MODEL_PATH = os.path.join("models", "skin_model.keras")

# Paste your direct download link here (Hugging Face, Dropbox, Google Drive direct, or GitHub Release)
# Or set it in your Streamlit Secrets as: MODEL_URL = "your_link"
MODEL_URL = st.secrets.get("MODEL_URL", "YOUR_DIRECT_DOWNLOAD_LINK_HERE")

@st.cache_resource
def load_model():
    """
    Load the trained SkinCareAI model from disk.
    If the model weights are not present, attempt to download them from MODEL_URL.
    """
    if not os.path.exists(MODEL_PATH):
        if MODEL_URL == "YOUR_DIRECT_DOWNLOAD_LINK_HERE":
            return None
        
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with st.spinner("📥 Downloading pre-trained model weights (~210MB)... This will only happen once."):
            import urllib.request
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                st.success("🎉 Model weights downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to download model weights: {e}")
                return None
                
    return keras.models.load_model(MODEL_PATH)


model = load_model()


# ==============================================================================
# Title
# ==============================================================================
st.markdown(
    "<h1 style='text-align:center;'>🔬 SkinCareAI — Dermatological Screening</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:gray;'>Powered by ResNet50 Transfer Learning | HAM10000 Dataset</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;'>Upload a skin lesion image and click <b>Analyse</b> to receive an AI-assisted triage assessment.</p>",
    unsafe_allow_html=True,
)

st.divider()


# ==============================================================================
# Main Layout
# ==============================================================================
info_col, image_col, result_col = st.columns([1, 2, 1.4])


# ==============================================================================
# Left Column — Project Information
# ==============================================================================
with info_col:
    st.markdown("""
<div class='glass-card'>
    <div class='glass-header'>📋 System Specifications</div>
    <div style='margin-bottom: 1rem;'>
        <div style='font-size: 0.8rem; color: #8892b0; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;'>Base Architecture</div>
        <div style='font-size: 0.95rem; font-weight: 600; color: #cbd5e1;'>ResNet50 (Transfer Learning)</div>
    </div>
    <div style='margin-bottom: 1rem;'>
        <div style='font-size: 0.8rem; color: #8892b0; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;'>Dataset</div>
        <div style='font-size: 0.95rem; font-weight: 600; color: #cbd5e1;'>HAM10000 (10,015 images)</div>
    </div>
    <div style='margin-bottom: 1rem;'>
        <div style='font-size: 0.8rem; color: #8892b0; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;'>Image Resolution</div>
        <div style='font-size: 0.95rem; font-weight: 600; color: #cbd5e1;'>224 × 224 pixels</div>
    </div>
    <div style='margin-bottom: 1.5rem;'>
        <div style='font-size: 0.8rem; color: #8892b0; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;'>Triage Urgency Levels</div>
        <div style='display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.3rem;'>
            <div style='background: rgba(231,76,60,0.12); border: 1px solid rgba(231,76,60,0.25); color: #ff6b6b; padding: 0.3rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.02em;'>🔴 URGENT (mel, bcc, akiec)</div>
            <div style='background: rgba(243,156,18,0.12); border: 1px solid rgba(243,156,18,0.25); color: #f39c12; padding: 0.3rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.02em;'>🟡 MONITOR (bkl, df)</div>
            <div style='background: rgba(39,174,96,0.12); border: 1px solid rgba(39,174,96,0.25); color: #2ecc71; padding: 0.3rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.02em;'>🟢 LOW RISK (nv, vasc)</div>
        </div>
    </div>
    <div class='glass-header'>🔬 Diagnostic Taxonomy</div>
    <div style='font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;'>
        <div style='margin-bottom: 0.25rem;'><span style='color: #64ffda; font-weight: 600;'>akiec</span>: Actinic Keratosis</div>
        <div style='margin-bottom: 0.25rem;'><span style='color: #64ffda; font-weight: 600;'>bcc</span>: Basal Cell Carcinoma</div>
        <div style='margin-bottom: 0.25rem;'><span style='color: #64ffda; font-weight: 600;'>bkl</span>: Benign Keratosis</div>
        <div style='margin-bottom: 0.25rem;'><span style='color: #64ffda; font-weight: 600;'>df</span>: Dermatofibroma</div>
        <div style='margin-bottom: 0.25rem;'><span style='color: #64ffda; font-weight: 600;'>mel</span>: Melanoma</div>
        <div style='margin-bottom: 0.25rem;'><span style='color: #64ffda; font-weight: 600;'>nv</span>: Melanocytic Nevi</div>
        <div style='margin-bottom: 0;'><span style='color: #64ffda; font-weight: 600;'>vasc</span>: Vascular Lesion</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# Middle Column — Image Upload
# ==============================================================================
with image_col:

    uploaded_file = st.file_uploader(
        "Upload Skin Lesion Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a dermoscopy or clinical photograph of the skin lesion.",
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Skin Lesion Image",
            width='stretch',
        )


# ==============================================================================
# Right Column — Prediction & Triage
# ==============================================================================
with result_col:

    st.subheader("🎯 Diagnosis")

    if uploaded_file is not None:

        if st.button("Analyse", width='stretch'):

            with st.spinner("Analysing skin lesion..."):

                # -------------------------------------------------------------
                # Model Not Loaded Guard
                # -------------------------------------------------------------
                if model is None:
                    st.error(
                        "⚠️ Model not found.\n\n"
                        f"Expected: `{MODEL_PATH}`\n\n"
                        "Run the training pipeline first:\n\n"
                        "```bash\npython src/train.py\n```"
                    )
                    st.stop()

                # -------------------------------------------------------------
                # Preprocessing
                # -------------------------------------------------------------
                img_array = np.array(image)
                img_tensor = preprocess_single_image(img_array)

                # -------------------------------------------------------------
                # Prediction
                # -------------------------------------------------------------
                prediction   = model.predict(img_tensor, verbose=0)
                predicted_idx = int(np.argmax(prediction))
                confidence    = float(np.max(prediction))
                predicted_cls = CLASS_NAMES[predicted_idx]

                # -------------------------------------------------------------
                # Triage Lookup
                # -------------------------------------------------------------
                triage_level, triage_color, triage_message = TRIAGE_MAP[predicted_cls]

                # -------------------------------------------------------------
                # Result Display — Class Name
                # -------------------------------------------------------------
                st.markdown(
                    f"<div class='result-class'>{predicted_cls.upper()}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='result-full'>{CLASS_FULL_NAMES[predicted_cls]}</div>",
                    unsafe_allow_html=True,
                )

                # -------------------------------------------------------------
                # Triage Badge
                # -------------------------------------------------------------
                triage_css = {
                    "URGENT":   "triage-urgent",
                    "MONITOR":  "triage-monitor",
                    "LOW RISK": "triage-low",
                }.get(triage_level, "triage-low")

                st.markdown(
                    f"<div class='{triage_css}'>🚦 {triage_level}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div class='triage-message'>{triage_message}</div>",
                    unsafe_allow_html=True,
                )

                st.divider()

                # -------------------------------------------------------------
                # Confidence Score
                # -------------------------------------------------------------
                st.subheader("Confidence")
                st.progress(confidence)
                st.write(f"### {confidence * 100:.2f}%")

                st.divider()

                # -------------------------------------------------------------
                # Class Probability Distribution
                # -------------------------------------------------------------
                st.subheader("Class Probabilities")

                probs = prediction[0]

                for i, cls in enumerate(CLASS_NAMES):
                    prob_val = float(probs[i])
                    is_top   = (i == predicted_idx)
                    label_class = "prob-label-top" if is_top else "prob-label"
                    pct_class   = "prob-pct-top" if is_top else "prob-pct"
                    checkmark   = " ✓" if is_top else ""
                    
                    st.markdown(
                        f"<div class='prob-label-container'>"
                        f"<span class='{label_class}'>{cls.upper()} ({CLASS_FULL_NAMES[cls]}){checkmark}</span>"
                        f"<span class='{pct_class}'>{prob_val * 100:.2f}%</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.progress(prob_val)

    else:
        st.info("📤 Upload an image to begin analysis.")


# ==============================================================================
# Footer
# ==============================================================================
st.divider()

st.caption(
    "⚕️ **Clinical Disclaimer**: SkinCareAI is an AI-assisted screening tool for "
    "educational and research purposes only. It does not constitute medical advice "
    "and must not be used as a substitute for professional clinical diagnosis. "
    "Always consult a qualified dermatologist for medical evaluation. | "
    "Developed using ResNet50 Transfer Learning on the HAM10000 dataset."
)
