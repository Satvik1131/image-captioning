"""
app/app.py
==========
Streamlit front-end for the Image Captioning project.

Usage (from project root):
    streamlit run app/app.py

Imports prediction utilities from src.predict so all model-loading and
greedy-decoding logic lives in one place.
"""

import os
import sys
import tempfile

import streamlit as st
from PIL import Image

# ── Resolve project root so `from src.predict import …` works regardless of CWD
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import (
    load_tokenizer,
    load_caption_model,
    build_feature_extractor,
    predict_caption,
    _read_max_length_from_h5,
    CAPTION_MODEL_PATH,
    TOKENIZER_PATH,
)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Image Captioning AI",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — dark glassmorphism theme
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Google Font ───────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global reset ──────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background gradient ───────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }

    /* ── Hide default Streamlit chrome ─────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Hero header ────────────────────────────────────────────── */
    .hero-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .hero-sub {
        text-align: center;
        color: rgba(255,255,255,0.55);
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 2.5rem;
        letter-spacing: 0.02em;
    }

    /* ── File uploader ──────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.04) !important;
        border: 2px dashed rgba(167,139,250,0.5) !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        transition: border-color 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(167,139,250,0.9) !important;
    }
    [data-testid="stFileUploader"] label {
        color: rgba(255,255,255,0.75) !important;
        font-weight: 500 !important;
    }

    /* ── Generate button ────────────────────────────────────────── */
    div[data-testid="stButton"] > button {
        width: 100%;
        padding: 0.75rem 1.5rem;
        font-size: 1.05rem;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #7c3aed, #3b82f6);
        color: #ffffff;
        cursor: pointer;
        letter-spacing: 0.04em;
        transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
        box-shadow: 0 4px 20px rgba(124,58,237,0.45);
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(124,58,237,0.65);
        opacity: 0.95;
    }
    div[data-testid="stButton"] > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(124,58,237,0.4);
    }

    /* ── Caption result box ─────────────────────────────────────── */
    .caption-box {
        background: linear-gradient(135deg,
            rgba(124,58,237,0.18) 0%,
            rgba(59,130,246,0.18) 100%);
        border: 1px solid rgba(167,139,250,0.35);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-top: 1.5rem;
        text-align: center;
        animation: fadeInUp 0.5s ease both;
    }
    .caption-label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(167,139,250,0.9);
        margin-bottom: 0.5rem;
    }
    .caption-text {
        font-size: 1.45rem;
        font-weight: 600;
        color: #f1f5f9;
        line-height: 1.5;
        letter-spacing: 0.01em;
    }

    /* ── Spinner colour ─────────────────────────────────────────── */
    .stSpinner > div {
        border-top-color: #7c3aed !important;
    }

    /* ── Fade-in animation ──────────────────────────────────────── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0);    }
    }

    /* ── Status / info messages ─────────────────────────────────── */
    .stAlert {
        border-radius: 12px !important;
    }

    /* ── Section divider ────────────────────────────────────────── */
    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin: 1.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (cached across reruns — loads only once per session)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load tokenizer, caption model and VGG16 once and cache them."""
    tokenizer  = load_tokenizer(TOKENIZER_PATH)
    vocab_size = len(tokenizer.word_index) + 1

    max_length = _read_max_length_from_h5(CAPTION_MODEL_PATH)
    if max_length is None:
        import json
        captions_path = os.path.join(
            PROJECT_ROOT, "data", "processed", "clean_captions.json"
        )
        if os.path.exists(captions_path):
            with open(captions_path, "r", encoding="utf-8") as f:
                captions_dict = json.load(f)
            lengths = [
                len(cap.split())
                for caps in captions_dict.values()
                for cap in caps
            ]
            max_length = max(lengths)
        else:
            max_length = 37

    caption_model = load_caption_model(CAPTION_MODEL_PATH, vocab_size, max_length)
    vgg_model     = build_feature_extractor()

    return caption_model, vgg_model, tokenizer, max_length


# ──────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">🖼️ Image Captioning AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Upload an image — let the model describe what it sees.</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOAD (with status banner)
# ──────────────────────────────────────────────────────────────────────────────

with st.spinner("⚙️  Loading models… (first run may take ~30 s)"):
    try:
        caption_model, vgg_model, tokenizer, max_length = load_all_models()
        models_ready = True
    except FileNotFoundError as e:
        st.error(
            f"**Model files not found.**\n\n"
            f"{e}\n\n"
            "Run the training pipeline first:\n"
            "```\npython src/train.py\n```"
        )
        models_ready = False
    except Exception as e:
        st.error(f"**Unexpected error loading models:** {e}")
        models_ready = False

if models_ready:
    st.success("✅ Models loaded and ready!", icon="🤖")

st.markdown("<hr>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# FILE UPLOADER
# ──────────────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    label="Drop your image here, or click to browse",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG · JPEG · PNG",
    disabled=not models_ready,
    label_visibility="visible",
)

# ──────────────────────────────────────────────────────────────────────────────
# IMAGE DISPLAY + INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

if uploaded_file is not None:

    # Show uploaded image centred
    image = Image.open(uploaded_file).convert("RGB")
    col_l, col_mid, col_r = st.columns([1, 4, 1])
    with col_mid:
        st.image(image, use_container_width=True, caption=uploaded_file.name)

    st.markdown("")

    # Generate Caption button
    if st.button("✨ Generate Caption", key="gen_btn", use_container_width=True):
        with st.spinner("🔍 Analysing image and generating caption…"):

            # Write the upload to a temp file so predict_caption can open it
            suffix = os.path.splitext(uploaded_file.name)[-1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                caption = predict_caption(
                    tmp_path,
                    caption_model,
                    vgg_model,
                    tokenizer,
                    max_length,
                )
            finally:
                os.unlink(tmp_path)   # clean up temp file

        # Display the result prominently
        caption_display = caption.capitalize()
        if caption_display and not caption_display.endswith("."):
            caption_display += "."

        st.markdown(
            f"""
            <div class="caption-box">
                <div class="caption-label">Generated Caption</div>
                <div class="caption-text">"{caption_display}"</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    # Placeholder hint when nothing is uploaded yet
    st.markdown(
        """
        <div style="
            text-align:center;
            color:rgba(255,255,255,0.3);
            font-size:0.95rem;
            padding: 2rem 0;
            letter-spacing:0.03em;">
            ⬆️ &nbsp; Upload an image above to get started
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center; color:rgba(255,255,255,0.25);
                font-size:0.78rem; letter-spacing:0.05em;">
        VGG16 + LSTM · Flickr8k · Greedy Decoding
    </div>
    """,
    unsafe_allow_html=True,
)
