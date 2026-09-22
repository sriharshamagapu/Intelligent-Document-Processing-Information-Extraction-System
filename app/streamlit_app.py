import sys
from pathlib import Path
import json
import hashlib
import html

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Load PyTorch before PaddleOCR/PaddlePaddle on Windows.
import torch

from src.ocr import run_ocr
from src.information_extraction import extract_document_info
from src.layoutlmv3_inference import LayoutLMv3Inference


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Document Processing",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING — FUTURISTIC DOCUMENT AI INTERFACE
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- GLOBAL ---------- */
    :root {
        --ink: #eaf2ff;
        --muted: #91a4c4;
        --cyan: #22d3ee;
        --blue: #4f7cff;
        --violet: #8b5cf6;
        --pink: #ec4899;
        --panel: rgba(10, 18, 38, .72);
        --line: rgba(120, 160, 255, .18);
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 12%, rgba(34,211,238,.14), transparent 25%),
            radial-gradient(circle at 88% 8%, rgba(139,92,246,.18), transparent 28%),
            radial-gradient(circle at 72% 78%, rgba(236,72,153,.08), transparent 25%),
            linear-gradient(135deg, #030712 0%, #071126 45%, #050816 100%);
        color: var(--ink);
        overflow-x: hidden;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        opacity: .20;
        background-image:
            linear-gradient(rgba(99,102,241,.10) 1px, transparent 1px),
            linear-gradient(90deg, rgba(99,102,241,.10) 1px, transparent 1px);
        background-size: 42px 42px;
        mask-image: linear-gradient(to bottom, black, transparent 92%);
        animation: gridDrift 18s linear infinite;
    }

    @keyframes gridDrift {
        from { transform: translate3d(0,0,0); }
        to { transform: translate3d(42px,42px,0); }
    }

    .stApp::after {
        content: "";
        position: fixed;
        width: 420px;
        height: 420px;
        right: -180px;
        top: 120px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(34,211,238,.10), transparent 68%);
        filter: blur(10px);
        pointer-events: none;
        animation: floatOrb 9s ease-in-out infinite;
        z-index: 0;
    }

    @keyframes floatOrb {
        0%,100% { transform: translate(0,0) scale(1); }
        50% { transform: translate(-90px,70px) scale(1.15); }
    }

    [data-testid="stHeader"] {
        background: rgba(3,7,18,.55) !important;
        backdrop-filter: blur(16px);
    }

    .main .block-container {
        position: relative;
        z-index: 1;
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    /* ---------- HERO ---------- */
    .hero-wrap {
        position: relative;
        padding: 34px 24px 28px;
        margin-bottom: 24px;
        border: 1px solid rgba(92,135,255,.24);
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(8,18,40,.88), rgba(17,13,48,.72));
        box-shadow: 0 25px 80px rgba(0,0,0,.34), inset 0 1px rgba(255,255,255,.05);
        overflow: hidden;
    }

    .hero-wrap::before {
        content: "";
        position: absolute;
        width: 620px;
        height: 2px;
        left: -120px;
        top: 0;
        background: linear-gradient(90deg, transparent, var(--cyan), var(--violet), transparent);
        box-shadow: 0 0 25px var(--cyan);
        animation: laser 5s linear infinite;
    }

    @keyframes laser {
        0% { transform: translateX(-20%); opacity: 0; }
        15% { opacity: 1; }
        80% { opacity: 1; }
        100% { transform: translateX(230%); opacity: 0; }
    }

    .hero-title {
        font-size: clamp(34px, 4vw, 58px);
        line-height: 1.05;
        font-weight: 900;
        text-align: center;
        letter-spacing: -2px;
        margin: 5px 0 10px;
        background: linear-gradient(100deg, #f8fbff 5%, #67e8f9 32%, #818cf8 58%, #f0abfc 82%, #fff 100%);
        background-size: 250% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titleFlow 6s linear infinite;
        filter: drop-shadow(0 0 22px rgba(79,124,255,.18));
    }

    @keyframes titleFlow {
        to { background-position: 250% center; }
    }

    .hero-subtitle {
        text-align: center;
        color: #9fb0cf;
        font-size: 15px;
        letter-spacing: .5px;
        margin-bottom: 18px;
    }

    .ai-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 9px 18px;
        border-radius: 999px;
        color: #dffbff;
        font-weight: 800;
        font-size: 13px;
        border: 1px solid rgba(34,211,238,.35);
        background: rgba(14,30,57,.78);
        box-shadow: 0 0 25px rgba(34,211,238,.12), inset 0 0 20px rgba(79,124,255,.08);
        animation: badgePulse 2.8s ease-in-out infinite;
    }

    @keyframes badgePulse {
        0%,100% { box-shadow: 0 0 18px rgba(34,211,238,.10), inset 0 0 20px rgba(79,124,255,.06); }
        50% { box-shadow: 0 0 34px rgba(34,211,238,.22), inset 0 0 24px rgba(139,92,246,.10); }
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050a18 0%, #071126 55%, #050816 100%) !important;
        border-right: 1px solid rgba(91,124,255,.22);
    }

    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    .pipeline-shell {
        position: relative;
        padding: 8px 2px 6px;
    }

    .pipeline-head {
        font-size: 19px;
        font-weight: 900;
        color: #f5f8ff;
        letter-spacing: .2px;
        margin-bottom: 18px;
    }

    .pipeline-head span {
        color: var(--cyan);
        text-shadow: 0 0 18px rgba(34,211,238,.55);
    }

    .pipeline {
        position: relative;
        padding-left: 10px;
    }

    .pipeline::before {
        content: "";
        position: absolute;
        left: 21px;
        top: 22px;
        bottom: 22px;
        width: 2px;
        background: linear-gradient(180deg, rgba(34,211,238,.75), rgba(99,102,241,.55), rgba(236,72,153,.45));
        box-shadow: 0 0 12px rgba(34,211,238,.30);
    }

    .pipe-step {
        position: relative;
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0 0 10px;
        padding: 10px 9px;
        border: 1px solid rgba(105,130,190,.16);
        border-radius: 14px;
        background: rgba(11,22,46,.60);
        backdrop-filter: blur(10px);
        transition: transform .25s ease, border-color .25s ease, background .25s ease, box-shadow .25s ease;
        animation: stepIn .65s both;
    }

    .pipe-step:nth-child(2) { animation-delay: .08s; }
    .pipe-step:nth-child(3) { animation-delay: .16s; }
    .pipe-step:nth-child(4) { animation-delay: .24s; }
    .pipe-step:nth-child(5) { animation-delay: .32s; }
    .pipe-step:nth-child(6) { animation-delay: .40s; }
    .pipe-step:nth-child(7) { animation-delay: .48s; }
    .pipe-step:nth-child(8) { animation-delay: .56s; }

    .pipe-step:hover {
        transform: translateX(5px);
        border-color: rgba(34,211,238,.42);
        background: rgba(15,31,61,.88);
        box-shadow: 0 8px 28px rgba(0,0,0,.28), 0 0 20px rgba(34,211,238,.08);
    }

    @keyframes stepIn {
        from { opacity: 0; transform: translateX(-14px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .pipe-icon {
        position: relative;
        z-index: 2;
        width: 27px;
        height: 27px;
        min-width: 27px;
        display: grid;
        place-items: center;
        border-radius: 9px;
        background: linear-gradient(135deg, #172b5d, #101a3b);
        border: 1px solid rgba(103,232,249,.32);
        box-shadow: 0 0 16px rgba(34,211,238,.12);
        font-size: 13px;
    }

    .pipe-text {
        color: #dce7fb;
        font-size: 12.5px;
        font-weight: 750;
        line-height: 1.25;
    }

    .pipe-live {
        margin-left: auto;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 12px #34d399;
        animation: livePulse 1.4s infinite;
    }

    @keyframes livePulse {
        0%,100% { opacity: .55; transform: scale(.85); }
        50% { opacity: 1; transform: scale(1.25); }
    }

    .pipeline-foot {
        margin-top: 18px;
        padding: 12px;
        border-radius: 14px;
        border: 1px solid rgba(139,92,246,.20);
        background: linear-gradient(135deg, rgba(49,24,92,.32), rgba(9,28,57,.35));
        color: #9fb0cf;
        font-size: 11px;
        line-height: 1.5;
    }

    .pipeline-foot b { color: #d8c7ff; }

    /* ---------- UPLOAD ---------- */
    [data-testid="stFileUploader"] {
        border-radius: 20px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, rgba(10,24,50,.84), rgba(23,15,52,.72)) !important;
        border: 1px dashed rgba(103,232,249,.35) !important;
        border-radius: 20px !important;
        box-shadow: inset 0 0 35px rgba(34,211,238,.035), 0 12px 35px rgba(0,0,0,.18);
        transition: all .3s ease;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(103,232,249,.90) !important;
        box-shadow: 0 0 35px rgba(34,211,238,.16), inset 0 0 35px rgba(34,211,238,.07);
        transform: translateY(-2px);
    }

    /* Bright, clearly visible Streamlit upload button */
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #00c6ff 0%, #6366f1 52%, #a855f7 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,.55) !important;
        border-radius: 12px !important;
        font-weight: 900 !important;
        min-height: 42px !important;
        padding: 0 22px !important;
        box-shadow: 0 0 22px rgba(0,198,255,.38), 0 7px 20px rgba(99,102,241,.28) !important;
        transition: transform .2s ease, box-shadow .2s ease, filter .2s ease !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        filter: brightness(1.16) !important;
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 0 30px rgba(0,198,255,.55), 0 10px 25px rgba(99,102,241,.35) !important;
    }

    [data-testid="stFileUploaderDropzone"] small {
        color: #9db3d7 !important;
    }

    /* ---------- BUTTONS ---------- */
    .stButton > button, .stDownloadButton > button {
        border: 1px solid rgba(103,232,249,.25) !important;
        border-radius: 13px !important;
        font-weight: 800 !important;
        color: #e9f7ff !important;
        background: linear-gradient(135deg, rgba(25,65,125,.95), rgba(73,40,138,.92)) !important;
        box-shadow: 0 8px 25px rgba(0,0,0,.22), 0 0 18px rgba(79,124,255,.10);
        transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease !important;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-3px) scale(1.01);
        border-color: rgba(103,232,249,.60) !important;
        box-shadow: 0 13px 32px rgba(0,0,0,.28), 0 0 28px rgba(34,211,238,.16);
    }

    /* ---------- CONTENT CARDS ---------- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(11,24,49,.78), rgba(20,16,49,.72)) !important;
        border: 1px solid rgba(109,139,207,.18) !important;
        border-radius: 17px !important;
        padding: 15px !important;
        box-shadow: 0 12px 32px rgba(0,0,0,.22), inset 0 1px rgba(255,255,255,.03);
        transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: rgba(34,211,238,.32) !important;
        box-shadow: 0 18px 40px rgba(0,0,0,.28), 0 0 25px rgba(34,211,238,.08);
    }

    div[data-testid="stMetricLabel"] { color: #91a4c4 !important; }
    div[data-testid="stMetricValue"] { color: #f4f8ff !important; }

    .result-card {
        padding: 20px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(9,21,43,.80), rgba(20,14,46,.70));
        border: 1px solid rgba(105,135,205,.20);
        box-shadow: 0 15px 45px rgba(0,0,0,.22), inset 0 1px rgba(255,255,255,.03);
        margin-bottom: 13px;
        animation: cardIn .55s ease both;
    }

    @keyframes cardIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .field-label { color: #8fa4c7; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .7px; margin-bottom: 4px; }
    .field-value { color: #eef6ff; font-size: 16px; font-weight: 650; word-break: break-word; }

    .section-glow {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 20px 0 10px;
        color: #eef6ff;
        font-size: 18px;
        font-weight: 850;
    }

    .section-glow::before {
        content: "";
        width: 4px;
        height: 24px;
        border-radius: 999px;
        background: linear-gradient(180deg, var(--cyan), var(--violet));
        box-shadow: 0 0 14px rgba(34,211,238,.42);
    }

    .footer {
        text-align: center;
        color: #667da5;
        padding: 30px 10px 15px;
        font-size: 12px;
        letter-spacing: .4px;
    }

    /* Streamlit text / headings on dark canvas */
    .stMarkdown, .stText, label, p, h1, h2, h3, h4 { color: inherit; }
    [data-testid="stExpander"] {
        background: rgba(7,17,35,.55);
        border: 1px solid rgba(105,135,205,.17);
        border-radius: 15px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(105,135,205,.18);
        border-radius: 14px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_layoutlmv3():
    return LayoutLMv3Inference()


# ============================================================
# HELPERS
# ============================================================

def clean_display_value(value):
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value).strip()
    if isinstance(value, list):
        return ", ".join(clean_display_value(x) for x in value if clean_display_value(x))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def pretty_label(value):
    text = str(value).replace("_", " ").replace("-", " ")
    return " ".join(word.capitalize() for word in text.split())


def flatten_scalar_fields(data, prefix=""):
    """Return useful scalar/list fields from a nested extraction dictionary."""
    output = []
    if not isinstance(data, dict):
        return output
    for key, value in data.items():
        if key in {"table_rows", "academic_information"}:
            continue
        label = f"{prefix} {key}".strip() if prefix else str(key)
        if isinstance(value, dict):
            output.extend(flatten_scalar_fields(value, label))
        elif isinstance(value, list):
            if value and all(not isinstance(x, (dict, list)) for x in value):
                output.append((label, ", ".join(str(x) for x in value)))
        elif value not in (None, "", False):
            output.append((label, value))
    return output


def normalize_table_rows(rows):
    """Convert common table representations into a displayable DataFrame."""
    if not rows:
        return None

    if isinstance(rows, pd.DataFrame):
        return rows

    if isinstance(rows, dict):
        # A dictionary of columns is already suitable for DataFrame in many cases.
        try:
            return pd.DataFrame(rows)
        except Exception:
            return None

    if not isinstance(rows, list):
        return None

    if not rows:
        return None

    if all(isinstance(row, dict) for row in rows):
        try:
            return pd.DataFrame(rows)
        except Exception:
            return None

    if all(isinstance(row, (list, tuple)) for row in rows):
        try:
            # Grade-card style rows are normalized only for presentation.
            if rows and len(rows[0]) == 4:
                return pd.DataFrame(
                    rows,
                    columns=["Course Code", "Name / Description", "Credits", "Grade"],
                )
            return pd.DataFrame(rows)
        except Exception:
            return None

    return None


def collect_tables(entities):
    tables = []
    if not isinstance(entities, dict):
        return tables

    for key, value in entities.items():
        if "table" in str(key).lower() and value:
            df = normalize_table_rows(value)
            if df is not None and not df.empty:
                tables.append((pretty_label(key), df))
    return tables


def make_ocr_layout_image(image, boxes, words, scores):
    output = image.copy()
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for index, box in enumerate(boxes, start=1):
        if not box or len(box) < 4:
            continue
        try:
            x1, y1, x2, y2 = [int(float(v)) for v in box[:4]]
        except Exception:
            continue
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        label = str(index)
        bbox = draw.textbbox((x1, y1), label, font=font)
        label_w = bbox[2] - bbox[0] + 6
        label_h = bbox[3] - bbox[1] + 4
        label_y = max(0, y1 - label_h)
        draw.rectangle([x1, label_y, x1 + label_w, y1], fill="green")
        draw.text((x1 + 3, label_y + 1), label, fill="white", font=font)
    return output


def make_semantic_image(image, layout_results):
    output = image.copy()
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for item in layout_results or []:
        label = str(item.get("label", "O"))
        if label == "O":
            continue
        box = item.get("box", [])
        if not box or len(box) < 4:
            continue
        try:
            x1, y1, x2, y2 = [int(float(v)) for v in box[:4]]
        except Exception:
            continue
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 18)), label, fill="red", font=font)
    return output


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-title">🤖 Intelligent Document AI</div>
        <div class="hero-subtitle">
            See your document transform from pixels → text → structure → intelligence
        </div>
        <div style="text-align:center;">
            <span class="ai-badge">
                ⚡ PaddleOCR &nbsp;•&nbsp; LayoutLMv3 &nbsp;•&nbsp; Generic AI Extraction &nbsp;•&nbsp; Live Analysis
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — ANIMATED PIPELINE
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="pipeline-shell">
            <div class="pipeline-head">✦ <span>AI</span> Processing Pipeline</div>
            <div class="pipeline">
                <div class="pipe-step"><div class="pipe-icon">📤</div><div class="pipe-text">Upload Document</div><div class="pipe-live"></div></div>
                <div class="pipe-step"><div class="pipe-icon">👁</div><div class="pipe-text">PaddleOCR Vision</div><div class="pipe-live"></div></div>
                <div class="pipe-step"><div class="pipe-icon">⌗</div><div class="pipe-text">Text + Coordinates</div><div class="pipe-live"></div></div>
                <div class="pipe-step"><div class="pipe-icon">◈</div><div class="pipe-text">Document Detection</div><div class="pipe-live"></div></div>
                <div class="pipe-step"><div class="pipe-icon">🧠</div><div class="pipe-text">LayoutLMv3 AI</div><div class="pipe-live"></div></div>
                <div class="pipe-step"><div class="pipe-icon">✦</div><div class="pipe-text">Generic Extraction</div><div class="pipe-live"></div></div>
                <div class="pipe-step"><div class="pipe-icon">📊</div><div class="pipe-text">Results & Download</div><div class="pipe-live"></div></div>
            </div>
            <div class="pipeline-foot">
                <b>● SYSTEM ONLINE</b><br>
                Vision engine connected · CPU inference ready · Dynamic document understanding enabled
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "📤 Upload any document image",
    type=["png", "jpg", "jpeg"],
    help="Upload a clear document image. The system extracts information dynamically from the document.",
)


if uploaded:
    file_bytes = uploaded.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    image = Image.open(uploaded).convert("RGB")

    st.subheader("🖼️ Uploaded Document")
    st.image(image, caption=uploaded.name, width="stretch")

    previous_hash = st.session_state.get("processed_file_hash")
    already_processed = previous_hash == file_hash

    run_button = st.button(
        "🚀 Run Document Intelligence",
        type="primary",
        width="stretch",
    )

    if run_button:
        if already_processed:
            st.info("⚡ This document is already processed. Showing stored results.")
        else:
            for key in [
                "processed_file_hash",
                "ocr_words",
                "ocr_boxes",
                "ocr_scores",
                "document_info",
                "layout_results",
            ]:
                st.session_state.pop(key, None)

            progress = st.progress(0)
            status = st.empty()

            # ---------------- OCR ----------------
            status.markdown("🔍 **Step 1/3 — Running PaddleOCR...**")
            try:
                with st.spinner("Extracting text and coordinates..."):
                    words, boxes, scores = run_ocr(image)
            except Exception as error:
                progress.empty()
                status.empty()
                st.error(f"❌ OCR processing failed: {error}")
                st.stop()

            progress.progress(33)

            # ---------------- Generic extraction ----------------
            status.markdown("🧠 **Step 2/3 — Extracting document information...**")
            try:
                with st.spinner("Understanding the uploaded document..."):
                    info = extract_document_info(words, boxes)
            except Exception as error:
                progress.empty()
                status.empty()
                st.error(f"❌ Information extraction failed: {error}")
                st.stop()

            progress.progress(66)

            # ---------------- LayoutLMv3 ----------------
            status.markdown("🤖 **Step 3/3 — Running LayoutLMv3...**")
            layout_results = []
            try:
                with st.spinner("Analyzing document layout and semantic regions..."):
                    layout_model = load_layoutlmv3()
                    # Keep generic OCR/extraction on the full document, but cap
                    # LayoutLMv3 inference input to reduce CPU latency on Windows.
                    layout_words = words[:256]
                    layout_boxes = boxes[:256]
                    layout_results = layout_model.predict(
                        image,
                        layout_words,
                        layout_boxes
                    )
            except Exception as error:
                # Generic extraction remains usable even if the optional semantic stage fails.
                st.warning(f"⚠️ LayoutLMv3 analysis could not be completed: {error}")

            progress.progress(100)
            progress.empty()
            status.empty()

            st.session_state["processed_file_hash"] = file_hash
            st.session_state["ocr_words"] = words
            st.session_state["ocr_boxes"] = boxes
            st.session_state["ocr_scores"] = scores
            st.session_state["document_info"] = info
            st.session_state["layout_results"] = layout_results

            st.success("🎉 Document intelligence completed successfully!")

    # ========================================================
    # LOAD RESULTS
    # ========================================================

    current_hash = st.session_state.get("processed_file_hash")
    if current_hash == file_hash:
        words = st.session_state.get("ocr_words", [])
        boxes = st.session_state.get("ocr_boxes", [])
        scores = st.session_state.get("ocr_scores", [])
        info = st.session_state.get("document_info", {})
        layout_results = st.session_state.get("layout_results", [])
    else:
        words = None
        boxes = None
        scores = None
        info = None
        layout_results = None

    # ========================================================
    # RESULTS
    # ========================================================

    if words is not None and boxes is not None and scores is not None and info is not None:
        combined_text = "\n".join(str(word) for word in words)
        average_confidence = sum(scores) / len(scores) if scores else 0.0
        document_type = info.get("document_type", "Unknown Document") if isinstance(info, dict) else "Unknown Document"
        entities = info.get("entities", {}) if isinstance(info, dict) else {}
        if not isinstance(entities, dict):
            entities = {}

        # --------------------------------------------------------
        # SUMMARY
        # --------------------------------------------------------
        st.success("✅ Document processing completed successfully.")
        st.subheader("📊 Document Processing Summary")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Text Regions", len(words))
        with c2:
            st.metric("OCR Confidence", f"{average_confidence:.2%}")
        with c3:
            st.metric("Characters", len(combined_text))
        with c4:
            st.metric("Document Type", str(document_type))

        # --------------------------------------------------------
        # GENERIC KEY INFORMATION — MAIN RESULT
        # --------------------------------------------------------
        st.subheader("📌 Extracted Information")
        st.caption("Relevant information detected from the uploaded document. Fields are generated dynamically.")

        label_values = entities.get("label_values", {})
        if not isinstance(label_values, dict):
            label_values = {}

        # Avoid displaying empty values.
        visible_fields = []
        for key, value in label_values.items():
            display_value = clean_display_value(value)
            if display_value:
                visible_fields.append((pretty_label(key), display_value))

        if visible_fields:
            field_columns = st.columns(2)
            for index, (key, value) in enumerate(visible_fields):
                with field_columns[index % 2]:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="field-label">{html.escape(key)}</div>
                            <div class="field-value">{html.escape(value)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No predefined key-value fields were detected. Other extracted entities are shown below.")

        # --------------------------------------------------------
        # OTHER GENERIC ENTITIES
        # --------------------------------------------------------
        st.subheader("🔎 Additional Extracted Information")

        generic_entity_keys = [
            "dates",
            "emails",
            "phone_numbers",
            "urls",
            "reference_numbers",
            "amounts",
            "numbered_entries",
        ]

        entity_sections_shown = 0
        for key in generic_entity_keys:
            values = entities.get(key, [])
            if not values:
                continue
            entity_sections_shown += 1
            st.markdown(f"**{pretty_label(key)}**")
            if isinstance(values, list) and values and isinstance(values[0], dict):
                st.dataframe(pd.DataFrame(values), width="stretch", hide_index=True)
            elif isinstance(values, list):
                st.dataframe(
                    pd.DataFrame({pretty_label(key): [clean_display_value(v) for v in values]}),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.write(clean_display_value(values))

        # Also show scalar fields returned outside label_values.
        extra_scalar_fields = []
        for key, value in flatten_scalar_fields(entities):
            if key.lower() in {"label values", "document type"}:
                continue
            display_value = clean_display_value(value)
            if display_value:
                extra_scalar_fields.append((pretty_label(key), display_value))

        if extra_scalar_fields:
            with st.expander("More detected fields"):
                extra_df = pd.DataFrame(extra_scalar_fields, columns=["Field", "Value"])
                st.dataframe(extra_df, width="stretch", hide_index=True)

        if entity_sections_shown == 0 and not extra_scalar_fields:
            st.info("No additional standard entities were detected.")

        # --------------------------------------------------------
        # TABLES — GENERIC, WITH GRADE CARD AS ONE POSSIBLE CASE
        # --------------------------------------------------------
        table_sections = collect_tables(entities)
        if table_sections:
            st.subheader("📊 Detected Tables")
            for title, table_df in table_sections:
                st.markdown(f"**{title}**")
                st.dataframe(table_df, width="stretch", hide_index=True)

        # --------------------------------------------------------
        # ACADEMIC INFORMATION — ONLY IF PRESENT
        # --------------------------------------------------------
        academic_information = entities.get("academic_information", {})
        if academic_information:
            with st.expander("🎓 Academic-specific Information"):
                if isinstance(academic_information, dict):
                    academic_rows = []
                    for key, value in academic_information.items():
                        academic_rows.append([pretty_label(key), clean_display_value(value)])
                    if academic_rows:
                        st.dataframe(
                            pd.DataFrame(academic_rows, columns=["Field", "Value"]),
                            width="stretch",
                            hide_index=True,
                        )
                else:
                    st.write(academic_information)

        # --------------------------------------------------------
        # AI / LAYOUT DETAILS
        # --------------------------------------------------------
        with st.expander("🤖 AI Document Understanding"):
            st.write(f"**Detected document type:** {document_type}")
            if layout_results:
                semantic_items = [
                    item for item in layout_results
                    if item.get("label", "O") != "O"
                ]
                st.write(
                    f"LayoutLMv3 produced **{len(semantic_items)} semantic regions** "
                    f"from **{len(layout_results)} processed OCR words**."
                )
            else:
                st.write("No LayoutLMv3 semantic regions are available.")

        if layout_results:
            with st.expander("🎯 LayoutLMv3 Semantic Predictions"):
                prediction_rows = []
                grouped = {}
                for item in layout_results:
                    label = item.get("label", "O")
                    if label == "O":
                        continue
                    confidence = float(item.get("confidence", 0.0))
                    prediction_rows.append(
                        {
                            "Word": item.get("word", ""),
                            "Label": label,
                            "Confidence": f"{confidence:.2%}",
                        }
                    )
                    grouped.setdefault(label, []).append(item.get("word", ""))

                if prediction_rows:
                    st.dataframe(
                        pd.DataFrame(prediction_rows),
                        width="stretch",
                        hide_index=True,
                    )
                    st.markdown("**Grouped semantic information**")
                    for label, values in grouped.items():
                        st.markdown(f"**{pretty_label(label)}:** {' '.join(map(str, values))}")
                else:
                    st.info("No non-O semantic predictions were produced.")

        # --------------------------------------------------------
        # OCR VISUALIZATION
        # --------------------------------------------------------
        with st.expander("🧩 OCR Layout Visualization"):
            st.caption("Green boxes represent text regions detected by PaddleOCR.")
            ocr_image = make_ocr_layout_image(image, boxes, words, scores)
            st.image(ocr_image, caption="PaddleOCR text regions", width="stretch")

        if layout_results:
            with st.expander("🧠 LayoutLMv3 Semantic Layout Visualization"):
                semantic_image = make_semantic_image(image, layout_results)
                st.image(semantic_image, caption="LayoutLMv3 semantic regions", width="stretch")

        # --------------------------------------------------------
        # OCR TEXT
        # --------------------------------------------------------
        with st.expander("📝 OCR Text"):
            if words:
                ocr_df = pd.DataFrame(
                    {
                        "Region": range(1, len(words) + 1),
                        "Text": words,
                        "Confidence": [f"{score:.2%}" for score in scores],
                    }
                )
                st.dataframe(ocr_df, width="stretch", hide_index=True)
            else:
                st.warning("No text was detected.")

            st.text_area("Combined OCR Text", combined_text, height=250)

        # --------------------------------------------------------
        # COMPLETE JSON
        # --------------------------------------------------------
        with st.expander("🔬 Complete Technical JSON"):
            complete_results = {
                "document_information": info,
                "layoutlmv3_predictions": layout_results,
            }
            st.json(complete_results)

        # --------------------------------------------------------
        # DOWNLOADS
        # --------------------------------------------------------
        st.subheader("📥 Download Results")

        st.download_button(
            "⬇️ Download OCR Text",
            combined_text,
            file_name="ocr_result.txt",
            mime="text/plain",
            width="stretch",
        )

        complete_results = {
            "document_information": info,
            "layoutlmv3_predictions": layout_results,
        }
        json_data = json.dumps(complete_results, indent=4, ensure_ascii=False, default=str)

        st.download_button(
            "⬇️ Download Complete AI Results (JSON)",
            json_data,
            file_name="document_ai_results.json",
            mime="application/json",
            width="stretch",
        )

        # --------------------------------------------------------
        # CLEAR
        # --------------------------------------------------------
        st.divider()
        if st.button("🗑️ Clear Current Document", width="stretch"):
            for key in [
                "processed_file_hash",
                "ocr_words",
                "ocr_boxes",
                "ocr_scores",
                "document_info",
                "layout_results",
            ]:
                st.session_state.pop(key, None)
            st.rerun()

else:
    st.markdown(
        """
        <div class="result-card" style="text-align:center; padding:45px;">
            <div style="font-size:55px;">📄</div>
            <h2>Upload a document to begin</h2>
            <p style="color:#64748b;">
                The system will automatically run OCR, document understanding,
                LayoutLMv3 analysis and generic information extraction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🤖 Intelligent Document Processing & Information Extraction System
        <br>
        PaddleOCR • LayoutLMv3 • Transformer-based Document Understanding
    </div>
    """,
    unsafe_allow_html=True,
)
