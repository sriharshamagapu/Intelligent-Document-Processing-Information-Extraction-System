import sys
from pathlib import Path
import json
import hashlib
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from PIL import Image, ImageDraw, ImageFont

# IMPORTANT:
# Load PyTorch before PaddleOCR/PaddlePaddle.
# This prevents the Windows DLL conflict.
import torch

from src.ocr import run_ocr
from src.information_extraction import extract_document_info
from src.layoutlmv3_inference import LayoutLMv3Inference


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Document Processing",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(99, 102, 241, 0.08),
            transparent 35%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(14, 165, 233, 0.08),
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #f8fafc,
            #eef2ff,
            #f8fafc
        );
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 5px;

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #7c3aed,
            #0891b2,
            #2563eb
        );

    background-size: 300% 300%;

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    animation:
        gradientMove 5s ease infinite;
}

@keyframes gradientMove {

    0% {
        background-position: 0% 50%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 50%;
    }

}

.hero-subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
    margin-bottom: 25px;
}

.ai-badge {
    display: inline-block;
    padding: 7px 16px;
    border-radius: 999px;

    background:
        linear-gradient(
            90deg,
            #dbeafe,
            #ede9fe
        );

    color: #3730a3;
    font-weight: 700;
    font-size: 13px;

    animation:
        pulseBadge 2s infinite;
}

@keyframes pulseBadge {

    0% {
        box-shadow:
            0 0 0 0
            rgba(99, 102, 241, 0.25);
    }

    70% {
        box-shadow:
            0 0 0 10px
            rgba(99, 102, 241, 0);
    }

    100% {
        box-shadow:
            0 0 0 0
            rgba(99, 102, 241, 0);
    }

}

div[data-testid="stMetric"] {
    background:
        rgba(255, 255, 255, 0.75);

    border:
        1px solid
        rgba(148, 163, 184, 0.25);

    border-radius: 16px;

    padding: 15px;

    box-shadow:
        0 8px 25px
        rgba(15, 23, 42, 0.06);

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    transform:
        translateY(-5px);

    box-shadow:
        0 15px 35px
        rgba(15, 23, 42, 0.12);
}

.stButton > button {
    border-radius: 12px;
    font-weight: 700;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

.stButton > button:hover {
    transform:
        translateY(-2px);

    box-shadow:
        0 8px 20px
        rgba(37, 99, 235, 0.20);
}

.result-card {
    padding: 18px;
    border-radius: 16px;

    background:
        rgba(255, 255, 255, 0.82);

    border:
        1px solid
        rgba(148, 163, 184, 0.22);

    box-shadow:
        0 8px 25px
        rgba(15, 23, 42, 0.05);

    margin-bottom: 12px;

    animation:
        fadeUp 0.6s ease;
}

@keyframes fadeUp {

    from {
        opacity: 0;
        transform:
            translateY(15px);
    }

    to {
        opacity: 1;
        transform:
            translateY(0);
    }

}

.field-card {
    padding: 16px;
    border-radius: 14px;

    background:
        rgba(255, 255, 255, 0.90);

    border:
        1px solid
        rgba(148, 163, 184, 0.22);

    margin-bottom: 10px;

    box-shadow:
        0 5px 18px
        rgba(15, 23, 42, 0.04);
}

.field-name {
    color: #64748b;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 4px;
}

.field-value {
    color: #0f172a;
    font-size: 16px;
    font-weight: 600;
}

.important-box {
    padding: 20px;
    border-radius: 18px;

    background:
        linear-gradient(
            135deg,
            rgba(239, 246, 255, 0.95),
            rgba(245, 243, 255, 0.95)
        );

    border:
        1px solid
        rgba(99, 102, 241, 0.20);

    margin-bottom: 20px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    font-size: 13px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD LAYOUTLMV3
# ============================================================

@st.cache_resource
def load_layoutlmv3():
    return LayoutLMv3Inference()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero-title">
    🤖 Intelligent Document AI
</div>

<div class="hero-subtitle">
    OCR • Layout Understanding • Information Extraction •
    Transformer-based Document Intelligence
</div>

<div style="text-align:center;">
    <span class="ai-badge">
        ⚡ PaddleOCR + LayoutLMv3 + AI Extraction
    </span>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ AI Processing Pipeline")

    st.markdown(
        """
**1️⃣ Upload Document**

↓

**2️⃣ PaddleOCR**

↓

**3️⃣ Text + Coordinates**

↓

**4️⃣ LayoutLMv3**

↓

**5️⃣ Document Understanding**

↓

**6️⃣ Information Extraction**

↓

**7️⃣ Clean Results**
"""
    )

    st.divider()

    st.success(
        "🤖 Trained LayoutLMv3 model is connected."
    )

    st.info(
        "The system is designed for generic "
        "document processing."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "📤 Upload a document image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ],
    help="Upload a clear document image."
)


# ============================================================
# DOCUMENT UPLOADED
# ============================================================

if uploaded:

    # ========================================================
    # FILE HASH
    # ========================================================

    file_bytes = uploaded.getvalue()

    file_hash = hashlib.md5(
        file_bytes
    ).hexdigest()


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = Image.open(
        uploaded
    ).convert("RGB")


    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    st.subheader(
        "🖼️ Uploaded Document"
    )

    st.image(
        image,
        caption=uploaded.name,
        width="stretch"
    )


    # ========================================================
    # CHECK CURRENT FILE
    # ========================================================

    processed_file_hash = st.session_state.get(
        "processed_file_hash"
    )

    current_file_processed = (
        processed_file_hash == file_hash
    )


    # ========================================================
    # PROCESS BUTTON
    # ========================================================

    run_button = st.button(
        "🚀 Run Document Intelligence",
        type="primary",
        width="stretch"
    )


    # ========================================================
    # PROCESS DOCUMENT
    # ========================================================

    if run_button:

        if current_file_processed:

            st.info(
                "⚡ This document is already processed. "
                "Using stored results."
            )

        else:

            # ------------------------------------------------
            # CLEAR OLD RESULTS
            # ------------------------------------------------

            for key in [
                "processed_file_hash",
                "ocr_words",
                "ocr_boxes",
                "ocr_scores",
                "document_info",
                "layout_results"
            ]:

                st.session_state.pop(
                    key,
                    None
                )


            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            progress = st.progress(0)

            status = st.empty()


            # ------------------------------------------------
            # STEP 1 - OCR
            # ------------------------------------------------

            status.markdown(
                "🔎 **Step 1/3 — Running PaddleOCR...**"
            )

            with st.spinner(
                "Analyzing document text..."
            ):

                try:

                    words, boxes, scores = run_ocr(
                        image
                    )

                except Exception as error:

                    progress.empty()
                    status.empty()

                    st.error(
                        f"❌ OCR processing failed: {error}"
                    )

                    st.stop()


            progress.progress(33)


            # ------------------------------------------------
            # STEP 2 - INFORMATION EXTRACTION
            # ------------------------------------------------

            status.markdown(
                "🧠 **Step 2/3 — Extracting document information...**"
            )

            with st.spinner(
                "Understanding document structure..."
            ):

                try:

                    info = extract_document_info(
                        words,
                        boxes
                    )

                except Exception as error:

                    progress.empty()
                    status.empty()

                    st.error(
                        f"❌ Information extraction failed: {error}"
                    )

                    st.stop()


            progress.progress(66)


            # ------------------------------------------------
            # STEP 3 - LAYOUTLMV3
            # ------------------------------------------------

            status.markdown(
                "🤖 **Step 3/3 — Running LayoutLMv3...**"
            )

            with st.spinner(
                "AI model is analyzing document layout..."
            ):

                try:

                    layout_model = load_layoutlmv3()

                    layout_results = layout_model.predict(
                        image,
                        words,
                        boxes
                    )

                except Exception as error:

                    progress.empty()
                    status.empty()

                    st.error(
                        f"❌ LayoutLMv3 processing failed: {error}"
                    )

                    st.stop()


            progress.progress(100)

            time.sleep(0.3)

            progress.empty()
            status.empty()


            # ------------------------------------------------
            # SAVE RESULTS
            # ------------------------------------------------

            st.session_state[
                "processed_file_hash"
            ] = file_hash

            st.session_state[
                "ocr_words"
            ] = words

            st.session_state[
                "ocr_boxes"
            ] = boxes

            st.session_state[
                "ocr_scores"
            ] = scores

            st.session_state[
                "document_info"
            ] = info

            st.session_state[
                "layout_results"
            ] = layout_results


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            st.success(
                "✅ Document intelligence completed successfully!"
            )

            # IMPORTANT:
            # No st.balloons() here.


    # ========================================================
    # LOAD STORED RESULTS
    # ========================================================

    processed_file_hash = st.session_state.get(
        "processed_file_hash"
    )

    current_file_processed = (
        processed_file_hash == file_hash
    )


    if current_file_processed:

        words = st.session_state.get(
            "ocr_words",
            []
        )

        boxes = st.session_state.get(
            "ocr_boxes",
            []
        )

        scores = st.session_state.get(
            "ocr_scores",
            []
        )

        info = st.session_state.get(
            "document_info",
            {}
        )

        layout_results = st.session_state.get(
            "layout_results",
            []
        )

    else:

        words = None
        boxes = None
        scores = None
        info = None
        layout_results = None


    # ========================================================
    # RESULTS
    # ========================================================

    if (
        words is not None
        and boxes is not None
        and scores is not None
        and info is not None
    ):

        # ====================================================
        # BASIC DATA
        # ====================================================

        if scores:

            average_confidence = (
                sum(scores)
                /
                len(scores)
            )

        else:

            average_confidence = 0.0


        combined_text = "\n".join(
            str(word)
            for word in words
        )


        entities = info.get(
            "entities",
            {}
        )


        document_type = info.get(
            "document_type",
            "Unknown Document"
        )


        label_values = entities.get(
            "label_values",
            {}
        )


        # ====================================================
        # SUCCESS
        # ====================================================

        st.success(
            "✅ Document processing completed successfully."
        )


        # ====================================================
        # MAIN EXTRACTION HEADER
        # ====================================================

        st.markdown(
            """
<div class="important-box">

<h2 style="margin-top:0;">
📌 Extracted Information
</h2>

<p style="color:#64748b;">
Important information detected from the document.
Raw OCR and technical AI output are available below
in expandable sections.
</p>

</div>
""",
            unsafe_allow_html=True
        )


        # ====================================================
        # DOCUMENT TYPE
        # ====================================================

        st.markdown(
            f"""
<div class="field-card">

<div class="field-name">
DOCUMENT TYPE
</div>

<div class="field-value">
{document_type}
</div>

</div>
""",
            unsafe_allow_html=True
        )


        # ====================================================
        # IMPORTANT FIELD ORDER
        # ====================================================

        preferred_order = [
            "name",
            "registration_number",
            "application_number",
            "reference_number",
            "transaction_id",
            "receipt_number",
            "date_of_birth",
            "date",
            "campus",
            "institute",
            "academic_year",
            "percentage",
            "disability_type",
            "issuing_authority",
            "fee_description",
            "payment_mode",
            "amount"
        ]


        displayed_fields = set()


        # ====================================================
        # IMPORTANT EXTRACTED FIELDS
        # ====================================================

        for field_name in preferred_order:

            if field_name not in label_values:
                continue

            value = label_values.get(
                field_name
            )

            if value is None:
                continue

            value = str(value).strip()

            if not value:
                continue

            displayed_fields.add(
                field_name
            )

            display_name = (
                field_name
                .replace(
                    "_",
                    " "
                )
                .title()
            )


            st.markdown(
                f"""
<div class="field-card">

<div class="field-name">
{display_name}
</div>

<div class="field-value">
{value}
</div>

</div>
""",
                unsafe_allow_html=True
            )


        # ====================================================
        # OTHER STRUCTURED FIELDS
        # ====================================================

        for field_name, value in label_values.items():

            if field_name in displayed_fields:
                continue

            if value is None:
                continue

            value = str(value).strip()

            if not value:
                continue

            display_name = (
                field_name
                .replace(
                    "_",
                    " "
                )
                .title()
            )


            st.markdown(
                f"""
<div class="field-card">

<div class="field-name">
{display_name}
</div>

<div class="field-value">
{value}
</div>

</div>
""",
                unsafe_allow_html=True
            )


        # ====================================================
        # NO STRUCTURED FIELDS
        # ====================================================

        if not label_values:

            st.info(
                "No structured key-value fields were confidently detected. "
                "You can inspect the AI and OCR details below."
            )


        # ====================================================
        # PROCESSING SUMMARY
        # ====================================================

        st.subheader(
            "📊 Processing Summary"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Text Regions",
                len(words)
            )

        with col2:

            st.metric(
                "OCR Confidence",
                f"{average_confidence:.2%}"
            )

        with col3:

            st.metric(
                "Characters",
                len(combined_text)
            )

        with col4:

            if layout_results:

                semantic_count = sum(
                    1
                    for item in layout_results
                    if item.get("label") != "O"
                )

            else:

                semantic_count = 0

            st.metric(
                "AI Regions",
                semantic_count
            )


        # ====================================================
        # AI UNDERSTANDING
        # ====================================================

        with st.expander(
            "🤖 View AI Document Understanding",
            expanded=False
        ):

            if layout_results:

                non_o_predictions = [
                    item
                    for item in layout_results
                    if item.get("label") != "O"
                ]


                if non_o_predictions:

                    st.info(
                        f"LayoutLMv3 detected "
                        f"**{len(non_o_predictions)} semantic regions** "
                        f"from {len(layout_results)} OCR words."
                    )


                    grouped_predictions = {}


                    for item in non_o_predictions:

                        label = item.get(
                            "label",
                            "O"
                        )

                        grouped_predictions.setdefault(
                            label,
                            []
                        )

                        grouped_predictions[
                            label
                        ].append(
                            item.get(
                                "word",
                                ""
                            )
                        )


                    st.markdown(
                        "### Semantic Information"
                    )


                    for label, values in grouped_predictions.items():

                        clean_label = (
                            label
                            .replace(
                                "B-",
                                ""
                            )
                            .replace(
                                "I-",
                                ""
                            )
                            .title()
                        )

                        combined_values = " ".join(
                            values
                        )


                        st.markdown(
                            f"""
<div class="result-card">

<strong>
{clean_label}
</strong>

<br><br>

{combined_values}

</div>
""",
                            unsafe_allow_html=True
                        )


                    prediction_rows = []


                    for item in non_o_predictions:

                        confidence_value = float(
                            item.get(
                                "confidence",
                                0.0
                            )
                        )

                        prediction_rows.append(
                            {
                                "Word": item.get(
                                    "word",
                                    ""
                                ),

                                "Label": item.get(
                                    "label",
                                    ""
                                ),

                                "Confidence": (
                                    f"{confidence_value:.2%}"
                                )
                            }
                        )


                    if prediction_rows:

                        st.markdown(
                            "### Detailed Predictions"
                        )

                        st.dataframe(
                            pd.DataFrame(
                                prediction_rows
                            ),
                            width="stretch",
                            hide_index=True
                        )

                else:

                    st.info(
                        "No non-O semantic predictions were produced."
                    )

            else:

                st.info(
                    "LayoutLMv3 results are not available."
                )


        # ====================================================
        # OCR LAYOUT
        # ====================================================

        with st.expander(
            "🖼️ View OCR Layout Visualization",
            expanded=False
        ):

            st.caption(
                "Bounding boxes show text regions detected by PaddleOCR."
            )

            layout_image = image.copy()

            draw = ImageDraw.Draw(
                layout_image
            )


            try:

                font = ImageFont.truetype(
                    "arial.ttf",
                    14
                )

            except Exception:

                font = ImageFont.load_default()


            for index, (
                box,
                text,
                score
            ) in enumerate(
                zip(
                    boxes,
                    words,
                    scores
                ),
                start=1
            ):

                if not box or len(box) < 4:
                    continue


                try:

                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])

                except (
                    TypeError,
                    ValueError
                ):

                    continue


                draw.rectangle(
                    [
                        x1,
                        y1,
                        x2,
                        y2
                    ],
                    outline="green",
                    width=3
                )


                label = str(index)

                text_bbox = draw.textbbox(
                    (
                        x1,
                        y1
                    ),
                    label,
                    font=font
                )


                label_width = (
                    text_bbox[2]
                    -
                    text_bbox[0]
                )

                label_height = (
                    text_bbox[3]
                    -
                    text_bbox[1]
                )


                label_y = max(
                    0,
                    y1
                    -
                    label_height
                    -
                    4
                )


                draw.rectangle(
                    [
                        x1,
                        label_y,
                        x1
                        +
                        label_width
                        +
                        6,
                        y1
                    ],
                    fill="green"
                )


                draw.text(
                    (
                        x1 + 3,
                        max(
                            0,
                            y1
                            -
                            label_height
                            -
                            2
                        )
                    ),
                    label,
                    fill="white",
                    font=font
                )


            st.image(
                layout_image,
                caption="PaddleOCR detected text regions",
                width="stretch"
            )


        # ====================================================
        # LAYOUTLM SEMANTIC LAYOUT
        # ====================================================

        with st.expander(
            "🧠 View LayoutLMv3 Semantic Layout",
            expanded=False
        ):

            if layout_results:

                semantic_image = image.copy()

                semantic_draw = ImageDraw.Draw(
                    semantic_image
                )


                for item in layout_results:

                    label = item.get(
                        "label",
                        "O"
                    )

                    box = item.get(
                        "box",
                        []
                    )


                    if label == "O":
                        continue


                    if not box or len(box) < 4:
                        continue


                    try:

                        x1 = int(box[0])
                        y1 = int(box[1])
                        x2 = int(box[2])
                        y2 = int(box[3])

                    except Exception:

                        continue


                    semantic_draw.rectangle(
                        [
                            x1,
                            y1,
                            x2,
                            y2
                        ],
                        outline="red",
                        width=3
                    )


                    semantic_draw.text(
                        (
                            x1,
                            max(
                                0,
                                y1 - 15
                            )
                        ),
                        label,
                        fill="red"
                    )


                st.image(
                    semantic_image,
                    caption="LayoutLMv3 semantic predictions",
                    width="stretch"
                )

            else:

                st.info(
                    "No LayoutLMv3 semantic layout is available."
                )


        # ====================================================
        # ADDITIONAL EXTRACTED DATA
        # ====================================================

        with st.expander(
            "📚 View Additional Extracted Data",
            expanded=False
        ):

            academic_information = entities.get(
                "academic_information",
                {}
            )


            if academic_information:

                st.markdown(
                    "### 🎓 Academic Information"
                )

                for key, value in academic_information.items():

                    display_key = (
                        key
                        .replace(
                            "_",
                            " "
                        )
                        .title()
                    )


                    if isinstance(
                        value,
                        list
                    ):

                        if value:

                            st.write(
                                f"**{display_key}:**"
                            )

                            for item in value:

                                st.write(
                                    f"- {item}"
                                )

                    else:

                        if value:

                            st.write(
                                f"**{display_key}:** {value}"
                            )


            table_rows = entities.get(
                "table_rows",
                []
            )


            if table_rows:

                st.markdown(
                    "### 📊 Detected Table-like Data"
                )

                st.dataframe(
                    pd.DataFrame(
                        table_rows
                    ),
                    width="stretch",
                    hide_index=True
                )


            additional_sections = [
                (
                    "📅 Dates",
                    "dates",
                    "Date"
                ),
                (
                    "📧 Email Addresses",
                    "emails",
                    "Email"
                ),
                (
                    "📱 Phone Numbers",
                    "phone_numbers",
                    "Phone Number"
                ),
                (
                    "🌐 URLs",
                    "urls",
                    "URL"
                ),
                (
                    "🔢 Reference Numbers",
                    "reference_numbers",
                    "Reference Number"
                ),
                (
                    "💰 Monetary Amounts",
                    "amounts",
                    "Amount"
                )
            ]


            for title, key, column_name in additional_sections:

                values = entities.get(
                    key,
                    []
                )

                if not values:
                    continue


                st.markdown(
                    f"### {title}"
                )


                st.dataframe(
                    pd.DataFrame(
                        {
                            column_name: values
                        }
                    ),
                    width="stretch",
                    hide_index=True
                )


            numbered_entries = entities.get(
                "numbered_entries",
                []
            )


            if numbered_entries:

                st.markdown(
                    "### 📋 Numbered Entries"
                )

                st.dataframe(
                    pd.DataFrame(
                        numbered_entries
                    ),
                    width="stretch",
                    hide_index=True
                )


        # ====================================================
        # RAW OCR TEXT
        # ====================================================

        with st.expander(
            "🔎 View OCR Text",
            expanded=False
        ):

            if words:

                ocr_df = pd.DataFrame(
                    {
                        "Region": range(
                            1,
                            len(words) + 1
                        ),

                        "Text": words,

                        "Confidence": [
                            f"{score:.2%}"
                            for score in scores
                        ]
                    }
                )


                st.dataframe(
                    ocr_df,
                    width="stretch",
                    hide_index=True
                )


                st.text_area(
                    "Combined OCR Text",
                    combined_text,
                    height=250
                )

            else:

                st.warning(
                    "No text was detected."
                )


        # ====================================================
        # COMPLETE TECHNICAL JSON
        # ====================================================

        with st.expander(
            "🧪 View Complete Technical JSON",
            expanded=False
        ):

            complete_results = {
                "document_information": info,
                "layoutlmv3_predictions": layout_results
            }


            st.json(
                complete_results
            )


        # ====================================================
        # DOWNLOADS
        # ====================================================

        st.subheader(
            "📥 Download Results"
        )

        download_col1, download_col2 = st.columns(2)


        with download_col1:

            st.download_button(
                "⬇️ Download OCR Text",
                combined_text,
                file_name="ocr_result.txt",
                mime="text/plain",
                width="stretch"
            )


        with download_col2:

            complete_results = {
                "document_information": info,
                "layoutlmv3_predictions": layout_results
            }


            json_data = json.dumps(
                complete_results,
                indent=4,
                ensure_ascii=False
            )


            st.download_button(
                "⬇️ Download Complete AI Results",
                json_data,
                file_name="document_ai_results.json",
                mime="application/json",
                width="stretch"
            )


        # ====================================================
        # CLEAR DOCUMENT
        # ====================================================

        st.divider()


        if st.button(
            "🗑️ Clear Current Document",
            width="stretch"
        ):

            for key in [
                "processed_file_hash",
                "ocr_words",
                "ocr_boxes",
                "ocr_scores",
                "document_info",
                "layout_results"
            ]:

                st.session_state.pop(
                    key,
                    None
                )


            st.rerun()


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    empty_col1, empty_col2, empty_col3 = st.columns(
        [1, 2, 1]
    )

    with empty_col2:

        st.markdown(
            "<div style='text-align:center;'>"
            "<div style='font-size:55px;'>📄</div>"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<h2 style='text-align:center;'>"
            "Upload a document to begin"
            "</h2>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p style='text-align:center; color:#64748b;'>"
            "Your document will pass through OCR, "
            "LayoutLMv3 and information extraction automatically."
            "</p>",
            unsafe_allow_html=True
        )

        st.info(
            "Supported formats: PNG, JPG and JPEG"
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
    unsafe_allow_html=True
)