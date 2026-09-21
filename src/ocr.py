from PIL import Image
import numpy as np
from paddleocr import PaddleOCR


# =========================================================
# OCR FUNCTION
# =========================================================

def run_ocr(image: Image.Image):

    # -----------------------------------------------------
    # Create a fresh OCR engine for this processing run.
    # This avoids PaddlePaddle tensor-memory problems
    # caused by reusing the engine across Streamlit reruns.
    # -----------------------------------------------------

    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=False,
        enable_mkldnn=False,
        cpu_threads=10,
        ocr_version="PP-OCRv4",
        rec_algorithm="SVTR_LCNet"
    )

    # -----------------------------------------------------
    # Convert PIL image to NumPy
    # -----------------------------------------------------

    image_array = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8
    )

    # Ensure contiguous memory
    image_array = np.ascontiguousarray(
        image_array
    )

    # -----------------------------------------------------
    # Run OCR
    # -----------------------------------------------------

    result = ocr.ocr(
        image_array,
        cls=True
    )

    words = []
    boxes = []
    scores = []

    if not result:
        return words, boxes, scores

    # =====================================================
    # PROCESS OCR RESULT
    # =====================================================

    for page in result:

        if page is None:
            continue

        for item in page:

            if not item:
                continue

            if len(item) < 2:
                continue

            box = item[0]
            text_info = item[1]

            if not text_info:
                continue

            if len(text_info) < 2:
                continue

            text = text_info[0]
            score = float(text_info[1])

            if not text:
                continue

            # -------------------------------------------------
            # PaddleOCR returns a 4-point polygon.
            # Convert it to:
            # [x1, y1, x2, y2]
            # -------------------------------------------------

            xs = [
                point[0]
                for point in box
            ]

            ys = [
                point[1]
                for point in box
            ]

            x1 = min(xs)
            y1 = min(ys)
            x2 = max(xs)
            y2 = max(ys)

            words.append(
                text
            )

            boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            )

            scores.append(
                score
            )

    return words, boxes, scores