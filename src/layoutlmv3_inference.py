import os
import numpy as np

from PIL import Image
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification
)
import torch


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "layoutlmv3-funsd"
)


class LayoutLMv3Inference:

    def __init__(self):

        print("=" * 70)
        print("LOADING TRAINED LAYOUTLMV3 MODEL")
        print("=" * 70)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained LayoutLMv3 model not found:\n{MODEL_PATH}"
            )

        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            apply_ocr=False
        )

        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            MODEL_PATH
        )

        self.device = torch.device("cpu")

        self.model.to(self.device)

        self.model.eval()

        self.id2label = {
            int(key): value
            for key, value in self.model.config.id2label.items()
        }

        print("LayoutLMv3 model loaded successfully.")
        print(f"Model path: {MODEL_PATH}")


    def predict(
        self,
        image,
        words,
        boxes
    ):

        if image is None:
            return []

        if not words:
            return []

        if not boxes:
            return []

        if not isinstance(image, Image.Image):

            image = Image.fromarray(
                np.asarray(image)
            )

        image = image.convert("RGB")

        # -----------------------------------------------------
        # Keep words and boxes aligned
        # -----------------------------------------------------

        count = min(
            len(words),
            len(boxes)
        )

        words = words[:count]

        boxes = boxes[:count]

        if count == 0:
            return []

        # -----------------------------------------------------
        # Normalize boxes to LayoutLMv3 0-1000 coordinate
        # system.
        # -----------------------------------------------------

        width, height = image.size

        width = max(
            int(width),
            1
        )

        height = max(
            int(height),
            1
        )

        normalized_boxes = []

        for box in boxes:

            try:

                x1, y1, x2, y2 = box

                x1 = max(
                    0,
                    min(
                        int(x1),
                        width
                    )
                )

                y1 = max(
                    0,
                    min(
                        int(y1),
                        height
                    )
                )

                x2 = max(
                    0,
                    min(
                        int(x2),
                        width
                    )
                )

                y2 = max(
                    0,
                    min(
                        int(y2),
                        height
                    )
                )

            except Exception:

                x1 = 0
                y1 = 0
                x2 = 0
                y2 = 0

            normalized_boxes.append(
                [
                    int(1000 * x1 / width),
                    int(1000 * y1 / height),
                    int(1000 * x2 / width),
                    int(1000 * y2 / height)
                ]
            )

        # -----------------------------------------------------
        # LayoutLMv3 processor
        # -----------------------------------------------------

        encoding = self.processor(
            image,
            text=words,
            boxes=normalized_boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        # -----------------------------------------------------
        # Move tensors to CPU
        # -----------------------------------------------------

        model_inputs = {}

        for key, value in encoding.items():

            if hasattr(value, "to"):

                model_inputs[key] = value.to(
                    self.device
                )

        # -----------------------------------------------------
        # Prediction
        # -----------------------------------------------------

        with torch.no_grad():

            outputs = self.model(
                **model_inputs
            )

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        prediction_ids = predictions[
            0
        ].cpu().numpy()

        prediction_probabilities = probabilities[
            0
        ].cpu().numpy()

        # -----------------------------------------------------
        # Correctly obtain word IDs
        # -----------------------------------------------------

        word_ids = encoding.word_ids(
            batch_index=0
        )

        results = []

        processed_word_ids = set()

        for token_index, word_id in enumerate(
            word_ids
        ):

            if word_id is None:
                continue

            if word_id in processed_word_ids:
                continue

            processed_word_ids.add(
                word_id
            )

            if word_id >= len(words):
                continue

            label_id = int(
                prediction_ids[
                    token_index
                ]
            )

            confidence = float(
                prediction_probabilities[
                    token_index,
                    label_id
                ]
            )

            label = self.id2label.get(
                label_id,
                str(label_id)
            )

            results.append(
                {
                    "word": words[word_id],
                    "box": boxes[word_id],
                    "label": label,
                    "confidence": confidence
                }
            )

        return results