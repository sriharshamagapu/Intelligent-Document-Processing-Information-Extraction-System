import os
import numpy as np

from datasets import load_from_disk
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
import evaluate
from PIL import Image


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "funsd"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "layoutlmv3-funsd"
)

MODEL_NAME = "microsoft/layoutlmv3-base"


print("=" * 70)
print("LOADING FUNSD DATASET")
print("=" * 70)

dataset = load_from_disk(DATASET_PATH)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

print(f"Training documents : {len(train_dataset)}")
print(f"Test documents     : {len(test_dataset)}")
print(f"Features           : {train_dataset.features}")


ner_feature = train_dataset.features["ner_tags"]

label_names = ner_feature.feature.names

label2id = {
    label: index
    for index, label in enumerate(label_names)
}

id2label = {
    index: label
    for index, label in enumerate(label_names)
}

num_labels = len(label_names)


print()
print("=" * 70)
print("LABEL INFORMATION")
print("=" * 70)

print("Labels:")

for index, label in enumerate(label_names):
    print(f"{index}: {label}")

print(f"Number of labels: {num_labels}")


print()
print("=" * 70)
print("LOADING LAYOUTLMV3 PROCESSOR")
print("=" * 70)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    apply_ocr=False
)


def preprocess_example(example):

    image = example["image"]

    words = example["words"]

    boxes = example["bboxes"]

    word_labels = example["ner_tags"]

    # ---------------------------------------------------------
    # Make sure image is a PIL RGB image.
    # ---------------------------------------------------------

    if not isinstance(image, Image.Image):
        image = Image.fromarray(
            np.asarray(image)
        )

    image = image.convert("RGB")

    # ---------------------------------------------------------
    # Make sure bounding boxes are integers.
    # ---------------------------------------------------------

    boxes = [
        [
            int(coordinate)
            for coordinate in box
        ]
        for box in boxes
    ]

    # ---------------------------------------------------------
    # LayoutLMv3 processor
    # ---------------------------------------------------------

    encoding = processor(
        image,
        text=words,
        boxes=boxes,
        word_labels=word_labels,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # The processor can return:
    #
    # [1, 3, 224, 224]
    #
    # for a single image.
    #
    # Trainer later creates a batch, which would become:
    #
    # [2, 1, 3, 224, 224]
    #
    # LayoutLMv3 expects:
    #
    # [2, 3, 224, 224]
    #
    # Therefore remove the unnecessary first dimension.
    # ---------------------------------------------------------

    if "pixel_values" in encoding:

        pixel_values = encoding["pixel_values"]

        if hasattr(pixel_values, "ndim"):

            if pixel_values.ndim == 4 and pixel_values.shape[0] == 1:

                encoding["pixel_values"] = pixel_values[0]

        elif isinstance(pixel_values, list):

            array = np.asarray(
                pixel_values
            )

            if array.ndim == 4 and array.shape[0] == 1:

                encoding["pixel_values"] = array[0]

    return encoding


print()
print("=" * 70)
print("PREPROCESSING TRAINING DATA")
print("=" * 70)

processed_train = train_dataset.map(
    preprocess_example,
    remove_columns=train_dataset.column_names,
    desc="Processing training documents"
)


print()
print("=" * 70)
print("PREPROCESSING TEST DATA")
print("=" * 70)

processed_test = test_dataset.map(
    preprocess_example,
    remove_columns=test_dataset.column_names,
    desc="Processing test documents"
)


print()
print("=" * 70)
print("DATASET PREPROCESSING COMPLETED")
print("=" * 70)

print(f"Processed training documents : {len(processed_train)}")
print(f"Processed test documents     : {len(processed_test)}")


# -------------------------------------------------------------
# Check image tensor shape before training
# -------------------------------------------------------------

sample_pixel_values = processed_train[0]["pixel_values"]

print()
print("Sample pixel_values shape:")

try:
    print(np.asarray(sample_pixel_values).shape)
except Exception:
    print("Unable to determine shape")


seqeval = evaluate.load("seqeval")


def compute_metrics(prediction):

    predictions, labels = prediction

    predictions = np.argmax(
        predictions,
        axis=2
    )

    true_predictions = []

    true_labels = []

    for prediction_row, label_row in zip(
        predictions,
        labels
    ):

        current_predictions = []

        current_labels = []

        for prediction_id, label_id in zip(
            prediction_row,
            label_row
        ):

            if label_id == -100:
                continue

            current_predictions.append(
                label_names[prediction_id]
            )

            current_labels.append(
                label_names[label_id]
            )

        true_predictions.append(
            current_predictions
        )

        true_labels.append(
            current_labels
        )

    results = seqeval.compute(
        predictions=true_predictions,
        references=true_labels
    )

    return {
        "precision": results.get(
            "overall_precision",
            0.0
        ),
        "recall": results.get(
            "overall_recall",
            0.0
        ),
        "f1": results.get(
            "overall_f1",
            0.0
        ),
        "accuracy": results.get(
            "overall_accuracy",
            0.0
        ),
    }


print()
print("=" * 70)
print("LOADING LAYOUTLMV3 MODEL")
print("=" * 70)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label
)


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=1,

    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    gradient_accumulation_steps=2,

    learning_rate=5e-5,

    weight_decay=0.01,

    warmup_ratio=0.1,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=10,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    save_total_limit=1,

    fp16=False,

    dataloader_num_workers=0,

    report_to="none",

    remove_unused_columns=False,
)


trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=processed_train,

    eval_dataset=processed_test,

    processing_class=processor,

    data_collator=default_data_collator,

    compute_metrics=compute_metrics,
)


print()
print("=" * 70)
print("STARTING LAYOUTLMV3 TRAINING")
print("=" * 70)

print()
print("Training on CPU.")
print("FUNSD training documents:", len(processed_train))
print("FUNSD test documents:", len(processed_test))
print()
print("This may take some time.")
print()


trainer.train()


print()
print("=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

metrics = trainer.evaluate()


for key, value in metrics.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


print()
print("=" * 70)
print("SAVING TRAINED MODEL")
print("=" * 70)

trainer.save_model(
    OUTPUT_DIR
)

processor.save_pretrained(
    OUTPUT_DIR
)


print()
print("Model saved to:")

print(OUTPUT_DIR)


print()
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)