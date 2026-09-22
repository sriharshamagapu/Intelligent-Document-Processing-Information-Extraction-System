Intelligent Document Processing & Information Extraction System

An AI-powered document understanding system that converts document images into structured information using PaddleOCR, LayoutLMv3, rule-based extraction, and a Streamlit dashboard.

The system is designed to handle different document types such as academic documents, receipts/invoices, certificates, and previously unseen/general documents.

🚀 Key Features

📤 Upload document images through a Streamlit interface

👁️ OCR using PaddleOCR

📍 Extract OCR text with bounding-box coordinates

🧠 Automatic document-type detection

🤖 LayoutLMv3 token classification for document understanding

🧾 Structured information extraction

🎓 Academic/grade-card extraction

🧾 Receipt/invoice extraction

📜 Certificate extraction

📊 Table extraction

🔎 Generic entity extraction

🖼️ OCR layout visualization

🎯 LayoutLMv3 semantic predictions

🧪 Technical JSON output

📥 Downloadable OCR and JSON results

🌐 Interactive futuristic Streamlit dashboard

🏗️ System Architecture

                 ┌──────────────────────┐
                 │   Document Image     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      PaddleOCR       │
                 │  Text + Coordinates  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Document Detection   │
                 │ Academic / Receipt / │
                 │ Certificate / Other  │
                 └──────────┬───────────┘
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
   ┌───────────────────┐        ┌────────────────────┐
   │ Generic Extraction│        │    LayoutLMv3      │
   │ Fields / Entities │        │ Semantic Prediction│
   └─────────┬─────────┘        └──────────┬─────────┘
             │                             │
             └──────────────┬──────────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Structured Results   │
                 │ JSON / Tables / UI   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Streamlit Dashboard  │
                 └──────────────────────┘

🧰 Technology Stack

Component

Technology

Programming Language

Python 3.11

Web Interface

Streamlit

OCR

PaddleOCR 2.7.3

OCR Engine

PaddlePaddle 2.6.2

Document AI

LayoutLMv3

Deep Learning

PyTorch 2.6.0 CPU

Transformers

Hugging Face Transformers

Dataset

FUNSD

Data Processing

NumPy, Pandas

Output

JSON, tables, downloadable results

📁 Project Structure

Intelligent Document Processing & Information Extraction System/
│
├── app/
│   └── streamlit_app.py
│
├── configs/
│
├── data/
│   └── funsd/
│
├── models/
│   └── layoutlmv3-funsd/
│
├── src/
│   ├── ocr.py
│   ├── information_extraction.py
│   ├── postprocess.py
│   ├── train_layoutlmv3.py
│   └── layoutlmv3_inference.py
│
├── tests/
│
├── .gitignore
├── README.md
└── requirements.txt

🔄 Processing Pipeline

1. Document Upload

The user uploads a PNG, JPG, or JPEG document through the Streamlit dashboard.

2. OCR

PaddleOCR detects:

Text

Text regions

Bounding boxes

Recognition confidence

3. Document Type Detection

The extracted text is analyzed to identify document categories such as:

Academic Document

Receipt / Invoice

Certificate

Identity Document

Application/Form

Unknown Document

4. Information Extraction

The system extracts relevant information using document-specific and generic extraction rules.

Examples include:

Names

Registration numbers

Course information

Semester

SGPA / CGPA

Dates

Phone numbers

URLs

Reference numbers

Amounts

Receipt information

Certificate information

5. LayoutLMv3

The trained LayoutLMv3 model uses:

Document image

OCR words

Bounding boxes

to perform token-level semantic classification.

6. Results

The dashboard presents:

Document type

Extracted fields

Additional entities

Tables

OCR text

OCR visualization

LayoutLMv3 predictions

Technical JSON

🤖 LayoutLMv3 Model

The project uses LayoutLMv3 for document-level understanding and token classification.

The model was trained using the FUNSD dataset.

FUNSD Dataset

Split

Documents

Training

149

Testing

50

Labels

The model uses the following FUNSD labels:

O
B-HEADER
I-HEADER
B-QUESTION
I-QUESTION
B-ANSWER
I-ANSWER

Training Result

The completed training run produced:

Evaluation Loss : 0.9683
Precision       : 0.4533
Recall          : 0.4724
F1 Score        : 0.4627
Accuracy        : 0.6699

These metrics are from the project's FUNSD training run and are intended as baseline results for this prototype.

📦 Installation

1. Clone the repository

git clone https://github.com/sriharshamagapu/Intelligent-Document-Processing-Information-Extraction-System.git
cd Intelligent-Document-Processing-Information-Extraction-System

2. Create a virtual environment

Windows

py -3.11 -m venv .venv
.venv\Scriptsctivate

Linux / macOS

python3.11 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

▶️ Run the Application

From the project root:

streamlit run app/streamlit_app.py

The application will open in the browser at the local Streamlit address shown in the terminal.

🧪 Tested Document Types

The application has been tested with multiple document categories.

🎓 Academic / Grade Card

The system can detect and extract information such as:

Student name

Registration number

Degree

Branch

Semester

SGPA

CGPA

Institution

Printed date

Course table

🧾 Receipt / Invoice

The system can detect receipt documents and extract information such as:

Date

Cashier / Clerk

Phone number

Amounts

Receipt-related information

📜 Certificate

The system can detect certificate documents and extract available information such as:

Date

Name

Institution

Certificate number

🖥️ General / Unknown Documents

Documents that do not match the predefined document categories are classified as:

Unknown Document

The system still performs OCR, generic entity extraction, table detection, visualization, and LayoutLMv3 processing.

📊 Example OCR Performance

Example test results from the application:

Document

Text Regions

OCR Confidence

Characters

Grade Card

51

~99%

~924

Receipt

17

94.57%

210

Certificate

52

96.01%

1547

General Screenshot

157

95.58%

1816

OCR confidence varies depending on document quality, resolution, layout, and text clarity.

🖥️ Streamlit Dashboard

The dashboard provides the following sections:

AI Processing Pipeline
        ↓
Document Upload
        ↓
PaddleOCR Vision
        ↓
Text + Coordinates
        ↓
Document Detection
        ↓
LayoutLMv3 AI
        ↓
Generic Extraction
        ↓
Results & Download

The interface also provides:

OCR visualization

LayoutLMv3 semantic visualization

OCR text

Complete technical JSON

Downloadable results

📄 Output Structure

A typical processing result contains:

{
  "document_information": {
    "Document Type": "Academic Document",
    "Fields": {}
  },
  "document_type": "Academic Document",
  "text": "...",
  "entities": {
    "dates": [],
    "emails": [],
    "phones": [],
    "amounts": [],
    "label_values": {},
    "table_data": []
  }
}

The actual fields depend on the uploaded document.

🧠 Important Model File Note

The trained LayoutLMv3 model file:

models/layoutlmv3-funsd/model.safetensors

is approximately 478 MB.

It is intentionally excluded from the normal Git repository because GitHub's standard repository file limit does not allow files of this size.

The source code required to train and run the model is included in the repository.

The training script is:

src/train_layoutlmv3.py

The inference implementation is:

src/layoutlmv3_inference.py

For a production deployment, the trained model should be stored using a suitable model/artifact repository or Git LFS.

🧪 Training the Model

The training pipeline uses the FUNSD dataset.

The main training file is:

src/train_layoutlmv3.py

The model is saved under:

models/layoutlmv3-funsd/

The inference module loads the trained model from that location.

🔬 Project Modules

src/ocr.py

Handles:

PaddleOCR initialization

Text recognition

Bounding-box extraction

OCR confidence

src/information_extraction.py

Handles:

Document classification

Field extraction

Academic extraction

Receipt extraction

Certificate extraction

Generic entity extraction

Table extraction

Numbered-entry extraction

src/train_layoutlmv3.py

Handles:

FUNSD dataset preparation

LayoutLMv3 preprocessing

Model training

Evaluation

src/layoutlmv3_inference.py

Handles:

Loading the trained LayoutLMv3 model

OCR word and bounding-box processing

Token classification

Semantic predictions

app/streamlit_app.py

Provides the complete interactive web dashboard.

🎯 Project Objectives

The main objectives of this project are:

Automate document text extraction.

Preserve document layout information using bounding boxes.

Identify different document types.

Extract structured information from unstructured documents.

Apply transformer-based document understanding.

Provide an interactive user interface.

Generate machine-readable JSON output.

Visualize OCR and semantic document information.

🔮 Future Improvements

Possible future improvements include:

Larger and more diverse training datasets

Improved LayoutLMv3 fine-tuning

Better table structure recognition

Improved receipt line-item extraction

More robust certificate field extraction

Improved phone-number/date entity filtering

Support for PDF documents

GPU acceleration

Model hosting using Hugging Face Hub or another model registry

Automated evaluation on a larger benchmark dataset

REST API deployment

Cloud deployment

⚠️ Limitations

This is an academic/research prototype.

OCR and information extraction accuracy can vary depending on:

Image quality

Blur

Rotation

Handwritten content

Complex layouts

Font styles

Tables

Unusual document formats

The extracted information should therefore be validated before being used in critical workflows.

👨‍💻 Project

Intelligent Document Processing & Information Extraction System

GitHub Repository:

https://github.com/sriharshamagapu/Intelligent-Document-Processing-Information-Extraction-System

License

This project is intended for educational and research purposes.