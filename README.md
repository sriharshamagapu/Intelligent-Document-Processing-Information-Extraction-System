# Intelligent Document Processing & Information Extraction System

Advanced OCR + LayoutLM/Transformers pipeline for extracting structured fields from receipts, invoices, and forms.

## Pipeline
1. Upload PDF/image
2. OCR with PaddleOCR/Tesseract
3. Normalize words and bounding boxes
4. Token classification with LayoutLMv3
5. Post-process entities into JSON
6. Evaluate with entity-level precision, recall, F1
7. Serve through Streamlit dashboard

## Quick start
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Dataset
Start with ICDAR2019 SROIE. Keep downloaded data under `data/` and do not commit large files.

## Disclaimer
This is a research/education prototype. Validate outputs before using them in business workflows.
