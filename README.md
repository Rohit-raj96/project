# Financial Statement Extractor

This repo now supports two extraction paths inside one Streamlit UI:

- `Local OCR (Free)`: runs the higher-accuracy PyTesseract pipeline on your machine, including template detection and normalization.
- `Google Document AI (Cloud)`: sends the uploaded file to an existing Document AI processor when you want a cloud pipeline.

The local extractor is based on the current higher-accuracy pipeline in `main.py` and related modules. The earlier first-level baseline pipeline has been moved under `archive/legacy-baseline/`.

## What The UI Does

- Upload a financial statement as PDF or image
- Choose either the local OCR pipeline or Google Document AI
- View extracted text, tables, and normalized fields
- Edit the resulting JSON directly in the browser
- Download the edited JSON and extracted text

## Main Files

```text
app.py                    Streamlit UI for both pipelines
main.py                   Local OCR pipeline entrypoint and shared helpers
ocr_extractor.py          OCR for images and scanned PDFs
pdf_text_extractor.py     Native-text PDF extraction and table extraction
table_builder.py          OCR pseudo-table construction
table_classifier.py       Table section classification
template_detector.py      Template/provider detection
json_normalizer.py        Final normalized financial statement schema
document_ai_client.py     Google Document AI client wrapper
parser.py                 Google Document AI parsing helpers
exporter.py               Google Document AI JSON exporter
archive/legacy-baseline/ Archived first-level baseline code, samples, and outputs
```

## Requirements

- Python 3.10+
- One virtual environment for the whole repo
- For local OCR mode:
  - Tesseract OCR installed
  - Poppler installed for scanned PDFs
  - Ghostscript may be needed depending on Camelot usage
- For cloud mode:
  - A Google Cloud project with Document AI enabled
  - An existing Document AI processor
  - `GOOGLE_APPLICATION_CREDENTIALS` or another ADC flow

## Setup

1. Create and activate one virtual environment for the repo.

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file from the template.

```powershell
Copy-Item .env.example .env
```

4. Edit `.env` only for the paths and cloud settings you need.

```dotenv
# Local OCR pipeline
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
POPPLER_PATH=C:\path\to\poppler\Library\bin

# Google Document AI pipeline
DOCUMENT_AI_PROJECT_ID=your-project-id
DOCUMENT_AI_LOCATION=us
DOCUMENT_AI_PROCESSOR_ID=your-processor-id
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account.json
```

5. Start the UI.

```powershell
streamlit run app.py --server.port 8502
```

6. Open the Streamlit URL, upload a statement, choose a mode, run extraction, review the output, edit the JSON, and download the result.

## Batch CLI

The original local batch pipeline still works:

```powershell
python main.py
```

It reads files from `dataset/` and writes output JSON files to `output/`.

## GitHub Notes

- Keep `.env` local only
- Do not commit Google service account keys
- Keep your one shared `venv/` local only
- The UI supports both the free local path and the optional cloud path, which is the main repo story for GitHub visitors

## Publish Flow

```powershell
git init
git add .
git commit -m "Initial commit: dual-mode financial statement extractor"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
