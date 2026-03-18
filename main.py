import json
import shutil
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

warnings.filterwarnings("ignore")

import pytesseract
from dotenv import load_dotenv

from config_utils import read_setting
from file_detector import detect_file_type
from pdf_text_extractor import extract_from_text_pdf
from ocr_extractor import extract_from_image, extract_from_scanned_pdf
from template_detector import detect_template
from table_builder import build_raw_structure, build_ocr_pseudo_tables
from table_classifier import classify_tables
from json_normalizer import normalize_to_output_json


ENV_FILE = Path(__file__).resolve().with_name(".env")
DATASET_DIR = "dataset"
OUTPUT_DIR = "output"


def load_runtime_environment() -> None:
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def get_runtime_settings() -> dict[str, str | bool | None]:
    load_runtime_environment()

    tesseract_cmd = read_setting("TESSERACT_CMD")
    poppler_path = read_setting("POPPLER_PATH")
    tesseract_on_path = shutil.which("tesseract")

    return {
        "env_file": str(ENV_FILE) if ENV_FILE.exists() else None,
        "tesseract_cmd": tesseract_cmd,
        "tesseract_on_path": tesseract_on_path,
        "tesseract_available": bool(tesseract_cmd or tesseract_on_path),
        "poppler_path": poppler_path,
    }


def configure_ocr_runtime() -> dict[str, str | bool | None]:
    settings = get_runtime_settings()
    if settings["tesseract_cmd"]:
        pytesseract.pytesseract.tesseract_cmd = str(settings["tesseract_cmd"])
    return settings


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_output_path(output_dir: Path, input_name: str) -> Path:
    return output_dir / f"{Path(input_name).stem}.json"


def serialize_result(data: dict, indent: int = 2) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=False)


def save_json_to_path(output_path: Path, data: dict):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(serialize_result(data))


def save_json(output_dir: Path, input_file: Path, data: dict):
    save_json_to_path(build_output_path(output_dir, input_file.name), data)


def empty_extraction_result() -> dict:
    return {
        "raw_text": {},
        "raw_tables": [],
    }


def merge_page_text_results(page_results: list[dict]) -> dict:
    """
    Merge OCR/text page results into a single raw_text structure.
    Expected page shape:
    {
      "page_num": 1,
      "text": "...",
      "lines": [...],
      "words": [...],          # optional for OCR pages
      "ocr_meta": {...},       # optional for OCR pages
      "source": "ocr|text"
    }
    """
    pages = sorted(page_results, key=lambda p: p.get("page_num", 0))
    return {
        "page_count": len(pages),
        "pages": pages,
    }


def normalize_text_pdf_pages_for_merge(text_result: dict, allowed_pages: set[int] | None = None) -> list[dict]:
    """
    Convert text-pdf extractor output into the same page structure used by OCR pages.
    Assumes text_result may contain:
      - raw_text.pages
      - or page_count + pages
      - or only a text blob (fallback)
    """
    raw_text = text_result.get("raw_text", text_result)

    if isinstance(raw_text, dict) and raw_text.get("pages"):
        pages = []
        for page in raw_text["pages"]:
            page_num = page.get("page_num")
            if allowed_pages and page_num not in allowed_pages:
                continue

            page_text = page.get("text", "")
            page_lines = page.get("lines") or [ln.strip() for ln in page_text.splitlines() if ln.strip()]
            pages.append({
                "page_num": page_num,
                "text": page_text,
                "lines": page_lines,
                "words": page.get("words", []),
                "source": "text",
            })
        return pages

    # fallback: no page-aware structure available
    text_blob = ""
    if isinstance(raw_text, dict):
        text_blob = raw_text.get("text", "") or raw_text.get("full_text", "") or ""
    elif isinstance(raw_text, str):
        text_blob = raw_text

    text_blob = text_blob or ""
    return [{
        "page_num": 1,
        "text": text_blob,
        "lines": [ln.strip() for ln in text_blob.splitlines() if ln.strip()],
        "words": [],
        "source": "text",
    }]


def filter_text_tables_by_pages(raw_tables: list[dict], allowed_pages: set[int] | None = None) -> list[dict]:
    if not allowed_pages:
        return raw_tables or []

    filtered = []
    for table in raw_tables or []:
        table_page = table.get("page")
        if table_page is None or table_page in allowed_pages:
            filtered.append(table)
    return filtered


def process_text_pdf(file_path: Path) -> dict:
    return extract_from_text_pdf(str(file_path))


def process_scanned_pdf(
    file_path: Path,
    poppler_path: str | None = None,
    page_numbers: list[int] | None = None,
) -> dict:
    raw_text = extract_from_scanned_pdf(
        str(file_path),
        poppler_path=poppler_path,
        page_numbers=page_numbers,
    )
    return {
        "raw_text": raw_text,
        "raw_tables": [],
    }


def process_image(file_path: Path) -> dict:
    raw_text = extract_from_image(str(file_path))
    return {
        "raw_text": raw_text,
        "raw_tables": [],
    }


def process_mixed_pdf(file_path: Path, detected: dict, poppler_path: str | None = None) -> dict:
    """
    Hybrid strategy:
    - run text extractor once for the whole PDF
    - run OCR once for the whole PDF
    - keep only the pages we need from each source
    """
    pages = detected.get("pages", [])
    text_page_nums = {p["page_num"] for p in pages if p.get("page_type") == "text_pdf"}
    scanned_page_nums = {p["page_num"] for p in pages if p.get("page_type") == "scanned_pdf"}

    text_result = empty_extraction_result()
    ocr_result = empty_extraction_result()

    if text_page_nums:
        text_result = process_text_pdf(file_path)

    if scanned_page_nums:
        ocr_result = process_scanned_pdf(
            file_path,
            poppler_path=poppler_path,
            page_numbers=sorted(scanned_page_nums),
        )

    merged_pages = []

    if text_page_nums:
        merged_pages.extend(
            normalize_text_pdf_pages_for_merge(text_result, allowed_pages=text_page_nums)
        )

    if scanned_page_nums:
        ocr_pages = (ocr_result.get("raw_text", {}) or {}).get("pages", [])
        for page in ocr_pages:
            if page.get("page_num") in scanned_page_nums:
                page["source"] = "ocr"
                merged_pages.append(page)

    merged_raw_text = merge_page_text_results(merged_pages)

    text_tables = filter_text_tables_by_pages(
        text_result.get("raw_tables", []),
        allowed_pages=text_page_nums,
    )

    return {
        "raw_text": merged_raw_text,
        "raw_tables": text_tables,
    }


def process_file(file_path: Path) -> dict:
    runtime_settings = configure_ocr_runtime()
    errors = []

    detected = detect_file_type(str(file_path))
    file_type = detected["document_type"]

    if file_type == "unsupported":
        return {
            "file_name": file_path.name,
            "file_type": "unsupported",
            "status": "skipped",
            "raw_text": {},
            "raw_tables": [],
            "classified_tables": [],
            "normalized_fields": {},
            "detected": detected,
            "errors": [f"Unsupported file type: {file_path.suffix.lower()}"],
        }

    extraction_result = empty_extraction_result()

    try:
        if file_type == "text_pdf":
            extraction_result = process_text_pdf(file_path)

        elif file_type == "scanned_pdf":
            extraction_result = process_scanned_pdf(
                file_path,
                poppler_path=runtime_settings["poppler_path"],
            )

        elif file_type == "mixed_pdf":
            extraction_result = process_mixed_pdf(
                file_path,
                detected,
                poppler_path=runtime_settings["poppler_path"],
            )

        elif file_type == "image":
            extraction_result = process_image(file_path)

    except Exception as e:
        errors.append(str(e))
        extraction_result = empty_extraction_result()

    built = build_raw_structure(file_type, extraction_result)

    template_info = detect_template(
        raw_text=built["raw_text"],
        raw_tables=built["raw_tables"],
    )
    # Build pseudo tables for OCR-like pages
    if file_type in {"scanned_pdf", "image", "mixed_pdf"}:
        pseudo_tables = build_ocr_pseudo_tables(built["raw_text"])
        built["raw_tables"].extend(pseudo_tables)

    classified_tables = classify_tables(built["raw_tables"])

    final_json = normalize_to_output_json(
        file_name=file_path.name,
        file_type=file_type,
        raw_text=built["raw_text"],
        raw_tables=built["raw_tables"],
        classified_tables=classified_tables,
        errors=errors,
    )

    # preserve detector metadata for debugging / QA
    if isinstance(final_json, dict):
        final_json["detected"] = detected
        final_json["template"] = template_info

    return final_json


def process_uploaded_file(file_name: str, file_bytes: bytes) -> dict:
    safe_name = Path(file_name).name or "uploaded_document"

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / safe_name
        temp_path.write_bytes(file_bytes)
        return process_file(temp_path)


def main():
    configure_ocr_runtime()

    dataset_path = Path(DATASET_DIR)
    output_path = Path(OUTPUT_DIR)

    ensure_dir(output_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    files = [p for p in dataset_path.iterdir() if p.is_file()]

    if not files:
        print("No files found in dataset folder.")
        return

    for file_path in files:
        try:
            result = process_file(file_path)
            save_json(output_path, file_path, result)
            print(f"Done: {file_path.name} -> {file_path.stem}.json")
        except Exception as e:
            fallback = {
                "file_name": file_path.name,
                "file_type": "unknown",
                "status": "failed",
                "raw_text": {},
                "raw_tables": [],
                "classified_tables": [],
                "normalized_fields": {},
                "errors": [str(e)],
            }
            save_json(output_path, file_path, fallback)
            print(f"Failed: {file_path.name} -> {e}")


if __name__ == "__main__":
    main()
