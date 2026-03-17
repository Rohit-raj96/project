from pathlib import Path
import re

from pypdf import PdfReader


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PDF_EXTENSIONS = {".pdf"}

PRINTABLE_RE = re.compile(r"[A-Za-z0-9]")
GARBLED_RE = re.compile(r"[\uFFFD]")


def _clean_text(text: str) -> str:
    return " ".join((text or "").split())


def _page_text_metrics(text: str) -> dict:
    cleaned = _clean_text(text)
    total_chars = len(cleaned)

    if total_chars == 0:
        return {
            "text_chars": 0,
            "printable_chars": 0,
            "printable_ratio": 0.0,
            "digit_ratio": 0.0,
            "garbled_ratio": 0.0,
            "likely_text": False,
        }

    printable_chars = len(PRINTABLE_RE.findall(cleaned))
    digit_chars = len(re.findall(r"\d", cleaned))
    garbled_chars = len(GARBLED_RE.findall(cleaned))

    printable_ratio = printable_chars / total_chars
    digit_ratio = digit_chars / total_chars
    garbled_ratio = garbled_chars / total_chars

    # Heuristic:
    # - enough text
    # - reasonable amount of printable characters
    # - not dominated by garbage chars
    likely_text = (
        total_chars >= 80
        and printable_ratio >= 0.45
        and garbled_ratio <= 0.05
    )

    return {
        "text_chars": total_chars,
        "printable_chars": printable_chars,
        "printable_ratio": round(printable_ratio, 3),
        "digit_ratio": round(digit_ratio, 3),
        "garbled_ratio": round(garbled_ratio, 3),
        "likely_text": likely_text,
    }


def _classify_pdf_pages(reader: PdfReader, sample_pages: int | None = None) -> list[dict]:
    total_pages = len(reader.pages)
    limit = min(total_pages, sample_pages) if sample_pages else total_pages

    page_results = []

    for idx in range(limit):
        page = reader.pages[idx]
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        metrics = _page_text_metrics(page_text)

        # Optional lightweight image hint
        image_count = 0
        try:
            image_count = len(getattr(page, "images", []) or [])
        except Exception:
            image_count = 0

        if metrics["likely_text"]:
            page_type = "text_pdf"
        else:
            page_type = "scanned_pdf"

        page_results.append({
            "page_num": idx + 1,
            "page_type": page_type,
            "text_chars": metrics["text_chars"],
            "printable_ratio": metrics["printable_ratio"],
            "digit_ratio": metrics["digit_ratio"],
            "garbled_ratio": metrics["garbled_ratio"],
            "image_count": image_count,
        })

    return page_results


def _derive_document_type(page_results: list[dict]) -> str:
    if not page_results:
        return "scanned_pdf"

    page_types = {p["page_type"] for p in page_results}

    if page_types == {"text_pdf"}:
        return "text_pdf"
    if page_types == {"scanned_pdf"}:
        return "scanned_pdf"
    return "mixed_pdf"


def detect_file_type(
    file_path: str,
    sample_pages: int | None = None,
) -> dict:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        return {
            "extension": suffix,
            "category": "image",
            "document_type": "image",
            "pages": [
                {
                    "page_num": 1,
                    "page_type": "image",
                }
            ],
        }

    if suffix in PDF_EXTENSIONS:
        try:
            reader = PdfReader(str(path))
            total_pages = len(reader.pages)
            page_results = _classify_pdf_pages(reader, sample_pages=sample_pages)
            document_type = _derive_document_type(page_results)

            return {
                "extension": suffix,
                "category": "pdf",
                "document_type": document_type,
                "page_count": total_pages,
                "sampled_pages": len(page_results),
                "pages": page_results,
            }

        except Exception as exc:
            return {
                "extension": suffix,
                "category": "pdf",
                "document_type": "scanned_pdf",
                "page_count": None,
                "sampled_pages": 0,
                "pages": [],
                "error": str(exc),
            }

    return {
        "extension": suffix,
        "category": "unsupported",
        "document_type": "unsupported",
        "pages": [],
    }