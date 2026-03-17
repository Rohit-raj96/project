import re
from pathlib import Path

import camelot
import pdfplumber
from pypdf import PdfReader


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def is_meaningful_cell(cell: str) -> bool:
    return bool(str(cell or "").strip())


def normalize_rows(rows: list[list]) -> list[list[str]]:
    normalized = []
    for row in rows:
        normalized.append([str(cell).strip() for cell in row])
    return normalized


def estimate_table_quality(rows: list[list[str]], accuracy: float | None = None) -> float:
    if not rows:
        return 0.0

    row_count = len(rows)
    avg_cols = sum(len(r) for r in rows) / max(1, row_count)
    non_empty_cells = sum(1 for row in rows for cell in row if is_meaningful_cell(cell))
    numeric_cells = sum(
        1 for row in rows for cell in row
        if re.search(r"\d", str(cell or ""))
    )

    score = (
        row_count * 1.0
        + avg_cols * 2.0
        + non_empty_cells * 0.15
        + numeric_cells * 0.25
    )

    if accuracy is not None:
        score += float(accuracy) * 0.1

    return round(score, 2)


def extract_pdf_text(pdf_path: str) -> dict:
    path = Path(pdf_path)
    pages = []

    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:
            text = ""

        pages.append({
            "page_num": i,
            "text": text,
            "lines": extract_lines(text),
            "words": [],
            "source": "text",
        })

    return {
        "page_count": len(pages),
        "pages": pages,
    }


def extract_camelot_tables(pdf_path: str) -> list[dict]:
    path = Path(pdf_path)
    all_tables = []

    for flavor in ["stream", "lattice"]:
        try:
            tables = camelot.read_pdf(str(path), pages="all", flavor=flavor)
        except Exception:
            continue

        for i, table in enumerate(tables, start=1):
            try:
                df = table.df.fillna("")
            except Exception:
                continue

            if df.empty:
                continue

            rows = normalize_rows(df.values.tolist())
            non_empty_count = sum(
                1 for row in rows for cell in row if is_meaningful_cell(cell)
            )
            if non_empty_count == 0:
                continue

            accuracy = None
            try:
                accuracy = round(float(table.accuracy), 2) if table.accuracy is not None else None
            except Exception:
                accuracy = None

            page_num = None
            try:
                page_num = int(getattr(table, "page", None))
            except Exception:
                page_num = None

            quality_score = estimate_table_quality(rows, accuracy=accuracy)

            all_tables.append({
                "table_id": f"camelot_{flavor}_{i}",
                "source": "camelot",
                "flavor": flavor,
                "page": page_num,
                "accuracy": accuracy,
                "quality_score": quality_score,
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "headers": [],
                "rows": rows,
            })

    return all_tables


def extract_pdfplumber_tables(pdf_path: str) -> list[dict]:
    path = Path(pdf_path)
    extracted = []

    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables()
                except Exception:
                    continue

                if not tables:
                    continue

                for table_idx, table in enumerate(tables, start=1):
                    if not table:
                        continue

                    rows = normalize_rows(table)
                    non_empty_count = sum(
                        1 for row in rows for cell in row if is_meaningful_cell(cell)
                    )
                    if non_empty_count == 0:
                        continue

                    col_count = max((len(r) for r in rows), default=0)
                    quality_score = estimate_table_quality(rows, accuracy=None)

                    extracted.append({
                        "table_id": f"pdfplumber_{page_idx}_{table_idx}",
                        "source": "pdfplumber",
                        "flavor": "text",
                        "page": page_idx,
                        "accuracy": None,
                        "quality_score": quality_score,
                        "shape": [len(rows), col_count],
                        "headers": [],
                        "rows": rows,
                    })
    except Exception:
        return []

    return extracted


def deduplicate_tables(tables: list[dict]) -> list[dict]:
    seen = set()
    deduped = []

    for table in sorted(
        tables,
        key=lambda t: (
            t.get("page") if t.get("page") is not None else 10**9,
            -(t.get("quality_score") or 0),
        ),
    ):
        rows = table.get("rows", [])
        signature = (
            table.get("page"),
            tuple(tuple(cell for cell in row) for row in rows[:20]),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(table)

    return deduped


def select_best_tables_per_page(tables: list[dict]) -> list[dict]:
    if not tables:
        return []

    grouped: dict[object, list[dict]] = {}
    for table in tables:
        page = table.get("page")
        grouped.setdefault(page, []).append(table)

    final_tables = []
    for page, page_tables in grouped.items():
        page_tables = sorted(
            page_tables,
            key=lambda t: (
                t.get("quality_score") or 0,
                t.get("accuracy") or 0,
            ),
            reverse=True,
        )

        if page is None:
            final_tables.extend(page_tables[:3])
            continue

        best_score = page_tables[0].get("quality_score") or 0
        keep = []
        for t in page_tables:
            score = t.get("quality_score") or 0
            if score >= best_score * 0.65:
                keep.append(t)

        final_tables.extend(keep)

    final_tables = sorted(
        final_tables,
        key=lambda t: (
            t.get("page") if t.get("page") is not None else 10**9,
            t.get("table_id", ""),
        ),
    )
    return final_tables


def extract_pdf_tables(pdf_path: str) -> list:
    camelot_tables = extract_camelot_tables(pdf_path)
    plumber_tables = extract_pdfplumber_tables(pdf_path)

    combined = camelot_tables + plumber_tables
    combined = deduplicate_tables(combined)
    combined = select_best_tables_per_page(combined)

    return combined


def extract_from_text_pdf(pdf_path: str) -> dict:
    text_result = extract_pdf_text(pdf_path)
    tables = extract_pdf_tables(pdf_path)

    return {
        "raw_text": text_result,
        "raw_tables": tables,
    }