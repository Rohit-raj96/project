import re
from typing import Any

from google.cloud import documentai


WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def extract_full_text(document: documentai.Document) -> str:
    return document.text or ""


def get_text_from_layout(layout: Any, document_text: str) -> str:
    if not layout or not layout.text_anchor:
        return ""

    text_segments = getattr(layout.text_anchor, "text_segments", None) or []
    parts: list[str] = []
    for segment in text_segments:
        start_index = int(getattr(segment, "start_index", 0) or 0)
        end_index = int(getattr(segment, "end_index", 0) or 0)
        parts.append(document_text[start_index:end_index])

    return normalize_text("".join(parts))


def row_to_values(row: Any, document_text: str) -> list[str]:
    return [get_text_from_layout(cell.layout, document_text) for cell in row.cells]


def normalize_row(row_values: list[str], width: int) -> list[str]:
    values = row_values[:width]
    if len(values) < width:
        values.extend([""] * (width - len(values)))
    return values


def build_table_columns(header_rows: list[list[str]], body_rows: list[list[str]]) -> list[str]:
    width = max((len(row) for row in header_rows + body_rows), default=0)
    if width == 0:
        return []

    if not header_rows:
        return [f"Column {index}" for index in range(1, width + 1)]

    normalized_headers = [normalize_row(row, width) for row in header_rows]
    columns: list[str] = []
    for index in range(width):
        values = [row[index] for row in normalized_headers if row[index]]
        columns.append(" / ".join(values) if values else f"Column {index + 1}")
    return columns


def extract_tables(document: documentai.Document) -> list[dict[str, Any]]:
    document_text = document.text or ""
    tables: list[dict[str, Any]] = []

    for page_index, page in enumerate(document.pages, start=1):
        page_number = page.page_number or page_index
        for table_index, table in enumerate(page.tables, start=1):
            header_rows = [row_to_values(row, document_text) for row in table.header_rows]
            body_rows = [row_to_values(row, document_text) for row in table.body_rows]
            columns = build_table_columns(header_rows, body_rows)
            normalized_body_rows = [normalize_row(row, len(columns)) for row in body_rows]

            tables.append(
                {
                    "page_number": page_number,
                    "table_number": table_index,
                    "columns": columns,
                    "rows": normalized_body_rows,
                }
            )

    return tables


def extract_form_fields(document: documentai.Document) -> list[dict[str, Any]]:
    document_text = document.text or ""
    form_fields: list[dict[str, Any]] = []

    for page_index, page in enumerate(document.pages, start=1):
        page_number = page.page_number or page_index
        for field in page.form_fields:
            field_name = get_text_from_layout(field.field_name, document_text)
            field_value = get_text_from_layout(field.field_value, document_text)

            if not field_name and not field_value:
                continue

            form_fields.append(
                {
                    "page_number": page_number,
                    "field_name": field_name,
                    "field_value": field_value,
                    "name_confidence": round(float(field.field_name.confidence or 0.0), 4),
                    "value_confidence": round(float(field.field_value.confidence or 0.0), 4),
                }
            )

    return form_fields
