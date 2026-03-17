import re
from statistics import median


def build_raw_structure(file_type: str, extraction_result: dict) -> dict:
    raw_text = extraction_result.get("raw_text", {})
    raw_tables = extraction_result.get("raw_tables", [])

    if file_type in {"image", "scanned_pdf"} and not raw_text:
        raw_text = extraction_result

    return {
        "raw_text": raw_text,
        "raw_tables": raw_tables,
    }


def normalize_cell_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def group_words_into_rows(words: list[dict], y_tolerance: int | None = None) -> list[list[dict]]:
    if not words:
        return []

    words = sorted(words, key=lambda w: (w["y"], w["x"]))
    heights = [max(1, int(w.get("h", 1))) for w in words]
    default_tol = max(8, int(median(heights) * 0.6))
    tol = y_tolerance or default_tol

    rows: list[list[dict]] = []

    for word in words:
        word_mid_y = word["y"] + word["h"] / 2
        placed = False

        for row in rows:
            row_mid_y = median([w["y"] + w["h"] / 2 for w in row])
            if abs(word_mid_y - row_mid_y) <= tol:
                row.append(word)
                placed = True
                break

        if not placed:
            rows.append([word])

    rows = [sorted(row, key=lambda w: w["x"]) for row in rows]
    rows = sorted(rows, key=lambda row: min(w["y"] for w in row))
    return rows


def infer_column_positions(rows: list[list[dict]], x_tolerance: int = 25) -> list[int]:
    x_positions = []

    for row in rows:
        for word in row:
            x_positions.append(int(word["x"]))

    if not x_positions:
        return []

    x_positions.sort()
    clusters: list[list[int]] = []

    for x in x_positions:
        if not clusters:
            clusters.append([x])
            continue

        cluster_center = sum(clusters[-1]) / len(clusters[-1])
        if abs(x - cluster_center) <= x_tolerance:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    centers = [int(sum(cluster) / len(cluster)) for cluster in clusters]

    # drop weak columns that appear only once when we already have many columns
    if len(centers) <= 2:
        return centers

    strong_centers = []
    for cluster, center in zip(clusters, centers):
        if len(cluster) >= 2 or len(centers) <= 4:
            strong_centers.append(center)

    return strong_centers or centers


def assign_words_to_columns(row: list[dict], column_positions: list[int]) -> list[str]:
    if not column_positions:
        text = " ".join(w["text"] for w in row).strip()
        return [normalize_cell_text(text)] if text else []

    cells = [[] for _ in column_positions]

    for word in row:
        x = word["x"]
        best_idx = min(
            range(len(column_positions)),
            key=lambda idx: abs(x - column_positions[idx]),
        )
        cells[best_idx].append(word)

    rendered = []
    for cell_words in cells:
        cell_words = sorted(cell_words, key=lambda w: w["x"])
        cell_text = " ".join(w["text"] for w in cell_words).strip()
        rendered.append(normalize_cell_text(cell_text))

    # trim trailing empty cells
    while rendered and not rendered[-1]:
        rendered.pop()

    return rendered


def rows_from_words(words: list[dict]) -> list[list[str]]:
    grouped_rows = group_words_into_rows(words)
    if not grouped_rows:
        return []

    column_positions = infer_column_positions(grouped_rows)

    rows = []
    for row in grouped_rows:
        rendered = assign_words_to_columns(row, column_positions)
        if any(cell.strip() for cell in rendered):
            rows.append(rendered)

    return rows


def rows_from_lines(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        parts = re.split(r"\s{2,}|\t", line)
        parts = [normalize_cell_text(p) for p in parts if normalize_cell_text(p)]
        if not parts:
            parts = [normalize_cell_text(line)]
        if any(parts):
            rows.append(parts)
    return rows


def estimate_table_quality(rows: list[list[str]]) -> float:
    if not rows:
        return 0.0

    non_empty_rows = sum(1 for row in rows if any(cell.strip() for cell in row))
    avg_cols = sum(len(row) for row in rows) / max(1, len(rows))
    numeric_cells = sum(
        1
        for row in rows
        for cell in row
        if re.search(r"\d", cell or "")
    )

    return round(
        non_empty_rows * 1.0
        + avg_cols * 2.0
        + numeric_cells * 0.25,
        2,
    )


def build_ocr_pseudo_tables(raw_text: dict) -> list:
    """
    Build pseudo tables from OCR output.

    Priority:
    1. Use OCR word bounding boxes when available
    2. Fall back to line splitting when boxes are unavailable
    """
    pages = raw_text.get("pages", [])
    pseudo_tables = []

    for page in pages:
        page_num = page.get("page_num")
        words = page.get("words", []) or []
        lines = page.get("lines", []) or []

        rows = rows_from_words(words) if words else rows_from_lines(lines)
        if not rows:
            continue

        max_cols = max((len(r) for r in rows), default=0)

        pseudo_tables.append({
            "table_id": f"ocr_page_{page_num}",
            "source": "ocr_words" if words else "ocr_lines",
            "flavor": "positional" if words else "pseudo",
            "page": page_num,
            "accuracy": page.get("ocr_meta", {}).get("mean_confidence"),
            "quality_score": estimate_table_quality(rows),
            "shape": [len(rows), max_cols],
            "headers": [],
            "rows": rows,
        })

    return pseudo_tables