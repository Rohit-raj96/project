import re
from collections import Counter


SECTION_PATTERNS = {
    "merchant_info": [
        r"\bmerchant\b",
        r"\bmerchant number\b",
        r"\bmerchant no\b",
        r"\bstore number\b",
        r"\bchain number\b",
        r"\bstatement period\b",
        r"\bclient group\b",
        r"\bsettlement bank\b",
        r"\baddress\b",
    ],
    "summary_table": [
        r"\bsummary\b",
        r"\btransaction summary\b",
        r"\bsales\b",
        r"\breturns\b",
        r"\bcredits?\b",
        r"\bcount\b",
        r"\bvolume\b",
        r"\bamount\b",
        r"\bnet sales\b",
        r"\bgross sales\b",
        r"\btotal sales\b",
    ],
    "deposit_table": [
        r"\bdeposit\b",
        r"\bdeposits\b",
        r"\bbatch\b",
        r"\bbatch number\b",
        r"\bsettlement date\b",
        r"\bdeposit date\b",
        r"\bdeposit amount\b",
        r"\breference number\b",
        r"\bfunding\b",
    ],
    "fee_table": [
        r"\bfee\b",
        r"\bfees\b",
        r"\bcharges\b",
        r"\bcharge\b",
        r"\bdiscount\b",
        r"\bassessment\b",
        r"\bassessments\b",
        r"\binterchange\b",
        r"\bprocessing charges?\b",
        r"\bnetwork fees?\b",
        r"\bauthorization fees?\b",
        r"\bother fees?\b",
        r"\bservice fee\b",
        r"\bmonthly fee\b",
        r"\brate\b",
    ],
    "card_brand_table": [
        r"\bvisa\b",
        r"\bmastercard\b",
        r"\bmaster card\b",
        r"\bamex\b",
        r"\bamerican express\b",
        r"\bdiscover\b",
        r"\binterac\b",
        r"\bdebit\b",
        r"\bcredit\b",
        r"\bcard type\b",
    ],
    "pricing_table": [
        r"\bqualified\b",
        r"\bmid-qualified\b",
        r"\bnon-qualified\b",
        r"\bper item\b",
        r"\bbasis points?\b",
        r"\btransaction fee\b",
        r"\bdiscount rate\b",
        r"\bpricing\b",
        r"\bprogram fee\b",
        r"\bbuyout\b",
        r"\bcriteria\b",
    ],
    "statement_meta": [
        r"\bstatement\b",
        r"\bpage\b",
        r"\bperiod\b",
        r"\bdate\b",
        r"\breport\b",
    ],
}


NUMERIC_RE = re.compile(r"\d")
MONEY_RE = re.compile(r"[\$\€\£]?\(?-?\d[\d,]*\.\d{2}\)?")
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[_|]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def table_rows_to_text(table: dict) -> tuple[list[list[str]], str]:
    rows = table.get("rows", []) or []
    clean_rows = [[str(cell or "").strip() for cell in row] for row in rows]
    text_blob = " ".join(" ".join(row) for row in clean_rows)
    return clean_rows, normalize_text(text_blob)


def compute_table_features(rows: list[list[str]], text_blob: str) -> dict:
    row_count = len(rows)
    col_count = max((len(r) for r in rows), default=0)

    cells = [cell for row in rows for cell in row if str(cell).strip()]
    numeric_cells = sum(1 for cell in cells if NUMERIC_RE.search(cell))
    money_cells = sum(1 for cell in cells if MONEY_RE.search(cell))
    date_cells = sum(1 for cell in cells if DATE_RE.search(cell))

    header_text = " ".join(rows[0]) if rows else ""
    header_text = normalize_text(header_text)

    return {
        "row_count": row_count,
        "col_count": col_count,
        "cell_count": len(cells),
        "numeric_cells": numeric_cells,
        "money_cells": money_cells,
        "date_cells": date_cells,
        "header_text": header_text,
        "text_len": len(text_blob),
    }


def score_section(section: str, text_blob: str, features: dict) -> float:
    score = 0.0

    for pattern in SECTION_PATTERNS.get(section, []):
        matches = len(re.findall(pattern, text_blob))
        score += matches * 3.0

        header_matches = len(re.findall(pattern, features["header_text"]))
        score += header_matches * 2.0

    row_count = features["row_count"]
    col_count = features["col_count"]
    numeric_cells = features["numeric_cells"]
    money_cells = features["money_cells"]
    date_cells = features["date_cells"]

    if section == "merchant_info":
        if row_count <= 12:
            score += 2.0
        if col_count <= 4:
            score += 1.0
        if date_cells >= 1:
            score += 0.5

    elif section == "summary_table":
        if numeric_cells >= 4:
            score += 2.0
        if money_cells >= 2:
            score += 2.0
        if row_count >= 2:
            score += 1.0

    elif section == "deposit_table":
        if date_cells >= 2:
            score += 4.0
        if money_cells >= 1:
            score += 2.0
        if numeric_cells >= 4:
            score += 1.0

    elif section == "fee_table":
        if money_cells >= 2:
            score += 3.0
        if numeric_cells >= 4:
            score += 1.5
        if col_count >= 3:
            score += 1.0

    elif section == "card_brand_table":
        if numeric_cells >= 3:
            score += 1.5
        if money_cells >= 1:
            score += 1.0

    elif section == "pricing_table":
        if money_cells >= 1:
            score += 1.5
        if numeric_cells >= 3:
            score += 1.5
        if col_count >= 3:
            score += 1.0

    elif section == "statement_meta":
        if row_count <= 10:
            score += 1.0

    return round(score, 2)


def apply_priority_rules(scores: dict, text_blob: str, features: dict) -> tuple[str, str]:
    """
    Resolve close calls with domain-specific rules.
    """
    row_count = features["row_count"]
    date_cells = features["date_cells"]
    money_cells = features["money_cells"]

    if scores["deposit_table"] >= 6 and date_cells >= 2:
        return "deposit_table", "matched deposit patterns with strong date signals"

    if scores["fee_table"] >= 6 and ("fee" in text_blob or "charges" in text_blob):
        return "fee_table", "matched fee patterns and monetary structure"

    if scores["card_brand_table"] >= 6 and scores["pricing_table"] >= 6:
        if any(word in text_blob for word in ["qualified", "non-qualified", "basis points", "discount rate"]):
            return "pricing_table", "pricing terms override card-brand-only classification"
        return "card_brand_table", "card brand signals stronger than pricing terms"

    if scores["merchant_info"] >= 5 and row_count <= 12:
        return "merchant_info", "compact header/info structure detected"

    if scores["summary_table"] >= 5 and money_cells >= 2:
        return "summary_table", "summary terms with transaction totals detected"

    best_label = max(scores, key=scores.get)
    return best_label, "best weighted score"


def simplify_label(label: str) -> str:
    """
    Keep compatibility with your older downstream naming where useful.
    """
    mapping = {
        "merchant_info": "header_or_info_table",
        "statement_meta": "header_or_info_table",
        "summary_table": "summary_table",
        "deposit_table": "deposit_table",
        "fee_table": "fee_table",
        "card_brand_table": "pricing_or_card_table",
        "pricing_table": "pricing_or_card_table",
    }
    return mapping.get(label, "unknown")


def classify_single_table(table: dict) -> dict:
    rows, text_blob = table_rows_to_text(table)
    features = compute_table_features(rows, text_blob)

    if not text_blob.strip():
        return {
            "table_type": "unknown",
            "section_label": "unknown",
            "confidence": 0.0,
            "scores": {},
            "reason": "empty table text",
        }

    scores = {
        section: score_section(section, text_blob, features)
        for section in SECTION_PATTERNS.keys()
    }

    best_section, reason = apply_priority_rules(scores, text_blob, features)
    best_score = scores.get(best_section, 0.0)

    sorted_scores = sorted(scores.values(), reverse=True)
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    confidence = round(best_score - second_score + min(best_score, 10) * 0.1, 2)

    return {
        "table_type": simplify_label(best_section),
        "section_label": best_section,
        "confidence": max(confidence, 0.0),
        "scores": scores,
        "reason": reason,
    }


def classify_tables(raw_tables: list) -> list:
    classified = []

    for table in raw_tables or []:
        table_copy = dict(table)
        result = classify_single_table(table)

        table_copy["table_type"] = result["table_type"]
        table_copy["section_label"] = result["section_label"]
        table_copy["classification_confidence"] = result["confidence"]
        table_copy["classification_reason"] = result["reason"]
        table_copy["classification_scores"] = result["scores"]

        classified.append(table_copy)

    return classified
