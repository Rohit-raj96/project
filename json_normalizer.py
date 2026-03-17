import re
from typing import Any


MONEY_RE = re.compile(r"\(?-?\$?\d[\d,]*\.\d{2}\)?")
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /#&().-]{2,40})\s*[:\-]\s*(.+?)\s*$")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def money_to_float(value: str) -> float | None:
    if value is None:
        return None

    text = clean_text(value)
    if not text:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    text = text.replace("$", "").replace(",", "").strip()

    try:
        amount = float(text)
        return -amount if negative else amount
    except Exception:
        return None


def first_money_in_text(text: str) -> float | None:
    if not text:
        return None
    m = MONEY_RE.search(text)
    return money_to_float(m.group(0)) if m else None


def extract_dates_from_text(text: str) -> list[str]:
    if not text:
        return []
    return DATE_RE.findall(text)


def normalize_key(text: str) -> str:
    text = clean_text(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def flatten_pages_to_lines(raw_text: dict) -> list[dict]:
    """
    Returns:
    [
      {"page_num": 1, "line": "..."},
      ...
    ]
    """
    lines_out = []

    pages = raw_text.get("pages", []) if isinstance(raw_text, dict) else []
    for page in pages:
        page_num = page.get("page_num")
        lines = page.get("lines", []) or []
        for line in lines:
            line = clean_text(line)
            if line:
                lines_out.append({"page_num": page_num, "line": line})

    return lines_out


def lines_to_text(lines: list[dict]) -> str:
    return "\n".join(item["line"] for item in lines)


def find_first_matching_line(lines: list[dict], patterns: list[str]) -> dict | None:
    for item in lines:
        text = item["line"].lower()
        if any(re.search(pattern, text) for pattern in patterns):
            return item
    return None


def extract_key_values_from_lines(lines: list[dict]) -> dict[str, str]:
    result = {}

    for item in lines:
        line = item["line"]
        m = KEY_VALUE_RE.match(line)
        if not m:
            continue

        key = normalize_key(m.group(1))
        value = clean_text(m.group(2))
        if key and value and key not in result:
            result[key] = value

    return result


def extract_merchant_info(lines: list[dict], classified_tables: list[dict]) -> dict[str, Any]:
    merchant = {
        "name": None,
        "store_number": None,
        "merchant_number": None,
        "chain_number": None,
        "client_group": None,
        "address": None,
    }

    kv = extract_key_values_from_lines(lines)

    key_map = {
        "merchant": "name",
        "merchant_name": "name",
        "store_number": "store_number",
        "merchant_number": "merchant_number",
        "merchant_no": "merchant_number",
        "chain_number": "chain_number",
        "client_group": "client_group",
        "address": "address",
    }

    for k, v in kv.items():
        if k in key_map and not merchant[key_map[k]]:
            merchant[key_map[k]] = v

    # Fallback: look in merchant/header tables
    info_tables = [
        t for t in classified_tables
        if t.get("section_label") in {"merchant_info", "statement_meta"}
        or t.get("table_type") == "header_or_info_table"
    ]

    for table in info_tables:
        for row in table.get("rows", []):
            cells = [clean_text(c) for c in row if clean_text(c)]
            joined = " | ".join(cells).lower()

            if not merchant["store_number"] and "store number" in joined:
                for c in cells:
                    if re.search(r"\d", c):
                        merchant["store_number"] = c
                        break

            if not merchant["merchant_number"] and ("merchant number" in joined or "merchant no" in joined):
                for c in cells:
                    if re.search(r"\d", c):
                        merchant["merchant_number"] = c
                        break

            if not merchant["chain_number"] and "chain number" in joined:
                for c in cells:
                    if re.search(r"\d", c):
                        merchant["chain_number"] = c
                        break

            if not merchant["client_group"] and "client group" in joined:
                for c in cells:
                    if c.lower() != "client group":
                        merchant["client_group"] = c
                        break

    return merchant


def extract_statement_meta(lines: list[dict]) -> dict[str, Any]:
    full_text = lines_to_text(lines)

    dates = extract_dates_from_text(full_text)
    statement_date = dates[0] if dates else None

    period_line = find_first_matching_line(lines, [r"\bstatement period\b", r"\bperiod\b"])
    statement_period = period_line["line"] if period_line else None

    return {
        "statement_date": statement_date,
        "statement_period": statement_period,
    }


def parse_summary_table(table: dict) -> dict[str, Any]:
    summary = {
        "sales_count": None,
        "sales_amount": None,
        "returns_count": None,
        "returns_amount": None,
        "net_sales_count": None,
        "net_sales_amount": None,
        "total_charges_fees": None,
    }

    for row in table.get("rows", []):
        cells = [clean_text(c) for c in row]
        row_text = " ".join(cells).lower()
        money_values = [money_to_float(c) for c in cells if money_to_float(c) is not None]
        digit_values = [c for c in cells if re.search(r"\d", c)]

        if "net sales" in row_text:
            if digit_values:
                for c in digit_values:
                    if c.isdigit():
                        summary["net_sales_count"] = int(c)
                        break
            if money_values:
                summary["net_sales_amount"] = money_values[-1]

        elif "return" in row_text or "credit" in row_text:
            if digit_values:
                for c in digit_values:
                    if c.isdigit():
                        summary["returns_count"] = int(c)
                        break
            if money_values:
                summary["returns_amount"] = money_values[-1]

        elif "sales" in row_text and "net sales" not in row_text:
            if digit_values:
                for c in digit_values:
                    if c.isdigit():
                        summary["sales_count"] = int(c)
                        break
            if money_values:
                summary["sales_amount"] = money_values[-1]

        elif "charges" in row_text or "fees" in row_text:
            if money_values:
                summary["total_charges_fees"] = money_values[-1]

    return summary


def extract_summary(classified_tables: list[dict]) -> dict[str, Any]:
    candidates = [
        t for t in classified_tables
        if t.get("section_label") == "summary_table"
        or t.get("table_type") == "summary_table"
    ]

    best = None
    best_score = -1
    for t in candidates:
        score = (t.get("classification_confidence") or 0) + (t.get("quality_score") or 0)
        if score > best_score:
            best = t
            best_score = score

    return parse_summary_table(best) if best else {
        "sales_count": None,
        "sales_amount": None,
        "returns_count": None,
        "returns_amount": None,
        "net_sales_count": None,
        "net_sales_amount": None,
        "total_charges_fees": None,
    }


def parse_deposit_rows(table: dict) -> list[dict]:
    deposits = []

    for row in table.get("rows", []):
        cells = [clean_text(c) for c in row if clean_text(c)]
        if not cells:
            continue

        row_text = " ".join(cells)
        dates = extract_dates_from_text(row_text)
        amounts = [money_to_float(c) for c in cells if money_to_float(c) is not None]

        if not dates and not amounts:
            continue

        deposit = {
            "batch_date": dates[0] if len(dates) >= 1 else None,
            "settlement_date": dates[1] if len(dates) >= 2 else None,
            "reference_number": None,
            "batch_number": None,
            "amount": amounts[-1] if amounts else None,
        }

        digitish = [c for c in cells if re.search(r"\d", c)]
        if digitish:
            deposit["reference_number"] = digitish[0]
            if len(digitish) > 1:
                deposit["batch_number"] = digitish[1]

        # avoid obvious header rows
        text_lower = row_text.lower()
        if any(h in text_lower for h in ["batch", "deposit", "settlement date", "reference number"]):
            continue

        if deposit["amount"] is not None or deposit["batch_date"] is not None:
            deposits.append(deposit)

    return deposits


def extract_deposits(classified_tables: list[dict]) -> list[dict]:
    deposit_tables = [
        t for t in classified_tables
        if t.get("section_label") == "deposit_table"
        or t.get("table_type") == "deposit_table"
    ]

    all_deposits = []
    for table in deposit_tables:
        all_deposits.extend(parse_deposit_rows(table))

    return all_deposits


def detect_fee_section(row_text: str) -> str:
    text = row_text.lower()
    if "authorization" in text:
        return "authorization"
    if "network" in text:
        return "network"
    if "pin" in text or "debit" in text:
        return "pin_debit"
    if "other fee" in text or "other fees" in text:
        return "other"
    return "credit_card_processing"


def parse_fee_rows(table: dict) -> list[dict]:
    fees = []

    for row in table.get("rows", []):
        cells = [clean_text(c) for c in row if clean_text(c)]
        if not cells:
            continue

        row_text = " ".join(cells)
        lower = row_text.lower()

        if any(h in lower for h in ["description", "rate", "fee amount", "charges", "fees"]) and len(cells) <= 5:
            continue

        money_values = [money_to_float(c) for c in cells if money_to_float(c) is not None]
        if not money_values and "fee" not in lower and "charge" not in lower and "discount" not in lower:
            continue

        rate = None
        for c in cells:
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", c)
            if m:
                rate = float(m.group(1))
                break

        item_count = None
        for c in cells:
            if c.isdigit():
                item_count = int(c)
                break

        description = cells[0] if cells else None

        fees.append({
            "section": detect_fee_section(row_text),
            "description": description,
            "item_count": item_count,
            "sales_amount": money_values[0] if len(money_values) >= 2 else None,
            "rate": rate,
            "fee_amount": money_values[-1] if money_values else None,
        })

    return fees


def extract_fees(classified_tables: list[dict]) -> list[dict]:
    fee_tables = [
        t for t in classified_tables
        if t.get("section_label") == "fee_table"
        or t.get("table_type") == "fee_table"
        or t.get("section_label") in {"pricing_table", "card_brand_table"}
    ]

    all_fees = []
    for table in fee_tables:
        all_fees.extend(parse_fee_rows(table))

    return all_fees


def run_validations(summary: dict, deposits: list[dict], fees: list[dict], classified_tables: list[dict]) -> dict[str, Any]:
    issues = []

    deposit_amounts = [d["amount"] for d in deposits if d.get("amount") is not None]
    fee_amounts = [f["fee_amount"] for f in fees if f.get("fee_amount") is not None]

    deposit_total = round(sum(deposit_amounts), 2) if deposit_amounts else None
    fee_total = round(sum(fee_amounts), 2) if fee_amounts else None

    net_sales = summary.get("net_sales_amount")
    total_charges = summary.get("total_charges_fees")

    if net_sales is None:
        issues.append("Missing net sales amount")
    if not classified_tables:
        issues.append("No classified tables found")
    if deposit_total is None and not deposits:
        issues.append("No deposit rows extracted")
    if fee_total is None and not fees:
        issues.append("No fee rows extracted")

    status = "pass"
    if len(issues) >= 3:
        status = "fail"
    elif issues:
        status = "review"

    return {
        "status": status,
        "issues": issues,
        "totals": {
            "deposits_sum": deposit_total,
            "fees_sum": fee_total,
            "summary_net_sales_amount": net_sales,
            "summary_total_charges_fees": total_charges,
        }
    }


def build_review_flags(classified_tables: list[dict]) -> list[dict]:
    flags = []

    for table in classified_tables:
        conf = table.get("classification_confidence", 0)
        if conf < 2.5:
            flags.append({
                "type": "low_table_classification_confidence",
                "table_id": table.get("table_id"),
                "page": table.get("page"),
                "section_label": table.get("section_label"),
                "confidence": conf,
            })

        acc = table.get("accuracy")
        if acc is not None and acc < 70:
            flags.append({
                "type": "low_ocr_accuracy",
                "table_id": table.get("table_id"),
                "page": table.get("page"),
                "accuracy": acc,
            })

    return flags


def normalize_to_output_json(
    file_name: str,
    file_type: str,
    raw_text: dict,
    raw_tables: list,
    classified_tables: list,
    errors: list | None = None
) -> dict:
    errors = errors or []

    lines = flatten_pages_to_lines(raw_text)
    merchant = extract_merchant_info(lines, classified_tables)
    meta = extract_statement_meta(lines)
    summary = extract_summary(classified_tables)
    deposits = extract_deposits(classified_tables)
    fees = extract_fees(classified_tables)
    validation = run_validations(summary, deposits, fees, classified_tables)
    review_flags = build_review_flags(classified_tables)

    normalized_fields = {
        "merchant": merchant,
        **meta,
        "summary": summary,
        "deposits": deposits,
        "fees": fees,
        "validation": validation,
        "review_flags": review_flags,
    }

    status = "success"
    if errors:
        status = "partial_success"
    if validation["status"] == "fail":
        status = "needs_review"

    return {
        "file_name": file_name,
        "file_type": file_type,
        "status": status,
        "raw_text": raw_text,
        "raw_tables": raw_tables,
        "classified_tables": classified_tables,
        "normalized_fields": normalized_fields,
        "errors": errors,
    }