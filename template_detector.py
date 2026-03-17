import re
from collections import Counter


PROVIDER_PATTERNS = {
    "elavon": [
        r"\belavon\b",
        r"\biplus\b",
        r"\bmerchant connect\b",
        r"\bpayment network fees\b",
        r"\bauthorization fees\b",
        r"\bcredit card processing charges\b",
    ],
    "global_payments": [
        r"\bglobal payments\b",
        r"\bglobalpayments\b",
        r"\bmerchant services provided by global payments\b",
        r"\bdiscount fee\b",
        r"\btransit routing\b",
    ],
    "touchbistro": [
        r"\btouchbistro\b",
        r"\bbuyout criteria\b",
        r"\bopen outcry\b",
    ],
    "open_outcry": [
        r"\bopen outcry\b",
        r"\bbuyout criteria\b",
    ],
}


SECTION_HINTS = {
    "summary_heavy": [
        r"\bsummary\b",
        r"\btransaction summary\b",
        r"\bnet sales\b",
        r"\bgross sales\b",
    ],
    "deposit_heavy": [
        r"\bdeposit\b",
        r"\bdeposits\b",
        r"\bsettlement date\b",
        r"\bbatch number\b",
        r"\breference number\b",
    ],
    "fee_heavy": [
        r"\bfee\b",
        r"\bfees\b",
        r"\bcharges\b",
        r"\bauthorization fees\b",
        r"\bnetwork fees\b",
        r"\bprocessing charges\b",
    ],
    "pricing_heavy": [
        r"\bqualified\b",
        r"\bnon-qualified\b",
        r"\bdiscount rate\b",
        r"\bbasis points\b",
        r"\bbuyout criteria\b",
    ],
}


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def collect_text(raw_text: dict, raw_tables: list[dict]) -> str:
    parts = []

    if isinstance(raw_text, dict):
        for page in raw_text.get("pages", []) or []:
            parts.append(page.get("text", ""))

    for table in raw_tables or []:
        for row in table.get("rows", []) or []:
            parts.append(" ".join(str(cell or "") for cell in row))

    return normalize_text("\n".join(parts))


def score_provider(text: str) -> dict[str, float]:
    scores = {}

    for provider, patterns in PROVIDER_PATTERNS.items():
        score = 0.0
        for pattern in patterns:
            matches = len(re.findall(pattern, text))
            score += matches * 3.0
        scores[provider] = round(score, 2)

    return scores


def score_shape(text: str) -> dict[str, float]:
    scores = {}

    for label, patterns in SECTION_HINTS.items():
        score = 0.0
        for pattern in patterns:
            matches = len(re.findall(pattern, text))
            score += matches * 1.5
        scores[label] = round(score, 2)

    return scores


def choose_provider(provider_scores: dict[str, float]) -> tuple[str, float]:
    if not provider_scores:
        return "unknown", 0.0

    ranked = sorted(provider_scores.items(), key=lambda x: x[1], reverse=True)
    best_provider, best_score = ranked[0]

    if best_score < 3:
        return "unknown", best_score

    return best_provider, best_score


def infer_document_kind(shape_scores: dict[str, float]) -> str:
    if not shape_scores:
        return "unknown"

    best_label = max(shape_scores, key=shape_scores.get)

    mapping = {
        "summary_heavy": "financial_statement",
        "deposit_heavy": "financial_statement",
        "fee_heavy": "financial_statement",
        "pricing_heavy": "pricing_or_buyout_doc",
    }
    return mapping.get(best_label, "unknown")


def detect_template(raw_text: dict, raw_tables: list[dict]) -> dict:
    text = collect_text(raw_text, raw_tables)

    provider_scores = score_provider(text)
    shape_scores = score_shape(text)

    provider, provider_score = choose_provider(provider_scores)
    document_kind = infer_document_kind(shape_scores)

    ranked = sorted(provider_scores.items(), key=lambda x: x[1], reverse=True)
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = round(max(provider_score - second_score, 0) + min(provider_score, 10) * 0.1, 2)

    template_name = provider
    if provider == "unknown" and document_kind == "pricing_or_buyout_doc":
        template_name = "pricing_or_buyout_unknown"

    return {
        "provider": provider,
        "template_name": template_name,
        "document_kind": document_kind,
        "confidence": confidence,
        "provider_scores": provider_scores,
        "shape_scores": shape_scores,
    }