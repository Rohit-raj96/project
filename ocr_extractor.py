import os
import re
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_path


DEFAULT_TESSERACT_CONFIGS = [
    "--oem 3 --psm 6",
    "--oem 3 --psm 4",
    "--oem 3 --psm 11",
]

NUMERIC_TOKEN_RE = re.compile(r"^[\$\(\)\-]?\d[\d,]*(\.\d{1,2})?%?$")
CURRENCY_HINT_RE = re.compile(r"[\$\€\£]|\d[\d,]*\.\d{2}")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pil_to_cv(pil_img: Image.Image):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def deskew_image(gray: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(gray < 250))
    if coords.size == 0:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.2:
        return gray

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def crop_borders(gray: np.ndarray, pad: int = 10) -> np.ndarray:
    inv = 255 - gray
    coords = cv2.findNonZero(inv)
    if coords is None:
        return gray

    x, y, w, h = cv2.boundingRect(coords)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray.shape[1], x + w + pad)
    y2 = min(gray.shape[0], y + h + pad)
    return gray[y1:y2, x1:x2]


def remove_table_lines(binary_img: np.ndarray) -> np.ndarray:
    cleaned = binary_img.copy()

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vertical = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, v_kernel, iterations=1)

    lines = cv2.bitwise_or(horizontal, vertical)
    cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(lines))
    return cleaned


def preprocess_for_ocr(img_bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    scale = 2.0 if max(gray.shape[:2]) < 2500 else 1.5
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = crop_borders(gray)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    gray = deskew_image(gray)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    otsu_no_lines = remove_table_lines(otsu)
    adaptive_no_lines = remove_table_lines(adaptive)

    return [
        gray,
        otsu,
        adaptive,
        otsu_no_lines,
        adaptive_no_lines,
    ]


def safe_int(value, default=0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def safe_float(value, default=0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def extract_words_and_lines_from_data(data: dict) -> tuple[list[dict], list[str], str]:
    n = len(data.get("text", []))
    words = []
    line_buckets: dict[tuple[int, int, int], list[dict]] = {}

    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf = safe_float(data["conf"][i], default=-1)

        if not text:
            continue
        if conf < 0:
            continue

        word = {
            "text": text,
            "conf": round(conf, 2),
            "x": safe_int(data["left"][i]),
            "y": safe_int(data["top"][i]),
            "w": safe_int(data["width"][i]),
            "h": safe_int(data["height"][i]),
            "block_num": safe_int(data["block_num"][i]),
            "par_num": safe_int(data["par_num"][i]),
            "line_num": safe_int(data["line_num"][i]),
        }
        words.append(word)

        key = (word["block_num"], word["par_num"], word["line_num"])
        line_buckets.setdefault(key, []).append(word)

    lines = []
    for _, bucket in sorted(
        line_buckets.items(),
        key=lambda item: (
            min(w["y"] for w in item[1]),
            min(w["x"] for w in item[1]),
        ),
    ):
        bucket = sorted(bucket, key=lambda w: w["x"])
        line_text = " ".join(w["text"] for w in bucket).strip()
        if line_text:
            lines.append(line_text)

    full_text = clean_text("\n".join(lines))
    return words, lines, full_text


def score_ocr_result(words: list[dict], lines: list[str], text: str) -> float:
    if not words:
        return -1e9

    confidences = [w["conf"] for w in words if w["conf"] >= 0]
    mean_conf = mean(confidences) if confidences else 0.0

    tokens = [w["text"] for w in words]
    numeric_tokens = sum(1 for t in tokens if NUMERIC_TOKEN_RE.match(t))
    currency_hits = sum(1 for t in tokens if CURRENCY_HINT_RE.search(t))
    alpha_num_chars = len(re.findall(r"[A-Za-z0-9$%.,()/-]", text))

    long_lines = sum(1 for line in lines if len(line) >= 20)
    unique_tokens = len(set(t.lower() for t in tokens if len(t) > 1))

    score = (
        mean_conf * 3.5
        + numeric_tokens * 2.5
        + currency_hits * 2.0
        + long_lines * 1.5
        + unique_tokens * 0.2
        + alpha_num_chars * 0.02
    )
    return score


def run_best_ocr(image_variants: list[np.ndarray], configs: list[str] | None = None) -> dict:
    configs = configs or DEFAULT_TESSERACT_CONFIGS
    best = {
        "text": "",
        "lines": [],
        "words": [],
        "ocr_score": -1e9,
        "ocr_config": None,
        "image_variant": None,
        "mean_confidence": None,
    }

    for variant_idx, img in enumerate(image_variants):
        for config in configs:
            try:
                data = pytesseract.image_to_data(
                    img,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )
                words, lines, text = extract_words_and_lines_from_data(data)
                score = score_ocr_result(words, lines, text)

                confidences = [w["conf"] for w in words if w["conf"] >= 0]
                mean_conf = round(mean(confidences), 2) if confidences else None

                if score > best["ocr_score"]:
                    best = {
                        "text": text,
                        "lines": lines,
                        "words": words,
                        "ocr_score": round(score, 2),
                        "ocr_config": config,
                        "image_variant": variant_idx,
                        "mean_confidence": mean_conf,
                    }
            except Exception:
                continue

    return best


def extract_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_page_result(page_num: int, best_ocr: dict, image_shape: tuple[int, int]) -> dict:
    height, width = image_shape[:2]
    return {
        "page_num": page_num,
        "text": best_ocr.get("text", ""),
        "lines": best_ocr.get("lines", []),
        "words": best_ocr.get("words", []),
        "ocr_meta": {
            "ocr_score": best_ocr.get("ocr_score"),
            "mean_confidence": best_ocr.get("mean_confidence"),
            "ocr_config": best_ocr.get("ocr_config"),
            "image_variant": best_ocr.get("image_variant"),
            "image_width": width,
            "image_height": height,
        },
    }


def extract_from_image(image_path: str) -> dict:
    pil_img = Image.open(image_path).convert("RGB")
    img_bgr = pil_to_cv(pil_img)

    variants = preprocess_for_ocr(img_bgr)
    best_ocr = run_best_ocr(variants)

    return {
        "page_count": 1,
        "pages": [
            build_page_result(1, best_ocr, variants[0].shape)
        ],
    }


def extract_from_scanned_pdf(pdf_path: str, dpi: int = 300, poppler_path: str | None = None) -> dict:
    poppler_path = poppler_path or os.getenv("POPPLER_PATH")

    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        poppler_path=poppler_path,
    )

    pages = []
    for i, pil_img in enumerate(images, start=1):
        img_bgr = pil_to_cv(pil_img.convert("RGB"))
        variants = preprocess_for_ocr(img_bgr)
        best_ocr = run_best_ocr(variants)
        pages.append(build_page_result(i, best_ocr, variants[0].shape))

    return {
        "page_count": len(pages),
        "pages": pages,
    }