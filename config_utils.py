import os

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

PLACEHOLDER_VALUES = {
    "your-google-project-id",
    "your-processor-id",
    "your-region",
}


def normalize_setting_value(value: object) -> str | None:
    normalized = str(value).strip()
    if not normalized or normalized in PLACEHOLDER_VALUES:
        return None
    return normalized


def read_setting(*keys: str) -> str | None:
    secret_value = read_streamlit_secret(*keys)
    if secret_value is not None:
        return secret_value

    for key in keys:
        value = os.getenv(key, "")
        normalized = normalize_setting_value(value)
        if normalized:
            return normalized

    return None


def read_streamlit_secret(*keys: str) -> str | None:
    if st is None:
        return None

    try:
        for key in keys:
            value = st.secrets.get(key)
            if value is None:
                continue

            normalized = normalize_setting_value(value)
            if normalized:
                return normalized
    except Exception:
        return None

    return None
