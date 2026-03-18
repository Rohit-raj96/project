import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

from requests import RequestsDependencyWarning

warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\..*")

import pandas as pd
import streamlit as st

from document_ai_client import DocumentAIClient, DocumentAISettings, load_settings
from exporter import document_to_json
from main import get_runtime_settings, process_uploaded_file as process_local_upload, serialize_result
from parser import extract_form_fields, extract_full_text, extract_tables

SUPPORTED_EXTENSIONS = ("pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff")
MIME_TYPE_BY_EXTENSION = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}
LOCAL_MODE = "Local OCR (Free)"
CLOUD_MODE = "Google Document AI (Cloud)"


@st.cache_resource(show_spinner=False)
def get_document_ai_client() -> DocumentAIClient:
    return DocumentAIClient()


def detect_mime_type(file_name: str, streamlit_mime_type: str | None) -> str:
    if streamlit_mime_type:
        return streamlit_mime_type
    return MIME_TYPE_BY_EXTENSION.get(Path(file_name).suffix.lower(), "application/octet-stream")


def build_download_name(file_name: str, suffix: str) -> str:
    return f"{Path(file_name).stem}{suffix}"


def build_upload_token(file_name: str, file_bytes: bytes) -> str:
    digest = hashlib.sha1(file_bytes).hexdigest()[:12]
    return f"{Path(file_name).name}:{digest}"


def collect_local_full_text(result: dict[str, Any]) -> str:
    pages = (result.get("raw_text", {}) or {}).get("pages", [])
    return "\n\n".join(page.get("text", "") for page in pages if page.get("text"))


def normalize_table_rows(rows: list[list[Any]], width: int) -> list[list[str]]:
    normalized = []
    for row in rows:
        values = [str(cell or "") for cell in row[:width]]
        if len(values) < width:
            values.extend([""] * (width - len(values)))
        normalized.append(values)
    return normalized


def table_to_dataframe(table: dict[str, Any]) -> pd.DataFrame:
    rows = table.get("rows", []) or []
    width = max((len(row) for row in rows), default=0)

    headers = [str(header or "").strip() for header in table.get("headers", []) or []]
    if headers and len(headers) == width and any(headers):
        columns = [header or f"Column {index}" for index, header in enumerate(headers, start=1)]
    else:
        columns = [f"Column {index}" for index in range(1, width + 1)]

    return pd.DataFrame(normalize_table_rows(rows, width), columns=columns)


def parse_json_editor(value: str) -> tuple[bool, str, str | None]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        return False, value, str(exc)

    normalized = json.dumps(parsed, indent=2, ensure_ascii=False)
    return True, normalized, None


def reset_editor(state_key: str, original_value: str) -> None:
    st.session_state[state_key] = original_value


def render_json_editor(state_key: str, original_key: str, file_name: str, text_payload: str | None = None) -> None:
    original_json = st.session_state[original_key]
    if state_key not in st.session_state:
        st.session_state[state_key] = original_json

    st.button(
        "Reset editor to original extraction",
        on_click=reset_editor,
        args=(state_key, original_json),
    )
    st.text_area(
        "Editable JSON output",
        key=state_key,
        height=520,
    )

    is_valid, normalized_json, error_message = parse_json_editor(st.session_state[state_key])
    if is_valid:
        st.success("JSON is valid and ready to download.")
    else:
        st.error(f"JSON editor contains an error: {error_message}")

    download_col, text_col = st.columns(2)
    download_col.download_button(
        label="Download edited JSON",
        data=normalized_json if is_valid else "",
        file_name=build_download_name(file_name, "_edited.json"),
        mime="application/json",
        disabled=not is_valid,
    )

    if text_payload:
        text_col.download_button(
            label="Download extracted text",
            data=text_payload,
            file_name=build_download_name(file_name, "_text.txt"),
            mime="text/plain",
        )


def render_runtime_status() -> None:
    runtime_settings = get_runtime_settings()

    st.sidebar.header("Pipeline")
    st.sidebar.caption("Choose the free local OCR flow or the optional cloud pipeline.")

    st.sidebar.subheader("Local OCR Runtime")
    if runtime_settings["tesseract_available"]:
        source = runtime_settings["tesseract_cmd"] or runtime_settings["tesseract_on_path"]
        st.sidebar.success(f"Tesseract available: `{source}`")
    else:
        st.sidebar.warning("Tesseract is not configured. Images and scanned PDFs will fail until it is installed.")

    poppler_path = runtime_settings["poppler_path"]
    if poppler_path:
        st.sidebar.write(f"Poppler path: `{poppler_path}`")
        st.sidebar.caption("Poppler is used only when a scanned PDF page must be converted into an image before OCR.")
    else:
        st.sidebar.info("Poppler is only needed for scanned PDFs or scanned pages inside mixed PDFs. Digital text PDFs do not use it.")

    st.sidebar.caption("Scanned PDFs are slower in local mode because each scanned page is converted to an image and OCR is tried multiple ways.")

    if runtime_settings["env_file"]:
        st.sidebar.write(f"Loaded `.env`: `{runtime_settings['env_file']}`")


def render_document_ai_status() -> None:
    st.sidebar.subheader("Document AI Runtime")
    st.sidebar.warning("Cloud mode can incur Google Cloud charges. Keep it optional.")

    try:
        settings = load_settings()
    except EnvironmentError as exc:
        st.sidebar.error(str(exc))
        st.sidebar.code(
            "\n".join(
                [
                    "Required cloud settings:",
                    "PROJECT_ID or DOCUMENT_AI_PROJECT_ID",
                    "REGION or DOCUMENT_AI_LOCATION",
                    "PROCESSOR_ID or DOCUMENT_AI_PROCESSOR_ID",
                    "",
                    "Authentication:",
                    "Google ADC or credentials configured for the runtime",
                ]
            )
        )
        return

    render_loaded_settings(settings)


def render_loaded_settings(settings: DocumentAISettings) -> None:
    st.sidebar.success("Document AI environment loaded.")
    st.sidebar.write(f"Project: `{settings.project_id}`")
    st.sidebar.write(f"Location: `{settings.location}`")
    st.sidebar.write(f"Processor: `{settings.processor_id}`")


def process_local_mode(uploaded_file: Any) -> None:
    file_bytes = uploaded_file.getvalue()
    with st.spinner("Running the local OCR pipeline..."):
        result = process_local_upload(uploaded_file.name, file_bytes)

    original_json = serialize_result(result)
    st.session_state["local_result"] = result
    st.session_state["local_file_name"] = uploaded_file.name
    st.session_state["local_result_token"] = build_upload_token(uploaded_file.name, file_bytes)
    st.session_state["local_original_json"] = original_json
    st.session_state["local_editable_json"] = original_json


def process_cloud_mode(uploaded_file: Any) -> None:
    client = get_document_ai_client()
    file_bytes = uploaded_file.getvalue()
    mime_type = detect_mime_type(uploaded_file.name, uploaded_file.type)

    with st.spinner("Sending document to Google Document AI..."):
        document = client.process_document(file_bytes, mime_type)

    raw_json = document_to_json(document)
    st.session_state["cloud_result"] = {
        "file_name": uploaded_file.name,
        "full_text": extract_full_text(document),
        "tables": extract_tables(document),
        "form_fields": extract_form_fields(document),
        "raw_json": raw_json,
    }
    st.session_state["cloud_result_token"] = build_upload_token(uploaded_file.name, file_bytes)
    st.session_state["cloud_original_json"] = raw_json
    st.session_state["cloud_editable_json"] = raw_json


def render_local_overview(result: dict[str, Any]) -> None:
    normalized_fields = result.get("normalized_fields", {}) or {}
    merchant = normalized_fields.get("merchant", {}) or {}
    statement_meta = {
        "statement_date": normalized_fields.get("statement_date"),
        "statement_period": normalized_fields.get("statement_period"),
    }
    summary = normalized_fields.get("summary", {}) or {}
    deposits = normalized_fields.get("deposits", []) or []
    fees = normalized_fields.get("fees", []) or []
    validation = normalized_fields.get("validation", {}) or {}
    review_flags = normalized_fields.get("review_flags", []) or []

    st.subheader("Detector and template")
    st.json(
        {
            "detected": result.get("detected", {}),
            "template": result.get("template", {}),
        }
    )

    st.subheader("Normalized fields")
    summary_col, merchant_col = st.columns(2)
    summary_col.write("Statement")
    summary_col.json(statement_meta)
    merchant_col.write("Merchant")
    merchant_col.json(merchant)

    if summary:
        st.write("Summary")
        st.dataframe(pd.DataFrame([summary]), width="stretch")

    if deposits:
        st.write("Deposits")
        st.dataframe(pd.DataFrame(deposits), width="stretch")

    if fees:
        st.write("Fees")
        st.dataframe(pd.DataFrame(fees), width="stretch")

    if validation:
        st.write("Validation")
        st.json(validation)

    if review_flags:
        st.write("Review flags")
        st.dataframe(pd.DataFrame(review_flags), width="stretch")


def render_local_pages(result: dict[str, Any], result_token: str) -> None:
    pages = (result.get("raw_text", {}) or {}).get("pages", [])
    if not pages:
        st.info("No page-level text was extracted.")
        return

    for page in pages:
        page_number = page.get("page_num", "?")
        with st.expander(f"Page {page_number}", expanded=page_number == 1):
            st.text_area(
                f"Extracted text for page {page_number}",
                value=page.get("text", ""),
                height=260,
                disabled=True,
                key=f"local_page_{result_token}_{page_number}",
            )


def render_local_tables(result: dict[str, Any]) -> None:
    classified_tables = result.get("classified_tables", []) or []
    raw_tables = result.get("raw_tables", []) or []

    classified_tab, raw_tab = st.tabs(["Classified Tables", "Raw Tables"])

    with classified_tab:
        if not classified_tables:
            st.info("No classified tables were produced.")
        for index, table in enumerate(classified_tables, start=1):
            label = (
                f"Table {index} - page {table.get('page', '?')} - "
                f"{table.get('section_label', table.get('table_type', 'unknown'))}"
            )
            with st.expander(label, expanded=index == 1):
                st.json(
                    {
                        "table_type": table.get("table_type"),
                        "section_label": table.get("section_label"),
                        "classification_confidence": table.get("classification_confidence"),
                        "classification_reason": table.get("classification_reason"),
                        "quality_score": table.get("quality_score"),
                        "shape": table.get("shape"),
                    }
                )
                dataframe = table_to_dataframe(table)
                if dataframe.empty:
                    st.info("This table has no row data.")
                else:
                    st.dataframe(dataframe, width="stretch")

    with raw_tab:
        if not raw_tables:
            st.info("No raw tables were produced.")
        for index, table in enumerate(raw_tables, start=1):
            with st.expander(f"Raw table {index} - page {table.get('page', '?')}", expanded=index == 1):
                st.json(
                    {
                        "source": table.get("source"),
                        "flavor": table.get("flavor"),
                        "accuracy": table.get("accuracy"),
                        "quality_score": table.get("quality_score"),
                        "shape": table.get("shape"),
                    }
                )
                dataframe = table_to_dataframe(table)
                if dataframe.empty:
                    st.info("This table has no row data.")
                else:
                    st.dataframe(dataframe, width="stretch")


def render_local_results() -> None:
    result = st.session_state.get("local_result")
    if not result:
        return

    file_name = st.session_state["local_file_name"]
    result_token = st.session_state.get("local_result_token", file_name)
    st.success(f"Local OCR processing completed for `{file_name}`.")

    metrics = st.columns(4)
    metrics[0].metric("Status", str(result.get("status", "unknown")))
    metrics[1].metric("File type", str(result.get("file_type", "unknown")))
    metrics[2].metric("Pages", int((result.get("raw_text", {}) or {}).get("page_count", 0)))
    metrics[3].metric("Tables", len(result.get("classified_tables", []) or []))

    overview_tab, pages_tab, tables_tab, edit_tab = st.tabs(
        ["Overview", "Pages", "Tables", "Edit & Download"]
    )

    with overview_tab:
        render_local_overview(result)

    with pages_tab:
        render_local_pages(result, result_token)

    with tables_tab:
        render_local_tables(result)

    with edit_tab:
        render_json_editor(
            state_key="local_editable_json",
            original_key="local_original_json",
            file_name=file_name,
            text_payload=collect_local_full_text(result),
        )


def render_cloud_tables(tables: list[dict[str, Any]]) -> None:
    if not tables:
        st.info("No tables were detected in the document.")
        return

    for table in tables:
        st.subheader(f"Page {table['page_number']} - Table {table['table_number']}")
        dataframe = pd.DataFrame(table["rows"], columns=table["columns"])
        st.dataframe(dataframe, width="stretch")


def render_cloud_form_fields(form_fields: list[dict[str, Any]]) -> None:
    if not form_fields:
        st.info("No form fields were detected in the document.")
        return

    st.dataframe(pd.DataFrame(form_fields), width="stretch")


def render_cloud_results() -> None:
    result = st.session_state.get("cloud_result")
    if not result:
        return

    file_name = result["file_name"]
    st.success(f"Google Document AI processing completed for `{file_name}`.")

    metrics = st.columns(4)
    metrics[0].metric("Text length", len(result["full_text"]))
    metrics[1].metric("Tables", len(result["tables"]))
    metrics[2].metric("Form fields", len(result["form_fields"]))
    metrics[3].metric("Mode", "Cloud")

    text_tab, tables_tab, fields_tab, edit_tab = st.tabs(
        ["Full Text", "Tables", "Form Fields", "Edit & Download"]
    )

    with text_tab:
        st.text_area("Extracted full text", value=result["full_text"], height=420, disabled=True)

    with tables_tab:
        render_cloud_tables(result["tables"])

    with fields_tab:
        render_cloud_form_fields(result["form_fields"])

    with edit_tab:
        render_json_editor(
            state_key="cloud_editable_json",
            original_key="cloud_original_json",
            file_name=file_name,
            text_payload=result["full_text"],
        )


def main() -> None:
    st.set_page_config(page_title="Financial Statement Extractor", layout="wide")
    st.title("Financial Statement Extractor")
    st.caption(
        "One UI, two pipelines: run the free local OCR extractor on your machine or switch "
        "to Google Document AI when cloud processing is appropriate."
    )

    render_runtime_status()

    mode = st.sidebar.radio("Execution mode", [LOCAL_MODE, CLOUD_MODE], index=0)
    if mode == CLOUD_MODE:
        render_document_ai_status()

    uploaded_file = st.file_uploader(
        "Upload a financial statement (PDF, PNG, JPG, BMP, or TIFF)",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        st.write(f"Selected file: `{uploaded_file.name}`")
        button_label = "Run local OCR pipeline" if mode == LOCAL_MODE else "Run Google Document AI pipeline"
        if st.button(button_label, type="primary"):
            try:
                if mode == LOCAL_MODE:
                    process_local_mode(uploaded_file)
                else:
                    process_cloud_mode(uploaded_file)
            except Exception as exc:
                st.exception(exc)

    if mode == LOCAL_MODE:
        render_local_results()
    else:
        render_cloud_results()


if __name__ == "__main__":
    main()
