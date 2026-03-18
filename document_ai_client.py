from dataclasses import dataclass
from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from dotenv import load_dotenv

from config_utils import read_setting

ENV_FILE = Path(__file__).resolve().with_name(".env")


def load_environment() -> None:
    load_dotenv(dotenv_path=ENV_FILE, override=False)


@dataclass(frozen=True)
class DocumentAISettings:
    project_id: str
    location: str
    processor_id: str

    @property
    def processor_name(self) -> str:
        return (
            f"projects/{self.project_id}/locations/{self.location}/processors/{self.processor_id}"
        )

    @property
    def api_endpoint(self) -> str | None:
        if self.location.lower() == "global":
            return None
        return f"{self.location}-documentai.googleapis.com"


def load_settings() -> DocumentAISettings:
    load_environment()

    env_mapping = {
        "project_id": ("PROJECT_ID", "DOCUMENT_AI_PROJECT_ID"),
        "location": ("REGION", "DOCUMENT_AI_LOCATION"),
        "processor_id": ("PROCESSOR_ID", "DOCUMENT_AI_PROCESSOR_ID"),
    }
    values = {field: read_setting(*keys) or "" for field, keys in env_mapping.items()}
    missing = [
        "PROJECT_ID or DOCUMENT_AI_PROJECT_ID"
        for field in ("project_id",)
        if not values[field]
    ]
    if not values["location"]:
        missing.append("REGION or DOCUMENT_AI_LOCATION")
    if not values["processor_id"]:
        missing.append("PROCESSOR_ID or DOCUMENT_AI_PROCESSOR_ID")

    if missing:
        raise EnvironmentError(
            "Missing required Streamlit secrets or environment variables: " + ", ".join(missing)
        )

    return DocumentAISettings(**values)


class DocumentAIClient:
    def __init__(self, settings: DocumentAISettings | None = None) -> None:
        self.settings = settings or load_settings()

        client_options = None
        if self.settings.api_endpoint:
            client_options = ClientOptions(api_endpoint=self.settings.api_endpoint)

        self._client = documentai.DocumentProcessorServiceClient(client_options=client_options)

    def process_document(self, file_bytes: bytes, mime_type: str) -> documentai.Document:
        raw_document = documentai.RawDocument(content=file_bytes, mime_type=mime_type)
        request = documentai.ProcessRequest(
            name=self.settings.processor_name,
            raw_document=raw_document,
        )
        result = self._client.process_document(request=request)
        return result.document
