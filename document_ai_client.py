import os
from dataclasses import dataclass
from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from dotenv import load_dotenv


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
        "project_id": "DOCUMENT_AI_PROJECT_ID",
        "location": "DOCUMENT_AI_LOCATION",
        "processor_id": "DOCUMENT_AI_PROCESSOR_ID",
    }
    values = {field: os.getenv(env_var, "").strip() for field, env_var in env_mapping.items()}
    missing = [env_var for field, env_var in env_mapping.items() if not values[field]]

    if missing:
        raise EnvironmentError(
            "Missing required environment variables: " + ", ".join(sorted(missing))
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
