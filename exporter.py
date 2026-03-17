import json
from typing import Any

from google.cloud import documentai
from google.protobuf.json_format import MessageToDict


def document_to_dict(document: documentai.Document) -> dict[str, Any]:
    return MessageToDict(document._pb, preserving_proto_field_name=True)


def document_to_json(document: documentai.Document, indent: int = 2) -> str:
    return json.dumps(document_to_dict(document), indent=indent, ensure_ascii=False)
