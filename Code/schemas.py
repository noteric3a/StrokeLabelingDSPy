"""
JSON schemas passed to Ollama structured output.
"""

from typing import Any, Dict
from config import ALLOWED_LABELS


def label_array_schema() -> Dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "enum": ALLOWED_LABELS},
        "minItems": 1,
        "maxItems": len(ALLOWED_LABELS) - 1,  # Allow all labels except "NONE"
    }


SINGLE_MODALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "modality": {"type": "string", "enum": ["CT", "CTA", "CTP"]},
        "labels": label_array_schema(),
        "reasoning": {"type": "string"},
    },
    "required": ["case_id", "modality", "labels", "reasoning"],
}


CT_SANITIZATION_SCHEMA = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "contamination_found": {"type": "boolean"},
        "sanitized_report": {"type": "string"},
        "removed_sections": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning": {"type": "string"},
    },
    "required": [
        "case_id",
        "contamination_found",
        "sanitized_report",
        "removed_sections",
        "reasoning",
    ],
}


COMBINED_SCHEMA = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "Combined_GT": label_array_schema(),
        "reasoning": {"type": "string"},
    },
    "required": ["case_id", "Combined_GT", "reasoning"],
}
