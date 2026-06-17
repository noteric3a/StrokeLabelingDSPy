"""
JSON schemas passed to Ollama structured output.
"""

from typing import Any, Dict

import config as cfg
from config import ALLOWED_LABELS


def label_array_schema() -> Dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "enum": ALLOWED_LABELS},
        "minItems": 1,
        "maxItems": len(ALLOWED_LABELS) - 1,  # Allow all labels except "NONE"
    }


def _generate_label_reasoning() -> bool:
    """Return whether label prompts should ask Ollama to generate reasoning."""
    return bool(getattr(cfg, "GENERATE_LABEL_REASONING", False))


def _generate_sanitizer_reasoning() -> bool:
    """Return whether the CT sanitizer should ask Ollama to generate reasoning."""
    return bool(getattr(cfg, "GENERATE_SANITIZER_REASONING", False))


def _single_modality_schema() -> Dict[str, Any]:
    properties: Dict[str, Any] = {
        "case_id": {"type": "string"},
        "modality": {"type": "string", "enum": ["CT", "CTA", "CTP"]},
        "labels": label_array_schema(),
    }
    required = ["case_id", "modality", "labels"]
    if _generate_label_reasoning():
        properties["reasoning"] = {"type": "string"}
        required.append("reasoning")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _ct_sanitization_schema() -> Dict[str, Any]:
    properties: Dict[str, Any] = {
        "case_id": {"type": "string"},
        "contamination_found": {"type": "boolean"},
        "sanitized_report": {"type": "string"},
        "removed_sections": {
            "type": "array",
            "items": {"type": "string"},
        },
    }
    required = [
        "case_id",
        "contamination_found",
        "sanitized_report",
        "removed_sections",
    ]
    if _generate_sanitizer_reasoning():
        properties["reasoning"] = {"type": "string"}
        required.append("reasoning")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _combined_schema() -> Dict[str, Any]:
    properties: Dict[str, Any] = {
        "case_id": {"type": "string"},
        "Combined_GT": label_array_schema(),
    }
    required = ["case_id", "Combined_GT"]
    if _generate_label_reasoning():
        properties["reasoning"] = {"type": "string"}
        required.append("reasoning")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


SINGLE_MODALITY_SCHEMA = _single_modality_schema()
CT_SANITIZATION_SCHEMA = _ct_sanitization_schema()
COMBINED_SCHEMA = _combined_schema()
