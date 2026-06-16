from __future__ import annotations

import ast
import re
from typing import Any, List

import pandas as pd

from Code import config as cfg


def clean_report(value: Any) -> str:
    """Convert NaN/None into an empty string so the model does not see 'nan' as text."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _pieces_from_labels(labels: Any) -> List[str]:
    """
    Convert model output into raw label pieces.

    Supports lists, comma-separated strings, stringified lists, and longer text
    that contains exact label names.
    """
    if labels is None:
        return []
    if isinstance(labels, float) and pd.isna(labels):
        return []
    if isinstance(labels, (list, tuple, set)):
        return [str(x) for x in labels]

    text = str(labels).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x) for x in parsed]
        except Exception:
            pass

    pieces = re.split(r"[,;\n/]+", text)

    allowed_hits = []
    upper_text = text.upper()
    for label in cfg.ALLOWED_LABELS:
        if re.search(rf"\b{re.escape(label)}\b", upper_text):
            allowed_hits.append(label)

    return pieces + allowed_hits


def normalize_labels(labels: Any) -> List[str]:
    """
    Normalize labels according to config.py.

    Guarantees:
    - never returns []
    - returns ["NONE"] when empty/invalid
    - removes NONE when mixed with real territory labels
    - deduplicates labels
    - accepts both list output and DSPy string output
    """
    cleaned: List[str] = []

    for piece in _pieces_from_labels(labels):
        label = str(piece).strip().upper()
        label = label.strip('"\'[](){}')
        label = re.sub(r"\s+", " ", label)
        label = re.sub(r"^(LABELS?|FINAL LABELS?|OUTPUT)\s*:\s*", "", label).strip()

        label = cfg.LABEL_ALIASES.get(label, label)

        compact = label.replace(" ", "")
        if compact in cfg.ALLOWED_LABELS:
            label = compact

        if label in cfg.ALLOWED_LABELS and label not in cleaned:
            cleaned.append(label)

    if not cleaned:
        return ["NONE"]

    if "NONE" in cleaned and len(cleaned) > 1:
        cleaned = [label for label in cleaned if label != "NONE"]

    return cleaned
