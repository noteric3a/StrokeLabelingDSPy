"""
Small helper functions shared across the project.
"""

from typing import Any, List
import pandas as pd
from config import ALLOWED_LABELS


def clean_report(value: Any) -> str:
    """Convert NaN/None into an empty string so the model does not see 'nan' as text."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_labels(labels: Any) -> List[str]:
    """
    Guarantee:
    - never []
    - ["NONE"] when empty/invalid
    - no NONE mixed with territory labels
    - deduplicated, allowed labels only
    """
    if not isinstance(labels, list):
        return ["NONE"]

    cleaned: List[str] = []
    for label in labels:
        label = str(label).strip().upper()
        if label in ALLOWED_LABELS and label not in cleaned:
            cleaned.append(label)

    if not cleaned:
        return ["NONE"]

    if "NONE" in cleaned and len(cleaned) > 1:
        cleaned = [label for label in cleaned if label != "NONE"]

    return cleaned
