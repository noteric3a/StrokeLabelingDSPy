import ast
import re
from typing import Any, List
import pandas as pd
from config import ALLOWED_LABELS


LABEL_ALIASES = {
    "RIGHT MCA": "RMCA",
    "LEFT MCA": "LMCA",
    "RIGHT MIDDLE CEREBRAL": "RMCA",
    "LEFT MIDDLE CEREBRAL": "LMCA",

    "RIGHT ACA": "RACA",
    "LEFT ACA": "LACA",
    "RIGHT ANTERIOR CEREBRAL": "RACA",
    "LEFT ANTERIOR CEREBRAL": "LACA",

    "RIGHT PCA": "RPCA",
    "LEFT PCA": "LPCA",
    "RIGHT POSTERIOR CEREBRAL": "RPCA",
    "LEFT POSTERIOR CEREBRAL": "LPCA",

    "RIGHT PICA": "RPICA",
    "LEFT PICA": "LPICA",

    "BASILAR": "BA",
    "BASILAR ARTERY": "BA",

    "RIGHT VERTEBRAL": "RVA",
    "LEFT VERTEBRAL": "LVA",
    "RIGHT VERTEBRAL ARTERY": "RVA",
    "LEFT VERTEBRAL ARTERY": "LVA",

    "RIGHT ICA": "RICA",
    "LEFT ICA": "LICA",
    "RIGHT INTERNAL CAROTID": "RICA",
    "LEFT INTERNAL CAROTID": "LICA",

    "RIGHT COMMON CAROTID": "RCA",
    "LEFT COMMON CAROTID": "LCA",

    "NEGATIVE": "NONE",
    "NORMAL": "NONE",
    "NO ACUTE STROKE": "NONE",
}


def clean_report(value: Any) -> str:
    """Convert NaN/None into an empty string so the model does not see 'nan' as text."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _pieces_from_labels(labels: Any) -> List[str]:
    """
    Convert model output into raw label pieces.

    Supports:
    - ["RMCA", "LMCA"]
    - "RMCA, LMCA"
    - "['RMCA', 'LMCA']"
    - "labels: RMCA, LMCA"
    """

    if labels is None:
        return []

    if isinstance(labels, float) and pd.isna(labels):
        return []

    if isinstance(labels, list):
        return [str(x) for x in labels]

    if isinstance(labels, tuple) or isinstance(labels, set):
        return [str(x) for x in labels]

    text = str(labels).strip()

    if not text:
        return []

    # Handle stringified Python/JSON-style lists.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x) for x in parsed]
        except Exception:
            pass

    # First split by common separators.
    pieces = re.split(r"[,;\n/]+", text)

    # Also extract exact allowed labels from longer text.
    allowed_hits = []
    for label in ALLOWED_LABELS:
        if re.search(rf"\b{re.escape(label)}\b", text.upper()):
            allowed_hits.append(label)

    return pieces + allowed_hits


def normalize_labels(labels: Any) -> List[str]:
    """
    Normalize labels.

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

        # Remove common prefixes the model may include.
        label = re.sub(r"^(LABELS?|FINAL LABELS?|OUTPUT)\s*:\s*", "", label).strip()

        # Alias mapping.
        label = LABEL_ALIASES.get(label, label)

        # Remove spaces for official labels like "R MCA" if needed.
        compact = label.replace(" ", "")

        if compact in ALLOWED_LABELS:
            label = compact

        if label in ALLOWED_LABELS and label not in cleaned:
            cleaned.append(label)

    if not cleaned:
        return ["NONE"]

    if "NONE" in cleaned and len(cleaned) > 1:
        cleaned = [label for label in cleaned if label != "NONE"]

    return cleaned