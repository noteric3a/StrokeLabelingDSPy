"""Confidence voting helpers for repeated LLM label runs.

The confidence score is a model-stability score: it measures how often the
same normalized label set wins across repeated samples. It is not a clinical
probability of correctness.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import config as cfg
from utils import normalize_labels


DEFAULT_CONFIDENCE_RUNS = 10
DEFAULT_CONFIDENCE_TEMPERATURE = 0.2
DEFAULT_MIN_CONFIDENCE_PERCENTAGE = 70.0
LOWEST_ALLOWED_MIN_CONFIDENCE_PERCENTAGE = 51.0

LABEL_ORDER = [
    "NONE",
    "RMCA", "LMCA",
    "RACA", "LACA",
    "RPCA", "LPCA",
    "RPICA", "LPICA",
    "BA",
    "RVA", "LVA",
    "RICA", "LICA",
]


def confidence_checking_enabled() -> bool:
    """Return True when config.py enables repeated-sample confidence checking."""
    return bool(getattr(cfg, "ENABLE_CONFIDENCE_CHECKING", False))


def confidence_run_count() -> int:
    """Return number of repeated model samples to run in confidence mode."""
    raw_value = getattr(cfg, "CONFIDENCE_RUNS", DEFAULT_CONFIDENCE_RUNS)
    try:
        runs = int(raw_value)
    except (TypeError, ValueError):
        runs = DEFAULT_CONFIDENCE_RUNS
    return max(1, runs)


def confidence_temperature() -> float:
    """Return model temperature used for confidence sampling."""
    raw_value = getattr(cfg, "CONFIDENCE_TEMPERATURE", DEFAULT_CONFIDENCE_TEMPERATURE)
    try:
        temperature = float(raw_value)
    except (TypeError, ValueError):
        temperature = DEFAULT_CONFIDENCE_TEMPERATURE
    return max(0.0, temperature)


def min_confidence_percentage() -> float:
    """Return minimum vote percentage needed to call a label confident.

    A blank config value falls back to DEFAULT_MIN_CONFIDENCE_PERCENTAGE. Values
    below 51 are raised to 51 so the winner must have at least a majority.
    """
    raw_value = getattr(cfg, "MIN_CONFIDENCE_PERCENTAGE", "")
    if raw_value is None or str(raw_value).strip() == "":
        threshold = DEFAULT_MIN_CONFIDENCE_PERCENTAGE
    else:
        try:
            threshold = float(raw_value)
        except (TypeError, ValueError):
            threshold = DEFAULT_MIN_CONFIDENCE_PERCENTAGE

    if threshold < LOWEST_ALLOWED_MIN_CONFIDENCE_PERCENTAGE:
        threshold = LOWEST_ALLOWED_MIN_CONFIDENCE_PERCENTAGE
    if threshold > 100.0:
        threshold = 100.0
    return threshold


def _label_sort_key(label: str) -> int:
    try:
        return LABEL_ORDER.index(label)
    except ValueError:
        return len(LABEL_ORDER) + 100


def _label_key(labels: Any) -> Tuple[str, ...]:
    normalized = normalize_labels(labels)
    return tuple(sorted(normalized, key=_label_sort_key))


def _extract_labels(result: Dict[str, Any], label_field: str) -> List[str]:
    if label_field in result:
        return normalize_labels(result.get(label_field))
    return normalize_labels(result.get("labels"))


def _labels_to_text(labels: Tuple[str, ...] | List[str]) -> str:
    """Return a compact human-readable label list."""
    return ", ".join(labels) if labels else "NONE"


def _reasoning_from_result(
    result: Dict[str, Any],
    reasoning_field: str = "reasoning",
) -> str:
    """Return the reasoning text from a model result, falling back safely."""
    return str(result.get(reasoning_field, result.get("reasoning", ""))).strip()


def _build_reasoning_union(
    successful_results: List[Tuple[int, Dict[str, Any]]],
    *,
    label_field: str,
    reasoning_field: str,
    winning_key: Tuple[str, ...],
    winning_count: int,
    total_votes: int,
    confidence: float,
    failed_runs: int,
) -> str:
    """Build a debug-only union of all confidence-sample reasonings.

    The normal *_reasoning field should stay canonical and concise.  This
    debug union is retained in the confidence summary for diagnostics, but it
    should not be used as the final reasoning text scanned by review checks.
    """
    winning_lines: List[str] = []
    alternate_lines: List[str] = []

    for run_number, result in successful_results:
        key = _label_key(_extract_labels(result, label_field))
        reasoning = _reasoning_from_result(result, reasoning_field)
        if not reasoning:
            reasoning = "No reasoning text was returned for this run."

        line = f"- Run {run_number} voted [{_labels_to_text(key)}]: {reasoning}"
        if key == winning_key:
            winning_lines.append(line)
        else:
            alternate_lines.append(line)

    header = (
        "Confidence reasoning union: "
        f"final labels [{_labels_to_text(winning_key)}] won "
        f"{winning_count}/{total_votes} successful confidence votes "
        f"({confidence}%)."
    )
    if failed_runs:
        header += f" {failed_runs} confidence run(s) failed and had no reasoning to include."

    sections = [header]

    if winning_lines:
        sections.append("Winning-label reasoning samples:\n" + "\n".join(winning_lines))
    else:
        sections.append(
            "Winning-label reasoning samples:\n"
            "- No reasoning text was returned for the winning label set."
        )

    if alternate_lines:
        sections.append("Alternate-label reasoning samples:\n" + "\n".join(alternate_lines))

    return "\n\n".join(sections)


def summarize_label_votes(
    results: Iterable[Any],
    *,
    label_field: str,
    reasoning_field: str = "reasoning",
) -> Dict[str, Any]:
    """Summarize repeated model outputs into a winning label and vote table.

    possible_answers is a JSON-friendly array of two-item arrays:
        [
          [["RMCA"], 80.0],
          [["RICA", "RMCA"], 20.0]
        ]

    winning_reasoning is the canonical reasoning from the first result that
    produced the winning label set.  The full confidence reasoning union is
    kept separately as reasoning_union for debugging, so deterministic review
    checks do not scan alternate or repetitive reasoning samples.
    """
    successful_results: List[Tuple[int, Dict[str, Any]]] = []
    failed_runs = 0

    for run_number, result in enumerate(results, start=1):
        if isinstance(result, dict):
            successful_results.append((run_number, result))
        else:
            failed_runs += 1

    if not successful_results:
        return {
            "final_label": ["NONE"],
            "possible_answers": [],
            "confidence_percentage": 0.0,
            "confidence_vote_count": 0,
            "confidence_total_votes": 0,
            "confidence_failed_runs": failed_runs,
            "confidence_threshold": min_confidence_percentage(),
            "is_confident": False,
            "winning_reasoning": "All confidence samples failed; defaulted to NONE.",
            "reasoning_union": "All confidence samples failed; defaulted to NONE.",
            "winning_result": {
                label_field: ["NONE"],
                reasoning_field: "All confidence samples failed; defaulted to NONE.",
            },
        }

    counts: Counter[Tuple[str, ...]] = Counter()
    first_seen_order: List[Tuple[str, ...]] = []
    first_result_for_key: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    for _, result in successful_results:
        key = _label_key(_extract_labels(result, label_field))
        if key not in counts:
            first_seen_order.append(key)
            first_result_for_key[key] = result
        counts[key] += 1

    def sort_key(item: Tuple[Tuple[str, ...], int]) -> Tuple[int, int]:
        key, count = item
        return (-count, first_seen_order.index(key))

    sorted_items = sorted(counts.items(), key=sort_key)
    winning_key, winning_count = sorted_items[0]

    total_votes = sum(counts.values())
    possible_answers = [
        [list(key), round((count / total_votes) * 100.0, 2)]
        for key, count in sorted_items
    ]

    confidence = round((winning_count / total_votes) * 100.0, 2)
    threshold = min_confidence_percentage()
    winning_result = first_result_for_key[winning_key]

    # Keep the actual final reasoning canonical: use one clean reasoning from
    # the first sample that produced the winning label set.  Do NOT place the
    # full confidence union in the normal reasoning field, because it contains
    # repeated and alternate-label explanations that can create false review
    # flags.
    winning_reasoning = _reasoning_from_result(winning_result, reasoning_field)
    if not winning_reasoning:
        winning_reasoning = (
            f"Final labels [{_labels_to_text(winning_key)}] won "
            f"{winning_count}/{total_votes} confidence votes ({confidence}%). "
            "No reasoning text was returned for the winning sample."
        )

    reasoning_union = _build_reasoning_union(
        successful_results,
        label_field=label_field,
        reasoning_field=reasoning_field,
        winning_key=winning_key,
        winning_count=winning_count,
        total_votes=total_votes,
        confidence=confidence,
        failed_runs=failed_runs,
    )

    return {
        "final_label": list(winning_key),
        "possible_answers": possible_answers,
        "confidence_percentage": confidence,
        "confidence_vote_count": winning_count,
        "confidence_total_votes": total_votes,
        "confidence_failed_runs": failed_runs,
        "confidence_threshold": threshold,
        "is_confident": confidence >= threshold,
        "winning_reasoning": winning_reasoning,
        "reasoning_union": reasoning_union,
        "winning_result": winning_result,
    }


def build_confidence_fields(prefix: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Build JSON fields for one label field's confidence summary."""
    return {
        f"{prefix}_final_label": summary.get("final_label", ["NONE"]),
        f"{prefix}_possible_answers": summary.get("possible_answers", []),
        f"{prefix}_confidence_percentage": summary.get("confidence_percentage", 0.0),
        f"{prefix}_is_confident": bool(summary.get("is_confident", False)),
        f"{prefix}_confidence_threshold": summary.get("confidence_threshold", min_confidence_percentage()),
        f"{prefix}_confidence_vote_count": summary.get("confidence_vote_count", 0),
        f"{prefix}_confidence_total_votes": summary.get("confidence_total_votes", 0),
        f"{prefix}_confidence_failed_runs": summary.get("confidence_failed_runs", 0),
    }


def confidence_fields_for_result(prefix: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Return confidence fields from a result dict that has _confidence_summary."""
    summary = result.get("_confidence_summary")
    if not isinstance(summary, dict):
        return {}
    return build_confidence_fields(prefix, summary)
