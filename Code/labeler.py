from __future__ import annotations
import asyncio
from collections import Counter
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Tuple
import config as cfg
from tqdm import tqdm

from dspy_programs import (
    StrokePrediction,
    initialize_dspy_programs,
    label_ct,
    label_cta,
    label_ctp,
    label_combined,
)

from review_checks import add_review_flags


# =============================================================================
# Optional semaphore helper
# =============================================================================

@asynccontextmanager
async def optional_semaphore(semaphore):
    """
    Use a semaphore if one is provided.

    Why this exists:
        You may want to limit how many reports run at the same time.

    Example:
        If max_concurrent = 4, only 4 case-labeling tasks run at once.

    If semaphore is None:
        This context manager does nothing.
    """

    if semaphore is None:
        yield
    else:
        async with semaphore:
            yield


# =============================================================================
# Confidence helpers
# =============================================================================

def _label_key(labels: List[str]) -> Tuple[str, ...]:
    """
    Convert a label list into a hashable sorted tuple.

    Counter needs hashable keys.

    Example:
        ["LMCA", "RMCA"] -> ("LMCA", "RMCA")
        ["RMCA", "LMCA"] -> ("LMCA", "RMCA")

    Sorting prevents the same label set from being counted separately just
    because the model returned the labels in a different order.
    """

    return tuple(sorted(str(label).strip().upper() for label in labels))


def _union_reasoning(final_key: Tuple[str, ...], attempts: List[StrokePrediction]) -> str:
    """
    Build final reasoning from repeated attempts.

    Your earlier pipeline wanted the final reasoning to be a union of the
    repeated runs in confidence mode.

    This function:
        1. Keeps reasoning from attempts that voted for the winning label.
        2. Adds alternate-label reasoning samples below a marker.
        3. Uses the exact marker your review checker knows how to ignore:

              Alternate-label reasoning samples:

    Why that marker matters:
        Your checker strips alternate-label reasoning before doing consistency
        checks so alternate votes do not accidentally trigger false flags.
    """

    winning_reasonings: List[str] = []
    alternate_reasonings: List[str] = []

    for result in attempts:
        current_key = _label_key(result.labels)
        text = result.reasoning.strip()

        if not text:
            continue

        if current_key == final_key:
            # Avoid repeating identical reasoning.
            if text not in winning_reasonings:
                winning_reasonings.append(text)
        else:
            # Keep a few alternate examples for debugging.
            alternate_text = f"Labels {list(current_key)}: {text}"
            if alternate_text not in alternate_reasonings:
                alternate_reasonings.append(alternate_text)

    # Keep only the first few reasoning samples so the spreadsheet does not
    # become enormous.
    final_text = "\n".join(f"- {r}" for r in winning_reasonings[:cfg.CONFIDENCE_REASONING_WINNING_LIMIT])

    if not final_text:
        final_text = "Winning label selected by repeated DSPy runs, but no reasoning was returned."

    if alternate_reasonings:
        final_text += "\n\nAlternate-label reasoning samples:\n"
        final_text += "\n".join(f"- {r}" for r in alternate_reasonings[:cfg.CONFIDENCE_REASONING_ALTERNATE_LIMIT])

    return final_text


async def run_prediction_with_confidence(
    sync_predict_fn: Callable[[], StrokePrediction],
) -> Dict[str, Any]:
    """
    Run a DSPy prediction once or multiple times.

    Args:
        sync_predict_fn:
            A normal synchronous function that returns StrokePrediction.

    Returns:
        A dictionary containing:
            labels
            reasoning
            is_confident
            confidence_percentage
            possible_answers

    Why asyncio.to_thread() is used:
        DSPy calls are synchronous/blocking. Wrapping them in asyncio.to_thread()
        lets the rest of your async pipeline keep working.
    """

    # Simple mode:
    # Run once and return 100% confidence.
    if not cfg.ENABLE_CONFIDENCE_CHECKING:
        result = await asyncio.to_thread(sync_predict_fn)

        return {
            "labels": result.labels,
            "reasoning": result.reasoning,
            "is_confident": True,
            "confidence_percentage": 100.0,
            "possible_answers": [(result.labels, 100.0)],
        }

    # Confidence mode:
    # Run the same prediction multiple times.
    attempts: List[StrokePrediction] = []

    for _ in range(cfg.CONFIDENCE_ATTEMPTS):
        result = await asyncio.to_thread(sync_predict_fn)
        attempts.append(result)

    # Count each unique label set.
    counts = Counter(_label_key(result.labels) for result in attempts)

    # Pick the most common label set.
    final_key, final_count = counts.most_common(1)[0]

    # Compute vote percentage.
    confidence_percentage = final_count / len(attempts) * 100

    # Apply your threshold.
    is_confident = confidence_percentage >= cfg.CONFIDENCE_THRESHOLD_PERCENTAGE

    # Store all possible answers for debugging.
    possible_answers = [
        (list(key), count / len(attempts) * 100)
        for key, count in counts.most_common()
    ]

    return {
        "labels": list(final_key),
        "reasoning": _union_reasoning(final_key, attempts),
        "is_confident": is_confident,
        "confidence_percentage": confidence_percentage,
        "possible_answers": possible_answers,
    }


def _apply_result(case: Dict[str, Any], prefix: str, result: Dict[str, Any]) -> None:
    """
    Write a modality result into the case dictionary.

    Example:
        prefix = "CT_GT"

    Writes:
        case["CT_GT"]
        case["CT_GT_reasoning"]
        case["CT_GT_is_confident"]
        case["CT_GT_confidence_percentage"]
        case["CT_GT_possible_answers"]

    This keeps the output format close to your manual pipeline.
    """

    case[prefix] = result["labels"]
    case[f"{prefix}_reasoning"] = result["reasoning"]
    case[f"{prefix}_is_confident"] = result["is_confident"]
    case[f"{prefix}_confidence_percentage"] = result["confidence_percentage"]
    case[f"{prefix}_possible_answers"] = result["possible_answers"]


# =============================================================================
# Individual modality labeling functions
# =============================================================================

async def label_ct_for_case(case: Dict[str, Any], semaphore=None) -> Dict[str, Any]:
    """
    Label the CT report for one case.

    This prefers New_CT_Report when it exists because your old pipeline used
    sanitized CT reports to remove CTA/CTP contamination.
    """

    async with optional_semaphore(semaphore):
        report = str(case.get("New_CT_Report") or case.get("CT_Report") or "")

        result = await run_prediction_with_confidence(
            lambda: label_ct(report)
        )

        _apply_result(case, "CT_GT", result)
        return case


async def label_cta_for_case(case: Dict[str, Any], semaphore=None) -> Dict[str, Any]:
    """
    Label the CTA report for one case.
    """

    async with optional_semaphore(semaphore):
        report = str(case.get("CTA_Report") or "")

        result = await run_prediction_with_confidence(
            lambda: label_cta(report)
        )

        _apply_result(case, "CTA_GT", result)
        return case


async def label_ctp_for_case(case: Dict[str, Any], semaphore=None) -> Dict[str, Any]:
    """
    Label the CTP report for one case.
    """

    async with optional_semaphore(semaphore):
        report = str(case.get("CTP_Report") or "")

        result = await run_prediction_with_confidence(
            lambda: label_ctp(report)
        )

        _apply_result(case, "CTP_GT", result)
        return case


async def label_combined_for_case(case: Dict[str, Any], semaphore=None) -> Dict[str, Any]:
    """
    Label the final Combined_GT for one case.

    Important:
        This should run after CT, CTA, and CTP have already been labeled.

    Combined uses:
        - CT report
        - CTA report
        - CTP report
        - MRI report if present
        - CT_GT / CTA_GT / CTP_GT
        - modality reasoning
    """

    async with optional_semaphore(semaphore):
        result = await run_prediction_with_confidence(
            lambda: label_combined(case)
        )

        # Combined is slightly special because your checker expects the
        # reasoning field to be named CT_Combined_GT_reasoning.
        case["Combined_GT"] = result["labels"]
        case["CT_Combined_GT_reasoning"] = result["reasoning"]
        case["Combined_GT_is_confident"] = result["is_confident"]
        case["Combined_GT_confidence_percentage"] = result["confidence_percentage"]
        case["Combined_GT_possible_answers"] = result["possible_answers"]

        return case


# =============================================================================
# Full case labeling
# =============================================================================

async def label_full_case(case: Dict[str, Any], semaphore=None) -> Dict[str, Any]:
    """
    Label one full case using the DSPy pipeline.

    Order:
        1. CT
        2. CTA
        3. CTP
        4. Combined
        5. Review flags

    This mirrors your manual pipeline structure.
    """

    await label_ct_for_case(case, semaphore)
    await label_cta_for_case(case, semaphore)
    await label_ctp_for_case(case, semaphore)
    await label_combined_for_case(case, semaphore)

    # Run your deterministic checker after labels/reasoning are present.
    case = add_review_flags(case)

    return case


async def label_cases(
    cases: List[Dict[str, Any]],
    max_concurrent: int = cfg.MAX_CONCURRENT_CASES,
) -> List[Dict[str, Any]]:
    """
    Label all cases with a single-line dynamic progress bar.

    The progress bar updates in place and includes the current date/time beside
    the bar.  Because this pipeline runs cases concurrently, results are
    returned as each case finishes, not necessarily in the original input order.
    """
    initialize_dspy_programs()

    total_cases = len(cases)
    if total_cases == 0:
        return []

    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        asyncio.create_task(label_full_case(case, semaphore))
        for case in cases
    ]

    results: List[Dict[str, Any]] = []

    def current_timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    progress_bar = tqdm(
        total=total_cases,
        desc="Labeling cases",
        unit="case",
        dynamic_ncols=True,
        leave=True,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] | {postfix}",
    )

    # Initial date/time display before the first case completes.
    progress_bar.set_postfix_str(f"now={current_timestamp()}", refresh=True)

    async def refresh_clock() -> None:
        """
        Refresh the date/time while cases are running.

        This keeps the timestamp live even during long model calls.  It updates
        the same tqdm line instead of printing new lines.
        """
        try:
            while progress_bar.n < total_cases:
                progress_bar.set_postfix_str(f"now={current_timestamp()}", refresh=True)
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            return

    clock_task = asyncio.create_task(refresh_clock())

    try:
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            results.append(result)
            progress_bar.update(1)
            progress_bar.set_postfix_str(f"now={current_timestamp()}", refresh=True)
    finally:
        clock_task.cancel()
        try:
            await clock_task
        except asyncio.CancelledError:
            pass
        progress_bar.set_postfix_str(f"finished={current_timestamp()}", refresh=True)
        progress_bar.close()

    return results

