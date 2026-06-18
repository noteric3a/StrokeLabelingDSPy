from __future__ import annotations
import asyncio
import json
import re
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
from schemas import CT_SANITIZATION_SCHEMA
from ollama_client import ollama_generate_sync


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
# CT sanitization helpers
# =============================================================================

CT_CONTAMINATION_KEYWORDS = (
    "cta",
    "ct angiogram",
    "ct angiography",
    "angiographic",
    "arterial phase",
    "mip",
    "3d reconstructed",
    "3-d reconstruction",
    "3d reconstruction",
    "vessel postprocessing",
    "ctp",
    "ct perfusion",
    "perfusion",
    "tmax",
    "cbf",
    "cbv",
    "mtt",
    "mismatch",
    "hypoperfusion",
    "penumbra",
    "infarct core",
    "core volume",
    "tissue at risk",
    "brain at risk",
    "rapid",
)


def clean_report_text(value: Any) -> str:
    """Convert None/NaN-like values into safe report text."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def ct_report_may_contain_cta_ctp(report: str) -> bool:
    """Cheap pre-screen so the sanitizer LLM only runs on suspicious CT reports."""
    text = (report or "").lower()
    return any(keyword in text for keyword in CT_CONTAMINATION_KEYWORDS)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}
    return bool(value)


def default_ct_sanitization_result(case_id: str, ct_report: str) -> Dict[str, Any]:
    """Result used when the report does not look contaminated."""
    return {
        "case_id": case_id,
        "contamination_found": False,
        "sanitized_report": ct_report,
        "removed_sections": [],
        "reasoning": "No CTA/CTP contamination keywords were detected, so the original CT report was used.",
    }


def safe_ct_sanitization_result(
    result: Any,
    case_id: str,
    original_ct_report: str,
) -> Dict[str, Any]:
    """Convert sanitizer failures or malformed outputs into a safe fallback."""
    if isinstance(result, Exception):
        return {
            "case_id": case_id,
            "contamination_found": False,
            "sanitized_report": original_ct_report,
            "removed_sections": [],
            "reasoning": f"FAILED CT sanitization: {repr(result)}. Original CT report was used.",
        }

    if not isinstance(result, dict):
        return {
            "case_id": case_id,
            "contamination_found": False,
            "sanitized_report": original_ct_report,
            "removed_sections": [],
            "reasoning": "Malformed CT sanitization result. Original CT report was used.",
        }

    sanitized_report = clean_report_text(result.get("sanitized_report", ""))
    if not sanitized_report:
        sanitized_report = original_ct_report

    removed_sections = result.get("removed_sections", [])
    if not isinstance(removed_sections, list):
        removed_sections = [str(removed_sections)] if removed_sections else []

    return {
        "case_id": str(result.get("case_id", case_id)),
        "contamination_found": _boolish(result.get("contamination_found", False)),
        "sanitized_report": sanitized_report,
        "removed_sections": [str(item) for item in removed_sections],
        "reasoning": clean_report_text(result.get("reasoning", "")),
    }


def build_ct_sanitization_prompt(case_id: str, ct_report: str) -> str:
    """Build the same CT sanitization prompt used in the earlier non-DSPy pipeline."""
    return f"""
You are cleaning a report that is supposed to be ONLY a non-contrast CT head/brain report.

Your job:
1. Detect whether the supplied CT report contains CTA/CT angiogram or CTP/perfusion findings mixed into it.
2. If contamination is present, remove ONLY the CTA/CTP/perfusion/angiographic findings and return a sanitized CT-only report.
3. If no contamination is present, return the original CT report exactly as the sanitized_report.

What counts as CTA/CTP contamination:
- Sentences or sections labeled CTA, CT ANGIOGRAM, CT angiography, angiographic, arterial phase, vessel postprocessing, MIP, 3D reconstruction, circle of Willis vessel evaluation, head/neck vessel evaluation.
- Vessel-only CTA findings such as occlusion, stenosis, thrombus, filling defect, flow cutoff, absent opacification, reconstitution, collateral flow, or delayed filling when they are clearly from CTA/angiography rather than a noncontrast CT hyperdense vessel sign.
- CT perfusion findings such as CTP, perfusion, Tmax, CBF, CBV, MTT, mismatch, core volume, hypoperfusion volume, penumbra, tissue at risk, RAPID, or perfusion maps.
- Impressions/recommendations that summarize CTA/CTP findings rather than CT findings.

What must be preserved:
- Noncontrast CT-visible findings, including hemorrhage, mass effect, midline shift, edema, hypodensity, loss of gray-white differentiation, ASPECTS, hyperdense MCA/vessel sign, infarct seen on CT, chronic infarcts, encephalomalacia, lacunar infarcts, atrophy, microvascular disease, hydrocephalus, and postoperative CT findings.
- CT report structure when possible: EXAMINATION, COMPARISON, TECHNIQUE, FINDINGS, IMPRESSION.
- Do not change wording of preserved CT-only findings except for minimal cleanup needed after removing contaminated sentences.
- Do not add new medical facts.

Important edge cases:
- Keep a noncontrast CT phrase like "hyperdense left MCA sign" because that is CT-visible.
- Remove a CTA phrase like "left M1 occlusion on CTA" because that is angiographic.
- Remove CTP values like "Tmax >6 seconds", "CBF <30%", "mismatch volume", or "hypoperfusion".
- If the entire supplied text is actually CTA/CTP and no CT-only findings remain, set sanitized_report to "No noncontrast CT-only findings are provided in the supplied report."

Before returning the sanitized CT report, scan it for CTA/CTP modality terms.
The final New_CT_Report must not contain:
CTA, CT angiogram, CT angiography, CTP, CT perfusion, perfusion, Tmax, CBF, CBV, mismatch, hypoperfusion, penumbra, infarct core, tissue at risk.

If a sentence contains both a CT-visible finding and CTA/CTP wording, preserve only the CT-visible finding and remove the CTA/CTP wording.
Example:
"There is hyperdensity in the right MCA compatible with subtotal occlusion seen on the CT angiogram"
should become:
"There is hyperdensity in the right MCA compatible with thrombus."

Case ID:
{case_id}

Original CT Report:
{ct_report}

Return exactly this JSON structure:
{{
  "case_id": "{case_id}",
  "contamination_found": true,
  "sanitized_report": "CT-only report text after removing CTA/CTP contamination, or the original report if no contamination was found",
  "removed_sections": ["short description or exact removed CTA/CTP sentence"],
  "reasoning": "Brief explanation of what was removed or why no removal was needed"
}}
""".strip()


def _json_from_text(text: str) -> Dict[str, Any] | None:
    """Extract a JSON object from model text."""
    if not text:
        return None
    text = str(text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None


def sanitize_ct_report_sync(case_id: str, ct_report: str) -> Dict[str, Any]:
    """Run the CT sanitizer through direct Ollama structured output."""
    raw = ollama_generate_sync(
        prompt=build_ct_sanitization_prompt(case_id, ct_report),
        schema=CT_SANITIZATION_SCHEMA,
        case_id=case_id,
        tag="CT_Sanitize",
        temperature=cfg.DSPY_TASK_TEMPERATURE,
    )
    parsed = _json_from_text(raw)
    if parsed is None:
        raise RuntimeError(f"CT sanitizer returned invalid JSON: {raw[:300]!r}")
    return parsed


async def sanitize_ct_report_for_case(case_id: str, ct_report: str, semaphore=None) -> Dict[str, Any]:
    """Async wrapper around the synchronous sanitizer."""
    async with optional_semaphore(semaphore):
        return await asyncio.to_thread(sanitize_ct_report_sync, case_id, ct_report)


# =============================================================================
# Individual modality labeling functions
# =============================================================================

async def label_ct_for_case(case: Dict[str, Any], semaphore=None) -> Dict[str, Any]:
    """
    Label the CT report for one case with the restored sanitization behavior.

    Controlled behavior from the earlier pipeline:
        1. Label the original CT report normally and keep it as CT_Original_GT.
        2. If the CT report contains CTA/CTP/perfusion keywords, run the CT sanitizer.
        3. If the sanitizer actually changed the report, rerun CT on New_CT_Report
           and use that result as CT_GT.
    """

    case_id = str(case.get("case_id", "unknown"))
    original_report = clean_report_text(case.get("CT_Report") or "")

    async with optional_semaphore(semaphore):
        original_result = await run_prediction_with_confidence(
            lambda: label_ct(original_report)
        )

    _apply_result(case, "CT_Original_GT", original_result)

    if ct_report_may_contain_cta_ctp(original_report):
        try:
            raw_sanitization_result = await sanitize_ct_report_for_case(
                case_id,
                original_report,
                semaphore,
            )
        except Exception as exc:
            raw_sanitization_result = exc
        sanitization_result = safe_ct_sanitization_result(
            raw_sanitization_result,
            case_id,
            original_report,
        )
    else:
        sanitization_result = default_ct_sanitization_result(case_id, original_report)

    new_ct_report = clean_report_text(sanitization_result.get("sanitized_report", original_report))
    ct_report_was_sanitized = (
        bool(sanitization_result.get("contamination_found"))
        and new_ct_report.strip() != original_report.strip()
    )

    # Keep New_CT_Report present for downstream Combined and spreadsheet review.
    case["New_CT_Report"] = new_ct_report if ct_report_was_sanitized else original_report
    case["CT_Report_Was_Sanitized"] = ct_report_was_sanitized
    case["CT_Sanitization_reasoning"] = str(sanitization_result.get("reasoning", ""))
    case["CT_Sanitization_removed_sections"] = sanitization_result.get("removed_sections", [])

    if ct_report_was_sanitized:
        async with optional_semaphore(semaphore):
            final_result = await run_prediction_with_confidence(
                lambda: label_ct(new_ct_report)
            )
    else:
        final_result = original_result

    _apply_result(case, "CT_GT", final_result)
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

