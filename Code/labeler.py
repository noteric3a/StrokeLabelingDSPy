"""
Main labeling workflow with pipelined case processing.

Pipeline behavior:
1. Run CT/CTA/CTP for a case in parallel.
2. Start that case's Combined step only after CT/CTA/CTP labels exist.
3. While that Combined step is running, start CT/CTA/CTP for the next case.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import MODEL_NAME, MAX_CONCURRENT_REQUESTS
import config as cfg
from cache import ProcessingCache
from ollama_client import ollama_generate_async
from prompts import (
    build_ct_prompt,
    build_cta_prompt,
    build_ctp_prompt,
    build_combined_prompt,
    build_ct_sanitization_prompt,
)
from schemas import SINGLE_MODALITY_SCHEMA, COMBINED_SCHEMA, CT_SANITIZATION_SCHEMA
from utils import clean_report, normalize_labels
from review_checks import add_review_flags
from confidence import confidence_checking_enabled, confidence_temperature, confidence_run_count, confidence_fields_for_result, summarize_label_votes


EMPTY_REASONING_FIELDS = {
    "CT_GT_reasoning": "",
    "CTA_GT_reasoning": "",
    "CTP_GT_reasoning": "",
    "CT_Combined_GT_reasoning": "",
    "CT_Original_GT_reasoning": "",
    "CT_Sanitization_reasoning": "",
}

# Top-level fields that should be omitted from the final output JSON when
# INCLUDE_REASONING_IN_JSON is False or --no-reasoning is used.
REASONING_JSON_FIELDS = set(EMPTY_REASONING_FIELDS) | {"reasoning"}


def _resolve_include_reasoning(include_reasoning: Optional[bool] = None) -> bool:
    """Return the effective reasoning-output setting.

    A function argument overrides config.py. If no argument is supplied, use
    config.INCLUDE_REASONING_IN_JSON, defaulting to True for backward
    compatibility with older config.py files.
    """
    if include_reasoning is None:
        return bool(getattr(cfg, "INCLUDE_REASONING_IN_JSON", True))
    return bool(include_reasoning)


def strip_reasoning_fields(case: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of one labeled case without reasoning text fields."""
    return {
        key: value
        for key, value in case.items()
        if key not in REASONING_JSON_FIELDS
    }


def prepare_case_for_json(
    case: Dict[str, Any],
    include_reasoning: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return one case in the form that should be written to output JSON."""
    case_with_review_flags = add_review_flags(case)
    if _resolve_include_reasoning(include_reasoning):
        return dict(case_with_review_flags)
    return strip_reasoning_fields(case_with_review_flags)


def prepare_cases_for_json(
    labeled_cases: List[Dict[str, Any]],
    include_reasoning: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Return all cases in the form that should be written to output JSON."""
    return [prepare_case_for_json(case, include_reasoning) for case in labeled_cases]


def failed_case(case_id: str, error: Exception | str) -> Dict[str, Any]:
    """Return a safe output row when a case cannot be fully processed."""
    case = {
        "case_id": case_id,
        "CT_GT": ["NONE"],
        "CTA_GT": ["NONE"],
        "CTP_GT": ["NONE"],
        "Combined_GT": ["NONE"],
        "New_CT_Report": "",
        "CT_Report_Was_Sanitized": False,
        "CT_Original_GT": ["NONE"],
        **EMPTY_REASONING_FIELDS,
        "reasoning": f"FAILED: {repr(error)}",
    }
    case = add_review_flags(case)
    case["Needs_Review"] = True
    case["Review_Flags"] = [f"Case failed during processing: {repr(error)}"] + case.get("Review_Flags", [])
    case["Review_Flag_Count"] = len(case["Review_Flags"])
    return case


def save_progress(
    output_file: str,
    labeled_cases: List[Dict[str, Any]],
    include_reasoning: Optional[bool] = None,
) -> None:
    """Save accumulated cases after each completed final case.

    The in-memory cases and debug logs can still keep reasoning. This controls
    only what is written to the final output JSON.
    """
    output_cases = prepare_cases_for_json(labeled_cases, include_reasoning)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_cases, f, indent=2)


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

    sanitized_report = clean_report(result.get("sanitized_report", ""))
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
        "reasoning": clean_report(result.get("reasoning", "")),
    }



COMBINED_FALLBACK_LABEL_ORDER = [
    "NONE",
    "RMCA", "LMCA",
    "RACA", "LACA",
    "RPCA", "LPCA",
    "RPICA", "LPICA",
    "BA",
    "RVA", "LVA",
    "RICA", "LICA",
]


def _combined_positive_fallback_enabled() -> bool:
    """Return whether code should prevent Combined_GT from erasing positive modality labels."""
    return bool(getattr(cfg, "ENFORCE_COMBINED_POSITIVE_FALLBACK", True))


def _sort_labels_for_combined(labels: List[str]) -> List[str]:
    order = {label: index for index, label in enumerate(COMBINED_FALLBACK_LABEL_ORDER)}
    unique: List[str] = []
    for label in labels:
        label = str(label).strip().upper()
        if label and label not in unique:
            unique.append(label)
    return sorted(unique, key=lambda label: order.get(label, 999))


def _positive_modality_label_union(partial_case: Dict[str, Any]) -> List[str]:
    """Return the union of non-NONE CT/CTA/CTP labels in stable label order."""
    labels: List[str] = []
    for field in ("CT_GT", "CTA_GT", "CTP_GT"):
        for label in normalize_labels(partial_case.get(field, ["NONE"])):
            if label != "NONE" and label not in labels:
                labels.append(label)
    labels = _sort_labels_for_combined(labels)
    return labels if labels else ["NONE"]


def apply_combined_none_positive_fallback(
    combined_result: Dict[str, Any],
    partial_case: Dict[str, Any],
) -> Dict[str, Any]:
    """Prevent Combined_GT from becoming NONE while modality labels are positive.

    The Combined prompt can still remove chronic/artifact/upstream labels, but in
    practice the model sometimes returns ["NONE"] even when CT_GT, CTA_GT, or
    CTP_GT has a concrete territory. This deterministic guard preserves the
    union of positive modality labels and records the model's original Combined
    output for review.
    """
    if not _combined_positive_fallback_enabled():
        return combined_result

    final_labels = normalize_labels(combined_result.get("Combined_GT"))
    positive_union = _positive_modality_label_union(partial_case)
    if final_labels != ["NONE"] or positive_union == ["NONE"]:
        return combined_result

    original_reasoning = str(combined_result.get("reasoning", "")).strip()
    fixed_result = dict(combined_result)
    fixed_result["Combined_GT_original_before_none_override"] = final_labels
    fixed_result["Combined_GT_none_overridden_from_modalities"] = True
    fixed_result["Combined_GT"] = positive_union
    fixed_result["reasoning"] = (
        "Code safety override: the Combined step returned NONE even though "
        "at least one modality label was positive. Combined_GT was set to the "
        f"union of positive CT/CTA/CTP labels: {positive_union}. "
        "Review this case if the positive modality label was chronic, artifact, "
        "or otherwise non-qualifying.\n\n"
        "Original Combined reasoning before override:\n"
        f"{original_reasoning or 'No Combined reasoning was returned.'}"
    )

    summary = fixed_result.get("_confidence_summary")
    if isinstance(summary, dict):
        fixed_summary = dict(summary)
        fixed_summary["model_final_label_before_none_override"] = summary.get("final_label", final_labels)
        fixed_summary["final_label"] = positive_union
        fixed_result["_confidence_summary"] = fixed_summary

    return fixed_result


async def generate_label_with_optional_confidence(
    prompt: str,
    schema: Dict[str, Any],
    case_id: str,
    tag: str,
    semaphore: asyncio.Semaphore,
    model: str,
    *,
    label_field: str = "labels",
) -> Dict[str, Any]:
    """Run one label prompt normally, or run repeated confidence samples.

    Normal mode does one deterministic call at temperature 0.

    Confidence mode does CONFIDENCE_RUNS calls at CONFIDENCE_TEMPERATURE. The
    most common label set becomes the final label. Vote percentages are stored
    in _confidence_summary and later copied into JSON fields.
    """
    if not confidence_checking_enabled():
        return await ollama_generate_async(
            prompt,
            schema,
            case_id,
            tag,
            semaphore,
            model,
            temperature=0,
        )

    runs = confidence_run_count()
    temp = confidence_temperature()
    print(f"Running {tag} confidence sampling for case {case_id}: {runs} runs at temperature {temp}")

    tasks = [
        ollama_generate_async(
            prompt,
            schema,
            case_id,
            f"{tag}_conf_{i + 1:02d}",
            semaphore,
            model,
            temperature=temp,
        )
        for i in range(runs)
    ]
    sampled_results = await asyncio.gather(*tasks, return_exceptions=True)
    summary = summarize_label_votes(sampled_results, label_field=label_field)

    final_result = dict(summary.get("winning_result", {}))
    if label_field == "Combined_GT":
        final_result["Combined_GT"] = summary["final_label"]
    else:
        final_result["labels"] = summary["final_label"]

    final_result["reasoning"] = summary.get("winning_reasoning", final_result.get("reasoning", ""))
    final_result["_confidence_summary"] = summary
    return final_result


def _dspy_backend_enabled() -> bool:
    """Return True when config selects compiled DSPy programs for labeling."""
    return str(getattr(cfg, "LABELING_BACKEND", "ollama")).strip().lower() == "dspy"


def _build_manual_modality_prompt(modality: str, case_id: str, report: str) -> str:
    builders = {
        "CT": build_ct_prompt,
        "CTA": build_cta_prompt,
        "CTP": build_ctp_prompt,
    }
    try:
        builder = builders[str(modality).upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported modality: {modality!r}") from exc
    return builder(case_id, report)


async def generate_modality_with_optional_confidence(
    *,
    modality: str,
    report: str,
    case_id: str,
    tag: str,
    semaphore: asyncio.Semaphore,
    model: str,
) -> Dict[str, Any]:
    """Run one modality through either the manual-prompt or DSPy backend."""
    modality = str(modality).upper()
    if not _dspy_backend_enabled():
        return await generate_label_with_optional_confidence(
            _build_manual_modality_prompt(modality, case_id, report),
            SINGLE_MODALITY_SCHEMA,
            case_id,
            tag,
            semaphore,
            model,
            label_field="labels",
        )

    from dspy_runtime import dspy_generate_modality_async

    if not confidence_checking_enabled():
        return await dspy_generate_modality_async(
            modality=modality,
            report=report,
            case_id=case_id,
            tag=tag,
            semaphore=semaphore,
            model=model,
            temperature=0.0,
        )

    runs = confidence_run_count()
    temp = confidence_temperature()
    print(
        f"Running DSPy {tag} confidence sampling for case {case_id}: "
        f"{runs} runs at temperature {temp}"
    )
    tasks = [
        dspy_generate_modality_async(
            modality=modality,
            report=report,
            case_id=case_id,
            tag=f"{tag}_conf_{i + 1:02d}",
            semaphore=semaphore,
            model=model,
            temperature=temp,
        )
        for i in range(runs)
    ]
    sampled_results = await asyncio.gather(*tasks, return_exceptions=True)
    summary = summarize_label_votes(sampled_results, label_field="labels")
    final_result = dict(summary.get("winning_result", {}))
    final_result["labels"] = summary["final_label"]
    final_result["reasoning"] = summary.get(
        "winning_reasoning", final_result.get("reasoning", "")
    )
    final_result["_confidence_summary"] = summary
    return final_result


async def generate_combined_with_optional_confidence(
    *,
    partial_case: Dict[str, Any],
    case_id: str,
    tag: str,
    semaphore: asyncio.Semaphore,
    model: str,
) -> Dict[str, Any]:
    """Run Combined through either the manual-prompt or DSPy backend."""
    ct_report = partial_case.get("New_CT_Report") or partial_case.get("ct_report", "")
    cta_report = partial_case.get("cta_report", "")
    ctp_report = partial_case.get("ctp_report", "")
    mri_report = partial_case.get("mri_report", "")
    ct_labels = normalize_labels(partial_case.get("CT_GT"))
    cta_labels = normalize_labels(partial_case.get("CTA_GT"))
    ctp_labels = normalize_labels(partial_case.get("CTP_GT"))

    use_dspy_combined = _dspy_backend_enabled() and bool(
        getattr(cfg, "DSPY_USE_FOR_COMBINED", False)
    )
    if not use_dspy_combined:
        return await generate_label_with_optional_confidence(
            build_combined_prompt(
                case_id,
                ct_report,
                cta_report,
                ctp_report,
                mri_report,
                ct_labels,
                cta_labels,
                ctp_labels,
            ),
            COMBINED_SCHEMA,
            case_id,
            tag,
            semaphore,
            model,
            label_field="Combined_GT",
        )

    from dspy_runtime import dspy_generate_combined_async

    call_kwargs = {
        "case_id": case_id,
        "ct_report": ct_report,
        "cta_report": cta_report,
        "ctp_report": ctp_report,
        "mri_report": mri_report,
        "ct_labels": ct_labels,
        "cta_labels": cta_labels,
        "ctp_labels": ctp_labels,
        "semaphore": semaphore,
        "model": model,
    }

    if not confidence_checking_enabled():
        return await dspy_generate_combined_async(
            **call_kwargs,
            tag=tag,
            temperature=0.0,
        )

    runs = confidence_run_count()
    temp = confidence_temperature()
    print(
        f"Running DSPy {tag} confidence sampling for case {case_id}: "
        f"{runs} runs at temperature {temp}"
    )
    tasks = [
        dspy_generate_combined_async(
            **call_kwargs,
            tag=f"{tag}_conf_{i + 1:02d}",
            temperature=temp,
        )
        for i in range(runs)
    ]
    sampled_results = await asyncio.gather(*tasks, return_exceptions=True)
    summary = summarize_label_votes(sampled_results, label_field="Combined_GT")
    final_result = dict(summary.get("winning_result", {}))
    final_result["Combined_GT"] = summary["final_label"]
    final_result["reasoning"] = summary.get(
        "winning_reasoning", final_result.get("reasoning", "")
    )
    final_result["_confidence_summary"] = summary
    return final_result


def add_confidence_fields_to_case(
    case: Dict[str, Any],
    prefix: str,
    result: Dict[str, Any],
) -> None:
    """Copy confidence summary fields from an LLM result into a case dict."""
    case.update(confidence_fields_for_result(prefix, result))



async def label_modalities_for_case(
    row: pd.Series,
    semaphore: asyncio.Semaphore,
    model: str = MODEL_NAME,
) -> Dict[str, Any]:
    """
    Run CT, CTA, and CTP for one case.

    CT safety behavior:
    1. Run the original CT label normally.
    2. If the CT report looks like it may contain CTA/CTP/perfusion text,
       ask the LLM to create a sanitized CT-only report.
    3. If contamination was found and the report changed, re-run CT labeling
       on the sanitized report and use that as CT_GT.
    """
    time_start = time.perf_counter()
    case_id = str(row["Case Name"])

    ct_report = clean_report(row.get("CT", ""))
    cta_report = clean_report(row.get("CTA", ""))
    ctp_report = clean_report(row.get("CTP", ""))
    mri_report = clean_report(row.get("MRI", ""))

    ct_task = generate_modality_with_optional_confidence(
        modality="CT",
        report=ct_report,
        case_id=case_id,
        tag="CT",
        semaphore=semaphore,
        model=model,
    )
    cta_task = generate_modality_with_optional_confidence(
        modality="CTA",
        report=cta_report,
        case_id=case_id,
        tag="CTA",
        semaphore=semaphore,
        model=model,
    )
    ctp_task = generate_modality_with_optional_confidence(
        modality="CTP",
        report=ctp_report,
        case_id=case_id,
        tag="CTP",
        semaphore=semaphore,
        model=model,
    )

    sanitizer_task = None
    if ct_report_may_contain_cta_ctp(ct_report):
        sanitizer_task = ollama_generate_async(
            build_ct_sanitization_prompt(case_id, ct_report),
            CT_SANITIZATION_SCHEMA,
            case_id,
            "CT_Sanitize",
            semaphore,
            model,
        )

    if sanitizer_task is not None:
        results = await asyncio.gather(
            ct_task,
            cta_task,
            ctp_task,
            sanitizer_task,
            return_exceptions=True,
        )
        ct_result, cta_result, ctp_result, raw_sanitization_result = results
        sanitization_result = safe_ct_sanitization_result(
            raw_sanitization_result,
            case_id,
            ct_report,
        )
    else:
        results = await asyncio.gather(ct_task, cta_task, ctp_task, return_exceptions=True)
        ct_result, cta_result, ctp_result = results
        sanitization_result = default_ct_sanitization_result(case_id, ct_report)

    # First convert exceptions into safe dicts.
    ct_result = safe_modality_result(ct_result, "CT")
    cta_result = safe_modality_result(cta_result, "CTA")
    ctp_result = safe_modality_result(ctp_result, "CTP")

    original_ct_result = ct_result
    original_ct_labels = normalize_labels(ct_result.get("labels"))
    original_ct_reasoning = ct_result.get("reasoning", "")
    new_ct_report = sanitization_result.get("sanitized_report", ct_report)
    ct_report_was_sanitized = (
        bool(sanitization_result.get("contamination_found"))
        and new_ct_report.strip() != ct_report.strip()
    )

    # If the sanitizer actually removed CTA/CTP material, re-run CT labeling
    # on the sanitized CT-only report and replace CT_GT with that new result.
    if ct_report_was_sanitized:
        try:
            sanitized_ct_raw = await generate_modality_with_optional_confidence(
                modality="CT",
                report=new_ct_report,
                case_id=case_id,
                tag="CT_Sanitized",
                semaphore=semaphore,
                model=model,
            )
        except Exception as e:
            sanitized_ct_raw = e
        ct_result = safe_modality_result(sanitized_ct_raw, "CT")

    # Then normalize labels safely.
    ct_labels = normalize_labels(ct_result.get("labels"))
    cta_labels = normalize_labels(cta_result.get("labels"))
    ctp_labels = normalize_labels(ctp_result.get("labels"))

    time_final = time.perf_counter()
    print(f"Finished CT/CTA/CTP for case {case_id} in {round(time_final - time_start, 3)}s")

    partial_case = {
        "case_id": case_id,
        "ct_report": ct_report,
        "cta_report": cta_report,
        "ctp_report": ctp_report,
        "mri_report": mri_report,
        "New_CT_Report": new_ct_report if ct_report_was_sanitized else None,
        "CT_Report_Was_Sanitized": ct_report_was_sanitized,
        "CT_Original_GT": original_ct_labels,
        "CT_GT": ct_labels,
        "CTA_GT": cta_labels,
        "CTP_GT": ctp_labels,
        "CT_Original_GT_reasoning": original_ct_reasoning,
        "CT_GT_reasoning": ct_result.get("reasoning", ""),
        "CTA_GT_reasoning": cta_result.get("reasoning", ""),
        "CTP_GT_reasoning": ctp_result.get("reasoning", ""),
        "CT_Sanitization_reasoning": sanitization_result.get("reasoning", "") if ct_report_was_sanitized else "",
    }

    # In confidence mode, the existing CT_GT/CTA_GT/CTP_GT fields remain the
    # final labels for compatibility. These explicit fields make the vote table
    # easy to inspect in JSON and Excel.
    add_confidence_fields_to_case(partial_case, "CT_Original_GT", original_ct_result)
    add_confidence_fields_to_case(partial_case, "CT_GT", ct_result)
    add_confidence_fields_to_case(partial_case, "CTA_GT", cta_result)
    add_confidence_fields_to_case(partial_case, "CTP_GT", ctp_result)

    return partial_case


async def label_combined_for_case(
    partial_case: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    model: str = MODEL_NAME,
) -> Dict[str, Any]:
    """
    Run Combined for one case.

    This function is only called after CT/CTA/CTP have already been analyzed
    and normalized for the same case.
    """
    time_start = time.perf_counter()
    case_id = partial_case["case_id"]

    combined_result = await generate_combined_with_optional_confidence(
        partial_case=partial_case,
        case_id=case_id,
        tag="Combined",
        semaphore=semaphore,
        model=model,
    )
    combined_result = apply_combined_none_positive_fallback(combined_result, partial_case)

    final_case = {
        "case_id": case_id,
        "CT_Report": partial_case.get("ct_report", ""),
        "CTA_Report": partial_case.get("cta_report", ""),
        "CTP_Report": partial_case.get("ctp_report", ""),
        "MRI_Report": partial_case.get("mri_report", ""),
        "New_CT_Report": partial_case.get("New_CT_Report") or partial_case.get("ct_report", ""),
        "CT_Report_Was_Sanitized": partial_case.get("CT_Report_Was_Sanitized", False),
        "CT_Original_GT": partial_case.get("CT_Original_GT", partial_case["CT_GT"]),
        "CT_GT": partial_case["CT_GT"],
        "CTA_GT": partial_case["CTA_GT"],
        "CTP_GT": partial_case["CTP_GT"],
        "Combined_GT": normalize_labels(combined_result.get("Combined_GT")),
        "CT_Original_GT_reasoning": partial_case.get("CT_Original_GT_reasoning", ""),
        "CT_GT_reasoning": partial_case["CT_GT_reasoning"],
        "CT_Sanitization_reasoning": partial_case.get("CT_Sanitization_reasoning", ""),
        "CTA_GT_reasoning": partial_case["CTA_GT_reasoning"],
        "CTP_GT_reasoning": partial_case["CTP_GT_reasoning"],
        "CT_Combined_GT_reasoning": combined_result.get("reasoning", ""),
        "Combined_GT_none_overridden_from_modalities": bool(combined_result.get("Combined_GT_none_overridden_from_modalities", False)),
        "Combined_GT_original_before_none_override": combined_result.get("Combined_GT_original_before_none_override", []),
    }

    # Carry modality confidence fields from the partial case into the final case.
    for key, value in partial_case.items():
        if (
            key.endswith("_final_label")
            or key.endswith("_possible_answers")
            or key.endswith("_confidence_percentage")
            or key.endswith("_is_confident")
            or key.endswith("_confidence_threshold")
            or key.endswith("_confidence_vote_count")
            or key.endswith("_confidence_total_votes")
            or key.endswith("_confidence_failed_runs")
        ):
            final_case[key] = value

    add_confidence_fields_to_case(final_case, "Combined_GT", combined_result)
    final_case = add_review_flags(final_case)
    if final_case.get("Combined_GT_none_overridden_from_modalities"):
        final_case["Needs_Review"] = True
        final_case["Review_Flags"] = [
            "Combined_GT was changed from NONE to the union of positive modality labels by code fallback"
        ] + final_case.get("Review_Flags", [])
        final_case["Review_Flag_Count"] = len(final_case["Review_Flags"])

    with open(cfg.RAW_OUTPUT_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n\n=== Case {case_id} ===\n")
        f.write(json.dumps(final_case, indent=2))

    time_final = time.perf_counter()
    print(f"Finished Combined for case {case_id} in {round(time_final - time_start, 3)}s")
    return final_case


async def safe_label_modalities_for_case(
    row: pd.Series,
    semaphore: asyncio.Semaphore,
    model: str,
) -> Dict[str, Any]:
    case_id = str(row["Case Name"])
    try:
        partial_case = await label_modalities_for_case(row, semaphore, model)
        return {"ok": True, "case_id": case_id, "partial_case": partial_case}

    except Exception as e:
        print(f"ERROR while running CT/CTA/CTP for case {case_id}: {e}")

        with open(cfg.FAILED_CASES_LOG, "a", encoding="utf-8") as f:
            f.write(f"{case_id} [CT/CTA/CTP]: {repr(e)}\n")

        final_case = failed_case(case_id, e)

        # NEW: make failed cases visible in raw_ollama_outputs_async.txt
        with open(cfg.RAW_OUTPUT_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== Case {case_id} ===\n")
            f.write(json.dumps(final_case, indent=2))

        return {"ok": False, "case_id": case_id, "final_case": final_case}


async def safe_label_combined_for_case(
    partial_case: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    model: str,
) -> Dict[str, Any]:
    """Run Combined and convert failures into a structured fallback case."""
    case_id = partial_case["case_id"]
    try:
        return await label_combined_for_case(partial_case, semaphore, model)
    except Exception as e:
        print(f"ERROR while running Combined for case {case_id}: {e}")
        fallback_combined_labels = _positive_modality_label_union(partial_case)
        final_case = {
            "case_id": case_id,
            "CT_Report": partial_case.get("ct_report", ""),
            "CTA_Report": partial_case.get("cta_report", ""),
            "CTP_Report": partial_case.get("ctp_report", ""),
            "MRI_Report": partial_case.get("mri_report", ""),
            "New_CT_Report": partial_case.get("New_CT_Report") or partial_case.get("ct_report", ""),
            "CT_Report_Was_Sanitized": partial_case.get("CT_Report_Was_Sanitized", False),
            "CT_Original_GT": partial_case.get("CT_Original_GT", partial_case.get("CT_GT", ["NONE"])),
            "CT_GT": partial_case.get("CT_GT", ["NONE"]),
            "CTA_GT": partial_case.get("CTA_GT", ["NONE"]),
            "CTP_GT": partial_case.get("CTP_GT", ["NONE"]),
            "Combined_GT": fallback_combined_labels,
            "CT_Original_GT_reasoning": partial_case.get("CT_Original_GT_reasoning", ""),
            "CT_GT_reasoning": partial_case.get("CT_GT_reasoning", ""),
            "CT_Sanitization_reasoning": partial_case.get("CT_Sanitization_reasoning", ""),
            "CTA_GT_reasoning": partial_case.get("CTA_GT_reasoning", ""),
            "CTP_GT_reasoning": partial_case.get("CTP_GT_reasoning", ""),
            "CT_Combined_GT_reasoning": (
                "FAILED Combined step. Code fallback used the union of positive "
                f"CT/CTA/CTP modality labels: {fallback_combined_labels}."
            ),
            "reasoning": f"FAILED Combined: {repr(e)}",
        }
        final_case = add_review_flags(final_case)
        final_case["Needs_Review"] = True
        final_case["Review_Flags"] = [f"Combined step failed: {repr(e)}"] + final_case.get("Review_Flags", [])
        final_case["Review_Flag_Count"] = len(final_case["Review_Flags"])

        with open(cfg.RAW_OUTPUT_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== Case {case_id} ===\n")
            f.write(json.dumps(final_case, indent=2))

        return final_case


async def label_one_case(
    row: pd.Series,
    semaphore: asyncio.Semaphore,
    model: str = MODEL_NAME,
) -> Dict[str, Any]:
    """
    Backward-compatible single-case workflow.

    Useful if another script imports label_one_case directly. This still runs
    CT/CTA/CTP first, then Combined only after those labels exist.
    """
    modality_result = await safe_label_modalities_for_case(row, semaphore, model)
    if not modality_result["ok"]:
        return modality_result["final_case"]
    return await safe_label_combined_for_case(modality_result["partial_case"], semaphore, model)


def find_case_row_by_id(
    df: pd.DataFrame,
    case_id: str,
    case_column: str = "Case Name",
) -> pd.Series:
    """
    Find exactly one case row by case ID.

    Matching is done against the spreadsheet's case_column after converting
    values to strings and stripping whitespace. If multiple rows match, the
    first row is used and a warning is printed.
    """
    if case_column not in df.columns:
        raise ValueError(
            f"Could not find case ID column {case_column!r}. "
            f"Available columns: {list(df.columns)}"
        )

    requested_case_id = str(case_id).strip()
    case_ids = df[case_column].astype(str).str.strip()
    matches = df.loc[case_ids == requested_case_id]

    if matches.empty:
        available_examples = case_ids.head(10).tolist()
        raise ValueError(
            f"Case ID {requested_case_id!r} was not found in column {case_column!r}. "
            f"First available case IDs: {available_examples}"
        )

    if len(matches) > 1:
        print(
            f"WARNING: Found {len(matches)} rows for case ID {requested_case_id!r}. "
            "Using the first match."
        )

    return matches.iloc[0]


async def label_case_by_id_async(
    input_file: str,
    case_id: str,
    output_file: Optional[str] = None,
    model: str = MODEL_NAME,
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,
    case_column: str = "Case Name",
    include_reasoning: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Find, analyze, and label one specific case from the spreadsheet.

    This is useful for debugging one difficult case without running the full
    spreadsheet pipeline. It uses the same safe single-case workflow as the
    full labeler: CT/CTA/CTP run first, then Combined runs only after those
    labels exist.

    If output_file is provided, the single labeled case is saved as a one-item
    JSON list so it has the same outer structure as label_spreadsheet_async.
    """
    df = pd.read_excel(input_file)
    row = find_case_row_by_id(df, case_id, case_column)

    semaphore = asyncio.Semaphore(max_concurrent_requests)
    normalized_case_id = str(row[case_column]).strip()

    print(f"\n===== Finding case by ID: {case_id} =====")
    print(f"===== Starting single-case labeling for: {normalized_case_id} =====")

    start_time = time.perf_counter()
    final_case = await label_one_case(row, semaphore, model)
    finish_time = time.perf_counter()

    if output_file is not None:
        save_progress(output_file, [final_case], include_reasoning=include_reasoning)
        print(f"Saved labeled case {normalized_case_id} to {output_file}")

    print(
        f"Finished single-case labeling for {normalized_case_id} "
        f"in {round(finish_time - start_time, 3)}s"
    )
    return final_case


def label_case_by_id(
    input_file: str,
    case_id: str,
    output_file: Optional[str] = None,
    model: str = MODEL_NAME,
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,
    case_column: str = "Case Name",
    include_reasoning: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for label_case_by_id_async.

    Use this from regular scripts. If you are already inside async code, call
    await label_case_by_id_async(...) instead.
    """
    return asyncio.run(
        label_case_by_id_async(
            input_file=input_file,
            case_id=case_id,
            output_file=output_file,
            model=model,
            max_concurrent_requests=max_concurrent_requests,
            case_column=case_column,
            include_reasoning=include_reasoning,
        )
    )

def safe_modality_result(result, modality):
    if isinstance(result, Exception):
        return {
            "modality": modality,
            "labels": ["NONE"],
            "reasoning": f"FAILED {modality}: {repr(result)}",
        }
    return result

async def label_spreadsheet_async(
    input_file: str,
    output_file: str,
    model: str = MODEL_NAME,
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,
    use_cache: bool = True,
    include_reasoning: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Label a spreadsheet with caching support for resumable runs.
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to output JSON file
        model: Ollama model to use
        max_concurrent_requests: Max concurrent API requests
        use_cache: Whether to skip already-processed cases
        include_reasoning: Whether to include reasoning fields in the final output JSON.
            None means use config.INCLUDE_REASONING_IN_JSON.
    """
    # Load cache if enabled
    cache = ProcessingCache() if use_cache else None
    
    df = pd.read_excel(input_file)
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    # Load existing output if resuming
    labeled_cases: List[Dict[str, Any]] = []
    if use_cache and Path(output_file).exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                labeled_cases = json.load(f)
            print(f"Loaded {len(labeled_cases)} previously processed cases from {output_file}")
        except (json.JSONDecodeError, IOError):
            labeled_cases = []
    
    pending_final_task: Optional[asyncio.Task[Dict[str, Any]]] = None

    start_time = time.perf_counter()
    skipped_count = 0
    processed_count = 0

    for index, row in df.iterrows():
        case_id = str(row["Case Name"])
        
        # Skip if already processed and caching is enabled
        if cache and cache.is_processed(case_id):
            skipped_count += 1
            continue
        
        print(f"\n===== Starting CT/CTA/CTP for case {index + 1}/{len(df)}: {case_id} =====")

        # Start current case CT/CTA/CTP immediately. If there is a previous
        # Combined task pending, it runs at the same time as these modality tasks.
        current_modalities_task = asyncio.create_task(
            safe_label_modalities_for_case(row, semaphore, model)
        )

        if pending_final_task is not None:
            previous_final_case, current_modality_result = await asyncio.gather(
                pending_final_task,
                current_modalities_task,
            )
            labeled_cases.append(previous_final_case)
            processed_count += 1
            if cache:
                cache.mark_processed(previous_final_case["case_id"])
            save_progress(output_file, labeled_cases, include_reasoning=include_reasoning)
        else:
            current_modality_result = await current_modalities_task

        # Start this case's Combined step only after this case's CT/CTA/CTP
        # have finished and produced labels.
        if current_modality_result["ok"]:
            print(f"===== Starting Combined for case {case_id} =====")
            pending_final_task = asyncio.create_task(
                safe_label_combined_for_case(
                    current_modality_result["partial_case"],
                    semaphore,
                    model,
                )
            )
        else:
            # No Combined task is created when CT/CTA/CTP failed.
            async def already_done(final_case: Dict[str, Any]) -> Dict[str, Any]:
                return final_case

            pending_final_task = asyncio.create_task(already_done(current_modality_result["final_case"]))

    # Flush the final case's Combined result.
    if pending_final_task is not None:
        final_case = await pending_final_task
        labeled_cases.append(final_case)
        processed_count += 1
        if cache:
            cache.mark_processed(final_case["case_id"])
        save_progress(output_file, labeled_cases, include_reasoning=include_reasoning)

    # Save cache state
    if cache:
        cache.save_cache()

    # If everything was skipped because of cache, still rewrite the output so
    # changing INCLUDE_REASONING_IN_JSON / --no-reasoning is reflected.
    if labeled_cases and skipped_count > 0 and processed_count == 0:
        save_progress(output_file, labeled_cases, include_reasoning=include_reasoning)

    finish_time = time.perf_counter()

    print(f"Saved {len(labeled_cases)} labeled cases to {output_file}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} already-processed cases (cached)")
    print(f"Total Time Taken: {str(round(finish_time - start_time, 3))}s")
    return labeled_cases
