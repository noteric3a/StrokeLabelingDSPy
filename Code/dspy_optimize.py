"""Continuously optimize CT/CTA/CTP prompts with DSPy GEPA.

The loop is deliberately resumable and conservative:

    current best program
        -> evaluate reports against ground truth
        -> GEPA reflects on exact-match failures and proposes a new instruction
        -> separately re-evaluate the candidate
        -> promote only when it is better
        -> repeat until the target accuracy, STOP file, Ctrl+C, or the configured round limit

By default, CT, CTA, and CTP are optimized independently from small zero-shot
base instructions.  Combined can also be selected explicitly, but its training
examples use the modality ground truths as preliminary labels (teacher forcing),
so a final end-to-end run should always be used for the definitive comparison.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import random
import re
import shutil
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

try:
    import dspy
except ImportError as exc:  # pragma: no cover - optional dependency.
    raise SystemExit(
        "DSPy is required for prompt optimization. Install requirements-dspy.txt "
        "or run: pip install dspy==3.2.1"
    ) from exc

import config as cfg
from dspy_programs import (
    MODALITY_PROGRAM_NAMES,
    canonical_program_name,
    create_adapter,
    create_lm,
    create_program,
    get_instructions,
    load_program,
    ollama_api_base,
    program_path,
    save_program,
)


CASE_COLUMN_CANDIDATES = (
    "Case Name",
    "case_id",
    "Case ID",
    "Case_Name",
    "CASE_ID",
    "Case",
)

REPORT_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "CT": ("CT", "CT Report", "CT_Report", "Noncontrast CT", "NCCT"),
    "CTA": ("CTA", "CTA Report", "CTA_Report", "CT Angiography"),
    "CTP": ("CTP", "CTP Report", "CTP_Report", "CT Perfusion"),
    "MRI": ("MRI", "MRI Report", "MRI_Report"),
}

GT_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "CT": ("CT GT", "CT_GT", "CT.GT", "CTGT", "CT Ground Truth", "CT_Ground_Truth"),
    "CTA": ("CTA GT", "CTA_GT", "CTA.GT", "CTAGT", "CTA Ground Truth", "CTA_Ground_Truth"),
    "CTP": ("CTP GT", "CTP_GT", "CTP.GT", "CTPGT", "CTP Ground Truth", "CTP_Ground_Truth"),
    "Combined": (
        "Combined GT",
        "Combined_GT",
        "Combined.GT",
        "CombinedGT",
        "Combined Ground Truth",
        "Combined_Ground_Truth",
    ),
}

NONE_ALIASES = {
    "",
    "NONE",
    "NEGATIVE",
    "NORMAL",
    "NO ACUTE",
    "NO ACUTE FINDING",
    "NO ACUTE FINDINGS",
    "NO LABEL",
    "NO LABELS",
    "NAN",
    "NULL",
}
NONE_ALIASES_COMPACT = {alias.replace(" ", "") for alias in NONE_ALIASES}

LABEL_ORDER = list(cfg.ALLOWED_LABELS)


@dataclass(frozen=True)
class OptimizerSettings:
    """All DSPy optimizer controls resolved from config.py."""

    reports: str | Path
    ground_truth: str | Path
    output_dir: str | Path
    base_prompts: Any
    base_prompt_file: str | Path | None
    programs: str | Sequence[str]
    task_model: str
    reflection_model: str | None
    api_base: str
    adapter: str
    max_tokens: int
    reflection_max_tokens: int
    num_ctx: int
    timeout: int
    num_retries: int
    validation_fraction: float
    test_fraction: float
    seed: int
    target_accuracy: float
    max_rounds: int
    metric_calls_per_round: int
    reflection_minibatch_size: int
    reflection_temperature: float
    num_threads: int
    selection_strategy: str
    max_consecutive_errors: int
    promote_f1_ties: bool
    track_gepa_stats: bool
    manual_report: str | Path | None
    evaluate_only: bool
    evaluate_test_each_round: bool
    include_full_reports: bool
    report_excerpt_chars: int
    reset: bool
    clear_stop: bool


def optimizer_settings_from_config() -> OptimizerSettings:
    """Build the optimizer settings object exclusively from config.py."""

    reflection_model = str(getattr(cfg, "DSPY_REFLECTION_MODEL", "") or "").strip() or None
    manual_report_raw = getattr(cfg, "DSPY_MANUAL_REPORT_FILE", "")
    manual_report = manual_report_raw if str(manual_report_raw).strip() else None
    base_prompt_file_raw = getattr(cfg, "DSPY_BASE_PROMPTS_FILE", "")
    base_prompt_file = (
        base_prompt_file_raw if str(base_prompt_file_raw).strip() else None
    )
    return OptimizerSettings(
        reports=getattr(cfg, "DSPY_OPTIMIZER_REPORTS_FILE", cfg.INPUT_REPORTS_FILE),
        ground_truth=getattr(cfg, "DSPY_OPTIMIZER_GROUND_TRUTH_FILE", cfg.GROUND_TRUTH_FILE),
        output_dir=getattr(cfg, "DSPY_OPTIMIZER_OUTPUT_DIR", cfg.DSPY_OPTIMIZATION_DIR),
        base_prompts=getattr(cfg, "DSPY_BASE_INSTRUCTIONS", {}),
        base_prompt_file=base_prompt_file,
        programs=getattr(cfg, "DSPY_OPTIMIZER_PROGRAMS", ("CT", "CTA", "CTP")),
        task_model=str(getattr(cfg, "DSPY_TASK_MODEL", cfg.MODEL_NAME)).strip(),
        reflection_model=reflection_model,
        api_base=str(getattr(cfg, "DSPY_OPTIMIZER_API_BASE", "") or "").strip(),
        adapter=str(getattr(cfg, "DSPY_OPTIMIZER_ADAPTER", getattr(cfg, "DSPY_ADAPTER", "json"))).strip(),
        max_tokens=int(getattr(cfg, "DSPY_OPTIMIZER_MAX_TOKENS", getattr(cfg, "DSPY_MAX_TOKENS", cfg.NUM_PREDICT))),
        reflection_max_tokens=int(getattr(cfg, "DSPY_REFLECTION_MAX_TOKENS", max(4000, getattr(cfg, "DSPY_MAX_TOKENS", 1670)))),
        num_ctx=int(getattr(cfg, "DSPY_OPTIMIZER_NUM_CTX", getattr(cfg, "DSPY_NUM_CTX", cfg.NUM_CTX))),
        timeout=int(getattr(cfg, "DSPY_OPTIMIZER_TIMEOUT_SECONDS", cfg.REQUEST_TIMEOUT_SECONDS)),
        num_retries=int(getattr(cfg, "DSPY_OPTIMIZER_NUM_RETRIES", 2)),
        validation_fraction=float(getattr(cfg, "DSPY_VALIDATION_FRACTION", 0.20)),
        test_fraction=float(getattr(cfg, "DSPY_TEST_FRACTION", 0.10)),
        seed=int(getattr(cfg, "DSPY_RANDOM_SEED", 42)),
        target_accuracy=float(getattr(cfg, "DSPY_TARGET_ACCURACY", 1.0)),
        max_rounds=int(getattr(cfg, "DSPY_MAX_ROUNDS", 0)),
        metric_calls_per_round=int(getattr(cfg, "DSPY_METRIC_CALLS_PER_ROUND", 64)),
        reflection_minibatch_size=int(getattr(cfg, "DSPY_REFLECTION_MINIBATCH_SIZE", 3)),
        reflection_temperature=float(getattr(cfg, "DSPY_REFLECTION_TEMPERATURE", 1.0)),
        num_threads=int(getattr(cfg, "DSPY_OPTIMIZER_NUM_THREADS", 1)),
        selection_strategy=str(getattr(cfg, "DSPY_SELECTION_STRATEGY", "worst")).strip().lower(),
        max_consecutive_errors=int(getattr(cfg, "DSPY_MAX_CONSECUTIVE_ERRORS", 3)),
        promote_f1_ties=bool(getattr(cfg, "DSPY_PROMOTE_F1_TIES", True)),
        track_gepa_stats=bool(getattr(cfg, "DSPY_TRACK_GEPA_STATS", False)),
        manual_report=manual_report,
        evaluate_only=bool(getattr(cfg, "DSPY_EVALUATE_ONLY", False)),
        evaluate_test_each_round=bool(getattr(cfg, "DSPY_EVALUATE_TEST_EACH_ROUND", False)),
        include_full_reports=bool(getattr(cfg, "DSPY_INCLUDE_FULL_REPORTS", False)),
        report_excerpt_chars=int(getattr(cfg, "DSPY_REPORT_EXCERPT_CHARS", 600)),
        reset=bool(getattr(cfg, "DSPY_RESET_OPTIMIZATION", False)),
        clear_stop=bool(getattr(cfg, "DSPY_CLEAR_STOP_FILE_ON_START", False)),
    )


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    ct_report: str
    cta_report: str
    ctp_report: str
    mri_report: str
    ct_gt: tuple[str, ...]
    cta_gt: tuple[str, ...]
    ctp_gt: tuple[str, ...]
    combined_gt: tuple[str, ...]

    def report_for(self, program_name: str) -> str:
        name = canonical_program_name(program_name)
        if name == "CT":
            return self.ct_report
        if name == "CTA":
            return self.cta_report
        if name == "CTP":
            return self.ctp_report
        raise ValueError("Combined uses multiple report fields.")

    def gold_for(self, program_name: str) -> list[str]:
        name = canonical_program_name(program_name)
        mapping = {
            "CT": self.ct_gt,
            "CTA": self.cta_gt,
            "CTP": self.ctp_gt,
            "Combined": self.combined_gt,
        }
        return list(mapping[name])


@dataclass
class EvaluationResult:
    program: str
    split: str
    correct: int
    total: int
    accuracy: float
    label_precision: float
    label_recall: float
    label_f1: float
    failures: int
    invalid_outputs: int
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    predictions_by_case: dict[str, list[str]] = field(default_factory=dict)

    @property
    def score_key(self) -> tuple[int, float, int, int]:
        # Exact-match correctness is primary. Label F1 only breaks exact ties.
        return (self.correct, self.label_f1, -self.failures, -self.invalid_outputs)

    def summary(self, include_mismatches: bool = False) -> dict[str, Any]:
        payload = {
            "program": self.program,
            "split": self.split,
            "correct": self.correct,
            "total": self.total,
            "accuracy": round(self.accuracy, 8),
            "accuracy_percent": round(self.accuracy * 100.0, 3),
            "label_precision": round(self.label_precision, 8),
            "label_recall": round(self.label_recall, 8),
            "label_f1": round(self.label_f1, 8),
            "failures": self.failures,
            "invalid_outputs": self.invalid_outputs,
            "mismatch_count": len(self.mismatches),
        }
        if include_mismatches:
            payload["mismatches"] = self.mismatches
        return payload


@dataclass
class ProgramState:
    program: str
    accepted_round: int = 0
    attempted_rounds: int = 0
    improvements: int = 0
    validation: dict[str, Any] = field(default_factory=dict)
    test: dict[str, Any] = field(default_factory=dict)
    instruction_sha256: str = ""


@dataclass
class OptimizerState:
    version: int
    created_at: str
    updated_at: str
    dataset_fingerprint: str
    reports_file: str
    ground_truth_file: str
    task_model: str
    reflection_model: str
    target_accuracy: float
    completed_rounds: int
    selected_programs: list[str]
    split: dict[str, list[str]]
    programs: dict[str, ProgramState]


# ---------------------------------------------------------------------------
# Data loading and normalization
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    normalized = {_normalize_column_name(column): str(column) for column in df.columns}
    for candidate in candidates:
        found = normalized.get(_normalize_column_name(candidate))
        if found is not None:
            return found
    return None


def _read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    suffix = table_path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(table_path)
    if suffix == ".csv":
        return pd.read_csv(table_path)
    raise ValueError(f"Unsupported table type: {table_path.suffix}")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _case_id(value: Any) -> str:
    return _clean_text(value).strip('"\'')


def _split_label_text(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed]
        except Exception:
            pass
    return [piece for piece in re.split(r"[,;|\n]+", text) if piece.strip()]


def normalize_label_value(value: Any) -> list[str]:
    """Normalize spreadsheet, DSPy, or JSON label values into stable exact sets."""
    if value is None:
        pieces: list[Any] = []
    elif isinstance(value, float) and pd.isna(value):
        pieces = []
    elif isinstance(value, (list, tuple, set)):
        pieces = list(value)
    else:
        pieces = _split_label_text(str(value))

    labels: list[str] = []
    for piece in pieces:
        label = str(piece).strip().strip('"\'[](){}').upper().replace(" ", "")
        readable = str(piece).strip().upper()
        if readable in NONE_ALIASES or label in NONE_ALIASES_COMPACT:
            label = "NONE"
        if label in cfg.ALLOWED_LABELS and label not in labels:
            labels.append(label)

    if not labels:
        return ["NONE"]
    if "NONE" in labels and len(labels) > 1:
        labels.remove("NONE")
    order = {label: index for index, label in enumerate(LABEL_ORDER)}
    return sorted(labels, key=lambda label: order.get(label, 999))


def _invalid_label_items(value: Any) -> list[str]:
    """Return unrecognized labels without treating missing/negative aliases as errors."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        pieces = list(value)
    else:
        pieces = _split_label_text(str(value))

    invalid: list[str] = []
    canonical: list[str] = []
    for piece in pieces:
        readable = str(piece).strip().upper()
        compact = readable.strip('"\'[](){}').replace(" ", "")
        if readable in NONE_ALIASES or compact in NONE_ALIASES_COMPACT:
            compact = "NONE"
        if compact not in cfg.ALLOWED_LABELS:
            invalid.append(str(piece).strip())
        elif compact not in canonical:
            canonical.append(compact)
    if "NONE" in canonical and len(canonical) > 1:
        invalid.append("NONE mixed with positive label(s)")
    return invalid


def _column_looks_like_labels(df: pd.DataFrame, column: str, sample_size: int = 50) -> bool:
    if column not in df.columns:
        return False
    values = [_clean_text(value) for value in df[column].dropna().head(sample_size).tolist()]
    values = [value for value in values if value]
    if not values:
        return True
    for value in values:
        pieces = _split_label_text(value)
        if not pieces:
            continue
        for piece in pieces:
            compact = str(piece).strip().strip('"\'[](){}').upper().replace(" ", "")
            readable = str(piece).strip().upper()
            if compact not in cfg.ALLOWED_LABELS and readable not in NONE_ALIASES:
                return False
    return True


def load_case_records(
    reports_file: str | Path,
    ground_truth_file: str | Path,
    *,
    require_combined_gt: bool = False,
) -> list[CaseRecord]:
    reports_df = _read_table(reports_file)
    gt_df = _read_table(ground_truth_file)

    reports_case_col = _find_column(reports_df, CASE_COLUMN_CANDIDATES)
    gt_case_col = _find_column(gt_df, CASE_COLUMN_CANDIDATES)
    if reports_case_col is None:
        raise ValueError(f"No case ID column found in reports file. Columns: {list(reports_df.columns)}")
    if gt_case_col is None:
        raise ValueError(f"No case ID column found in ground-truth file. Columns: {list(gt_df.columns)}")

    report_columns: dict[str, str | None] = {
        name: _find_column(reports_df, candidates)
        for name, candidates in REPORT_COLUMN_CANDIDATES.items()
    }
    missing_reports = [name for name in MODALITY_PROGRAM_NAMES if report_columns[name] is None]
    if missing_reports:
        raise ValueError(
            f"Missing report column(s) {missing_reports}. Reports columns: {list(reports_df.columns)}"
        )

    separate_files = Path(reports_file).resolve() != Path(ground_truth_file).resolve()
    gt_columns: dict[str, str | None] = {}
    for name, candidates in GT_COLUMN_CANDIDATES.items():
        column = _find_column(gt_df, candidates)
        if column is None and separate_files:
            fallback = _find_column(gt_df, (name,))
            if fallback is not None and _column_looks_like_labels(gt_df, fallback):
                column = fallback
        gt_columns[name] = column

    required_gt_names = ["CT", "CTA", "CTP"] + (["Combined"] if require_combined_gt else [])
    missing_gt = [name for name in required_gt_names if gt_columns[name] is None]
    if missing_gt:
        raise ValueError(
            f"Missing ground-truth column(s) {missing_gt}. Ground-truth columns: {list(gt_df.columns)}"
        )

    def unique_lookup(df: pd.DataFrame, case_column: str, source_name: str) -> dict[str, pd.Series]:
        lookup: dict[str, pd.Series] = {}
        duplicates: list[str] = []
        for _, row in df.iterrows():
            identifier = _case_id(row.get(case_column))
            if not identifier:
                continue
            if identifier in lookup:
                duplicates.append(identifier)
                continue
            lookup[identifier] = row
        if duplicates:
            examples = sorted(set(duplicates))[:10]
            raise ValueError(
                f"Duplicate case IDs in {source_name}; exact optimization requires one row per case. "
                f"Examples: {examples}"
            )
        return lookup

    reports_lookup = unique_lookup(reports_df, reports_case_col, "reports file")
    gt_lookup = unique_lookup(gt_df, gt_case_col, "ground-truth file")

    records: list[CaseRecord] = []
    missing_report_cases: list[str] = []
    for identifier, gt_row in gt_lookup.items():
        report_row = reports_lookup.get(identifier)
        if report_row is None:
            missing_report_cases.append(identifier)
            continue

        def report(name: str) -> str:
            column = report_columns.get(name)
            return _clean_text(report_row.get(column, "")) if column else ""

        def gold(name: str) -> tuple[str, ...]:
            column = gt_columns.get(name)
            if column is None:
                # Combined is optional unless it was explicitly selected.
                return tuple()
            raw_value = gt_row.get(column)
            invalid = _invalid_label_items(raw_value)
            if invalid:
                raise ValueError(
                    f"Invalid {name} ground-truth label(s) for case {identifier}: {invalid}. "
                    f"Allowed labels: {cfg.ALLOWED_LABELS}"
                )
            return tuple(normalize_label_value(raw_value))

        ct_gt = gold("CT")
        cta_gt = gold("CTA")
        ctp_gt = gold("CTP")
        combined_gt = gold("Combined")
        if not combined_gt:
            union: list[str] = []
            for labels in (ct_gt, cta_gt, ctp_gt):
                for label in labels:
                    if label != "NONE" and label not in union:
                        union.append(label)
            combined_gt = tuple(normalize_label_value(union))

        records.append(
            CaseRecord(
                case_id=identifier,
                ct_report=report("CT"),
                cta_report=report("CTA"),
                ctp_report=report("CTP"),
                mri_report=report("MRI"),
                ct_gt=ct_gt,
                cta_gt=cta_gt,
                ctp_gt=ctp_gt,
                combined_gt=combined_gt,
            )
        )

    if missing_report_cases:
        print(
            f"WARNING: skipped {len(missing_report_cases)} ground-truth case(s) with no matching report row. "
            f"Examples: {missing_report_cases[:5]}"
        )
    if not records:
        raise ValueError("No report/ground-truth case IDs matched.")
    records.sort(key=lambda record: record.case_id)
    return records


def dataset_fingerprint(records: Sequence[CaseRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        payload = asdict(record)
        digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Split and example construction
# ---------------------------------------------------------------------------


def _fraction_count(total: int, fraction: float) -> int:
    if fraction <= 0 or total <= 1:
        return 0
    count = int(round(total * fraction))
    return max(1, min(count, total - 1))


def create_split(
    records: Sequence[CaseRecord],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("Split fractions cannot be negative.")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1.0.")

    identifiers = [record.case_id for record in records]
    rng = random.Random(seed)
    rng.shuffle(identifiers)

    total = len(identifiers)
    test_count = _fraction_count(total, test_fraction)
    validation_count = _fraction_count(total, validation_fraction)

    # Fractions are defined against the full dataset (for example, the defaults
    # are approximately 70% train / 20% validation / 10% test). For very small
    # datasets, rounded holdout counts can consume every case. Reduce the larger
    # holdout until at least one true training case remains.
    while test_count + validation_count >= total and (test_count or validation_count):
        if validation_count >= test_count and validation_count > 0:
            validation_count -= 1
        elif test_count > 0:
            test_count -= 1

    test_ids = identifiers[:test_count]
    val_ids = identifiers[test_count : test_count + validation_count]
    train_ids = identifiers[test_count + validation_count :]

    if not train_ids:
        raise ValueError("The requested split left no training cases.")
    if not val_ids:
        # Explicit experiment mode: optimization and promotion use the train set.
        val_ids = list(train_ids)

    return {"train": sorted(train_ids), "validation": sorted(val_ids), "test": sorted(test_ids)}


def records_for_split(records: Sequence[CaseRecord], split_ids: Sequence[str]) -> list[CaseRecord]:
    wanted = set(split_ids)
    return [record for record in records if record.case_id in wanted]


def example_for_record(record: CaseRecord, program_name: str) -> dspy.Example:
    name = canonical_program_name(program_name)
    common = {
        "labels": record.gold_for(name),
        "case_id": record.case_id,
        "program_name": name,
    }
    if name in MODALITY_PROGRAM_NAMES:
        return dspy.Example(report=record.report_for(name), **common).with_inputs("report")

    # Combined is optional. Ground-truth modality labels are used as preliminary
    # labels during optimization, which isolates the Combined prompt itself.
    return dspy.Example(
        ct_report=record.ct_report,
        cta_report=record.cta_report,
        ctp_report=record.ctp_report,
        mri_report=record.mri_report,
        ct_labels=list(record.ct_gt),
        cta_labels=list(record.cta_gt),
        ctp_labels=list(record.ctp_gt),
        **common,
    ).with_inputs(
        "ct_report",
        "cta_report",
        "ctp_report",
        "mri_report",
        "ct_labels",
        "cta_labels",
        "ctp_labels",
    )


def examples_for_records(records: Sequence[CaseRecord], program_name: str) -> list[dspy.Example]:
    return [example_for_record(record, program_name) for record in records]


# ---------------------------------------------------------------------------
# Metric and evaluation
# ---------------------------------------------------------------------------


def _prediction_field(prediction: Any, field_name: str, default: Any) -> Any:
    if isinstance(prediction, dict):
        return prediction.get(field_name, default)
    try:
        return prediction[field_name]
    except Exception:
        pass
    value = getattr(prediction, field_name, default)
    return value if not callable(value) else default


def _raw_labels(prediction: Any) -> Any:
    return _prediction_field(prediction, "labels", ["NONE"])


def _invalid_prediction_labels(raw: Any) -> list[str]:
    if not isinstance(raw, (list, tuple, set)):
        return [str(raw)]
    invalid: list[str] = []
    canonical: list[str] = []
    for item in raw:
        label = str(item).strip().upper().replace(" ", "")
        if label not in cfg.ALLOWED_LABELS:
            invalid.append(str(item))
        elif label not in canonical:
            canonical.append(label)
    if "NONE" in canonical and len(canonical) > 1:
        invalid.append("NONE mixed with positive label(s)")
    return invalid


def make_gepa_metric(program_name: str) -> Callable[..., dspy.Prediction]:
    name = canonical_program_name(program_name)

    def metric(
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> dspy.Prediction:
        del trace, pred_trace
        expected = normalize_label_value(gold["labels"])
        raw_predicted = _raw_labels(pred)
        predicted = normalize_label_value(raw_predicted)
        missing = [label for label in expected if label not in predicted]
        extra = [label for label in predicted if label not in expected]
        invalid = _invalid_prediction_labels(raw_predicted)
        exact = expected == predicted and not invalid

        if exact:
            feedback = (
                f"Correct {name} exact label set {expected}. Preserve the general rule that produced this "
                "answer; do not memorize the case ID or report wording."
            )
        else:
            feedback_lines = [
                f"Incorrect {name} exact-match classification.",
                f"Expected: {expected}",
                f"Predicted: {predicted}",
            ]
            if missing:
                feedback_lines.append(f"Missing labels: {missing}")
            if extra:
                feedback_lines.append(f"Extra labels: {extra}")
            if invalid:
                feedback_lines.append(f"Invalid raw labels: {invalid}")
            feedback_lines.extend(
                [
                    "Inspect the report in the predictor trace and propose a general, modality-specific rule "
                    "that fixes this error without harming already-correct examples.",
                    "Do not encode case IDs, exact report sentences, or ground-truth answers as memorized exceptions.",
                ]
            )
            feedback = "\n".join(feedback_lines)

        if pred_name:
            feedback = f"Predictor {pred_name}:\n{feedback}"
        return dspy.Prediction(score=1.0 if exact else 0.0, feedback=feedback)

    return metric


def _evaluate_one(
    program: Any,
    example: dspy.Example,
    *,
    adapter: Any,
    lm: Any,
    report_excerpt_chars: int,
    include_full_reports: bool,
) -> dict[str, Any]:
    identifier = str(example["case_id"])
    expected = normalize_label_value(example["labels"])
    inputs = dict(example.inputs())
    try:
        local_program = copy.deepcopy(program)
        with dspy.context(lm=lm, adapter=adapter):
            prediction = local_program(**inputs)
        raw = _raw_labels(prediction)
        predicted = normalize_label_value(raw)
        invalid = _invalid_prediction_labels(raw)
        reasoning = str(_prediction_field(prediction, "reasoning", "")).strip()
        error = ""
    except Exception as exc:
        raw = None
        predicted = ["NONE"]
        invalid = []
        reasoning = ""
        error = repr(exc)

    exact = not error and not invalid and expected == predicted
    report_payload: dict[str, Any] = {}
    for key, value in inputs.items():
        if not key.endswith("report") and key != "report":
            continue
        text = str(value or "")
        if include_full_reports:
            report_payload[key] = text
        else:
            report_payload[f"{key}_excerpt"] = text[:report_excerpt_chars]

    return {
        "case_id": identifier,
        "expected": expected,
        "predicted": predicted,
        "raw_predicted": raw,
        "invalid_labels": invalid,
        "reasoning": reasoning,
        "exact": exact,
        "error": error,
        **report_payload,
    }


def evaluate_program(
    program: Any,
    examples: Sequence[dspy.Example],
    *,
    program_name: str,
    split_name: str,
    adapter: Any,
    lm: Any,
    num_threads: int,
    report_excerpt_chars: int,
    include_full_reports: bool,
) -> EvaluationResult:
    name = canonical_program_name(program_name)
    if not examples:
        return EvaluationResult(
            program=name,
            split=split_name,
            correct=0,
            total=0,
            accuracy=0.0,
            label_precision=0.0,
            label_recall=0.0,
            label_f1=0.0,
            failures=0,
            invalid_outputs=0,
        )

    outputs: list[dict[str, Any]] = []
    workers = max(1, int(num_threads))
    if workers == 1:
        for example in examples:
            outputs.append(
                _evaluate_one(
                    program,
                    example,
                    adapter=adapter,
                    lm=lm,
                    report_excerpt_chars=report_excerpt_chars,
                    include_full_reports=include_full_reports,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_one,
                    program,
                    example,
                    adapter=adapter,
                    lm=lm,
                    report_excerpt_chars=report_excerpt_chars,
                    include_full_reports=include_full_reports,
                )
                for example in examples
            ]
            for future in as_completed(futures):
                outputs.append(future.result())

    outputs.sort(key=lambda item: item["case_id"])
    correct = sum(1 for output in outputs if output["exact"])
    failures = sum(1 for output in outputs if output["error"])
    invalid_outputs = sum(1 for output in outputs if output["invalid_labels"])

    true_positive = false_positive = false_negative = 0
    predictions_by_case: dict[str, list[str]] = {}
    for output in outputs:
        expected_set = set(output["expected"])
        predicted_set = set(output["predicted"])
        # Treat NONE as the negative class for label-level diagnostics.
        expected_set.discard("NONE")
        predicted_set.discard("NONE")
        true_positive += len(expected_set & predicted_set)
        false_positive += len(predicted_set - expected_set)
        false_negative += len(expected_set - predicted_set)
        predictions_by_case[output["case_id"]] = list(output["predicted"])

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mismatches = [output for output in outputs if not output["exact"]]

    return EvaluationResult(
        program=name,
        split=split_name,
        correct=correct,
        total=len(outputs),
        accuracy=correct / len(outputs),
        label_precision=precision,
        label_recall=recall,
        label_f1=f1,
        failures=failures,
        invalid_outputs=invalid_outputs,
        mismatches=mismatches,
        predictions_by_case=predictions_by_case,
    )


# ---------------------------------------------------------------------------
# Persistence and reports
# ---------------------------------------------------------------------------


def _instruction_hash(program: Any) -> str:
    return hashlib.sha256(get_instructions(program).encode("utf-8")).hexdigest()


def _state_path(output_dir: Path) -> Path:
    return output_dir / "state.json"


def _history_path(output_dir: Path) -> Path:
    return output_dir / "history.jsonl"


def _stop_path(output_dir: Path) -> Path:
    return output_dir / "STOP"


def _program_state_from_json(payload: dict[str, Any]) -> ProgramState:
    return ProgramState(**payload)


def load_optimizer_state(output_dir: Path) -> OptimizerState | None:
    path = _state_path(output_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["programs"] = {
        name: _program_state_from_json(value)
        for name, value in payload.get("programs", {}).items()
    }
    return OptimizerState(**payload)


def save_optimizer_state(output_dir: Path, state: OptimizerState) -> None:
    state.updated_at = _utc_now()
    path = _state_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(asdict(state), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def append_history(output_dir: Path, payload: dict[str, Any]) -> None:
    path = _history_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_prompt_bundle(
    output_dir: Path,
    programs: dict[str, Any],
    evaluations: dict[str, EvaluationResult],
    state: OptimizerState,
) -> None:
    bundle = {
        "version": 1,
        "updated_at": _utc_now(),
        "dspy_version": getattr(dspy, "__version__", "unknown"),
        "dataset_fingerprint": state.dataset_fingerprint,
        "task_model": state.task_model,
        "programs": {},
    }
    for name, program in programs.items():
        evaluation = evaluations.get(name)
        bundle["programs"][name] = {
            "instructions": get_instructions(program),
            "instruction_sha256": _instruction_hash(program),
            "program_file": str(program_path(output_dir / "best_programs", name)),
            "validation": evaluation.summary() if evaluation else {},
        }
    path = output_dir / "best_prompts.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_manual_baseline(
    path: str | Path | None,
    *,
    validation_case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load manual scores, filtering report details to the exact validation IDs.

    ``validate.py`` report.json files include one detail row per case.  Using
    those details avoids comparing a DSPy validation subset with a manual score
    calculated over the full spreadsheet.  Older summary-only reports are still
    displayed, but are marked non-comparable and receive no accuracy delta.
    """
    if path is None or not str(path).strip():
        return {}
    report_path = Path(path)
    if not report_path.exists():
        print(f"WARNING: manual baseline report not found: {report_path}")
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: could not read manual baseline report: {exc}")
        return {}

    target_ids = {str(case_id) for case_id in (validation_case_ids or [])}
    summary = payload.get("summary", {})
    field_results = payload.get("field_results", {})
    baseline: dict[str, Any] = {}
    field_mapping = {"CT": "CT_GT", "CTA": "CTA_GT", "CTP": "CTP_GT", "Combined": "Combined_GT"}

    for program_name, field_name in field_mapping.items():
        detail_rows = field_results.get(field_name, {}).get("details", [])
        if target_ids and isinstance(detail_rows, list):
            detail_by_case = {
                str(row.get("case_id")): row
                for row in detail_rows
                if isinstance(row, dict) and str(row.get("case_id", "")) in target_ids
            }
            if detail_by_case:
                correct = sum(1 for row in detail_by_case.values() if bool(row.get("match", False)))
                total = len(detail_by_case)
                missing_ids = sorted(target_ids - set(detail_by_case))
                accuracy = correct / total if total else 0.0
                baseline[program_name] = {
                    "accuracy": accuracy,
                    "accuracy_percent": round(accuracy * 100.0, 3),
                    "correct": correct,
                    "total": total,
                    "scope": "same-validation-case-ids",
                    "comparable": not missing_ids,
                    "missing_case_ids": missing_ids,
                }
                if missing_ids:
                    print(
                        f"WARNING: manual {program_name} baseline is missing "
                        f"{len(missing_ids)} validation case(s); no delta will be reported."
                    )
                continue

        item = summary.get(field_name, {})
        raw_accuracy = item.get("accuracy")
        if raw_accuracy is None:
            continue
        accuracy = float(raw_accuracy)
        if accuracy > 1.0:
            accuracy /= 100.0
        baseline[program_name] = {
            "accuracy": accuracy,
            "accuracy_percent": round(accuracy * 100.0, 3),
            "correct": item.get("correct"),
            "total": item.get("total"),
            "scope": "full-manual-report-summary",
            "comparable": False,
            "note": "Summary-only baseline was not filtered to the DSPy validation case IDs.",
        }
    return baseline


def aggregate_evaluations(evaluations: dict[str, EvaluationResult]) -> dict[str, Any]:
    total = sum(result.total for result in evaluations.values())
    correct = sum(result.correct for result in evaluations.values())
    accuracy = correct / total if total else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "accuracy_percent": round(accuracy * 100.0, 3),
    }


def write_latest_report(
    output_dir: Path,
    *,
    state: OptimizerState,
    evaluations: dict[str, EvaluationResult],
    test_evaluations: dict[str, EvaluationResult] | None,
    manual_baseline: dict[str, Any],
    status: str,
) -> None:
    validation_payload: dict[str, Any] = {}
    for name, result in evaluations.items():
        item = result.summary(include_mismatches=True)
        if name in manual_baseline:
            item["manual_baseline"] = manual_baseline[name]
            if bool(manual_baseline[name].get("comparable", False)):
                item["accuracy_delta_vs_manual"] = round(
                    result.accuracy - float(manual_baseline[name]["accuracy"]), 8
                )
        validation_payload[name] = item

    payload = {
        "status": status,
        "generated_at": _utc_now(),
        "state": asdict(state),
        "validation": validation_payload,
        "validation_overall": aggregate_evaluations(evaluations),
        "test": {
            name: result.summary(include_mismatches=True)
            for name, result in (test_evaluations or {}).items()
        },
        "test_overall": aggregate_evaluations(test_evaluations or {}),
        "manual_baseline": manual_baseline,
        "stop_file": str(_stop_path(output_dir)),
    }
    json_path = output_dir / "latest_report.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(json_path)

    lines = [
        f"DSPy optimization status: {status}",
        f"Completed outer rounds: {state.completed_rounds}",
        "",
        "Validation exact-match accuracy:",
    ]
    for name in state.selected_programs:
        result = evaluations[name]
        line = f"- {name}: {result.correct}/{result.total} ({result.accuracy * 100.0:.2f}%)"
        if name in manual_baseline:
            manual = float(manual_baseline[name]["accuracy"])
            line += f" | manual {manual * 100.0:.2f}%"
            if bool(manual_baseline[name].get("comparable", False)):
                line += f" | delta {(result.accuracy - manual) * 100.0:+.2f} pp"
            else:
                line += " | delta unavailable (different or incomplete case scope)"
        lines.append(line)
    aggregate = aggregate_evaluations(evaluations)
    lines.append(
        f"- Overall: {aggregate['correct']}/{aggregate['total']} ({aggregate['accuracy_percent']:.2f}%)"
    )
    lines.extend(
        [
            "",
            f"To stop a continuous run cleanly, create: {_stop_path(output_dir)}",
            "Ctrl+C is also handled and preserves the latest best programs.",
        ]
    )
    (output_dir / "latest_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_evaluation_table(evaluations: dict[str, EvaluationResult], title: str) -> None:
    print("\n" + title)
    print("=" * len(title))
    for name, result in evaluations.items():
        print(
            f"{name:8s} {result.correct:5d}/{result.total:<5d} "
            f"exact={result.accuracy * 100.0:7.3f}%  label-F1={result.label_f1 * 100.0:7.3f}%  "
            f"failures={result.failures}"
        )
    aggregate = aggregate_evaluations(evaluations)
    print(
        f"OVERALL  {aggregate['correct']:5d}/{aggregate['total']:<5d} "
        f"exact={aggregate['accuracy_percent']:7.3f}%"
    )


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------


def load_base_prompt_overrides(source: Mapping[str, Any] | str | Path | None) -> dict[str, str]:
    """Load starting instructions from a config mapping or optional JSON path."""
    if source is None:
        return {}
    if isinstance(source, Mapping):
        payload: Any = dict(source)
    else:
        if not str(source).strip():
            return {}
        prompt_path = Path(source)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Base prompt file not found: {prompt_path}")
        payload = json.loads(prompt_path.read_text(encoding="utf-8"))

    if isinstance(payload, dict) and isinstance(payload.get("programs"), dict):
        payload = payload["programs"]
    if not isinstance(payload, dict):
        raise ValueError(
            "config.DSPY_BASE_INSTRUCTIONS must map CT/CTA/CTP/Combined to instruction text."
        )

    output: dict[str, str] = {}
    for raw_name, raw_value in payload.items():
        name = canonical_program_name(raw_name)
        if isinstance(raw_value, dict):
            raw_value = raw_value.get("instructions", "")
        instruction = str(raw_value or "").strip()
        if not instruction:
            raise ValueError(
                f"Base instruction for {name} cannot be blank in config.DSPY_BASE_INSTRUCTIONS."
            )
        output[name] = instruction
    return output


def _parse_programs(raw: str | Sequence[str]) -> list[str]:
    pieces = raw.split(",") if isinstance(raw, str) else list(raw)
    names: list[str] = []
    for piece in pieces:
        if not str(piece).strip():
            continue
        name = canonical_program_name(piece)
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("config.DSPY_OPTIMIZER_PROGRAMS must select at least one program.")
    return names


def _normalize_target_accuracy(value: float) -> float:
    target = float(value)
    if target > 1.0:
        target /= 100.0
    if not 0.0 < target <= 1.0:
        raise ValueError("target accuracy must be in (0, 1] or a percentage in (0, 100].")
    return target


def _initial_state(
    *,
    records: Sequence[CaseRecord],
    reports_file: str,
    ground_truth_file: str,
    task_model: str,
    reflection_model: str,
    target_accuracy: float,
    selected_programs: list[str],
    split: dict[str, list[str]],
) -> OptimizerState:
    now = _utc_now()
    return OptimizerState(
        version=1,
        created_at=now,
        updated_at=now,
        dataset_fingerprint=dataset_fingerprint(records),
        reports_file=str(Path(reports_file).resolve()),
        ground_truth_file=str(Path(ground_truth_file).resolve()),
        task_model=task_model,
        reflection_model=reflection_model,
        target_accuracy=target_accuracy,
        completed_rounds=0,
        selected_programs=selected_programs,
        split=split,
        programs={name: ProgramState(program=name) for name in selected_programs},
    )


def _reset_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    archive_root = output_dir.parent / f"{output_dir.name}_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    shutil.move(str(output_dir), str(archive))
    print(f"Archived previous optimizer state to: {archive}")


def _choose_program(
    selected_programs: Sequence[str],
    evaluations: dict[str, EvaluationResult],
    *,
    target_accuracy: float,
    completed_rounds: int,
    strategy: str,
) -> str | None:
    remaining = [name for name in selected_programs if evaluations[name].accuracy < target_accuracy]
    if not remaining:
        return None
    if strategy == "round-robin":
        return remaining[completed_rounds % len(remaining)]
    return min(
        remaining,
        key=lambda name: (
            evaluations[name].accuracy,
            evaluations[name].label_f1,
            evaluations[name].correct,
            selected_programs.index(name),
        ),
    )


def _candidate_is_better(candidate: EvaluationResult, current: EvaluationResult, promote_f1_ties: bool) -> bool:
    if candidate.correct > current.correct:
        return True
    if candidate.correct < current.correct:
        return False
    if not promote_f1_ties:
        return False
    return candidate.score_key > current.score_key


def _evaluate_all(
    programs: dict[str, Any],
    example_sets: dict[str, list[dspy.Example]],
    *,
    split_name: str,
    adapter: Any,
    task_lm: Any,
    num_threads: int,
    report_excerpt_chars: int,
    include_full_reports: bool,
) -> dict[str, EvaluationResult]:
    results: dict[str, EvaluationResult] = {}
    for name, program in programs.items():
        results[name] = evaluate_program(
            program,
            example_sets[name],
            program_name=name,
            split_name=split_name,
            adapter=adapter,
            lm=task_lm,
            num_threads=num_threads,
            report_excerpt_chars=report_excerpt_chars,
            include_full_reports=include_full_reports,
        )
    return results


def run_optimizer(args: OptimizerSettings) -> int:
    if args.max_rounds < 0:
        raise ValueError("max_rounds cannot be negative; use 0 for an unbounded run.")
    if args.metric_calls_per_round < 1:
        raise ValueError("metric_calls_per_round must be at least 1.")
    if args.reflection_minibatch_size < 1:
        raise ValueError("reflection_minibatch_size must be at least 1.")
    if args.num_threads < 1:
        raise ValueError("num_threads must be at least 1.")
    if args.max_consecutive_errors < 1:
        raise ValueError("max_consecutive_errors must be at least 1.")
    if args.report_excerpt_chars < 0:
        raise ValueError("report_excerpt_chars cannot be negative.")

    if not str(args.reports).strip():
        raise ValueError("config.DSPY_OPTIMIZER_REPORTS_FILE cannot be blank.")
    if not str(args.ground_truth).strip():
        raise ValueError("config.DSPY_OPTIMIZER_GROUND_TRUTH_FILE cannot be blank.")
    if not str(args.task_model).strip():
        raise ValueError("config.DSPY_TASK_MODEL cannot be blank.")
    if args.adapter not in {"json", "chat"}:
        raise ValueError("config.DSPY_OPTIMIZER_ADAPTER must be 'json' or 'chat'.")
    if args.selection_strategy not in {"worst", "round-robin"}:
        raise ValueError(
            "config.DSPY_SELECTION_STRATEGY must be 'worst' or 'round-robin'."
        )

    selected_programs = _parse_programs(args.programs)
    base_prompt_overrides = load_base_prompt_overrides(args.base_prompts)
    if args.base_prompt_file is not None:
        base_prompt_overrides.update(
            load_base_prompt_overrides(args.base_prompt_file)
        )
    target_accuracy = _normalize_target_accuracy(args.target_accuracy)
    output_dir = Path(args.output_dir)
    if args.reset:
        _reset_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "best_programs").mkdir(parents=True, exist_ok=True)
    (output_dir / "rounds").mkdir(parents=True, exist_ok=True)
    (output_dir / "gepa_logs").mkdir(parents=True, exist_ok=True)

    stop_file = _stop_path(output_dir)
    if args.clear_stop and stop_file.exists():
        stop_file.unlink()
    if stop_file.exists() and not args.evaluate_only:
        print(f"STOP file already exists: {stop_file}. Remove it or set "
            "config.DSPY_CLEAR_STOP_FILE_ON_START = True.")
        return 0

    records = load_case_records(
        args.reports,
        args.ground_truth,
        require_combined_gt="Combined" in selected_programs,
    )
    fingerprint = dataset_fingerprint(records)
    existing_state = load_optimizer_state(output_dir)

    if existing_state and not args.reset:
        split = existing_state.split
        # Keep the saved split only when all referenced IDs still exist.
        current_ids = {record.case_id for record in records}
        split_ids = set().union(*[set(ids) for ids in split.values()])
        if split_ids != current_ids:
            print("Dataset membership changed; regenerating the train/validation/test split.")
            split = create_split(
                records,
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
                seed=args.seed,
            )
        elif existing_state.dataset_fingerprint != fingerprint:
            print(
                "WARNING: report or ground-truth content changed for existing case IDs. "
                "Keeping the fixed case split, re-evaluating saved programs, and continuing from the previous best."
            )
    else:
        split = create_split(
            records,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )

    if existing_state and not args.reset:
        state = existing_state
        state.dataset_fingerprint = fingerprint
        state.reports_file = str(Path(args.reports).resolve())
        state.ground_truth_file = str(Path(args.ground_truth).resolve())
        state.task_model = args.task_model
        state.reflection_model = args.reflection_model or args.task_model
        state.target_accuracy = target_accuracy
        state.selected_programs = selected_programs
        state.split = split
        for name in selected_programs:
            state.programs.setdefault(name, ProgramState(program=name))
        state.programs = {name: state.programs[name] for name in selected_programs}
    else:
        state = _initial_state(
            records=records,
            reports_file=args.reports,
            ground_truth_file=args.ground_truth,
            task_model=args.task_model,
            reflection_model=args.reflection_model or args.task_model,
            target_accuracy=target_accuracy,
            selected_programs=selected_programs,
            split=split,
        )

    train_records = records_for_split(records, split["train"])
    validation_records = records_for_split(records, split["validation"])
    test_records = records_for_split(records, split["test"])
    print(
        f"Loaded {len(records)} matched cases: train={len(train_records)}, "
        f"validation={len(validation_records)}, test={len(test_records)}"
    )
    if set(split["train"]) == set(split["validation"]):
        print(
            "WARNING: validation_fraction=0 (or an undersized dataset) made validation equal training. "
            "Reaching 100% will be training-set accuracy, not evidence of generalization."
        )

    train_examples = {
        name: examples_for_records(train_records, name) for name in selected_programs
    }
    validation_examples = {
        name: examples_for_records(validation_records, name) for name in selected_programs
    }
    test_examples = {
        name: examples_for_records(test_records, name) for name in selected_programs
    }

    api_base = args.api_base or getattr(cfg, "DSPY_OLLAMA_API_BASE", "") or ollama_api_base(cfg.OLLAMA_URL)
    # The optimizer intentionally disables DSPy/LiteLLM response caching. Each
    # training, validation, candidate, and reflection call should hit the local
    # model so prompt changes and repeated evaluations are measured directly.
    task_lm = create_lm(
        model=args.task_model,
        api_base=api_base,
        temperature=0.0,
        max_tokens=args.max_tokens,
        num_ctx=args.num_ctx,
        timeout_seconds=args.timeout,
        cache=False,
        num_retries=args.num_retries,
    )
    reflection_lm = create_lm(
        model=args.reflection_model or args.task_model,
        api_base=api_base,
        temperature=args.reflection_temperature,
        max_tokens=args.reflection_max_tokens,
        num_ctx=args.num_ctx,
        timeout_seconds=args.timeout,
        cache=False,
        num_retries=args.num_retries,
    )
    adapter = create_adapter(args.adapter)
    dspy.configure(lm=task_lm, adapter=adapter)

    programs: dict[str, Any] = {}
    best_dir = output_dir / "best_programs"
    for name in selected_programs:
        saved = program_path(best_dir, name)
        if saved.exists() and not args.reset:
            programs[name] = load_program(best_dir, name, require_saved=True)
        else:
            programs[name] = create_program(name, instructions=base_prompt_overrides.get(name))
            save_program(programs[name], saved)

    evaluations = _evaluate_all(
        programs,
        validation_examples,
        split_name="validation",
        adapter=adapter,
        task_lm=task_lm,
        num_threads=args.num_threads,
        report_excerpt_chars=args.report_excerpt_chars,
        include_full_reports=args.include_full_reports,
    )
    print_evaluation_table(evaluations, "Current best validation scores")

    for name, result in evaluations.items():
        program_state = state.programs[name]
        program_state.validation = result.summary()
        program_state.instruction_sha256 = _instruction_hash(programs[name])
    save_optimizer_state(output_dir, state)
    save_prompt_bundle(output_dir, programs, evaluations, state)

    manual_baseline = load_manual_baseline(
        args.manual_report,
        validation_case_ids=split["validation"],
    )
    test_evaluations: dict[str, EvaluationResult] = {}
    if args.evaluate_only or args.evaluate_test_each_round:
        test_evaluations = _evaluate_all(
            programs,
            test_examples,
            split_name="test",
            adapter=adapter,
            task_lm=task_lm,
            num_threads=args.num_threads,
            report_excerpt_chars=args.report_excerpt_chars,
            include_full_reports=args.include_full_reports,
        ) if test_records else {}
        for name, result in test_evaluations.items():
            state.programs[name].test = result.summary()
        if test_evaluations:
            save_optimizer_state(output_dir, state)
    write_latest_report(
        output_dir,
        state=state,
        evaluations=evaluations,
        test_evaluations=test_evaluations,
        manual_baseline=manual_baseline,
        status="evaluate-only" if args.evaluate_only else "running",
    )
    if args.evaluate_only:
        if test_evaluations:
            print_evaluation_table(test_evaluations, "Held-out test scores")
        return 0

    consecutive_errors = 0
    status = "running"
    try:
        while True:
            if stop_file.exists():
                status = "stopped-by-file"
                print(f"Detected STOP file: {stop_file}")
                break
            selected = _choose_program(
                selected_programs,
                evaluations,
                target_accuracy=target_accuracy,
                completed_rounds=state.completed_rounds,
                strategy=args.selection_strategy,
            )
            if selected is None:
                status = "target-reached"
                print(f"All selected programs reached {target_accuracy * 100.0:.3f}% validation accuracy.")
                break
            if args.max_rounds > 0 and state.completed_rounds >= args.max_rounds:
                status = "max-rounds-reached"
                break

            round_number = state.completed_rounds + 1
            current_eval = evaluations[selected]
            print(
                f"\n===== DSPy outer round {round_number}: optimizing {selected} "
                f"from {current_eval.accuracy * 100.0:.3f}% ====="
            )
            state.programs[selected].attempted_rounds += 1
            round_dir = output_dir / "rounds" / f"round_{round_number:05d}_{selected.lower()}"
            round_dir.mkdir(parents=True, exist_ok=True)

            try:
                optimizer = dspy.GEPA(
                    metric=make_gepa_metric(selected),
                    max_metric_calls=args.metric_calls_per_round,
                    reflection_minibatch_size=args.reflection_minibatch_size,
                    candidate_selection_strategy="current_best",
                    reflection_lm=reflection_lm,
                    skip_perfect_score=True,
                    add_format_failure_as_feedback=True,
                    use_merge=False,
                    num_threads=args.num_threads,
                    failure_score=0.0,
                    perfect_score=1.0,
                    log_dir=str(output_dir / "gepa_logs" / f"round_{round_number:05d}_{selected.lower()}"),
                    track_stats=args.track_gepa_stats,
                    seed=args.seed + round_number,
                )
                candidate = optimizer.compile(
                    student=copy.deepcopy(programs[selected]),
                    trainset=train_examples[selected],
                    valset=validation_examples[selected],
                )
                candidate_path = round_dir / "candidate_program.json"
                save_program(candidate, candidate_path)
                candidate_eval = evaluate_program(
                    candidate,
                    validation_examples[selected],
                    program_name=selected,
                    split_name="validation",
                    adapter=adapter,
                    lm=task_lm,
                    num_threads=args.num_threads,
                    report_excerpt_chars=args.report_excerpt_chars,
                    include_full_reports=args.include_full_reports,
                )
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                state.completed_rounds = round_number
                error_payload = {
                    "event": "round-error",
                    "timestamp": _utc_now(),
                    "round": round_number,
                    "program": selected,
                    "error": repr(exc),
                }
                append_history(output_dir, error_payload)
                (round_dir / "error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
                save_optimizer_state(output_dir, state)
                print(f"Round {round_number} failed: {exc}")
                if consecutive_errors >= args.max_consecutive_errors:
                    status = "too-many-consecutive-errors"
                    break
                continue

            promoted = _candidate_is_better(
                candidate_eval,
                current_eval,
                promote_f1_ties=args.promote_f1_ties,
            )
            if promoted:
                programs[selected] = candidate
                evaluations[selected] = candidate_eval
                save_program(candidate, program_path(best_dir, selected))
                state.programs[selected].accepted_round = round_number
                state.programs[selected].improvements += 1
                outcome = "promoted"
                print(
                    f"PROMOTED {selected}: {current_eval.correct}/{current_eval.total} "
                    f"({current_eval.accuracy * 100.0:.3f}%) -> "
                    f"{candidate_eval.correct}/{candidate_eval.total} "
                    f"({candidate_eval.accuracy * 100.0:.3f}%)"
                )
            else:
                outcome = "rejected"
                print(
                    f"REJECTED {selected} candidate: current {current_eval.accuracy * 100.0:.3f}% "
                    f"vs candidate {candidate_eval.accuracy * 100.0:.3f}%"
                )

            state.completed_rounds = round_number
            state.programs[selected].validation = evaluations[selected].summary()
            state.programs[selected].instruction_sha256 = _instruction_hash(programs[selected])

            round_report = {
                "event": "round-complete",
                "timestamp": _utc_now(),
                "round": round_number,
                "program": selected,
                "outcome": outcome,
                "before": current_eval.summary(),
                "candidate": candidate_eval.summary(include_mismatches=True),
                "best_after": evaluations[selected].summary(),
                "candidate_instructions": get_instructions(candidate),
                "best_instructions_after": get_instructions(programs[selected]),
            }
            (round_dir / "round_report.json").write_text(
                json.dumps(round_report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            append_history(output_dir, round_report)
            save_optimizer_state(output_dir, state)
            save_prompt_bundle(output_dir, programs, evaluations, state)

            if args.evaluate_test_each_round and test_records:
                test_evaluations = _evaluate_all(
                    programs,
                    test_examples,
                    split_name="test",
                    adapter=adapter,
                    task_lm=task_lm,
                    num_threads=args.num_threads,
                    report_excerpt_chars=args.report_excerpt_chars,
                    include_full_reports=args.include_full_reports,
                )
            write_latest_report(
                output_dir,
                state=state,
                evaluations=evaluations,
                test_evaluations=test_evaluations,
                manual_baseline=manual_baseline,
                status="running",
            )
            print_evaluation_table(evaluations, "Best validation scores after round")

    except KeyboardInterrupt:
        status = "stopped-by-keyboard"
        print("\nKeyboard interrupt received. Preserving the latest best programs and state.")

    # Evaluate the held-out test split only at the end by default. It never
    # participates in candidate promotion or stopping.
    if test_records:
        test_evaluations = _evaluate_all(
            programs,
            test_examples,
            split_name="test",
            adapter=adapter,
            task_lm=task_lm,
            num_threads=args.num_threads,
            report_excerpt_chars=args.report_excerpt_chars,
            include_full_reports=args.include_full_reports,
        )
        for name, result in test_evaluations.items():
            state.programs[name].test = result.summary()
        print_evaluation_table(test_evaluations, "Final held-out test scores")

    save_optimizer_state(output_dir, state)
    save_prompt_bundle(output_dir, programs, evaluations, state)
    write_latest_report(
        output_dir,
        state=state,
        evaluations=evaluations,
        test_evaluations=test_evaluations,
        manual_baseline=manual_baseline,
        status=status,
    )
    print(f"\nOptimization status: {status}")
    print(f"Best programs: {best_dir}")
    print(f"Prompt bundle: {output_dir / 'best_prompts.json'}")
    print(f"Latest report: {output_dir / 'latest_report.json'}")
    return 0


# ---------------------------------------------------------------------------
# Config-driven entry point
# ---------------------------------------------------------------------------


def reject_command_line_arguments(arguments: Sequence[str] | None = None) -> None:
    """Keep config.py as the sole control surface for optimization runs."""
    supplied = list(sys.argv[1:] if arguments is None else arguments)
    if supplied:
        raise SystemExit(
            "Command-line arguments are disabled. Edit the DSPy optimization "
            f"settings in config.py instead. Unexpected arguments: {supplied}"
        )


def print_config_summary(settings: OptimizerSettings) -> None:
    """Print the resolved high-level optimizer configuration."""
    print("DSPy optimizer configuration")
    print("============================")
    print(f"Reports:       {settings.reports}")
    print(f"Ground truth:  {settings.ground_truth}")
    print(f"Output:        {settings.output_dir}")
    print(f"Programs:      {', '.join(_parse_programs(settings.programs))}")
    print(f"Task model:    {settings.task_model}")
    print(f"Reflect model: {settings.reflection_model or settings.task_model}")
    print(
        f"Target:        "
        f"{_normalize_target_accuracy(settings.target_accuracy) * 100.0:.3f}%"
    )
    print(f"Max rounds:    {settings.max_rounds or 'unbounded'}")


def run_from_config() -> int:
    """Run DSPy optimization using only values declared in config.py."""
    settings = optimizer_settings_from_config()
    print_config_summary(settings)
    return run_optimizer(settings)


def main() -> None:
    reject_command_line_arguments()
    try:
        exit_code = run_from_config()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"DSPy optimizer configuration error: {exc}") from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
