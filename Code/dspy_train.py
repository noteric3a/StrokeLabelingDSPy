"""
dspy_train.py

DSPy optimization script that trains from TWO spreadsheets:

    1. reports_file: contains CT / CTA / CTP report text
    2. ground_truth_file: contains answer-key labels

Important safety design:
    The model NEVER receives answer-key labels in its prompt.

How this works:
    - DSPy Example objects contain only report_text.
    - Gold labels are kept in Python-only module-level dictionaries.
    - The metric looks up gold labels after the model predicts.
    - MIPRO is run in instruction-only mode with zero labeled/bootstrapped demos
      by default, so few-shot demonstrations do not expose answers.

Example:
    python Code/dspy_train.py --report-type CTA
    python Code/dspy_train.py --report-type CTA --baseline-only
    python Code/dspy_train.py --report-type CTA --smoke-test
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import dspy
import pandas as pd

import config as cfg
from dspy_programs import configure_dspy, CTLabeler, CTALabeler, CTPLabeler
from utils import normalize_labels


# =============================================================================
# Python-only answer-key store
# =============================================================================
# These maps let the metric/evaluator use the answer key WITHOUT storing gold
# labels inside dspy.Example objects.  Only report_text is passed to the model.

_GOLD_BY_REPORT_KEY: Dict[str, str] = {}
_RAW_GOLD_BY_REPORT_KEY: Dict[str, Any] = {}
_CASE_ID_BY_REPORT_KEY: Dict[str, str] = {}


def _report_key(report_text: Any) -> str:
    """Stable key for matching an Example's report text to Python-only gold labels."""
    return " ".join(str(report_text or "").split())


def _example_value(example: Any, field: str, default: Any = "") -> Any:
    """Read a DSPy Example field without accidentally returning bound methods."""
    try:
        value = example[field]
    except Exception:
        try:
            value = example.get(field, default)
        except Exception:
            value = getattr(example, field, default)
    return default if callable(value) else value


def _report_text_for_example(example: Any) -> str:
    return str(_example_value(example, "report_text", "") or "")


def _case_id_for_example(example: Any) -> str:
    return _CASE_ID_BY_REPORT_KEY.get(_report_key(_report_text_for_example(example)), "")


def _raw_gold_for_example(example: Any) -> Any:
    return _RAW_GOLD_BY_REPORT_KEY.get(_report_key(_report_text_for_example(example)), "NONE")


def _gold_labels_for_example(example: Any) -> Set[str]:
    return normalize_gt(_GOLD_BY_REPORT_KEY.get(_report_key(_report_text_for_example(example)), "NONE"))


# =============================================================================
# Normalization helpers
# =============================================================================


def normalize_gt(value: Any) -> Set[str]:
    """Normalize a ground-truth label cell into a set of uppercase labels."""
    if pd.isna(value):
        return {"NONE"}
    text = str(value).strip()
    if not text or text.lower() in {"negative", "normal", "none"}:
        return {"NONE"}
    labels = {part.strip().upper() for part in text.split(",") if part.strip()}
    if not labels:
        return {"NONE"}
    if "NONE" in labels and len(labels) > 1:
        labels.remove("NONE")
    return labels


def normalize_pred(value: Any) -> Set[str]:
    return set(normalize_labels(value))


def normalize_case_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _normalize_col_name(name: Any) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _find_column(df: pd.DataFrame, candidates: Sequence[str], *, purpose: str) -> str:
    existing = list(df.columns)
    for col in candidates:
        if col in df.columns:
            return col

    normalized_existing = {_normalize_col_name(col): col for col in existing}
    for col in candidates:
        key = _normalize_col_name(col)
        if key in normalized_existing:
            return normalized_existing[key]

    raise ValueError(
        f"Could not find {purpose} column.\n"
        f"Tried: {list(candidates)}\n"
        f"Available columns: {existing}"
    )


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _report_column_candidates(report_type: str) -> List[str]:
    report_type = report_type.upper()
    training = cfg.TRAINING_COLUMN_CANDIDATES.get(report_type, {}).get("report", [])
    report_like = cfg.REPORT_COLUMN_CANDIDATES.get(f"{report_type}_Report", [])
    return _dedupe_keep_order([
        *training,
        *report_like,
        report_type,
        f"{report_type} Report",
        f"{report_type}_Report",
        f"{report_type} text",
        f"{report_type}_Text",
    ])


def _gt_column_candidates(report_type: str) -> List[str]:
    report_type = report_type.upper()
    training = cfg.TRAINING_COLUMN_CANDIDATES.get(report_type, {}).get("ground_truth", [])
    modality = []
    if hasattr(cfg, "MODALITY_COLUMN_CANDIDATES"):
        modality = list(cfg.MODALITY_COLUMN_CANDIDATES.get(report_type, {}).get("ground_truth", []))
    return _dedupe_keep_order([
        *training,
        *modality,
        f"{report_type} GT",
        f"{report_type}_GT",
        f"{report_type} Ground Truth",
        f"{report_type}_Ground_Truth",
        report_type,
    ])


def _signature_instructions(report_type: str) -> str:
    report_type = report_type.upper()
    if report_type == "CT":
        return getattr(cfg, "CT_SIGNATURE_INSTRUCTIONS", "")
    if report_type == "CTA":
        return getattr(cfg, "CTA_SIGNATURE_INSTRUCTIONS", "")
    if report_type == "CTP":
        return getattr(cfg, "CTP_SIGNATURE_INSTRUCTIONS", "")
    return ""


# =============================================================================
# DSPy metric
# =============================================================================


def exact_match_metric(example, pred, trace=None) -> float:
    """Return 1.0 only when predicted labels exactly match hidden gold labels."""
    gold = _gold_labels_for_example(example)
    predicted = normalize_pred(getattr(pred, "labels", "NONE"))
    return 1.0 if gold == predicted else 0.0


# =============================================================================
# DSPy optimizer helpers
# =============================================================================


def _mipro_v2_class():
    try:
        return dspy.MIPROv2
    except ImportError as exc:
        raise ImportError(
            "MIPROv2 requires optuna. Install it with: pip install \"dspy[optuna]\" "
            "or run this script with --baseline-only while debugging predictions."
        ) from exc
    except AttributeError:
        try:
            from dspy.teleprompt import MIPROv2
            return MIPROv2
        except ImportError as exc:
            raise ImportError(
                "MIPROv2 requires optuna. Install it with: pip install \"dspy[optuna]\" "
                "or run this script with --baseline-only while debugging predictions."
            ) from exc


def get_optimizer(metric):
    MIPROv2 = _mipro_v2_class()
    return MIPROv2(metric=metric, auto="light")


def compile(optimizer, program, trainset, devset, *, allow_demos: bool = False):
    """Compile while keeping answer-key labels out of all model prompts."""
    compile_kwargs = {"trainset": trainset, "valset": devset}

    if not allow_demos and bool(getattr(cfg, "DSPY_INSTRUCTION_ONLY_OPTIMIZATION", True)):
        compile_kwargs.update({
            "max_bootstrapped_demos": int(getattr(cfg, "DSPY_MAX_BOOTSTRAPPED_DEMOS", 0)),
            "max_labeled_demos": int(getattr(cfg, "DSPY_MAX_LABELED_DEMOS", 0)),
        })

    try:
        return optimizer.compile(program.deepcopy(), **compile_kwargs)
    except TypeError as exc:
        message = str(exc)
        if not allow_demos and any(k in message for k in ("max_bootstrapped_demos", "max_labeled_demos")):
            raise TypeError(
                "This DSPy/MIPROv2 version did not accept max_bootstrapped_demos=0 "
                "and max_labeled_demos=0. Stopping instead of falling back because "
                "fallback could create few-shot demos. Upgrade DSPy or rerun with "
                "--allow-demos if you accept model-generated demos."
            ) from exc
        raise


# =============================================================================
# Loading examples from separate reports + ground-truth files
# =============================================================================


def load_examples(
    reports_file: str,
    ground_truth_file: str,
    report_type: str,
    max_cases: Optional[int] = None,
) -> List[dspy.Example]:
    """Load report-only DSPy Examples and store answer key in Python-only maps."""
    report_type = report_type.upper()
    if report_type not in {"CT", "CTA", "CTP"}:
        raise ValueError("report_type must be CT, CTA, or CTP")

    reports_path = Path(reports_file)
    gt_path = Path(ground_truth_file)
    if not reports_path.exists():
        raise FileNotFoundError(f"Reports file not found: {reports_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    reports_df = pd.read_excel(reports_path)
    gt_df = pd.read_excel(gt_path)

    reports_case_col = _find_column(reports_df, cfg.CASE_ID_COLUMNS, purpose="case ID in reports file")
    gt_case_col = _find_column(gt_df, cfg.CASE_ID_COLUMNS, purpose="case ID in ground-truth file")
    report_col = _find_column(reports_df, _report_column_candidates(report_type), purpose=f"{report_type} report text in reports file")
    gt_col = _find_column(gt_df, _gt_column_candidates(report_type), purpose=f"{report_type} ground-truth labels in ground-truth file")

    print("\nResolved training columns")
    print(f"  Reports file:      {reports_path}")
    print(f"  Ground-truth file: {gt_path}")
    print(f"  Reports case col:  {reports_case_col}")
    print(f"  Reports text col:  {report_col}")
    print(f"  GT case col:       {gt_case_col}")
    print(f"  GT label col:      {gt_col}")

    gt_by_case: Dict[str, Any] = {}
    for _, row in gt_df.iterrows():
        case_key = normalize_case_id(row.get(gt_case_col))
        if case_key:
            gt_by_case[case_key] = row.get(gt_col)

    _GOLD_BY_REPORT_KEY.clear()
    _RAW_GOLD_BY_REPORT_KEY.clear()
    _CASE_ID_BY_REPORT_KEY.clear()

    examples: List[dspy.Example] = []
    missing_gt = 0
    missing_report = 0

    for _, row in reports_df.iterrows():
        case_id = normalize_case_id(row.get(reports_case_col))
        report_text = "" if pd.isna(row.get(report_col)) else str(row.get(report_col)).strip()
        if not case_id:
            continue
        if not report_text:
            missing_report += 1
            continue
        if case_id not in gt_by_case:
            missing_gt += 1
            continue

        raw_gold = gt_by_case[case_id]
        normalized_gold = ", ".join(sorted(normalize_gt(raw_gold)))
        key = _report_key(report_text)
        _GOLD_BY_REPORT_KEY[key] = normalized_gold
        _RAW_GOLD_BY_REPORT_KEY[key] = raw_gold
        _CASE_ID_BY_REPORT_KEY[key] = case_id

        # Critical: the DSPy Example contains only the model input.
        # It does NOT contain labels, reasoning, or case_id.
        examples.append(dspy.Example(report_text=report_text).with_inputs("report_text"))

    if max_cases:
        examples = examples[:max_cases]

    print(f"Loaded {len(examples)} {report_type} examples")
    print(f"Skipped rows missing report text: {missing_report}")
    print(f"Skipped rows missing ground truth: {missing_gt}")
    print("Gold labels are hidden from DSPy Example objects and used only by Python metrics.")

    if not examples:
        raise ValueError(
            "No training examples were loaded. Check shared case IDs and column names."
        )

    return examples


# =============================================================================
# Splitting examples
# =============================================================================


def split_examples(examples: List[dspy.Example], seed: int = 42):
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    n = len(examples)
    if n < 3:
        raise ValueError(f"Only {n} examples were loaded. Use more cases so train/dev/test sets are not empty.")
    train_end = max(1, int(n * 0.70))
    dev_end = max(train_end + 1, int(n * 0.85)) if n >= 4 else train_end + 1
    dev_end = min(dev_end, n)
    trainset = examples[:train_end]
    devset = examples[train_end:dev_end] or examples[:train_end]
    testset = examples[dev_end:] or devset
    return trainset, devset, testset


# =============================================================================
# Evaluation and debugging
# =============================================================================


def prediction_debug_row(*, example: dspy.Example, pred: Any | None = None, error: Exception | None = None) -> Dict[str, Any]:
    gold = _gold_labels_for_example(example)
    report_text = _report_text_for_example(example)
    row: Dict[str, Any] = {
        "case_id": _case_id_for_example(example),
        "gold_raw_for_metric_only": str(_raw_gold_for_example(example)),
        "gold_for_metric_only": sorted(gold),
        "gold_labels_stored_in_dspy_example": bool(_example_value(example, "labels", None) is not None),
        "report_text_preview": report_text[:1000],
    }
    if error is not None:
        row.update({
            "status": "error",
            "match": False,
            "error_type": type(error).__name__,
            "error": str(error)[:5000],
        })
        return row

    raw_labels = getattr(pred, "labels", "NONE")
    raw_reasoning = getattr(pred, "reasoning", "")
    predicted = normalize_pred(raw_labels)
    row.update({
        "status": "ok",
        "match": gold == predicted,
        "predicted": sorted(predicted),
        "raw_labels": str(raw_labels),
        "raw_reasoning": str(raw_reasoning),
        "raw_prediction_repr": repr(pred)[:3000],
    })
    return row


def evaluate(
    program,
    examples: List[dspy.Example],
    name: str,
    error_log_path: Optional[Path] = None,
    history_on_error_path: Optional[Path] = None,
    history_size: int = 3,
    prediction_log_path: Optional[Path] = None,
) -> float:
    correct = 0
    errors: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for ex in examples:
        try:
            pred = program(report_text=_report_text_for_example(ex))
            row = prediction_debug_row(example=ex, pred=pred)
            rows.append(row)
            if row["match"]:
                correct += 1
        except Exception as exc:
            row = prediction_debug_row(example=ex, error=exc)
            rows.append(row)
            errors.append(row)
            if history_on_error_path and bool(getattr(cfg, "DSPY_SAVE_HISTORY_ON_ERROR", True)):
                append_dspy_history_on_error(history_on_error_path, case_id=row.get("case_id", ""), error=exc, n=history_size)

    total = len(examples)
    acc = correct / total if total else 0.0
    print(f"{name}: {correct}/{total} = {acc:.2%} ({total - correct - len(errors)} wrong, {len(errors)} parse/runtime errors)")

    if prediction_log_path:
        prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_log_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(f"Saved {name} prediction details to {prediction_log_path}")
    if errors and error_log_path:
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        error_log_path.write_text(json.dumps(errors, indent=2, default=str), encoding="utf-8")
        print(f"Saved {name} evaluation errors to {error_log_path}")
    return acc


# =============================================================================
# Logging helpers
# =============================================================================


def make_optimization_run_dir(report_type: str, iteration: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("Files") / "Results" / "DSPy_Optimization_Runs" / report_type.upper() / f"{timestamp}_iter_{iteration:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_run_layout(run_dir: Path) -> Dict[str, Path]:
    layout = {"root": run_dir, "debug": run_dir / "debug", "prompts": run_dir / "prompts"}
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def save_dspy_history(path: Path, n: int = 50) -> None:
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            dspy.inspect_history(n=n)
        text = buffer.getvalue()
    except Exception as exc:
        text = f"Could not inspect DSPy history: {exc}\n"
    path.write_text(text, encoding="utf-8")


def append_dspy_history_on_error(path: Path, *, case_id: str, error: Exception, n: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            dspy.inspect_history(n=n)
        history_text = buffer.getvalue()
    except Exception as history_exc:
        history_text = f"Could not inspect DSPy history: {history_exc}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"Case ID: {case_id}\n")
        f.write(f"Error type: {type(error).__name__}\n")
        f.write(f"Error: {str(error)[:5000]}\n")
        f.write("\n--- dspy.inspect_history() ---\n")
        f.write(history_text)
        f.write("\n")


def save_program(program, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        program.save(str(path))
    except Exception as exc:
        path.with_suffix(".error.txt").write_text(str(exc), encoding="utf-8")


def extract_program_instructions(program: Any) -> str:
    for attrs in (("predict", "signature", "instructions"), ("signature", "instructions")):
        value = program
        try:
            for attr in attrs:
                value = getattr(value, attr)
            if value:
                return str(value)
        except Exception:
            pass
    try:
        signature = getattr(getattr(program, "predict", None), "signature", None)
        if isinstance(signature, dict):
            return str(signature.get("instructions", ""))
    except Exception:
        pass
    return ""


def save_examples_debug(path: Path, *, trainset, devset, testset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for split_name, examples in (("train", trainset), ("dev", devset), ("test", testset)):
        for ex in examples:
            rows.append({
                "split": split_name,
                "case_id": _case_id_for_example(ex),
                "gold_labels_for_metric_only": sorted(_gold_labels_for_example(ex)),
                "gold_labels_stored_in_dspy_example": bool(_example_value(ex, "labels", None) is not None),
                "report_text_preview": _report_text_for_example(ex)[:500],
            })
    path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


def save_accuracy_report(path: Path, summary: Dict[str, Any]) -> None:
    test_count = int(summary.get("test_examples") or 0)

    def pct(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.2%}"

    def count(value: Any) -> str:
        if value is None or not test_count:
            return "N/A"
        return f"{round(float(value) * test_count)}/{test_count}"

    lines = [
        f"Report type: {summary.get('report_type')}",
        f"Created at: {summary.get('created_at')}",
        f"Run folder: {summary.get('timestamped_run_dir')}",
        "",
        f"Baseline accuracy: {pct(summary.get('baseline_accuracy'))} ({count(summary.get('baseline_accuracy'))})",
        f"Optimized accuracy: {pct(summary.get('optimized_accuracy'))} ({count(summary.get('optimized_accuracy'))})",
        "",
        f"Max bootstrapped demos: {summary.get('max_bootstrapped_demos')}",
        f"Max labeled demos: {summary.get('max_labeled_demos')}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Training one modality
# =============================================================================


def _program_for_report_type(report_type: str):
    report_type = report_type.upper()
    if report_type == "CT":
        return CTLabeler(), "ct_labeler.json"
    if report_type == "CTA":
        return CTALabeler(), "cta_labeler.json"
    if report_type == "CTP":
        return CTPLabeler(), "ctp_labeler.json"
    raise ValueError("report_type must be CT, CTA, or CTP")


def train_one(
    report_type: str,
    reports_file: str,
    ground_truth_file: str,
    max_cases: Optional[int] = None,
    iteration: int = 1,
    save_run_logs: bool = True,
    history_size: int = 50,
    baseline_only: bool = False,
    smoke_test: bool = False,
    allow_demos: bool = False,
) -> Dict[str, Any]:
    report_type = report_type.upper()
    configure_dspy()

    run_dir: Optional[Path] = None
    layout: Optional[Dict[str, Path]] = None
    if save_run_logs:
        run_dir = make_optimization_run_dir(report_type, iteration)
        layout = make_run_layout(run_dir)
        (layout["prompts"] / "before_prompt.txt").write_text(_signature_instructions(report_type), encoding="utf-8")

    examples = load_examples(reports_file=reports_file, ground_truth_file=ground_truth_file, report_type=report_type, max_cases=max_cases)
    trainset, devset, testset = split_examples(examples)
    program, save_name = _program_for_report_type(report_type)

    print(f"\nTraining {report_type}")
    print(f"Train: {len(trainset)} | Dev: {len(devset)} | Test: {len(testset)}")

    if layout:
        save_program(program, layout["prompts"] / "before_program.json")
        save_examples_debug(layout["debug"] / "loaded_examples.json", trainset=trainset, devset=devset, testset=testset)

    if smoke_test:
        smoke_acc = evaluate(
            program,
            [testset[0]],
            f"{report_type} smoke test",
            error_log_path=(layout["debug"] / "smoke_test_errors.json") if layout else None,
            history_on_error_path=(layout["debug"] / "smoke_test_dspy_history_on_errors.txt") if layout else None,
            history_size=getattr(cfg, "DSPY_ERROR_HISTORY_SIZE", 3),
            prediction_log_path=(layout["debug"] / "smoke_test_predictions.json") if layout else None,
        )
        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "iteration": iteration,
            "report_type": report_type,
            "mode": "smoke_test",
            "smoke_test_accuracy": smoke_acc,
            "timestamped_run_dir": str(run_dir) if run_dir else None,
            "gold_labels_hidden_from_dspy_examples": True,
        }
        if layout:
            save_dspy_history(layout["debug"] / "dspy_history_after_smoke_test.txt", n=history_size)
            (layout["root"] / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    baseline_acc = evaluate(
        program,
        testset,
        f"{report_type} baseline",
        error_log_path=(layout["debug"] / "baseline_errors.json") if layout else None,
        history_on_error_path=(layout["debug"] / "baseline_dspy_history_on_errors.txt") if layout else None,
        history_size=getattr(cfg, "DSPY_ERROR_HISTORY_SIZE", 3),
        prediction_log_path=(layout["debug"] / "baseline_predictions.json") if layout else None,
    )
    if layout:
        save_dspy_history(layout["debug"] / "dspy_history_after_baseline_eval.txt", n=history_size)

    base_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "report_type": report_type,
        "reports_file": str(reports_file),
        "ground_truth_file": str(ground_truth_file),
        "max_cases": max_cases,
        "total_examples": len(examples),
        "train_examples": len(trainset),
        "dev_examples": len(devset),
        "test_examples": len(testset),
        "baseline_accuracy": baseline_acc,
        "gold_labels_hidden_from_dspy_examples": True,
        "instruction_only_optimization": not allow_demos,
        "max_bootstrapped_demos": 0 if not allow_demos else None,
        "max_labeled_demos": 0 if not allow_demos else None,
        "timestamped_run_dir": str(run_dir) if run_dir else None,
    }

    if baseline_only:
        summary = {**base_summary, "mode": "baseline_only", "optimized_accuracy": None, "active_saved_program": None}
        if layout:
            (layout["root"] / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            save_accuracy_report(layout["root"] / "accuracy_report.txt", summary)
        print("Baseline-only mode: skipped MIPRO optimization.")
        return summary

    optimizer = get_optimizer(exact_match_metric)

    try:
        optimized_program = compile(optimizer, program, trainset, devset, allow_demos=allow_demos)
    except Exception as exc:
        if layout:
            save_dspy_history(layout["debug"] / "dspy_history_after_compile_error.txt", n=history_size)
            (layout["debug"] / "compile_error.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
        raise RuntimeError(
            "DSPy optimization failed during compile. Check debug/compile_error.txt and "
            "debug/dspy_history_after_compile_error.txt. Gold labels are not stored in DSPy "
            "Examples, so any answer leakage should be absent unless --allow-demos was used."
        ) from exc

    if layout:
        save_dspy_history(layout["debug"] / "dspy_history_after_compile.txt", n=history_size)

    optimized_acc = evaluate(
        optimized_program,
        testset,
        f"{report_type} optimized",
        error_log_path=(layout["debug"] / "optimized_errors.json") if layout else None,
        history_on_error_path=(layout["debug"] / "optimized_dspy_history_on_errors.txt") if layout else None,
        history_size=getattr(cfg, "DSPY_ERROR_HISTORY_SIZE", 3),
        prediction_log_path=(layout["debug"] / "optimized_predictions.json") if layout else None,
    )

    if layout:
        save_dspy_history(layout["debug"] / "dspy_history_after_optimized_eval.txt", n=history_size)
        save_program(optimized_program, layout["prompts"] / "after_program.json")
        (layout["prompts"] / "after_prompt.txt").write_text(extract_program_instructions(optimized_program), encoding="utf-8")

    Path(cfg.DSPY_PROGRAM_DIR).mkdir(parents=True, exist_ok=True)
    active_save_path = Path(cfg.DSPY_PROGRAM_DIR) / save_name
    optimized_program.save(str(active_save_path))

    summary = {
        **base_summary,
        "optimized_accuracy": optimized_acc,
        "active_saved_program": str(active_save_path),
    }
    if layout:
        summary["folders"] = {"debug": str(layout["debug"]), "prompts": str(layout["prompts"])}
        summary["prompt_files"] = {
            "before_prompt": str(layout["prompts"] / "before_prompt.txt"),
            "after_prompt": str(layout["prompts"] / "after_prompt.txt"),
        }
        (layout["root"] / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        save_accuracy_report(layout["root"] / "accuracy_report.txt", summary)

    print(f"Saved optimized {report_type} program to {active_save_path}")
    if layout:
        print(f"Saved this optimization run to {run_dir}")
        print(f"  Debug files:  {layout['debug']}")
        print(f"  Prompt files: {layout['prompts']}")
    return summary


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Optimize DSPy stroke labelers without leaking answer keys into prompts.")
    parser.add_argument("--reports", "--reports-file", dest="reports_file", default=getattr(cfg, "TRAINING_REPORTS_FILE", cfg.INPUT_REPORT_FILE), help="Path to reports Excel file containing CT/CTA/CTP report text.")
    parser.add_argument("--ground-truth", default=cfg.GROUND_TRUTH_FILE, help="Path to ground-truth answer key Excel file.")
    parser.add_argument("--report-type", choices=["CT", "CTA", "CTP"], required=True, help="Which modality to optimize.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional number of cases for faster testing.")
    parser.add_argument("--loop", action="store_true", help="Keep optimizing repeatedly until stopped with Ctrl+C.")
    parser.add_argument("--no-run-logs", action="store_true", help="Disable timestamped optimization folders.")
    parser.add_argument("--history-size", type=int, default=50, help="Number of visible DSPy history items to save per stage.")
    parser.add_argument("--baseline-only", action="store_true", help="Evaluate the baseline and save debug logs without running MIPRO.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one test prediction and stop.")
    parser.add_argument("--allow-demos", action="store_true", help="Allow MIPRO to build demos. Default is OFF to avoid answer leakage.")
    args = parser.parse_args()

    iteration = 1
    try:
        while True:
            train_one(
                report_type=args.report_type,
                reports_file=args.reports_file,
                ground_truth_file=args.ground_truth,
                max_cases=args.max_cases,
                iteration=iteration,
                save_run_logs=not args.no_run_logs,
                history_size=args.history_size,
                baseline_only=args.baseline_only,
                smoke_test=args.smoke_test,
                allow_demos=args.allow_demos,
            )
            if not args.loop:
                break
            iteration += 1
    except KeyboardInterrupt:
        print("\nStopped DSPy optimization loop.")


if __name__ == "__main__":
    main()
