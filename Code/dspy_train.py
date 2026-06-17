"""
dspy_train.py

Optional DSPy optimization script.

This version trains from TWO spreadsheets:

    1. A reports spreadsheet that contains the report text, for example:
       Files/Report/New Reports.xlsx

    2. A ground-truth key spreadsheet that contains the answer labels, for example:
       Files/GT/GroundTruthKeyNew.xlsx

The files are joined by case ID / Case Name.

Example:

    python dspy_train.py \
      --reports "Files/Report/New Reports.xlsx" \
      --ground-truth "Files/GT/GroundTruthKeyNew.xlsx" \
      --report-type CTA \
      --max-cases 120

Continuous prompt optimization loop:

    python dspy_train.py \
      --reports "Files/Report/New Reports.xlsx" \
      --ground-truth "Files/GT/GroundTruthKeyNew.xlsx" \
      --report-type CTA \
      --loop

Stop loop mode with Ctrl+C.
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

# Import base DSPy programs and model configuration.
from dspy_programs import (
    configure_dspy,
    CTLabeler,
    CTALabeler,
    CTPLabeler,
)

# Reuse your existing label normalizer.
from utils import normalize_labels


# =============================================================================
# Normalization helpers
# =============================================================================


def normalize_gt(value: Any) -> Set[str]:
    """
    Normalize ground truth labels.

    Examples:
        "RMCA"       -> {"RMCA"}
        "RMCA, LMCA" -> {"RMCA", "LMCA"}
        "negative"   -> {"NONE"}
        empty cell    -> {"NONE"}
    """

    if pd.isna(value):
        return {"NONE"}

    text = str(value).strip()

    if not text:
        return {"NONE"}

    if text.lower() == "negative":
        return {"NONE"}

    return {
        part.strip().upper()
        for part in text.split(",")
        if part.strip()
    } or {"NONE"}



def normalize_pred(value: Any) -> Set[str]:
    return set(normalize_labels(value))



def normalize_case_id(value: Any) -> str:
    """
    Normalize case IDs so that the reports spreadsheet and ground-truth key can
    be joined reliably.
    """

    if pd.isna(value):
        return ""

    text = str(value).strip()

    # Excel sometimes reads numeric IDs as floats, e.g. 123.0.
    if text.endswith(".0"):
        text = text[:-2]

    return text



def _normalize_col_name(name: Any) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())



def _find_column(df: pd.DataFrame, candidates: Sequence[str], *, purpose: str) -> str:
    """
    Find a dataframe column using exact matching first, then relaxed matching.
    This avoids failures when the spreadsheet uses CT vs CT_Report, CTA GT vs
    CTA_GT, etc.
    """

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
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out



def _report_column_candidates(report_type: str) -> List[str]:
    """
    Report-text column candidates.

    For New Reports.xlsx, columns are usually exactly CT, CTA, CTP, MRI.
    Older versions used CT_Report, CTA_Report, etc.
    """

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
    """
    Ground-truth label column candidates.
    """

    report_type = report_type.upper()
    training = cfg.TRAINING_COLUMN_CANDIDATES.get(report_type, {}).get("ground_truth", [])
    modality = []

    if hasattr(cfg, "MODALITY_COLUMN_CANDIDATES"):
        modality = list(
            cfg.MODALITY_COLUMN_CANDIDATES.get(report_type, {}).get("ground_truth", [])
        )

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
    """
    Metric used by DSPy during optimization.

    Returns 1.0 if the predicted label set exactly matches ground truth,
    otherwise 0.0.
    """

    gold = normalize_gt(example.labels)
    predicted = normalize_pred(getattr(pred, "labels", "NONE"))

    return 1.0 if gold == predicted else 0.0


# =============================================================================
# Optimizer compatibility helper
# =============================================================================


def get_optimizer(metric):
    """
    Create a DSPy optimizer.

    DSPy versions sometimes expose MIPROv2 in slightly different places.
    auto="light" is a good starting point for local Ollama optimization.
    """

    try:
        return dspy.MIPROv2(metric=metric, auto="light")
    except AttributeError:
        from dspy.teleprompt import MIPROv2
        return MIPROv2(metric=metric, auto="light")


# =============================================================================
# Loading examples from separate reports + ground-truth files
# =============================================================================


def load_examples(
    reports_file: str,
    ground_truth_file: str,
    report_type: str,
    max_cases: Optional[int] = None,
) -> List[dspy.Example]:
    """
    Load DSPy examples by joining:

        reports_file      -> contains report text columns: CT, CTA, CTP, MRI
        ground_truth_file -> contains answer-key columns: CT_GT, CTA_GT, CTP_GT

    The join key is Case Name / case_id / Case ID.
    """

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

    reports_case_col = _find_column(
        reports_df,
        cfg.CASE_ID_COLUMNS,
        purpose="case ID in reports file",
    )
    gt_case_col = _find_column(
        gt_df,
        cfg.CASE_ID_COLUMNS,
        purpose="case ID in ground-truth file",
    )

    report_col = _find_column(
        reports_df,
        _report_column_candidates(report_type),
        purpose=f"{report_type} report text in reports file",
    )
    gt_col = _find_column(
        gt_df,
        _gt_column_candidates(report_type),
        purpose=f"{report_type} ground-truth labels in ground-truth file",
    )

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
        if not case_key:
            continue
        gt_by_case[case_key] = row.get(gt_col)

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

        labels = gt_by_case[case_id]

        ex = dspy.Example(
            case_id=case_id,
            report_text=report_text,
            labels=", ".join(sorted(normalize_gt(labels))),
        ).with_inputs("report_text")

        examples.append(ex)

    if max_cases:
        examples = examples[:max_cases]

    print(f"Loaded {len(examples)} {report_type} examples")
    print(f"Skipped rows missing report text: {missing_report}")
    print(f"Skipped rows missing ground truth: {missing_gt}")

    if not examples:
        raise ValueError(
            "No training examples were loaded. This would make DSPy's trainset empty.\n"
            "Check that the reports file and ground-truth key share the same case IDs, "
            "and that the report/ground-truth column names are correct."
        )

    return examples


# =============================================================================
# Splitting examples
# =============================================================================


def split_examples(examples: List[dspy.Example], seed: int = 42):
    """
    Split examples into train/dev/test sets.
    """

    examples = list(examples)
    random.Random(seed).shuffle(examples)

    n = len(examples)

    if n < 3:
        raise ValueError(
            f"Only {n} examples were loaded. Use more cases so train/dev/test "
            "sets are not empty."
        )

    train_end = max(1, int(n * 0.70))
    dev_end = max(train_end + 1, int(n * 0.85)) if n >= 4 else train_end + 1
    dev_end = min(dev_end, n)

    trainset = examples[:train_end]
    devset = examples[train_end:dev_end]
    testset = examples[dev_end:]

    # If the dataset is small, keep DSPy from receiving an empty validation set.
    if not devset:
        devset = trainset

    # If the dataset is small, still evaluate on something instead of crashing.
    if not testset:
        testset = devset

    if not trainset:
        raise ValueError("Trainset cannot be empty. Load more examples or increase --max-cases.")

    return trainset, devset, testset


# =============================================================================
# Evaluation
# =============================================================================


def evaluate(
    program,
    examples: List[dspy.Example],
    name: str,
    error_log_path: Optional[Path] = None,
    history_on_error_path: Optional[Path] = None,
    history_size: int = 3,
) -> float:
    """
    Evaluate a DSPy program on exact-match accuracy.

    Important: local Ollama thinking models can occasionally return a response
    that DSPy's adapter cannot parse. During evaluation, that should count as
    a wrong answer instead of crashing the entire training run.
    """

    correct = 0
    errors: List[Dict[str, Any]] = []

    for ex in examples:
        try:
            pred = program(report_text=ex.report_text)
            if exact_match_metric(ex, pred) == 1.0:
                correct += 1
        except Exception as exc:
            case_id = str(getattr(ex, "case_id", ""))
            errors.append({
                "case_id": case_id,
                "error_type": type(exc).__name__,
                "error": str(exc)[:5000],
                "ground_truth": str(getattr(ex, "labels", "")),
                "report_text_preview": str(getattr(ex, "report_text", ""))[:1000],
            })

            if history_on_error_path and bool(getattr(cfg, "DSPY_SAVE_HISTORY_ON_ERROR", True)):
                append_dspy_history_on_error(
                    history_on_error_path,
                    case_id=case_id,
                    error=exc,
                    n=history_size,
                )

            continue

    total = len(examples)
    acc = correct / total if total else 0.0

    if errors:
        print(f"{name}: {correct}/{total} = {acc:.2%} ({len(errors)} parse/runtime errors counted as wrong)")
        if error_log_path:
            error_log_path.parent.mkdir(parents=True, exist_ok=True)
            error_log_path.write_text(json.dumps(errors, indent=2, default=str), encoding="utf-8")
            print(f"Saved {name} evaluation errors to {error_log_path}")
    else:
        print(f"{name}: {correct}/{total} = {acc:.2%}")

    return acc


# =============================================================================
# Logging helpers
# =============================================================================


def make_optimization_run_dir(report_type: str, iteration: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path("Files")
        / "Results"
        / "DSPy_Optimization_Runs"
        / report_type.upper()
        / f"{timestamp}_iter_{iteration:04d}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir



def save_dspy_history(path: Path, n: int = 50) -> None:
    """
    Save visible DSPy LM history. DSPy does not always expose every internal
    candidate prompt, but this captures what inspect_history can show.
    """

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            dspy.inspect_history(n=n)
        text = buffer.getvalue()
    except Exception as exc:
        text = f"Could not inspect DSPy history: {exc}\n"

    path.write_text(text, encoding="utf-8")





def append_dspy_history_on_error(path: Path, *, case_id: str, error: Exception, n: int = 3) -> None:
    """Append the latest visible DSPy LM history for a failed example.

    This is for debugging truncation/parse failures. It captures the prompt and
    response that DSPy can expose via dspy.inspect_history().
    """
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
) -> Dict[str, Any]:
    """
    Optimize one modality-specific DSPy program.
    """

    report_type = report_type.upper()

    # Configure the Ollama model through DSPy.
    configure_dspy()

    run_dir: Optional[Path] = None
    if save_run_logs:
        run_dir = make_optimization_run_dir(report_type, iteration)
        (run_dir / "base_signature_instructions.txt").write_text(
            _signature_instructions(report_type),
            encoding="utf-8",
        )

    # Load examples by joining the reports file to the answer key.
    examples = load_examples(
        reports_file=reports_file,
        ground_truth_file=ground_truth_file,
        report_type=report_type,
        max_cases=max_cases,
    )

    # Split into train/dev/test.
    trainset, devset, testset = split_examples(examples)

    # Select the correct base program.
    program, save_name = _program_for_report_type(report_type)

    print(f"\nTraining {report_type}")
    print(f"Train: {len(trainset)} | Dev: {len(devset)} | Test: {len(testset)}")

    if run_dir:
        save_program(program, run_dir / "baseline_program.json")

    # Evaluate the unoptimized program first.
    baseline_acc = evaluate(
        program,
        testset,
        f"{report_type} baseline",
        error_log_path=(run_dir / "baseline_evaluation_errors.json") if run_dir else None,
        history_on_error_path=(run_dir / "baseline_dspy_history_on_errors.txt") if run_dir else None,
        history_size=getattr(cfg, "DSPY_ERROR_HISTORY_SIZE", 3),
    )

    if run_dir:
        save_dspy_history(run_dir / "dspy_history_after_baseline_eval.txt", n=history_size)

    # Create optimizer.
    optimizer = get_optimizer(exact_match_metric)

    # Compile/optimize the program.
    try:
        optimized_program = optimizer.compile(
            program.deepcopy(),
            trainset=trainset,
            valset=devset,
        )
    except Exception as exc:
        if run_dir:
            save_dspy_history(run_dir / "dspy_history_after_compile_error.txt", n=history_size)
            (run_dir / "compile_error.txt").write_text(
                f"{type(exc).__name__}: {exc}",
                encoding="utf-8",
            )
        raise RuntimeError(
            "DSPy optimization failed during compile. The examples loaded correctly, "
            "but the model/adapter failed while testing candidate prompts. "
            "This is usually caused by truncated or unparsable model output. "
            "Check *_dspy_history_on_errors.txt / dspy_history_after_compile_error.txt. If the history shows long reasoning_content or no final labels field, use a non-thinking Ollama model or a custom DSPy LM wrapper that passes think=False."
        ) from exc

    if run_dir:
        save_dspy_history(run_dir / "dspy_history_after_compile.txt", n=history_size)

    # Evaluate optimized program.
    optimized_acc = evaluate(
        optimized_program,
        testset,
        f"{report_type} optimized",
        error_log_path=(run_dir / "optimized_evaluation_errors.json") if run_dir else None,
        history_on_error_path=(run_dir / "optimized_dspy_history_on_errors.txt") if run_dir else None,
        history_size=getattr(cfg, "DSPY_ERROR_HISTORY_SIZE", 3),
    )

    if run_dir:
        save_dspy_history(run_dir / "dspy_history_after_optimized_eval.txt", n=history_size)
        save_program(optimized_program, run_dir / "optimized_program.json")

    # Save active optimized program. Main labeling will load this automatically.
    Path(cfg.DSPY_PROGRAM_DIR).mkdir(parents=True, exist_ok=True)
    active_save_path = Path(cfg.DSPY_PROGRAM_DIR) / save_name
    optimized_program.save(str(active_save_path))

    summary = {
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
        "optimized_accuracy": optimized_acc,
        "active_saved_program": str(active_save_path),
        "timestamped_run_dir": str(run_dir) if run_dir else None,
    }

    if run_dir:
        (run_dir / "optimization_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    print(f"Saved optimized {report_type} program to {active_save_path}")
    if run_dir:
        print(f"Saved this optimization run to {run_dir}")

    return summary


# =============================================================================
# Command-line entry point
# =============================================================================


def main():
    """
    Parse command-line args and train one modality.
    """

    parser = argparse.ArgumentParser(description="Optimize DSPy stroke labelers.")

    parser.add_argument(
        "--reports",
        "--reports-file",
        dest="reports_file",
        default=getattr(cfg, "TRAINING_REPORTS_FILE", cfg.INPUT_REPORT_FILE),
        help="Path to reports Excel file containing CT/CTA/CTP report text.",
    )

    parser.add_argument(
        "--ground-truth",
        default=cfg.GROUND_TRUTH_FILE,
        help="Path to ground-truth answer key Excel file.",
    )

    parser.add_argument(
        "--report-type",
        choices=["CT", "CTA", "CTP"],
        required=True,
        help="Which modality to optimize.",
    )

    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional number of cases for faster testing.",
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep optimizing repeatedly until stopped with Ctrl+C.",
    )

    parser.add_argument(
        "--no-run-logs",
        action="store_true",
        help="Disable timestamped optimization folders.",
    )

    parser.add_argument(
        "--history-size",
        type=int,
        default=50,
        help="Number of visible DSPy history items to save per stage.",
    )

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
            )

            if not args.loop:
                break

            iteration += 1

    except KeyboardInterrupt:
        print("\nStopped DSPy optimization loop.")


if __name__ == "__main__":
    main()
