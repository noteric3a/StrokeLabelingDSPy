"""
dspy_train.py

Optional DSPy optimization script.

This is the file that makes DSPy more than just a cleaner prompt wrapper.

It lets you train/optimize a DSPy module using your ground truth spreadsheet.

Recommended order:

    1. Optimize CT first.
    2. Optimize CTA second.
    3. Optimize CTP third.
    4. Do Combined last after CT/CTA/CTP are stable.

Example:

    python dspy_train.py \
      --ground-truth Files/Ground_truths.xlsx \
      --report-type CT \
      --max-cases 120

This will:

    1. Load CT examples from the spreadsheet.
    2. Split them into train/dev/test.
    3. Evaluate the baseline DSPy CTLabeler.
    4. Optimize the CTLabeler with MIPROv2.
    5. Evaluate the optimized program.
    6. Save it to optimized_programs/ct_labeler.json.

Then main_dspy.py will automatically try to load that saved program.
"""

from __future__ import annotations
import argparse
import contextlib
import io
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Set
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

    This should match your validator's behavior as closely as possible.

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
    }


def normalize_pred(value: Any) -> Set[str]:
    return set(normalize_labels(value))


# =============================================================================
# DSPy metric
# =============================================================================

def exact_match_metric(example, pred, trace=None) -> float:
    """
    Metric used by DSPy during optimization.

    DSPy tries to improve this score.

    This metric returns:
        1.0 if predicted label set exactly matches ground truth.
        0.0 otherwise.

    This mirrors the strict style of your validator.
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
    This helper tries both common locations.

    auto="light" is recommended first because local Ollama optimization can be slow.
    """

    try:
        return dspy.MIPROv2(metric=metric, auto="light")
    except AttributeError:
        from dspy.teleprompt import MIPROv2
        return MIPROv2(metric=metric, auto="light")


# =============================================================================
# Loading examples
# =============================================================================

def _first_existing(row, candidates: List[str]) -> Any:
    """
    Return first non-empty value from candidate column names.
    """

    for col in candidates:
        if col in row and not pd.isna(row.get(col)):
            return row.get(col)

    return ""


def load_examples(
    ground_truth_file: str,
    report_type: str,
    max_cases: int | None = None,
) -> List[dspy.Example]:
    """
    Load DSPy examples from your ground truth spreadsheet.

    Each DSPy example needs:
        - report_text input
        - labels output

    The labels are the answer key.

    Args:
        ground_truth_file:
            Path to your ground truth spreadsheet.

        report_type:
            CT, CTA, or CTP.

        max_cases:
            Optional limit for faster experiments.
    """

    df = pd.read_excel(ground_truth_file)

    report_type = report_type.upper()

    # Decide which columns to use based on config.py.
    if report_type not in cfg.TRAINING_COLUMN_CANDIDATES:
        raise ValueError("report_type must be CT, CTA, or CTP")
    report_col_candidates = cfg.TRAINING_COLUMN_CANDIDATES[report_type]["report"]
    gt_col_candidates = cfg.TRAINING_COLUMN_CANDIDATES[report_type]["ground_truth"]

    examples: List[dspy.Example] = []

    for _, row in df.iterrows():
        case_id = str(_first_existing(row, cfg.CASE_ID_COLUMNS) or "").strip()

        report_text = str(_first_existing(row, report_col_candidates) or "").strip()
        labels = _first_existing(row, gt_col_candidates)

        # Skip examples without case id or report text.
        if not case_id or not report_text:
            continue

        # DSPy examples store both input and expected output.
        #
        # .with_inputs("report_text") tells DSPy:
        #     report_text is the input field.
        #
        # labels remains the expected answer used by the metric.
        ex = dspy.Example(
            case_id=case_id,
            report_text=report_text,
            labels=", ".join(sorted(normalize_gt(labels))),
        ).with_inputs("report_text")

        examples.append(ex)

    if max_cases:
        examples = examples[:max_cases]

    return examples


# =============================================================================
# Splitting examples
# =============================================================================

def split_examples(examples: List[dspy.Example], seed: int = 42):
    """
    Split examples into train/dev/test sets.

    Important:
        For the full multi-modality experiment, split by case_id first so CT,
        CTA, CTP, and Combined from the same case do not leak across sets.

    Since this script optimizes one modality at a time, this simple shuffle split
    is acceptable for early testing.
    """

    examples = list(examples)
    random.Random(seed).shuffle(examples)

    n = len(examples)

    train_end = int(n * 0.70)
    dev_end = int(n * 0.85)

    trainset = examples[:train_end]
    devset = examples[train_end:dev_end]
    testset = examples[dev_end:]

    return trainset, devset, testset


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(program, examples: List[dspy.Example], name: str) -> float:
    """
    Evaluate a DSPy program on a set of examples.

    This prints exact-match accuracy.
    """

    correct = 0

    for ex in examples:
        pred = program(report_text=ex.report_text)

        if exact_match_metric(ex, pred) == 1.0:
            correct += 1

    total = len(examples)
    acc = correct / total if total else 0.0

    print(f"{name}: {correct}/{total} = {acc:.2%}")

    return acc



# =============================================================================
# Optimization logging helpers
# =============================================================================

def timestamp() -> str:
    return datetime.now().strftime(getattr(cfg, "RUN_TIMESTAMP_FORMAT", "%Y%m%d_%H%M%S"))


def make_optimization_run_dir(report_type: str, iteration: int | None = None) -> Path:
    """Create a timestamped folder for one DSPy optimization attempt."""
    base = Path(getattr(cfg, "DSPY_OPTIMIZATION_LOG_DIR", "Files/Results/DSPy_Optimization_Runs"))
    name = timestamp()
    if iteration is not None:
        name = f"{name}_iter_{iteration:04d}"
    run_dir = base / report_type.upper() / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_program_snapshot(program, path: Path) -> None:
    """Save a DSPy program snapshot, falling back to repr if save() fails."""
    try:
        program.save(str(path))
    except Exception as exc:
        save_text(path.with_suffix(".txt"), f"Could not save DSPy program as JSON: {exc}\n\n{repr(program)}")


def dump_dspy_history(path: Path) -> None:
    """Save DSPy's visible LM interaction history for prompt-audit/debugging."""
    n = int(getattr(cfg, "DSPY_INSPECT_HISTORY_N", 200))
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            result = dspy.inspect_history(n=n)
        text = buffer.getvalue()
        if result is not None:
            text += "\n" + str(result)
        if not text.strip():
            text = "DSPy inspect_history returned no visible history for this version/configuration."
    except Exception as exc:
        text = f"Could not inspect DSPy history: {repr(exc)}"
    save_text(path, text)


def instruction_text_for(report_type: str) -> str:
    report_type = report_type.upper()
    if report_type == "CT":
        return cfg.CT_SIGNATURE_INSTRUCTIONS
    if report_type == "CTA":
        return cfg.CTA_SIGNATURE_INSTRUCTIONS
    if report_type == "CTP":
        return cfg.CTP_SIGNATURE_INSTRUCTIONS
    return ""


# =============================================================================
# Training one modality
# =============================================================================

def train_one(
    report_type: str,
    ground_truth_file: str,
    max_cases: int | None = None,
    *,
    run_dir: Path | None = None,
    iteration: int | None = None,
):
    """
    Optimize one modality-specific DSPy program.

    When run_dir is provided, this also logs the base instructions, baseline
    program, optimized program, DSPy LM history, and summary metrics for the
    experiment run.
    """

    report_type = report_type.upper()
    run_dir = run_dir or make_optimization_run_dir(report_type, iteration)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Configure the Ollama model through DSPy.
    configure_dspy()

    # Load examples from the answer key.
    examples = load_examples(
        ground_truth_file=ground_truth_file,
        report_type=report_type,
        max_cases=max_cases,
    )

    # Split into train/dev/test.
    trainset, devset, testset = split_examples(examples)

    # Select the correct base program.
    if report_type == "CT":
        program = CTLabeler()
        save_name = "ct_labeler.json"
    elif report_type == "CTA":
        program = CTALabeler()
        save_name = "cta_labeler.json"
    elif report_type == "CTP":
        program = CTPLabeler()
        save_name = "ctp_labeler.json"
    else:
        raise ValueError("report_type must be CT, CTA, or CTP")

    print(f"\nTraining {report_type}")
    print(f"Train: {len(trainset)} | Dev: {len(devset)} | Test: {len(testset)}")
    print(f"Optimization log folder: {run_dir}")

    save_text(run_dir / "base_signature_instructions.txt", instruction_text_for(report_type))
    save_program_snapshot(program, run_dir / "baseline_program.json")

    # Evaluate the unoptimized program first.
    baseline_accuracy = evaluate(program, testset, f"{report_type} baseline")
    dump_dspy_history(run_dir / "dspy_history_after_baseline_eval.txt")

    # Create optimizer.
    optimizer = get_optimizer(exact_match_metric)

    # Compile/optimize the program.
    optimized_program = optimizer.compile(
        program.deepcopy(),
        trainset=trainset,
        valset=devset,
    )
    dump_dspy_history(run_dir / "dspy_history_after_compile.txt")

    # Evaluate optimized program.
    optimized_accuracy = evaluate(optimized_program, testset, f"{report_type} optimized")
    dump_dspy_history(run_dir / "dspy_history_after_optimized_eval.txt")

    # Save optimized program to the normal active DSPy program directory.
    Path(cfg.DSPY_PROGRAM_DIR).mkdir(parents=True, exist_ok=True)
    save_path = Path(cfg.DSPY_PROGRAM_DIR) / save_name
    optimized_program.save(str(save_path))

    # Also save a timestamped snapshot so no iteration is lost.
    save_program_snapshot(optimized_program, run_dir / "optimized_program.json")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "report_type": report_type,
        "ground_truth_file": ground_truth_file,
        "max_cases": max_cases,
        "train_examples": len(trainset),
        "dev_examples": len(devset),
        "test_examples": len(testset),
        "baseline_accuracy": baseline_accuracy,
        "optimized_accuracy": optimized_accuracy,
        "active_saved_program": str(save_path),
        "timestamped_run_dir": str(run_dir),
        "note": (
            "This folder preserves the final optimized program for this iteration and "
            "the visible DSPy LM prompt/history via dspy.inspect_history. Some DSPy "
            "versions do not expose every discarded internal candidate prompt directly."
        ),
    }
    save_json_file(run_dir / "optimization_summary.json", summary)

    print(f"Saved active optimized {report_type} program to {save_path}")
    print(f"Saved timestamped optimization logs to {run_dir}")
    return summary


def train_loop(
    report_type: str,
    ground_truth_file: str,
    max_cases: int | None = None,
    sleep_seconds: float = 0,
) -> None:
    """Keep optimizing repeatedly until the user stops the process with Ctrl+C."""
    iteration = 1
    print("Running continuous DSPy optimization. Press Ctrl+C to stop.")
    while True:
        run_dir = make_optimization_run_dir(report_type, iteration)
        train_one(
            report_type=report_type,
            ground_truth_file=ground_truth_file,
            max_cases=max_cases,
            run_dir=run_dir,
            iteration=iteration,
        )
        iteration += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

# =============================================================================
# Command-line entry point
# =============================================================================

def main():
    """
    Parse command-line args and train one modality.
    """

    parser = argparse.ArgumentParser(description="Optimize DSPy stroke labelers.")

    parser.add_argument(
        "--ground-truth",
        default=cfg.GROUND_TRUTH_FILE,
        help="Path to ground truth Excel file.",
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
        help="Keep running optimization attempts until stopped with Ctrl+C.",
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=getattr(cfg, "DSPY_TRAIN_LOOP_SLEEP_SECONDS", 0),
        help="Seconds to pause between looped optimization attempts.",
    )

    args = parser.parse_args()

    try:
        if args.loop:
            train_loop(
                report_type=args.report_type,
                ground_truth_file=args.ground_truth,
                max_cases=args.max_cases,
                sleep_seconds=args.sleep_seconds,
            )
        else:
            train_one(
                report_type=args.report_type,
                ground_truth_file=args.ground_truth,
                max_cases=args.max_cases,
            )
    except KeyboardInterrupt:
        print("\nStopped continuous DSPy optimization.")


if __name__ == "__main__":
    main()
