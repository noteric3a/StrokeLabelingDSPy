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
import random
from pathlib import Path
from typing import Any, List, Set
import dspy
import pandas as pd
from Code import config as cfg

# Import base DSPy programs and model configuration.
from Code.dspy_programs import (
    configure_dspy,
    CTLabeler,
    CTALabeler,
    CTPLabeler,
)

# Reuse your existing label normalizer.
from Code.utils import normalize_labels


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
# Training one modality
# =============================================================================

def train_one(report_type: str, ground_truth_file: str, max_cases: int | None = None):
    """
    Optimize one modality-specific DSPy program.
    """

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

    # Evaluate the unoptimized program first.
    evaluate(program, testset, f"{report_type} baseline")

    # Create optimizer.
    optimizer = get_optimizer(exact_match_metric)

    # Compile/optimize the program.
    #
    # Depending on DSPy version, compile() may accept trainset and valset.
    # If your installed version errors here, check your DSPy version and
    # adjust according to the current DSPy docs.
    optimized_program = optimizer.compile(
        program.deepcopy(),
        trainset=trainset,
        valset=devset,
    )

    # Evaluate optimized program.
    evaluate(optimized_program, testset, f"{report_type} optimized")

    # Save optimized program.
    Path(cfg.DSPY_PROGRAM_DIR).mkdir(parents=True, exist_ok=True)
    save_path = Path(cfg.DSPY_PROGRAM_DIR) / save_name

    optimized_program.save(str(save_path))

    print(f"Saved optimized {report_type} program to {save_path}")


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

    args = parser.parse_args()

    train_one(
        report_type=args.report_type,
        ground_truth_file=args.ground_truth,
        max_cases=args.max_cases,
    )


if __name__ == "__main__":
    main()
