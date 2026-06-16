"""
main.py

Minimal runner for the DSPy stroke-labeling pipeline.

This file is intentionally simple so you can test DSPy without rewriting your
entire existing main.py.

Basic usage:

    python main.py \
      --input Files/Report/Filtered_Reports.xlsx \
      --output Files/Results/labeled_cases_dspy.json \
      --ground-truth Files/GT/Ground_truths.xlsx \
      --max-concurrent 4

What it does:

    1. Reads cases from an Excel file.
    2. Runs the DSPy labeler.
    3. Saves labeled cases to JSON.
    4. Optionally runs your existing validator if a ground-truth file is given.

You may need to adjust the column names inside load_cases_from_excel() to match
your exact spreadsheet.
"""

from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

# Import code.
from Code.labeler import label_cases
from Code.validate import check_answers


def _first_existing(row, candidates: List[str]) -> Any:
    """
    Return the first non-empty value from a list of possible column names.

    Why this exists:
        Different versions of your spreadsheet may use slightly different names.

    Example:
        CT report column could be:
            "CT Report"
            "CT_Report"
            "CT"
    """

    for col in candidates:
        if col in row and not pd.isna(row.get(col)):
            return row.get(col)

    return ""


def load_cases_from_excel(path: str) -> List[Dict[str, Any]]:
    """
    Load report cases from an Excel spreadsheet.

    Adjust the column candidate lists below if your spreadsheet uses different
    column names.

    Required output format for each case:
        {
            "case_id": "...",
            "CT_Report": "...",
            "CTA_Report": "...",
            "CTP_Report": "...",
            "MRI_Report": "..."
        }
    """

    df = pd.read_excel(path)

    cases: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        # Try common case-id column names.
        case_id = str(
            _first_existing(row, ["Case Name", "case_id", "Case ID", "ID"])
            or ""
        ).strip()

        # Skip rows without a case identifier.
        if not case_id:
            continue

        # Build the case dictionary expected by labeler_dspy.py.
        case = {
            "case_id": case_id,

            # CT report text.
            "CT_Report": str(_first_existing(row, ["CT Report", "CT_Report", "CT text", "CT_Text"]) or ""),

            # CTA report text.
            "CTA_Report": str(_first_existing(row, ["CTA Report", "CTA_Report", "CTA text", "CTA_Text"]) or ""),

            # CTP report text.
            "CTP_Report": str(_first_existing(row, ["CTP Report", "CTP_Report", "CTP text", "CTP_Text"]) or ""),

            # MRI is optional.
            "MRI_Report": str(_first_existing(row, ["MRI Report", "MRI_Report", "MRI text", "MRI_Text"]) or ""),
        }

        cases.append(case)

    return cases


def save_json(cases: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save labeled cases to a JSON file.

    This output should be compatible with your converter/validator if the field
    names match your existing pipeline.
    """

    path = Path(output_path)

    # Ensure parent folder exists.
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)


async def main():
    """
    Main async entry point.
    """

    parser = argparse.ArgumentParser(description="Run DSPy stroke labeling.")

    parser.add_argument(
        "--input",
        required=True,
        help="Input Excel file containing report text.",
    )

    parser.add_argument(
        "--output",
        default="Files/labeled_cases_dspy.json",
        help="Output JSON path.",
    )

    parser.add_argument(
        "--ground-truth",
        default="",
        help="Optional ground truth Excel/CSV file for validation.",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum number of cases to process at the same time.",
    )

    args = parser.parse_args()

    # Load report cases.
    cases = load_cases_from_excel(args.input)

    print(f"Loaded {len(cases)} cases.")

    # Run DSPy labeling.
    labeled = await label_cases(
        cases,
        max_concurrent=args.max_concurrent,
    )

    # Save JSON.
    save_json(labeled, args.output)

    print(f"Saved labeled cases to {args.output}")

    # Optionally run your existing answer-key validator.
    if args.ground_truth:
        check_answers(
            json_file=args.output,
            ground_truth_file=args.ground_truth,
            report_path="Files/report_dspy.txt",
            json_report_path="Files/report_dspy.json",
        )


if __name__ == "__main__":
    asyncio.run(main())
