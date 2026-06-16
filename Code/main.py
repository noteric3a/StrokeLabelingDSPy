"""
main.py

Run the DSPy stroke-labeling pipeline.

This version is configured so you can run it with no command-line arguments:

    python Code/main.py

It uses the default paths and settings from config.py.

You can still override any path if needed:

    python Code/main.py --input Files/Report/Other_File.xlsx
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import config as cfg
from labeler import label_cases
from validate import check_answers


def _first_existing(row, candidates: List[str]) -> Any:
    """Return the first non-empty value from a list of possible column names."""
    for col in candidates:
        if col in row and not pd.isna(row.get(col)):
            return row.get(col)
    return ""


def load_cases_from_excel(path: str) -> List[Dict[str, Any]]:
    """
    Load report cases from an Excel spreadsheet.

    Column names are controlled by config.py:
        - CASE_ID_COLUMNS
        - REPORT_COLUMN_CANDIDATES
    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input report file was not found: {input_path}\n"
            "Fix this by either:\n"
            "  1. putting your report spreadsheet at config.INPUT_REPORT_FILE, or\n"
            "  2. changing INPUT_REPORT_FILE in config.py, or\n"
            "  3. passing --input path/to/your_file.xlsx"
        )

    df = pd.read_excel(input_path)

    cases: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        case_id = str(_first_existing(row, cfg.CASE_ID_COLUMNS) or "").strip()

        # Skip rows without a case identifier.
        if not case_id:
            continue

        case = {"case_id": case_id}

        # Pull each report type using the candidate column names from config.py.
        for output_field, candidates in cfg.REPORT_COLUMN_CANDIDATES.items():
            case[output_field] = str(_first_existing(row, candidates) or "")

        # Guarantee MRI_Report exists even if config.py does not include it.
        case.setdefault("MRI_Report", "")

        cases.append(case)

    return cases


def save_json(cases: List[Dict[str, Any]], output_path: str) -> None:
    """Save labeled cases to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)


async def main() -> None:
    """Main async entry point."""
    parser = argparse.ArgumentParser(description="Run DSPy stroke labeling.")

    parser.add_argument(
        "--input",
        default=cfg.INPUT_REPORT_FILE,
        help=f"Input Excel file containing report text. Default: {cfg.INPUT_REPORT_FILE}",
    )

    parser.add_argument(
        "--output",
        default=cfg.OUTPUT_JSON_FILE,
        help=f"Output JSON path. Default: {cfg.OUTPUT_JSON_FILE}",
    )

    parser.add_argument(
        "--ground-truth",
        default=cfg.GROUND_TRUTH_FILE,
        help=f"Optional ground truth Excel/CSV file for validation. Default: {cfg.GROUND_TRUTH_FILE}",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=cfg.MAX_CONCURRENT_CASES,
        help=f"Maximum number of cases to process at the same time. Default: {cfg.MAX_CONCURRENT_CASES}",
    )

    args = parser.parse_args()

    print("Running DSPy stroke labeling with:")
    print(f"  input:          {args.input}")
    print(f"  output:         {args.output}")
    print(f"  ground truth:   {args.ground_truth or '<none>'}")
    print(f"  max concurrent: {args.max_concurrent}")

    cases = load_cases_from_excel(args.input)
    print(f"Loaded {len(cases)} cases.")

    labeled = await label_cases(
        cases,
        max_concurrent=args.max_concurrent,
    )

    save_json(labeled, args.output)
    print(f"Saved labeled cases to {args.output}")

    # Run validation/conversion if a ground-truth path is configured or provided.
    # validate.py safely skips validation if the file is missing or blank, but
    # still attempts JSON -> Excel conversion.
    if args.ground_truth:
        check_answers(
            json_file=args.output,
            ground_truth_file=args.ground_truth,
            report_path=cfg.TEXT_REPORT_FILE,
            json_report_path=cfg.JSON_REPORT_FILE,
        )


if __name__ == "__main__":
    asyncio.run(main())
