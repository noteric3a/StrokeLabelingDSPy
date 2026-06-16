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
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

import config as cfg
from labeler import label_cases
from validate import check_answers

# Keep the one-line progress bar clean by hiding repeated DSPy warning messages.
logging.getLogger("dspy").setLevel(logging.ERROR)
logging.getLogger("dspy.clients.lm").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)


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


def create_timestamped_run_paths(args) -> tuple[str, str, str, Path | None]:
    """Return output/report paths, optionally inside a timestamped run folder."""
    timestamped = bool(getattr(cfg, "USE_TIMESTAMPED_RUN_FOLDERS", True)) and not args.no_timestamped_output
    if not timestamped:
        return args.output, cfg.TEXT_REPORT_FILE, cfg.JSON_REPORT_FILE, None

    run_stamp = datetime.now().strftime(getattr(cfg, "RUN_TIMESTAMP_FORMAT", "%Y%m%d_%H%M%S"))
    run_dir = Path(getattr(cfg, "RUN_OUTPUT_ROOT", "Files/Results/DSPy_Runs")) / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_dir / Path(args.output).name
    text_report_path = run_dir / Path(cfg.TEXT_REPORT_FILE).name
    json_report_path = run_dir / Path(cfg.JSON_REPORT_FILE).name

    # Redirect logs into the same run folder so each test run is self-contained.
    cfg.OLLAMA_WRAPPER_LOG = str(run_dir / "ollama_wrapper_log.jsonl")
    cfg.BAD_JSON_LOG = str(run_dir / "bad_json_log.jsonl")

    return str(output_path), str(text_report_path), str(json_report_path), run_dir


def save_run_metadata(run_dir: Path | None, args, *, output_path: str, text_report_path: str, json_report_path: str) -> None:
    """Write a small metadata file describing this experiment run."""
    if run_dir is None:
        return

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": args.input,
        "output_json": output_path,
        "ground_truth": args.ground_truth or "",
        "text_report": text_report_path,
        "json_report": json_report_path,
        "max_concurrent": args.max_concurrent,
        "model": cfg.DSPY_MODEL,
        "temperature": cfg.DSPY_TEMPERATURE,
        "max_tokens": cfg.DSPY_MAX_TOKENS,
        "confidence_enabled": cfg.ENABLE_CONFIDENCE_CHECKING,
        "confidence_attempts": cfg.CONFIDENCE_ATTEMPTS,
        "confidence_threshold_percentage": cfg.CONFIDENCE_THRESHOLD_PERCENTAGE,
        "ct_sanitization_restored": True,
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


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

    parser.add_argument(
        "--no-timestamped-output",
        action="store_true",
        help="Disable timestamped run folders and write exactly to --output/config report paths.",
    )

    args = parser.parse_args()
    output_path, text_report_path, json_report_path, run_dir = create_timestamped_run_paths(args)

    print("Running DSPy stroke labeling with:")
    print(f"  input:          {args.input}")
    print(f"  output:         {output_path}")
    print(f"  ground truth:   {args.ground_truth or '<none>'}")
    print(f"  max concurrent: {args.max_concurrent}")
    if run_dir is not None:
        print(f"  run folder:     {run_dir}")

    cases = load_cases_from_excel(args.input)
    print(f"Loaded {len(cases)} cases.")

    labeled = await label_cases(
        cases,
        max_concurrent=args.max_concurrent,
    )

    save_json(labeled, output_path)
    save_run_metadata(run_dir, args, output_path=output_path, text_report_path=text_report_path, json_report_path=json_report_path)
    print(f"Saved labeled cases to {output_path}")

    # Run validation/conversion if a ground-truth path is configured or provided.
    # validate.py safely skips validation if the file is missing or blank, but
    # still attempts JSON -> Excel conversion.
    if args.ground_truth:
        check_answers(
            json_file=output_path,
            ground_truth_file=args.ground_truth,
            report_path=text_report_path,
            json_report_path=json_report_path,
        )


if __name__ == "__main__":
    asyncio.run(main())
