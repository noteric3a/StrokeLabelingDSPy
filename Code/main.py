"""Run the DSPy stroke-labeling pipeline using config.py settings only."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import config as cfg
from labeler import label_cases
from validate import check_answers

logging.getLogger("dspy").setLevel(logging.ERROR)
logging.getLogger("dspy.clients.lm").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)


def _first_existing(row, candidates: List[str]) -> Any:
    for column in candidates:
        if column in row and not pd.isna(row.get(column)):
            return row.get(column)
    return ""


def load_cases_from_excel(path: str) -> List[Dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input report file was not found: {input_path}\n"
            "Change INPUT_REPORT_FILE in config.py."
        )

    dataframe = pd.read_excel(input_path)
    cases: List[Dict[str, Any]] = []
    for _, row in dataframe.iterrows():
        case_id = str(_first_existing(row, cfg.CASE_ID_COLUMNS) or "").strip()
        if not case_id:
            continue

        case: Dict[str, Any] = {"case_id": case_id}
        for output_field, candidates in cfg.REPORT_COLUMN_CANDIDATES.items():
            case[output_field] = str(_first_existing(row, candidates) or "")
        case.setdefault("MRI_Report", "")
        cases.append(case)
    return cases


def save_json(cases: List[Dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cases, indent=2), encoding="utf-8")


def create_run_paths() -> tuple[str, str, str, Path | None]:
    if not cfg.USE_TIMESTAMPED_RUN_FOLDERS:
        return cfg.OUTPUT_JSON_FILE, cfg.TEXT_REPORT_FILE, cfg.JSON_REPORT_FILE, None

    stamp = datetime.now().strftime(cfg.RUN_TIMESTAMP_FORMAT)
    run_dir = Path(cfg.RUN_OUTPUT_ROOT) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_dir / Path(cfg.OUTPUT_JSON_FILE).name
    text_report_path = run_dir / Path(cfg.TEXT_REPORT_FILE).name
    json_report_path = run_dir / Path(cfg.JSON_REPORT_FILE).name

    # Keep local Ollama diagnostics with the corresponding labeling run.
    cfg.OLLAMA_WRAPPER_LOG = str(run_dir / "ollama_wrapper_log.jsonl")
    cfg.BAD_JSON_LOG = str(run_dir / "bad_json_log.jsonl")
    return str(output_path), str(text_report_path), str(json_report_path), run_dir


def save_run_metadata(
    run_dir: Path | None,
    *,
    output_path: str,
    text_report_path: str,
    json_report_path: str,
) -> None:
    if run_dir is None:
        return

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": cfg.INPUT_REPORT_FILE,
        "output_json": output_path,
        "ground_truth": cfg.GROUND_TRUTH_FILE or "",
        "text_report": text_report_path,
        "json_report": json_report_path,
        "max_concurrent": cfg.MAX_CONCURRENT_CASES,
        "model": cfg.DSPY_MODEL,
        "task_temperature": cfg.DSPY_TASK_TEMPERATURE,
        "max_tokens": cfg.DSPY_MAX_TOKENS,
        "confidence_enabled": cfg.ENABLE_CONFIDENCE_CHECKING,
        "confidence_attempts": cfg.CONFIDENCE_ATTEMPTS,
        "confidence_threshold_percentage": cfg.CONFIDENCE_THRESHOLD_PERCENTAGE,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


async def main() -> None:
    output_path, text_report_path, json_report_path, run_dir = create_run_paths()

    print("Running DSPy stroke labeling with config.py settings:")
    print(f"  input:          {cfg.INPUT_REPORT_FILE}")
    print(f"  output:         {output_path}")
    print(f"  ground truth:   {cfg.GROUND_TRUTH_FILE or '<none>'}")
    print(f"  max concurrent: {cfg.MAX_CONCURRENT_CASES}")
    if run_dir is not None:
        print(f"  run folder:     {run_dir}")

    cases = load_cases_from_excel(cfg.INPUT_REPORT_FILE)
    print(f"Loaded {len(cases)} cases.")

    labeled = await label_cases(cases, max_concurrent=cfg.MAX_CONCURRENT_CASES)
    save_json(labeled, output_path)
    save_run_metadata(
        run_dir,
        output_path=output_path,
        text_report_path=text_report_path,
        json_report_path=json_report_path,
    )
    print(f"Saved labeled cases to {output_path}")

    if cfg.GROUND_TRUTH_FILE:
        check_answers(
            json_file=output_path,
            ground_truth_file=cfg.GROUND_TRUTH_FILE,
            report_path=text_report_path,
            json_report_path=json_report_path,
        )


if __name__ == "__main__":
    asyncio.run(main())
