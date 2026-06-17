"""
Entry point for the stroke labeler.

Most runtime settings live in config.py. Command-line arguments are optional
one-run overrides for those config defaults.

Run from config.py defaults:
    python main.py

Optional one-run overrides:
    python main.py --input Filtered_Reports.xlsx --ground-truth Radiology_Reports_GroundTruth_Subset.xlsx --output labeled_cases.json
    python main.py --no-reasoning
    python main.py --optimize-prompts
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import config as cfg
from config import MODEL_NAME, MAX_CONCURRENT_REQUESTS, INPUT_REPORTS_FILE, GROUND_TRUTH_FILE, OUTPUT_JSON_FILE
from labeler import label_spreadsheet_async, label_case_by_id_async
from validate import check_answers
from convert import convert
import prompts


def _config_value(name: str, default: Any = None) -> Any:
    """Read a config.py value with a safe fallback for older configs."""
    return getattr(cfg, name, default)


def _resolve_arg(cli_value: Any, config_name: str, default: Any = None) -> Any:
    """CLI value overrides config.py only when the user supplied it."""
    if cli_value is not None:
        return cli_value
    return _config_value(config_name, default)


def _resolve_bool_arg(cli_value: bool | None, config_name: str, default: bool = False) -> bool:
    """Resolve optional CLI boolean overrides against config.py defaults."""
    if cli_value is not None:
        return bool(cli_value)
    return bool(_config_value(config_name, default))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async stroke territory labeler using Ollama. Defaults are configured in config.py."
    )

    # File/model/concurrency overrides. Defaults intentionally stay None here so
    # config.py is the single source of default values.
    parser.add_argument("--input", default=None, help="Override config.INPUT_REPORTS_FILE for this run.")
    parser.add_argument("--ground-truth", default=None, help="Override config.GROUND_TRUTH_FILE for this run.")
    parser.add_argument("--output", default=None, help="Override config.OUTPUT_JSON_FILE for this run.")
    parser.add_argument("--model", default=None, help="Override config.MODEL_NAME for this run.")
    parser.add_argument("--concurrency", type=int, default=None, help="Override config.MAX_CONCURRENT_REQUESTS for this run.")

    validation_group = parser.add_mutually_exclusive_group()
    validation_group.add_argument(
        "--run-validation",
        dest="run_validation",
        action="store_true",
        default=None,
        help="Run validation after labeling, overriding config.RUN_VALIDATION_AFTER_LABELING=True.",
    )
    validation_group.add_argument(
        "--skip-validation",
        dest="run_validation",
        action="store_false",
        default=None,
        help="Skip validation after labeling, overriding config.RUN_VALIDATION_AFTER_LABELING=False.",
    )

    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--use-cache",
        dest="use_cache",
        action="store_true",
        default=None,
        help="Use the processing cache, overriding config.USE_PROCESSING_CACHE=True.",
    )
    cache_group.add_argument(
        "--skip-cache",
        dest="use_cache",
        action="store_false",
        default=None,
        help="Ignore the processing cache and reprocess all cases, overriding config.USE_PROCESSING_CACHE=False.",
    )

    optimized_prompt_group = parser.add_mutually_exclusive_group()
    optimized_prompt_group.add_argument(
        "--use-optimized-prompts",
        dest="use_optimized_prompts",
        action="store_true",
        default=None,
        help="Insert DSPy-optimized prompt guidance, overriding config.USE_OPTIMIZED_PROMPTS=True.",
    )
    optimized_prompt_group.add_argument(
        "--no-optimized-prompts",
        dest="use_optimized_prompts",
        action="store_false",
        default=None,
        help="Disable DSPy-optimized prompt guidance, overriding config.USE_OPTIMIZED_PROMPTS=False.",
    )

    parser.add_argument(
        "--optimize-prompts",
        action="store_true",
        default=None,
        help="Use DSPy to optimize prompt guidance, overriding config.RUN_PROMPT_OPTIMIZATION=True.",
    )
    parser.add_argument(
        "--optimized-prompts-output",
        default=None,
        help="Override config.OPTIMIZED_PROMPTS_FILE / DSPY_OPTIMIZED_PROMPTS_OUTPUT_FILE.",
    )
    parser.add_argument(
        "--dspy-modalities",
        default=None,
        help="Override config.DSPY_MODALITIES. Example: CT,CTA,CTP,COMBINED.",
    )
    parser.add_argument(
        "--dspy-optimizer",
        default=None,
        choices=["MIPROv2", "BootstrapFewShot", "LabeledFewShot"],
        help="Override config.DSPY_OPTIMIZER.",
    )
    parser.add_argument(
        "--dspy-auto",
        default=None,
        choices=["light", "medium", "heavy"],
        help="Override config.DSPY_AUTO for MIPROv2 optimization budget.",
    )
    parser.add_argument(
        "--dspy-train-limit",
        type=int,
        default=None,
        help="Override config.DSPY_TRAIN_LIMIT.",
    )
    parser.add_argument(
        "--dspy-val-limit",
        type=int,
        default=None,
        help="Override config.DSPY_VAL_LIMIT.",
    )
    parser.add_argument(
        "--dspy-seed",
        type=int,
        default=None,
        help="Override config.DSPY_SEED.",
    )
    parser.add_argument(
        "--dspy-lm",
        default=None,
        help="Override config.DSPY_LM, e.g. ollama_chat/qwen3.6:latest.",
    )
    parser.add_argument(
        "--dspy-api-base",
        default=None,
        help="Override config.DSPY_API_BASE. For Ollama, use http://localhost:11434.",
    )

    confidence_group = parser.add_mutually_exclusive_group()
    confidence_group.add_argument(
        "--confidence-checking",
        dest="confidence_checking",
        action="store_true",
        default=None,
        help="Enable confidence checking, overriding config.ENABLE_CONFIDENCE_CHECKING=True.",
    )
    confidence_group.add_argument(
        "--no-confidence-checking",
        dest="confidence_checking",
        action="store_false",
        default=None,
        help="Disable confidence checking, overriding config.ENABLE_CONFIDENCE_CHECKING=False.",
    )

    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--include-reasoning",
        dest="include_reasoning",
        action="store_true",
        default=None,
        help="Include *_reasoning fields, overriding config.INCLUDE_REASONING_IN_JSON=True.",
    )
    reasoning_group.add_argument(
        "--no-reasoning",
        dest="include_reasoning",
        action="store_false",
        default=None,
        help="Omit *_reasoning fields, overriding config.INCLUDE_REASONING_IN_JSON=False.",
    )

    return parser.parse_args()


def create_run_folder() -> Path:
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_folder = cfg.FINISHED_CASES_ROOT / run_name
    run_folder.mkdir(parents=True, exist_ok=False)
    return run_folder


def resolve_include_reasoning(cli_value: bool | None) -> bool:
    """CLI value overrides config.py; config.py defaults to True if missing."""
    return _resolve_bool_arg(cli_value, "INCLUDE_REASONING_IN_JSON", True)


def resolve_confidence_checking(cli_value: bool | None) -> bool:
    """CLI value overrides config.py; config.py defaults to False if missing."""
    return _resolve_bool_arg(cli_value, "ENABLE_CONFIDENCE_CHECKING", False)


def write_run_information(
    run_folder: Path,
    model: str,
    concurrency: int,
    include_reasoning: bool,
    confidence_checking: bool,
    use_optimized_prompts: bool,
    run_validation: bool,
    use_cache: bool,
) -> None:
    """Write model, prompt and config metadata into information.txt inside run_folder."""
    info_path = run_folder / "information.txt"
    with info_path.open("w", encoding="utf-8") as f:
        f.write(f"Run created: {datetime.now().isoformat()}\n")
        f.write(f"Effective model: {model}\n")
        f.write(f"Config MODEL_NAME: {cfg.MODEL_NAME}\n")
        f.write(f"OLLAMA_URL: {cfg.OLLAMA_URL}\n")
        f.write(f"NUM_PREDICT: {_config_value('NUM_PREDICT', '<not set>')}\n")
        f.write(f"NUM_CTX: {_config_value('NUM_CTX', '<not set>')}\n")
        f.write(f"MAX_CONCURRENT_REQUESTS: {_config_value('MAX_CONCURRENT_REQUESTS', '<not set>')}\n")
        f.write(f"Effective concurrency: {concurrency}\n")
        f.write(f"Output JSON includes reasoning: {include_reasoning}\n")
        f.write(f"Confidence checking enabled: {confidence_checking}\n")
        f.write(f"DSPy optimized prompts enabled: {use_optimized_prompts}\n")
        f.write(f"Run validation after labeling: {run_validation}\n")
        f.write(f"Use processing cache: {use_cache}\n")
        f.write(f"DSPy optimized prompts file: {_config_value('OPTIMIZED_PROMPTS_FILE', '<not set>')}\n")
        f.write(f"DSPY_MODALITIES: {_config_value('DSPY_MODALITIES', '<not set>')}\n")
        f.write(f"DSPY_OPTIMIZER: {_config_value('DSPY_OPTIMIZER', '<not set>')}\n")
        f.write(f"DSPY_AUTO: {_config_value('DSPY_AUTO', '<not set>')}\n")
        f.write(f"DSPY_TRAIN_LIMIT: {_config_value('DSPY_TRAIN_LIMIT', '<not set>')}\n")
        f.write(f"DSPY_VAL_LIMIT: {_config_value('DSPY_VAL_LIMIT', '<not set>')}\n")
        f.write(f"DSPY_SEED: {_config_value('DSPY_SEED', '<not set>')}\n")
        f.write(f"DSPY_LM: {_config_value('DSPY_LM', '<not set>')}\n")
        f.write(f"DSPY_API_BASE: {_config_value('DSPY_API_BASE', '<not set>')}\n")
        f.write(f"CONFIDENCE_RUNS: {_config_value('CONFIDENCE_RUNS', '<not set>')}\n")
        f.write(f"CONFIDENCE_TEMPERATURE: {_config_value('CONFIDENCE_TEMPERATURE', '<not set>')}\n")
        f.write(f"MIN_CONFIDENCE_PERCENTAGE: {_config_value('MIN_CONFIDENCE_PERCENTAGE', '<not set>')}\n\n")

        f.write("Allowed labels:\n")
        try:
            f.write(prompts.labels_text() + "\n\n")
        except Exception:
            f.write("<could not read allowed labels>\n\n")

        f.write("Base prompt rules (truncated):\n")
        try:
            base = prompts.base_rules()
            f.write(base[:1000] + ("...\n" if len(base) > 1000 else "\n"))
        except Exception:
            f.write("<could not read base rules>\n")

        f.write("\nPrompt templates used:\n")
        f.write("- build_ct_sanitization_prompt(case_id, ct_report)\n")
        f.write("- build_ct_prompt(case_id, ct_report)\n")
        f.write("- build_cta_prompt(case_id, cta_report)\n")
        f.write("- build_ctp_prompt(case_id, ctp_report)\n")
        f.write("- build_combined_prompt(case_id, ct_report, cta_report, ctp_report, mri_report, ct_labels, cta_labels, ctp_labels)\n")

        f.write('\n=== build_ct_sanitization_prompt (placeholder) ===\n')
        try:
            f.write(prompts.build_ct_sanitization_prompt("CASE_ID_PLACEHOLDER", "<CT report text goes here>") + "\n")
        except Exception as e:
            f.write(f"<error rendering build_ct_sanitization_prompt: {e}>\n")

        f.write('\n=== build_ct_prompt (placeholder) ===\n')
        try:
            f.write(prompts.build_ct_prompt("CASE_ID_PLACEHOLDER", "<CT report text goes here>") + "\n")
        except Exception as e:
            f.write(f"<error rendering build_ct_prompt: {e}>\n")

        f.write('\n=== build_cta_prompt (placeholder) ===\n')
        try:
            f.write(prompts.build_cta_prompt("CASE_ID_PLACEHOLDER", "<CTA report text goes here>") + "\n")
        except Exception as e:
            f.write(f"<error rendering build_cta_prompt: {e}>\n")

        f.write('\n=== build_ctp_prompt (placeholder) ===\n')
        try:
            f.write(prompts.build_ctp_prompt("CASE_ID_PLACEHOLDER", "<CTP report text goes here>") + "\n")
        except Exception as e:
            f.write(f"<error rendering build_ctp_prompt: {e}>\n")

        f.write('\n=== build_combined_prompt (placeholder) ===\n')
        try:
            f.write(
                prompts.build_combined_prompt(
                    "CASE_ID_PLACEHOLDER",
                    "<CT report text>",
                    "<CTA report text>",
                    "<CTP report text>",
                    "<MRI report text>",
                    ["RMCA"],
                    ["NONE"],
                    ["NONE"],
                )
                + "\n"
            )
        except Exception as e:
            f.write(f"<error rendering build_combined_prompt: {e}>\n")

    print(f"Wrote run information to: {info_path}")


async def run_pipeline(args: argparse.Namespace) -> None:
    input_file = _resolve_arg(args.input, "INPUT_REPORTS_FILE", INPUT_REPORTS_FILE)
    ground_truth_file = _resolve_arg(args.ground_truth, "GROUND_TRUTH_FILE", GROUND_TRUTH_FILE)
    output_file = _resolve_arg(args.output, "OUTPUT_JSON_FILE", OUTPUT_JSON_FILE)
    model = str(_resolve_arg(args.model, "MODEL_NAME", MODEL_NAME))
    concurrency = int(_resolve_arg(args.concurrency, "MAX_CONCURRENT_REQUESTS", MAX_CONCURRENT_REQUESTS))

    include_reasoning = resolve_include_reasoning(args.include_reasoning)
    confidence_checking = resolve_confidence_checking(args.confidence_checking)
    use_optimized_prompts = _resolve_bool_arg(args.use_optimized_prompts, "USE_OPTIMIZED_PROMPTS", True)
    run_validation = _resolve_bool_arg(args.run_validation, "RUN_VALIDATION_AFTER_LABELING", True)
    use_cache = _resolve_bool_arg(args.use_cache, "USE_PROCESSING_CACHE", True)

    # Runtime modules already read these config flags, so update the effective
    # values after applying CLI overrides.
    cfg.ENABLE_CONFIDENCE_CHECKING = confidence_checking
    cfg.USE_OPTIMIZED_PROMPTS = use_optimized_prompts

    run_folder = create_run_folder()
    debug_folder = run_folder / "Debug"
    debug_folder.mkdir(parents=True, exist_ok=True)

    cfg.RAW_OUTPUT_LOG = debug_folder / "raw_ollama_outputs_async.txt"
    cfg.OLLAMA_WRAPPER_LOG = debug_folder / "ollama_outputs_async.txt"
    cfg.BAD_JSON_LOG = debug_folder / "bad_json_outputs_async.txt"
    cfg.FAILED_CASES_LOG = debug_folder / "failed_cases_async.txt"

    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = run_folder / output_path.name

    write_run_information(
        run_folder,
        model,
        concurrency,
        include_reasoning,
        confidence_checking,
        use_optimized_prompts,
        run_validation,
        use_cache,
    )

    await label_spreadsheet_async(
        input_file=input_file,
        output_file=str(output_path),
        model=model,
        max_concurrent_requests=concurrency,
        use_cache=use_cache,
        include_reasoning=include_reasoning,
    )

    if run_validation:
        check_answers(
            json_file=str(output_path),
            ground_truth_file=ground_truth_file,
            report_path=str(run_folder / "report.txt"),
            json_report_path=str(run_folder / "report.json"),
        )

    print(f"Created run folder: {run_folder}")
    print(f"Saved output JSON: {output_path}")


def main() -> None:
    args = parse_args()
    run_prompt_optimization = _resolve_bool_arg(args.optimize_prompts, "RUN_PROMPT_OPTIMIZATION", False)
    if run_prompt_optimization:
        from dspy_prompt_optimizer import optimize_prompts_from_args

        optimize_prompts_from_args(args)
        return

    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
