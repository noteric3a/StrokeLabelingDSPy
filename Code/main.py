"""Config-driven entry point for labeling, DSPy optimization, and validation.

Edit ``config.py`` and run ``python main.py``. ``config.RUN_MODE`` selects the
workflow. Command-line arguments are deliberately rejected so the exact run is
reproducible from one configuration file.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import config as cfg
import prompts
from labeler import label_spreadsheet_async
from validate import check_answers


@dataclass(frozen=True)
class LabelRunSettings:
    """Normal-labeling settings resolved from config.py."""

    input_file: str | Path
    ground_truth_file: str | Path
    output_file: str | Path
    model: str
    concurrency: int
    skip_validation: bool
    use_cache: bool
    backend: str
    dspy_program_dir: str | Path
    require_dspy_programs: bool
    dspy_combined: bool
    include_reasoning: bool
    confidence_checking: bool


def labeling_settings_from_config() -> LabelRunSettings:
    """Resolve every former labeling command-line option from config.py."""
    backend = str(getattr(cfg, "LABELING_BACKEND", "ollama")).strip().lower()
    if backend not in {"ollama", "dspy"}:
        raise ValueError("config.LABELING_BACKEND must be 'ollama' or 'dspy'.")

    model = str(getattr(cfg, "MODEL_NAME", "")).strip()
    if not model:
        raise ValueError("config.MODEL_NAME cannot be blank.")

    concurrency = int(getattr(cfg, "MAX_CONCURRENT_REQUESTS", 1))
    if concurrency < 1:
        raise ValueError("config.MAX_CONCURRENT_REQUESTS must be at least 1.")

    input_file = getattr(cfg, "INPUT_REPORTS_FILE", "")
    output_file = getattr(cfg, "OUTPUT_JSON_FILE", "")
    if not str(input_file).strip():
        raise ValueError("config.INPUT_REPORTS_FILE cannot be blank.")
    if not str(output_file).strip():
        raise ValueError("config.OUTPUT_JSON_FILE cannot be blank.")

    return LabelRunSettings(
        input_file=input_file,
        ground_truth_file=getattr(cfg, "GROUND_TRUTH_FILE", ""),
        output_file=output_file,
        model=model,
        concurrency=concurrency,
        skip_validation=bool(getattr(cfg, "SKIP_VALIDATION", False)),
        use_cache=bool(getattr(cfg, "USE_PROCESSING_CACHE", True)),
        backend=backend,
        dspy_program_dir=getattr(cfg, "DSPY_PROGRAM_DIR", cfg.FILES_DIR / "dspy_optimization" / "best_programs"),
        require_dspy_programs=bool(getattr(cfg, "DSPY_REQUIRE_OPTIMIZED_PROGRAMS", False)),
        dspy_combined=bool(getattr(cfg, "DSPY_USE_FOR_COMBINED", False)),
        include_reasoning=bool(getattr(cfg, "INCLUDE_REASONING_IN_JSON", True)),
        confidence_checking=bool(getattr(cfg, "ENABLE_CONFIDENCE_CHECKING", False)),
    )


def configured_run_mode() -> str:
    """Return and validate config.RUN_MODE."""
    mode = str(getattr(cfg, "RUN_MODE", "label")).strip().lower()
    if mode not in {"label", "optimize", "validate"}:
        raise ValueError(
            "config.RUN_MODE must be 'label', 'optimize', or 'validate'."
        )
    return mode


def reject_command_line_arguments(arguments: Sequence[str] | None = None) -> None:
    """Keep config.py as the sole control surface for executable workflows."""
    supplied = list(sys.argv[1:] if arguments is None else arguments)
    if supplied:
        raise SystemExit(
            "Command-line arguments are disabled. Edit config.py instead. "
            f"Unexpected arguments: {supplied}"
        )


def create_run_folder() -> Path:
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_folder = cfg.FINISHED_CASES_ROOT / run_name
    run_folder.mkdir(parents=True, exist_ok=False)
    return run_folder


def resolve_include_reasoning(value: bool | None = None) -> bool:
    """Compatibility helper; config.py is used when no explicit value is supplied."""
    if value is None:
        return bool(getattr(cfg, "INCLUDE_REASONING_IN_JSON", True))
    return bool(value)


def resolve_confidence_checking(value: bool | None = None) -> bool:
    """Compatibility helper; config.py is used when no explicit value is supplied."""
    if value is None:
        return bool(getattr(cfg, "ENABLE_CONFIDENCE_CHECKING", False))
    return bool(value)


def apply_labeling_settings(settings: LabelRunSettings) -> None:
    """Apply settings used dynamically by labeler/dspy_runtime modules."""
    cfg.LABELING_BACKEND = settings.backend
    cfg.DSPY_PROGRAM_DIR = Path(settings.dspy_program_dir)
    cfg.DSPY_REQUIRE_OPTIMIZED_PROGRAMS = settings.require_dspy_programs
    cfg.DSPY_USE_FOR_COMBINED = settings.dspy_combined
    cfg.ENABLE_CONFIDENCE_CHECKING = settings.confidence_checking


def preflight_dspy_backend() -> None:
    """Fail before processing cases when the selected DSPy runtime is unusable."""
    if str(getattr(cfg, "LABELING_BACKEND", "ollama")).strip().lower() != "dspy":
        return
    try:
        from dspy_programs import load_program
    except ImportError as exc:
        raise SystemExit(
            "DSPy backend selected, but DSPy is not installed. "
            "Install requirements-dspy.txt first."
        ) from exc

    names = ["CT", "CTA", "CTP"]
    if bool(getattr(cfg, "DSPY_USE_FOR_COMBINED", False)):
        names.append("Combined")
    require_saved = bool(getattr(cfg, "DSPY_REQUIRE_OPTIMIZED_PROGRAMS", False))
    try:
        for name in names:
            load_program(cfg.DSPY_PROGRAM_DIR, name, require_saved=require_saved)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"DSPy backend preflight failed: {exc}") from exc


def write_run_information(run_folder: Path, settings: LabelRunSettings) -> None:
    """Write effective config, prompt, and model metadata for reproducibility."""

    info_path = run_folder / "information.txt"
    with info_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Run created: {datetime.now().isoformat()}\n")
        handle.write("Run mode: label\n")
        handle.write(f"Input reports: {settings.input_file}\n")
        handle.write(f"Ground truth: {settings.ground_truth_file}\n")
        handle.write(f"Configured output JSON: {settings.output_file}\n")
        handle.write(f"Model: {settings.model}\n")
        handle.write(f"OLLAMA_URL: {cfg.OLLAMA_URL}\n")
        handle.write(f"Labeling backend: {settings.backend}\n")
        handle.write(f"DSPy program directory: {settings.dspy_program_dir}\n")
        handle.write(f"DSPy adapter: {getattr(cfg, 'DSPY_ADAPTER', '<not set>')}\n")
        handle.write(
            "Require compiled DSPy programs: "
            f"{settings.require_dspy_programs}\n"
        )
        handle.write(f"Use DSPy for Combined: {settings.dspy_combined}\n")
        handle.write(f"NUM_PREDICT: {getattr(cfg, 'NUM_PREDICT', '<not set>')}\n")
        handle.write(f"NUM_CTX: {getattr(cfg, 'NUM_CTX', '<not set>')}\n")
        handle.write(f"Concurrency: {settings.concurrency}\n")
        handle.write(f"Use processing cache: {settings.use_cache}\n")
        handle.write(f"Skip validation: {settings.skip_validation}\n")
        handle.write(
            f"Output JSON includes reasoning: {settings.include_reasoning}\n"
        )
        handle.write(
            f"Confidence checking enabled: {settings.confidence_checking}\n"
        )
        handle.write(
            f"CONFIDENCE_RUNS: {getattr(cfg, 'CONFIDENCE_RUNS', '<not set>')}\n"
        )
        handle.write(
            "CONFIDENCE_TEMPERATURE: "
            f"{getattr(cfg, 'CONFIDENCE_TEMPERATURE', '<not set>')}\n"
        )
        handle.write(
            "MIN_CONFIDENCE_PERCENTAGE: "
            f"{getattr(cfg, 'MIN_CONFIDENCE_PERCENTAGE', '<not set>')}\n\n"
        )

        handle.write("Allowed labels:\n")
        try:
            handle.write(prompts.labels_text() + "\n\n")
        except Exception:
            handle.write("<could not read allowed labels>\n\n")

        handle.write("Manual base prompt rules (truncated):\n")
        try:
            base = prompts.base_rules()
            handle.write(base[:1000] + ("...\n" if len(base) > 1000 else "\n"))
        except Exception:
            handle.write("<could not read base rules>\n")

        handle.write("\nPrompt templates used:\n")
        handle.write("- build_ct_sanitization_prompt(case_id, ct_report)\n")
        handle.write("- build_ct_prompt(case_id, ct_report)\n")
        handle.write("- build_cta_prompt(case_id, cta_report)\n")
        handle.write("- build_ctp_prompt(case_id, ctp_report)\n")
        handle.write(
            "- build_combined_prompt(case_id, ct_report, cta_report, ctp_report, "
            "mri_report, ct_labels, cta_labels, ctp_labels)\n"
        )

        renderers = (
            (
                "build_ct_sanitization_prompt",
                lambda: prompts.build_ct_sanitization_prompt(
                    "CASE_ID_PLACEHOLDER", "<CT report text goes here>"
                ),
            ),
            (
                "build_ct_prompt",
                lambda: prompts.build_ct_prompt(
                    "CASE_ID_PLACEHOLDER", "<CT report text goes here>"
                ),
            ),
            (
                "build_cta_prompt",
                lambda: prompts.build_cta_prompt(
                    "CASE_ID_PLACEHOLDER", "<CTA report text goes here>"
                ),
            ),
            (
                "build_ctp_prompt",
                lambda: prompts.build_ctp_prompt(
                    "CASE_ID_PLACEHOLDER", "<CTP report text goes here>"
                ),
            ),
            (
                "build_combined_prompt",
                lambda: prompts.build_combined_prompt(
                    "CASE_ID_PLACEHOLDER",
                    "<CT report text>",
                    "<CTA report text>",
                    "<CTP report text>",
                    "<MRI report text>",
                    ["RMCA"],
                    ["NONE"],
                    ["NONE"],
                ),
            ),
        )
        for name, renderer in renderers:
            handle.write(f"\n=== {name} (placeholder) ===\n")
            try:
                handle.write(renderer() + "\n")
            except Exception as exc:
                handle.write(f"<error rendering {name}: {exc}>\n")

        if settings.backend == "dspy":
            handle.write("\nDSPy instructions selected for this run:\n")
            try:
                from dspy_programs import get_instructions, load_program

                program_names = ["CT", "CTA", "CTP"]
                if settings.dspy_combined:
                    program_names.append("Combined")
                for program_name in program_names:
                    program = load_program(
                        settings.dspy_program_dir,
                        program_name,
                        require_saved=settings.require_dspy_programs,
                    )
                    handle.write(f"\n=== DSPy {program_name} ===\n")
                    handle.write(get_instructions(program) + "\n")
            except Exception as exc:
                handle.write(f"<could not load DSPy instructions: {exc}>\n")

    print(f"Wrote run information to: {info_path}")


async def run_pipeline(settings: LabelRunSettings | None = None) -> None:
    """Run normal labeling using config.py settings."""
    settings = settings or labeling_settings_from_config()
    apply_labeling_settings(settings)
    preflight_dspy_backend()

    input_path = Path(settings.input_file)
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(
            f"Configured input report file does not exist: {input_path}"
        )

    run_folder = create_run_folder()
    debug_folder = run_folder / "Debug"
    debug_folder.mkdir(parents=True, exist_ok=True)

    cfg.RAW_OUTPUT_LOG = debug_folder / "raw_ollama_outputs_async.txt"
    cfg.OLLAMA_WRAPPER_LOG = debug_folder / "ollama_outputs_async.txt"
    cfg.BAD_JSON_LOG = debug_folder / "bad_json_outputs_async.txt"
    cfg.FAILED_CASES_LOG = debug_folder / "failed_cases_async.txt"
    cfg.DSPY_RUNTIME_LOG = debug_folder / "dspy_runtime_outputs.jsonl"

    output_path = Path(settings.output_file)
    if not output_path.is_absolute():
        output_path = run_folder / output_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_run_information(run_folder, settings)

    await label_spreadsheet_async(
        input_file=str(input_path),
        output_file=str(output_path),
        model=settings.model,
        max_concurrent_requests=settings.concurrency,
        use_cache=settings.use_cache,
        include_reasoning=settings.include_reasoning,
    )

    if not settings.skip_validation:
        check_answers(
            json_file=str(output_path),
            ground_truth_file=str(settings.ground_truth_file),
            report_path=str(run_folder / "report.txt"),
            json_report_path=str(run_folder / "report.json"),
        )

    print(f"Created run folder: {run_folder}")
    print(f"Saved output JSON: {output_path}")


def run_labeling_from_config() -> None:
    """Synchronous normal-labeling entry point."""
    asyncio.run(run_pipeline(labeling_settings_from_config()))


def main() -> None:
    reject_command_line_arguments()
    try:
        mode = configured_run_mode()
        if mode == "label":
            run_labeling_from_config()
        elif mode == "optimize":
            # Lazy import keeps the manual Ollama pipeline usable without DSPy.
            from dspy_optimize import run_from_config

            raise SystemExit(run_from_config())
        else:
            from validate import run_from_config

            run_from_config()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc


if __name__ == "__main__":
    main()
