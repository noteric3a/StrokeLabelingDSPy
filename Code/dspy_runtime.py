"""Runtime bridge from the existing async labeler to compiled DSPy programs.

This module intentionally imports DSPy lazily.  The original raw-Ollama backend
continues to work without the optional DSPy dependency installed.
"""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import config as cfg
from utils import normalize_labels


_LOG_LOCK = threading.Lock()


def dspy_backend_enabled() -> bool:
    return str(getattr(cfg, "LABELING_BACKEND", "ollama")).strip().lower() == "dspy"


def _program_directory() -> Path:
    raw = getattr(cfg, "DSPY_PROGRAM_DIR", cfg.FILES_DIR / "dspy_optimization" / "best_programs")
    return Path(raw)


def _program_mtime(program_name: str) -> float:
    from dspy_programs import program_path

    path = program_path(_program_directory(), program_name)
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


@lru_cache(maxsize=16)
def _load_program_template(
    program_dir_text: str,
    program_name: str,
    require_saved: bool,
    _mtime: float,
):
    from dspy_programs import load_program

    return load_program(program_dir_text, program_name, require_saved=require_saved)


@lru_cache(maxsize=32)
def _load_lm(
    model: str,
    api_base: str,
    temperature: float,
    max_tokens: int,
    num_ctx: int,
    timeout_seconds: int,
    cache: bool,
):
    from dspy_programs import create_lm

    return create_lm(
        model=model,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        num_ctx=num_ctx,
        timeout_seconds=timeout_seconds,
        cache=cache,
    )


@lru_cache(maxsize=4)
def _load_adapter(adapter_name: str):
    from dspy_programs import create_adapter

    return create_adapter(adapter_name)


def clear_runtime_cache() -> None:
    """Clear cached program templates and LMs, useful after optimizer promotion."""
    _load_program_template.cache_clear()
    _load_lm.cache_clear()
    _load_adapter.cache_clear()


def _prediction_value(prediction: Any, field: str, default: Any) -> Any:
    # DSPy Prediction inherits Example, whose ``labels`` method collides with a
    # user-defined output field named ``labels``. Prefer mapping access first.
    if isinstance(prediction, dict):
        return prediction.get(field, default)
    try:
        return prediction[field]
    except Exception:
        pass
    try:
        value = getattr(prediction, field)
        return default if callable(value) else value
    except Exception:
        return default


def _runtime_log_path() -> Path:
    configured = getattr(cfg, "DSPY_RUNTIME_LOG", None)
    if configured:
        return Path(configured)
    return Path(getattr(cfg, "RAW_OUTPUT_LOG", cfg.FILES_DIR_DEBUG / "raw_ollama_outputs_async.txt")).with_name(
        "dspy_runtime_outputs.jsonl"
    )


def _write_runtime_log(payload: dict[str, Any]) -> None:
    path = _runtime_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _run_program_sync(
    *,
    program_name: str,
    inputs: dict[str, Any],
    case_id: str,
    tag: str,
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError(
            "DSPy backend was selected, but DSPy is not installed. "
            "Install requirements-dspy.txt or run: pip install dspy==3.2.1"
        ) from exc

    from dspy_programs import ollama_api_base

    program_dir = _program_directory()
    require_saved = bool(getattr(cfg, "DSPY_REQUIRE_OPTIMIZED_PROGRAMS", True))
    template = _load_program_template(
        str(program_dir),
        program_name,
        require_saved,
        _program_mtime(program_name),
    )
    # Predictors can retain trace/debug state.  A cheap deepcopy keeps parallel
    # label calls independent while sharing the LM client.
    program = copy.deepcopy(template)

    api_base = str(
        getattr(cfg, "DSPY_OLLAMA_API_BASE", "")
        or ollama_api_base(getattr(cfg, "OLLAMA_URL", "http://localhost:11434/api/generate"))
    )
    max_tokens = int(getattr(cfg, "DSPY_MAX_TOKENS", getattr(cfg, "NUM_PREDICT", 1670)))
    num_ctx = int(getattr(cfg, "DSPY_NUM_CTX", getattr(cfg, "NUM_CTX", 20000)))
    timeout_seconds = int(getattr(cfg, "REQUEST_TIMEOUT_SECONDS", 600))
    # Repeated confidence samples must be real model calls. Reusing the DSPy/
    # LiteLLM cache for a nonzero-temperature request would make every vote a
    # copy of the first sample and falsely report 100% stability.
    cache = bool(getattr(cfg, "DSPY_LM_CACHE", True)) and float(temperature) == 0.0
    adapter_name = str(getattr(cfg, "DSPY_ADAPTER", "json"))

    lm = _load_lm(
        model,
        api_base,
        round(float(temperature), 6),
        max_tokens,
        num_ctx,
        timeout_seconds,
        cache,
    )
    adapter = _load_adapter(adapter_name)

    with dspy.context(lm=lm, adapter=adapter):
        prediction = program(**inputs)

    labels = normalize_labels(_prediction_value(prediction, "labels", ["NONE"]))
    reasoning = str(_prediction_value(prediction, "reasoning", "")).strip()
    result: Dict[str, Any] = {
        "case_id": str(case_id),
        "reasoning": reasoning,
    }
    if str(program_name).lower() == "combined":
        result["Combined_GT"] = labels
    else:
        result["modality"] = str(program_name).upper()
        result["labels"] = labels

    _write_runtime_log(
        {
            "case_id": str(case_id),
            "tag": tag,
            "program": program_name,
            "temperature": temperature,
            "result": result,
        }
    )
    return result


async def dspy_generate_modality_async(
    *,
    modality: str,
    report: str,
    case_id: str,
    tag: str,
    semaphore: asyncio.Semaphore,
    model: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    async with semaphore:
        print(f"Running DSPy {tag} for case {case_id}")
        return await asyncio.to_thread(
            _run_program_sync,
            program_name=modality,
            inputs={"report": report},
            case_id=case_id,
            tag=tag,
            model=model,
            temperature=temperature,
        )


async def dspy_generate_combined_async(
    *,
    case_id: str,
    ct_report: str,
    cta_report: str,
    ctp_report: str,
    mri_report: str,
    ct_labels: list[str],
    cta_labels: list[str],
    ctp_labels: list[str],
    tag: str,
    semaphore: asyncio.Semaphore,
    model: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    async with semaphore:
        print(f"Running DSPy {tag} for case {case_id}")
        return await asyncio.to_thread(
            _run_program_sync,
            program_name="Combined",
            inputs={
                "ct_report": ct_report,
                "cta_report": cta_report,
                "ctp_report": ctp_report,
                "mri_report": mri_report,
                "ct_labels": normalize_labels(ct_labels),
                "cta_labels": normalize_labels(cta_labels),
                "ctp_labels": normalize_labels(ctp_labels),
            },
            case_id=case_id,
            tag=tag,
            model=model,
            temperature=temperature,
        )
