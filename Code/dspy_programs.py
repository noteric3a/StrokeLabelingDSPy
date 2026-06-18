from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List
import hashlib
import json
import math
import re

import dspy

import config as cfg
from utils import normalize_labels


INPUT_DESCRIPTIONS = {
    "ct_report": "CT brain report text.",
    "cta_report": "CTA report text.",
    "ctp_report": "CT perfusion report text.",
    "mri_report": "MRI report text if available.",
    "ct_labels": "Already predicted CT labels.",
    "cta_labels": "Already predicted CTA labels.",
    "ctp_labels": "Already predicted CTP labels.",
    "ct_reasoning": "CT reasoning.",
    "cta_reasoning": "CTA reasoning.",
    "ctp_reasoning": "CTP reasoning.",
}

OUTPUT_DESCRIPTIONS = {
    "labels": "Only comma-separated labels from the allowed list, such as RMCA or NONE.",
    "reasoning": "Exactly one short sentence; no step-by-step analysis.",
    "combined_reasoning": "Exactly one short sentence explaining the final combined labels.",
}


def _program_signature_instructions(program: Any) -> str:
    """Return the currently loaded/optimized instruction for one DSPy program."""
    for attrs in (("predict", "signature", "instructions"), ("signature", "instructions")):
        value = program
        try:
            for attr in attrs:
                value = getattr(value, attr)
            if value:
                return str(value)
        except Exception:
            continue
    return ""


def current_cta_supplement(program: Any | None = None) -> str:
    """Return the active CTA supplement, falling back to the config seed."""
    if program is not None:
        optimized = _program_signature_instructions(program).strip()
        if optimized:
            return optimized

    loaded = _PROGRAMS.get("cta") if "_PROGRAMS" in globals() else None
    if loaded is not None:
        optimized = _program_signature_instructions(loaded).strip()
        if optimized:
            return optimized
    return cfg.CTA_SIGNATURE_INSTRUCTIONS.strip()


def effective_cta_prompt(supplement: str | None = None) -> str:
    """Return the immutable CTA base plus the selected supplemental instruction."""
    active_supplement = (supplement or cfg.CTA_SIGNATURE_INSTRUCTIONS).strip()
    return (
        "IMMUTABLE CTA BASE PROMPT:\n"
        f"{cfg.CTA_BASE_PROMPT.strip()}\n\n"
        "DSPY-OPTIMIZABLE CTA SUPPLEMENT:\n"
        f"{active_supplement}"
    )


@dataclass
class StrokePrediction:
    """Clean output object used by the rest of the pipeline."""
    labels: List[str]
    reasoning: str


# =============================================================================
# Full-report and context-window safety
# =============================================================================

_EXPECTED_FULL_REPORT: ContextVar[str | None] = ContextVar(
    "dspy_expected_full_report",
    default=None,
)
_EXPECTED_CTA_BASE_PROMPT: ContextVar[str | None] = ContextVar(
    "dspy_expected_cta_base_prompt",
    default=None,
)


def _normalize_line_endings(value: Any) -> str:
    """Normalize only line endings; preserve every other report character."""
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n")


def _iter_request_text(value: Any) -> Iterable[str]:
    """Yield textual values from an OpenAI-style prompt/messages structure."""
    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_request_text(nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_request_text(nested)
        return

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            yield from _iter_request_text(model_dump())
            return
        except Exception:
            pass


def _request_text(prompt: Any, messages: Any) -> str:
    parts: List[str] = []
    if prompt is not None:
        parts.extend(_iter_request_text(prompt))
    if messages is not None:
        parts.extend(_iter_request_text(messages))
    return "\n".join(parts)


def _safe_positive_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _estimate_prompt_tokens(model: str, prompt: Any, messages: Any) -> tuple[int, str]:
    """Estimate provider input tokens before sending the request.

    LiteLLM's token counter is preferred.  If the local model is unknown to the
    counter, use a deliberately conservative two-characters-per-token estimate.
    The estimate is used only to fail before a request could overflow num_ctx; it
    never shortens the prompt.
    """
    try:
        import litellm

        if messages is not None:
            count = litellm.token_counter(model=model, messages=messages)
        else:
            count = litellm.token_counter(model=model, text=str(prompt or ""))
        count = _safe_positive_int(count)
        if count:
            return count, "LiteLLM token_counter"
    except Exception:
        pass

    text = _request_text(prompt, messages)
    return max(1, math.ceil(len(text) / 2)), "conservative 2-characters/token estimate"


@contextmanager
def require_full_report(report_text: Any):
    """Require the exact report to survive DSPy's prompt construction unchanged."""
    report = _normalize_line_endings(report_text)
    token = _EXPECTED_FULL_REPORT.set(report if report else None)
    try:
        yield
    finally:
        _EXPECTED_FULL_REPORT.reset(token)


@contextmanager
def require_cta_base_prompt(base_prompt: Any):
    """Require the immutable CTA base prompt in every CTA task-model request."""
    base = _normalize_line_endings(base_prompt)
    token = _EXPECTED_CTA_BASE_PROMPT.set(base if base else None)
    try:
        yield
    finally:
        _EXPECTED_CTA_BASE_PROMPT.reset(token)


class ContextSafeOllamaLM(dspy.LM):
    """DSPy LM that refuses to send clipped or context-overflowing task inputs.

    The check happens immediately before DSPy calls LiteLLM/Ollama, so it also
    covers optimizer-internal evaluations and bootstrapped-demo generation.
    """

    def _validate_request_integrity(
        self,
        prompt: Any,
        messages: Any,
        call_kwargs: Dict[str, Any],
    ) -> None:
        request_text = _normalize_line_endings(_request_text(prompt, messages))
        expected_report = _EXPECTED_FULL_REPORT.get()
        expected_cta_base = _EXPECTED_CTA_BASE_PROMPT.get()

        if expected_cta_base and expected_cta_base not in request_text:
            digest = hashlib.sha256(expected_cta_base.encode("utf-8")).hexdigest()
            raise RuntimeError(
                "CTA base-prompt integrity check failed before the LM request: "
                f"the exact {len(expected_cta_base)}-character immutable base prompt "
                f"(sha256={digest}) was not present in DSPy's outgoing CTA prompt. "
                "Training stopped instead of evaluating a candidate without the base."
            )

        if expected_report and expected_report not in request_text:
            digest = hashlib.sha256(expected_report.encode("utf-8")).hexdigest()
            raise RuntimeError(
                "Full-report integrity check failed before the LM request: "
                f"the exact {len(expected_report)}-character report "
                f"(sha256={digest}) was not present in DSPy's outgoing prompt. "
                "Training stopped instead of evaluating a truncated report."
            )

        merged = {**getattr(self, "kwargs", {}), **call_kwargs}
        context_window = _safe_positive_int(
            merged.get("num_ctx"),
            _safe_positive_int(getattr(cfg, "DSPY_CONTEXT_WINDOW", 0)),
        )
        max_output_tokens = _safe_positive_int(
            merged.get("max_tokens")
            or merged.get("max_completion_tokens")
            or merged.get("num_predict")
        )

        if not context_window:
            raise RuntimeError(
                "DSPY_CONTEXT_WINDOW/num_ctx is not configured. Refusing to send "
                "a training request because Ollama could silently use a smaller default context."
            )

        estimated_prompt_tokens, method = _estimate_prompt_tokens(
            self.model,
            prompt,
            messages,
        )
        available_prompt_tokens = context_window - max_output_tokens
        if available_prompt_tokens <= 0 or estimated_prompt_tokens > available_prompt_tokens:
            raise RuntimeError(
                "LM request would exceed the configured context window and could truncate input. "
                f"Estimated prompt={estimated_prompt_tokens} tokens ({method}), "
                f"reserved output={max_output_tokens}, num_ctx={context_window}. "
                "Increase DSPY_CONTEXT_WINDOW, lower DSPY_PROMPT_MAX_TOKENS/DSPY_MAX_TOKENS, "
                "or reduce MIPRO demos/view-data batch size. The report was not shortened."
            )

    def forward(self, prompt=None, messages=None, **kwargs):
        self._validate_request_integrity(prompt, messages, kwargs)
        return super().forward(prompt=prompt, messages=messages, **kwargs)

    async def aforward(self, prompt=None, messages=None, **kwargs):
        self._validate_request_integrity(prompt, messages, kwargs)
        return await super().aforward(prompt=prompt, messages=messages, **kwargs)


def create_dspy_lm(
    *,
    model: str,
    api_base: str,
    temperature: float,
    max_tokens: int,
    context_window: int,
) -> ContextSafeOllamaLM:
    """Create one context-safe DSPy/Ollama LM from config-controlled values."""
    kwargs: Dict[str, Any] = {
        "api_base": api_base,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # LiteLLM's Ollama provider forwards num_ctx inside the native options object.
        "num_ctx": context_window,
        "think": False,
    }
    if bool(getattr(cfg, "DSPY_DISABLE_CACHE", True)):
        kwargs["cache"] = False

    try:
        return ContextSafeOllamaLM(model, **kwargs)
    except TypeError:
        # Compatibility fallbacks may remove DSPy-only conveniences, but never
        # remove num_ctx: silently dropping it would reintroduce truncation risk.
        kwargs.pop("cache", None)
        try:
            return ContextSafeOllamaLM(model, **kwargs)
        except TypeError:
            kwargs.pop("think", None)
            return ContextSafeOllamaLM(model, **kwargs)


def _disable_dspy_cache_if_configured() -> None:
    """Best-effort cache disable for reproducible training/evaluation runs.

    Your project-level ProcessingCache is not used by dspy_train.py, but DSPy
    itself can cache LM calls depending on version/settings.  This helper turns
    off DSPy's memory/disk caches when config.DSPY_DISABLE_CACHE is true.  It is
    intentionally defensive so the code still runs across DSPy versions.
    """
    if not bool(getattr(cfg, "DSPY_DISABLE_CACHE", True)):
        return

    configure_cache = getattr(dspy, "configure_cache", None)
    if callable(configure_cache):
        for kwargs in (
            {"enable_memory_cache": False, "enable_disk_cache": False},
            {"enable_disk_cache": False, "enable_memory_cache": False},
        ):
            try:
                configure_cache(**kwargs)
                return
            except TypeError:
                continue
            except Exception:
                return

    # Older/newer DSPy versions may expose settings differently.  Try silently.
    try:
        dspy.settings.configure(cache=False)
    except Exception:
        pass


def configure_dspy() -> None:
    """Configure DSPy with full-report and context-window enforcement."""
    _disable_dspy_cache_if_configured()

    lm = create_dspy_lm(
        model=cfg.DSPY_MODEL,
        api_base=cfg.DSPY_API_BASE,
        temperature=float(cfg.DSPY_TASK_TEMPERATURE),
        max_tokens=int(cfg.DSPY_MAX_TOKENS),
        context_window=int(cfg.DSPY_CONTEXT_WINDOW),
    )

    dspy.configure(lm=lm)
    _disable_dspy_cache_if_configured()


def clean_labels(raw_labels: Any) -> List[str]:
    """Normalize model labels using config.ALLOWED_LABELS and config.LABEL_ALIASES."""
    labels = normalize_labels(raw_labels)
    cleaned: List[str] = []
    for label in labels:
        label = str(label).strip().upper()
        if label in cfg.ALLOWED_LABELS and label not in cleaned:
            cleaned.append(label)
    if not cleaned:
        return ["NONE"]
    if "NONE" in cleaned and len(cleaned) > 1:
        cleaned = [label for label in cleaned if label != "NONE"]
    return cleaned


def _prediction_field(pred: Any, field: str, default: Any = "") -> Any:
    """Read a DSPy Prediction field without returning a bound method."""
    for reader in (
        lambda obj: obj[field],
        lambda obj: obj.get(field),
        lambda obj: getattr(obj, field),
    ):
        try:
            value = reader(pred)
        except Exception:
            continue
        if not callable(value):
            return value
    return default


def prediction_to_result(pred: Any) -> StrokePrediction:
    """Convert a raw DSPy Prediction into the clean project output shape."""
    labels = clean_labels(_prediction_field(pred, "labels", "NONE"))
    reasoning = str(_prediction_field(pred, "reasoning", "") or "").strip()
    if not reasoning:
        reasoning = "No reasoning returned by DSPy program."
    return StrokePrediction(labels=labels, reasoning=reasoning)




# =============================================================================
# Fallback helpers for DSPy adapter parse failures
# =============================================================================

def _json_from_text(text: str) -> Dict[str, Any] | None:
    """
    Extract a JSON object from text.

    This is intentionally defensive because local thinking models may sometimes
    wrap JSON in extra text even when asked for structured output.
    """
    if not text:
        return None

    text = str(text).strip()

    # Direct JSON object.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # First {...} block.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def _labels_from_reasoning_text(text: str) -> List[str]:
    """
    Last-resort label extraction from reasoning_content or plain text.

    This prevents a full crash when the model clearly states labels but DSPy
    receives an empty JSON response.
    """
    if not text:
        return ["NONE"]

    # Prefer explicit "Labels: ..." line when present.
    labels_match = re.search(
        r"(?:final\s+labels?|labels?|output)\s*:\s*([A-Z,\s]+)",
        text,
        flags=re.IGNORECASE,
    )
    if labels_match:
        return clean_labels(labels_match.group(1))

    # Otherwise, extract allowed labels mentioned anywhere.
    found = []
    upper = text.upper()
    for label in cfg.ALLOWED_LABELS:
        if label == "NONE":
            continue
        if re.search(rf"\b{re.escape(label)}\b", upper):
            found.append(label)

    return clean_labels(found or ["NONE"])


def _direct_ollama_fallback(
    report_text: str,
    modality: str,
    *,
    cta_supplement: str | None = None,
) -> StrokePrediction:
    """
    Retry the failed DSPy call through the direct Ollama client.

    Why this exists:
        Some Ollama thinking models occasionally return an empty final "text"
        field while putting the useful answer in reasoning_content. DSPy's
        JSONAdapter cannot parse that and raises AdapterParseError.

        The direct Ollama client uses think=False and a JSON schema, which is
        more reliable for this labeling task.
    """
    try:
        from ollama_client import ollama_generate_sync
        from schemas import SINGLE_MODALITY_SCHEMA

        instruction_map = {
            "CT": cfg.CT_SIGNATURE_INSTRUCTIONS,
            # The direct fallback must use the same immutable base and the
            # currently loaded optimized supplement as the DSPy program.
            "CTA": effective_cta_prompt(cta_supplement),
            "CTP": cfg.CTP_SIGNATURE_INSTRUCTIONS,
        }

        instruction = instruction_map.get(modality, "")
        prompt = f"""
You are labeling acute stroke-related vascular territories.

{instruction}

Return ONLY valid JSON matching this shape:
{{
  "case_id": "fallback",
  "modality": "{modality}",
  "labels": ["NONE"],
  "reasoning": "brief reason"
}}

Report:
{report_text or ""}
"""

        raw = ollama_generate_sync(
            prompt=prompt,
            schema=SINGLE_MODALITY_SCHEMA,
            case_id="fallback",
            tag=f"DSPY_{modality}_FALLBACK",
            temperature=cfg.DSPY_TASK_TEMPERATURE,
        )

        parsed = _json_from_text(raw)
        if parsed:
            return StrokePrediction(
                labels=clean_labels(parsed.get("labels", "NONE")),
                reasoning=str(parsed.get("reasoning", "Recovered by direct Ollama fallback.")).strip(),
            )

        return StrokePrediction(
            labels=_labels_from_reasoning_text(raw),
            reasoning="Recovered labels from direct Ollama fallback text because JSON parsing failed.",
        )

    except Exception as fallback_error:
        return StrokePrediction(
            labels=["NONE"],
            reasoning=f"DSPy call failed and direct fallback also failed: {fallback_error}",
        )


def _is_dspy_parse_error(exc: Exception) -> bool:
    """Return True when DSPy failed because it could not parse the LM response."""
    message = str(exc)
    exc_type = type(exc).__name__
    return (
        "AdapterParseError" in exc_type
        or "AdapterParseError" in message
        or "empty or null response" in message
        or "failed to parse" in message.lower()
        or "Expected to find output fields" in message
    )


def _safe_program_call(program: Any, report_text: str, modality: str) -> StrokePrediction:
    """
    Call a DSPy program safely.

    If DSPy succeeds, return the DSPy prediction.
    If DSPy fails because the adapter cannot parse the model response, retry
    through the direct Ollama fallback instead of crashing the entire run.
    """
    try:
        pred = program(report_text=report_text or "")
        return prediction_to_result(pred)
    except Exception as exc:
        message = str(exc)
        if _is_dspy_parse_error(exc):
            supplement = current_cta_supplement(program) if modality == "CTA" else None
            fallback = _direct_ollama_fallback(
                report_text or "",
                modality,
                cta_supplement=supplement,
            )
            fallback.reasoning = (
                f"DSPy parse fallback used for {modality}. Original error: {message[:300]}\n\n"
                f"{fallback.reasoning}"
            )
            return fallback

        raise


def _direct_ollama_combined_fallback(case: Dict[str, Any]) -> StrokePrediction:
    """Direct Ollama fallback for Combined labeling."""
    try:
        from ollama_client import ollama_generate_sync
        from schemas import COMBINED_SCHEMA

        prompt = f"""
You are creating the final Combined_GT stroke-territory label.

{cfg.COMBINED_SIGNATURE_INSTRUCTIONS}

Return ONLY valid JSON matching this shape:
{{
  "case_id": "{case.get('case_id', 'fallback')}",
  "Combined_GT": ["NONE"],
  "reasoning": "brief reason"
}}

CT report:
{case.get("New_CT_Report") or case.get("CT_Report") or ""}

CTA report:
{case.get("CTA_Report") or ""}

CTP report:
{case.get("CTP_Report") or ""}

MRI report:
{case.get("MRI_Report") or ""}

Already predicted modality labels:
CT: {case.get("CT_GT", ["NONE"])}
CTA: {case.get("CTA_GT", ["NONE"])}
CTP: {case.get("CTP_GT", ["NONE"])}

Modality reasoning:
CT: {case.get("CT_GT_reasoning", "")}
CTA: {case.get("CTA_GT_reasoning", "")}
CTP: {case.get("CTP_GT_reasoning", "")}
"""

        raw = ollama_generate_sync(
            prompt=prompt,
            schema=COMBINED_SCHEMA,
            case_id=str(case.get("case_id", "fallback")),
            tag="DSPY_COMBINED_FALLBACK",
            temperature=cfg.DSPY_TASK_TEMPERATURE,
        )

        parsed = _json_from_text(raw)
        if parsed:
            return StrokePrediction(
                labels=clean_labels(parsed.get("Combined_GT", parsed.get("labels", "NONE"))),
                reasoning=str(parsed.get("reasoning", "Recovered by direct Ollama fallback.")).strip(),
            )

        return StrokePrediction(
            labels=_labels_from_reasoning_text(raw),
            reasoning="Recovered combined labels from direct Ollama fallback text because JSON parsing failed.",
        )

    except Exception as fallback_error:
        return StrokePrediction(
            labels=["NONE"],
            reasoning=f"DSPy combined call failed and direct fallback also failed: {fallback_error}",
        )


class CTStrokeSignature(dspy.Signature):
    report_text: str = dspy.InputField(desc=INPUT_DESCRIPTIONS["ct_report"])
    labels: str = dspy.OutputField(desc=OUTPUT_DESCRIPTIONS["labels"])
    reasoning: str = dspy.OutputField(desc=OUTPUT_DESCRIPTIONS["reasoning"])


class CTAStrokeSignature(dspy.Signature):
    cta_base_prompt: str = dspy.InputField(
        desc=(
            "Immutable authoritative CTA labeling prompt supplied by code. "
            "Apply it exactly to report_text; the optimized signature instruction "
            "is only a supplemental interpretation layer and may not replace it."
        )
    )
    report_text: str = dspy.InputField(desc=INPUT_DESCRIPTIONS["cta_report"])
    labels: str = dspy.OutputField(desc=OUTPUT_DESCRIPTIONS["labels"])
    reasoning: str = dspy.OutputField(desc=OUTPUT_DESCRIPTIONS["reasoning"])


class CTPStrokeSignature(dspy.Signature):
    report_text: str = dspy.InputField(desc=INPUT_DESCRIPTIONS["ctp_report"])
    labels: str = dspy.OutputField(desc=OUTPUT_DESCRIPTIONS["labels"])
    reasoning: str = dspy.OutputField(desc=OUTPUT_DESCRIPTIONS["reasoning"])


class CombinedStrokeSignature(dspy.Signature):
    ct_report: str = dspy.InputField(desc=INPUT_DESCRIPTIONS["ct_report"])
    cta_report: str = dspy.InputField(desc=INPUT_DESCRIPTIONS["cta_report"])
    ctp_report: str = dspy.InputField(desc=INPUT_DESCRIPTIONS["ctp_report"])
    mri_report: str = dspy.InputField(desc=INPUT_DESCRIPTIONS["mri_report"])
    ct_labels: str = dspy.InputField(desc=INPUT_DESCRIPTIONS["ct_labels"])
    cta_labels: str = dspy.InputField(desc=INPUT_DESCRIPTIONS["cta_labels"])
    ctp_labels: str = dspy.InputField(desc=INPUT_DESCRIPTIONS["ctp_labels"])
    ct_reasoning: str = dspy.InputField(desc=INPUT_DESCRIPTIONS["ct_reasoning"])
    cta_reasoning: str = dspy.InputField(desc=INPUT_DESCRIPTIONS["cta_reasoning"])
    ctp_reasoning: str = dspy.InputField(desc=INPUT_DESCRIPTIONS["ctp_reasoning"])
    labels: str = dspy.OutputField(desc=OUTPUT_DESCRIPTIONS["labels"])
    reasoning: str = dspy.OutputField(desc=OUTPUT_DESCRIPTIONS["combined_reasoning"])


# DSPy reads the class docstring as part of the signature instructions.  The
# docstrings are assigned from config.py so you can edit task rules in one place.
CTStrokeSignature.__doc__ = cfg.CT_SIGNATURE_INSTRUCTIONS
CTAStrokeSignature.__doc__ = cfg.CTA_SIGNATURE_INSTRUCTIONS
CTPStrokeSignature.__doc__ = cfg.CTP_SIGNATURE_INSTRUCTIONS
CombinedStrokeSignature.__doc__ = cfg.COMBINED_SIGNATURE_INSTRUCTIONS


class CTLabeler(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CTStrokeSignature)

    def forward(self, report_text: str):
        with require_full_report(report_text):
            return self.predict(report_text=report_text)


class CTALabeler(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CTAStrokeSignature)

    def forward(self, report_text: str):
        # The manual base prompt is a normal runtime input, so MIPRO/GEPA can
        # optimize only CTAStrokeSignature.instructions (the supplement).
        # Each accepted supplement replaces the previous one; the base prompt
        # remains byte-for-byte identical on every CTA prediction.
        with require_full_report(report_text), require_cta_base_prompt(cfg.CTA_BASE_PROMPT):
            return self.predict(
                cta_base_prompt=cfg.CTA_BASE_PROMPT,
                report_text=report_text,
            )

class CTPLabeler(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CTPStrokeSignature)

    def forward(self, report_text: str):
        with require_full_report(report_text):
            return self.predict(report_text=report_text)


class CombinedLabeler(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(CombinedStrokeSignature)

    def forward(
        self,
        ct_report: str,
        cta_report: str,
        ctp_report: str,
        mri_report: str,
        ct_labels: str,
        cta_labels: str,
        ctp_labels: str,
        ct_reasoning: str,
        cta_reasoning: str,
        ctp_reasoning: str,
    ):
        return self.predict(
            ct_report=ct_report,
            cta_report=cta_report,
            ctp_report=ctp_report,
            mri_report=mri_report,
            ct_labels=ct_labels,
            cta_labels=cta_labels,
            ctp_labels=ctp_labels,
            ct_reasoning=ct_reasoning,
            cta_reasoning=cta_reasoning,
            ctp_reasoning=ctp_reasoning,
        )


_PROGRAMS: Dict[str, dspy.Module] = {}


def _program_path(name: str) -> Path:
    return Path(cfg.DSPY_PROGRAM_DIR) / f"{name}.json"


def _load_or_create(name: str, cls):
    if name in _PROGRAMS:
        return _PROGRAMS[name]
    program = cls()
    path = _program_path(name)
    if path.exists():
        try:
            program.load(str(path))
            print(f"Loaded optimized DSPy program: {path}")
        except Exception as e:
            print(f"⚠️ Could not load optimized DSPy program {path}: {e}")
            print("Using unoptimized DSPy program instead.")
    _PROGRAMS[name] = program
    return program


def initialize_dspy_programs() -> None:
    """
    Configure DSPy and load/create all programs.

    Important fix:
        cfg.DSPY_PROGRAM_NAMES stores save-file names like "ct_labeler",
        but the label functions use simple runtime keys like "ct".

        We store BOTH:
            _PROGRAMS["ct_labeler"] -> CT program
            _PROGRAMS["ct"]         -> same CT program

        This prevents KeyError: 'ct'.
    """
    configure_dspy()

    ct_program = _load_or_create(cfg.DSPY_PROGRAM_NAMES["CT"], CTLabeler)
    cta_program = _load_or_create(cfg.DSPY_PROGRAM_NAMES["CTA"], CTALabeler)
    ctp_program = _load_or_create(cfg.DSPY_PROGRAM_NAMES["CTP"], CTPLabeler)
    combined_program = _load_or_create(cfg.DSPY_PROGRAM_NAMES["Combined"], CombinedLabeler)

    # Runtime aliases used by label_ct / label_cta / label_ctp / label_combined.
    _PROGRAMS["ct"] = ct_program
    _PROGRAMS["cta"] = cta_program
    _PROGRAMS["ctp"] = ctp_program
    _PROGRAMS["combined"] = combined_program


def _program_for(runtime_key: str, config_key: str, cls):
    """
    Return a loaded program safely.

    This makes the public label_* functions robust even if someone calls them
    directly without calling initialize_dspy_programs() first.
    """
    if runtime_key not in _PROGRAMS:
        # If the runtime alias is missing, initialize everything.
        initialize_dspy_programs()

    if runtime_key in _PROGRAMS:
        return _PROGRAMS[runtime_key]

    # Final fallback: load/create from config save name and register alias.
    program = _load_or_create(cfg.DSPY_PROGRAM_NAMES[config_key], cls)
    _PROGRAMS[runtime_key] = program
    return program


def label_ct(report_text: str) -> StrokePrediction:
    """Label CT report text with DSPy, falling back to direct Ollama if parsing fails."""
    program = _program_for("ct", "CT", CTLabeler)
    return _safe_program_call(program, report_text or "", "CT")


def label_cta(report_text: str) -> StrokePrediction:
    """Label CTA report text with DSPy, falling back to direct Ollama if parsing fails."""
    program = _program_for("cta", "CTA", CTALabeler)
    return _safe_program_call(program, report_text or "", "CTA")


def label_ctp(report_text: str) -> StrokePrediction:
    """Label CTP report text with DSPy, falling back to direct Ollama if parsing fails."""
    program = _program_for("ctp", "CTP", CTPLabeler)
    return _safe_program_call(program, report_text or "", "CTP")


def label_combined(case: Dict[str, Any]) -> StrokePrediction:
    """Label the final combined case with DSPy, falling back to direct Ollama if parsing fails."""
    program = _program_for("combined", "Combined", CombinedLabeler)

    try:
        pred = program(
            ct_report=case.get("New_CT_Report") or case.get("CT_Report") or "",
            cta_report=case.get("CTA_Report") or "",
            ctp_report=case.get("CTP_Report") or "",
            mri_report=case.get("MRI_Report") or "",
            ct_labels=", ".join(case.get("CT_GT", ["NONE"])),
            cta_labels=", ".join(case.get("CTA_GT", ["NONE"])),
            ctp_labels=", ".join(case.get("CTP_GT", ["NONE"])),
            ct_reasoning=case.get("CT_GT_reasoning", ""),
            cta_reasoning=case.get("CTA_GT_reasoning", ""),
            ctp_reasoning=case.get("CTP_GT_reasoning", ""),
        )
        return prediction_to_result(pred)

    except Exception as exc:
        message = str(exc)
        if _is_dspy_parse_error(exc):
            fallback = _direct_ollama_combined_fallback(case)
            fallback.reasoning = (
                f"DSPy parse fallback used for Combined. Original error: {message[:300]}\\n\\n"
                f"{fallback.reasoning}"
            )
            return fallback

        raise
