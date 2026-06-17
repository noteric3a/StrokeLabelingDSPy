from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import re

import dspy

import config as cfg
from utils import normalize_labels


@dataclass
class StrokePrediction:
    """Clean output object used by the rest of the pipeline."""
    labels: List[str]
    reasoning: str


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
    """Configure DSPy to use the model settings from config.py."""
    _disable_dspy_cache_if_configured()

    lm_kwargs = dict(
        api_base=cfg.DSPY_API_BASE,
        temperature=cfg.DSPY_TEMPERATURE,
        max_tokens=cfg.DSPY_MAX_TOKENS,
        think=False,  # Disable automatic thinking in DSPy/Ollama calls.
    )

    # Some DSPy versions accept cache=False on LM; others do not.  Try it,
    # then fall back without the argument if that version rejects it.
    if bool(getattr(cfg, "DSPY_DISABLE_CACHE", True)):
        lm_kwargs["cache"] = False

    try:
        lm = dspy.LM(cfg.DSPY_MODEL, **lm_kwargs)
    except TypeError:
        lm_kwargs.pop("cache", None)
        lm = dspy.LM(cfg.DSPY_MODEL, **lm_kwargs)

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


def _direct_ollama_fallback(report_text: str, modality: str) -> StrokePrediction:
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
            # CTA uses a short optimizable signature plus fixed rules supplied
            # through the cta_rules input. The direct fallback has no signature
            # input channel, so include the same effective prompt text here.
            "CTA": getattr(cfg, "CTA_EFFECTIVE_PROMPT_PREVIEW", cfg.CTA_SIGNATURE_INSTRUCTIONS),
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
            temperature=cfg.DSPY_TEMPERATURE,
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
            fallback = _direct_ollama_fallback(report_text or "", modality)
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
            temperature=cfg.DSPY_TEMPERATURE,
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
    report_text: str = dspy.InputField(desc=cfg.DSPY_INPUT_DESCRIPTIONS["ct_report"])
    labels: str = dspy.OutputField(desc=cfg.DSPY_OUTPUT_DESCRIPTIONS["labels"])
    reasoning: str = dspy.OutputField(desc=cfg.DSPY_OUTPUT_DESCRIPTIONS["reasoning"])


class CTAStrokeSignature(dspy.Signature):
    cta_rules: str = dspy.InputField(
        desc=(
            "Fixed CTA labeling rules supplied by code, not by the optimizer. "
            "These rules are authoritative and must be applied to report_text."
        )
    )
    report_text: str = dspy.InputField(desc=cfg.DSPY_INPUT_DESCRIPTIONS["cta_report"])
    labels: str = dspy.OutputField(desc=cfg.DSPY_OUTPUT_DESCRIPTIONS["labels"])
    reasoning: str = dspy.OutputField(desc=cfg.DSPY_OUTPUT_DESCRIPTIONS["reasoning"])


class CTPStrokeSignature(dspy.Signature):
    report_text: str = dspy.InputField(desc=cfg.DSPY_INPUT_DESCRIPTIONS["ctp_report"])
    labels: str = dspy.OutputField(desc=cfg.DSPY_OUTPUT_DESCRIPTIONS["labels"])
    reasoning: str = dspy.OutputField(desc=cfg.DSPY_OUTPUT_DESCRIPTIONS["reasoning"])


class CombinedStrokeSignature(dspy.Signature):
    ct_report: str = dspy.InputField(desc=cfg.DSPY_INPUT_DESCRIPTIONS["ct_report"])
    cta_report: str = dspy.InputField(desc=cfg.DSPY_INPUT_DESCRIPTIONS["cta_report"])
    ctp_report: str = dspy.InputField(desc=cfg.DSPY_INPUT_DESCRIPTIONS["ctp_report"])
    mri_report: str = dspy.InputField(desc=cfg.DSPY_INPUT_DESCRIPTIONS["mri_report"])
    ct_labels: str = dspy.InputField(desc=cfg.DSPY_INPUT_DESCRIPTIONS["ct_labels"])
    cta_labels: str = dspy.InputField(desc=cfg.DSPY_INPUT_DESCRIPTIONS["cta_labels"])
    ctp_labels: str = dspy.InputField(desc=cfg.DSPY_INPUT_DESCRIPTIONS["ctp_labels"])
    ct_reasoning: str = dspy.InputField(desc=cfg.DSPY_INPUT_DESCRIPTIONS["ct_reasoning"])
    cta_reasoning: str = dspy.InputField(desc=cfg.DSPY_INPUT_DESCRIPTIONS["cta_reasoning"])
    ctp_reasoning: str = dspy.InputField(desc=cfg.DSPY_INPUT_DESCRIPTIONS["ctp_reasoning"])
    labels: str = dspy.OutputField(desc=cfg.DSPY_OUTPUT_DESCRIPTIONS["labels"])
    reasoning: str = dspy.OutputField(desc=cfg.DSPY_OUTPUT_DESCRIPTIONS["combined_reasoning"])


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
        return self.predict(report_text=report_text)


class CTALabeler(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CTAStrokeSignature)

    def forward(self, report_text: str):
        # Keep the CTA rule text outside the optimizable signature instruction.
        # MIPRO can rewrite CTAStrokeSignature.__doc__, but it cannot delete this
        # fixed input, so candidate prompts cannot collapse into a short summary.
        return self.predict(cta_rules=cfg.CTA_FIXED_RULES, report_text=report_text)

class CTPLabeler(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CTPStrokeSignature)

    def forward(self, report_text: str):
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
