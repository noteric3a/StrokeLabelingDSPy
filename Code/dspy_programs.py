from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import dspy

from Code import config as cfg
from Code.utils import normalize_labels


@dataclass
class StrokePrediction:
    """Clean output object used by the rest of the pipeline."""
    labels: List[str]
    reasoning: str


def configure_dspy() -> None:
    """Configure DSPy to use the model settings from config.py."""
    lm = dspy.LM(
        cfg.DSPY_MODEL,
        api_base=cfg.DSPY_API_BASE,
        temperature=cfg.DSPY_TEMPERATURE,
        max_tokens=cfg.DSPY_MAX_TOKENS,
    )
    dspy.configure(lm=lm)


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


def prediction_to_result(pred: Any) -> StrokePrediction:
    """Convert a raw DSPy Prediction into the clean project output shape."""
    labels = clean_labels(getattr(pred, "labels", "NONE"))
    reasoning = str(getattr(pred, "reasoning", "") or "").strip()
    if not reasoning:
        reasoning = "No reasoning returned by DSPy program."
    return StrokePrediction(labels=labels, reasoning=reasoning)


class CTStrokeSignature(dspy.Signature):
    report_text: str = dspy.InputField(desc=cfg.DSPY_INPUT_DESCRIPTIONS["ct_report"])
    labels: str = dspy.OutputField(desc=cfg.DSPY_OUTPUT_DESCRIPTIONS["labels"])
    reasoning: str = dspy.OutputField(desc=cfg.DSPY_OUTPUT_DESCRIPTIONS["reasoning"])


class CTAStrokeSignature(dspy.Signature):
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
        self.predict = dspy.ChainOfThought(CTStrokeSignature)

    def forward(self, report_text: str):
        return self.predict(report_text=report_text)


class CTALabeler(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(CTAStrokeSignature)

    def forward(self, report_text: str):
        return self.predict(report_text=report_text)


class CTPLabeler(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(CTPStrokeSignature)

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
    configure_dspy()
    _load_or_create(cfg.DSPY_PROGRAM_NAMES["CT"], CTLabeler)
    _load_or_create(cfg.DSPY_PROGRAM_NAMES["CTA"], CTALabeler)
    _load_or_create(cfg.DSPY_PROGRAM_NAMES["CTP"], CTPLabeler)
    _load_or_create(cfg.DSPY_PROGRAM_NAMES["Combined"], CombinedLabeler)


def label_ct(report_text: str) -> StrokePrediction:
    program = _load_or_create(cfg.DSPY_PROGRAM_NAMES["CT"], CTLabeler)
    pred = program(report_text=report_text or "")
    return prediction_to_result(pred)


def label_cta(report_text: str) -> StrokePrediction:
    program = _load_or_create(cfg.DSPY_PROGRAM_NAMES["CTA"], CTALabeler)
    pred = program(report_text=report_text or "")
    return prediction_to_result(pred)


def label_ctp(report_text: str) -> StrokePrediction:
    program = _load_or_create(cfg.DSPY_PROGRAM_NAMES["CTP"], CTPLabeler)
    pred = program(report_text=report_text or "")
    return prediction_to_result(pred)


def label_combined(case: Dict[str, Any]) -> StrokePrediction:
    program = _load_or_create(cfg.DSPY_PROGRAM_NAMES["Combined"], CombinedLabeler)
    ct_labels = ", ".join(clean_labels(case.get("CT_GT", ["NONE"])))
    cta_labels = ", ".join(clean_labels(case.get("CTA_GT", ["NONE"])))
    ctp_labels = ", ".join(clean_labels(case.get("CTP_GT", ["NONE"])))
    pred = program(
        ct_report=str(case.get("New_CT_Report") or case.get("CT_Report") or ""),
        cta_report=str(case.get("CTA_Report") or ""),
        ctp_report=str(case.get("CTP_Report") or ""),
        mri_report=str(case.get("MRI_Report") or ""),
        ct_labels=ct_labels,
        cta_labels=cta_labels,
        ctp_labels=ctp_labels,
        ct_reasoning=str(case.get("CT_GT_reasoning") or ""),
        cta_reasoning=str(case.get("CTA_GT_reasoning") or ""),
        ctp_reasoning=str(case.get("CTP_GT_reasoning") or ""),
    )
    return prediction_to_result(pred)
