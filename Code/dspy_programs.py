"""
dspy_programs.py

This file is the DSPy replacement for your old prompts.py approach.

In the manual pipeline, prompts.py probably did something like:

    1. Build a giant prompt string.
    2. Insert the radiology report into the prompt.
    3. Send that prompt to Ollama.
    4. Parse the model's response.

In this DSPy version, we instead define structured tasks called "Signatures."

A DSPy Signature tells the model:

    - what inputs it receives
    - what outputs it must produce
    - what rules/instructions it should follow

Then a DSPy Module uses that Signature to call the model.

The rest of your project does not need to know the difference. This file returns
the same kind of result your old pipeline returned:

    - labels
    - reasoning

Those values can still be written into:

    - CT_GT
    - CTA_GT
    - CTP_GT
    - Combined_GT
    - CT_GT_reasoning
    - CTA_GT_reasoning
    - CTP_GT_reasoning
    - CT_Combined_GT_reasoning
"""

from __future__ import annotations

# dataclass gives us a small, clean object for returning predictions.
from dataclasses import dataclass

# Path is used for loading saved optimized DSPy programs.
from pathlib import Path

# Any/Dict/List are just type hints that make the code easier to read.
from typing import Any, Dict, List

# DSPy is the prompt/program optimization library.
import dspy

# These settings should be copied into your real config.py.
from Code.config import (
    ALLOWED_LABELS,
    DSPY_API_BASE,
    DSPY_MAX_TOKENS,
    DSPY_MODEL,
    DSPY_PROGRAM_DIR,
    DSPY_TEMPERATURE,
)

# Reuse your existing label normalization function.
# This is important because your validator/checker already expects labels in
# the same normalized format.
from Code.utils import normalize_labels


# =============================================================================
# Small return object
# =============================================================================

@dataclass
class StrokePrediction:
    """
    Small wrapper for the output of a DSPy labeling call.

    Instead of passing around raw DSPy objects everywhere, we convert DSPy's
    Prediction object into this simple structure.

    Attributes:
        labels:
            A list of final labels, such as ["RMCA"] or ["RMCA", "RACA"].

        reasoning:
            A short explanation from the model.
    """

    labels: List[str]
    reasoning: str


# =============================================================================
# DSPy setup
# =============================================================================

def configure_dspy() -> None:
    """
    Configure DSPy to use your local Ollama model.

    This should be called once before running the DSPy labelers.

    In your old pipeline, you probably directly called your own Ollama client.
    In this pipeline, DSPy handles the model call, but still sends requests to
    the same local Ollama server.

    Important:
        Ollama must already be running.

    Usually:
        ollama serve
    """

    # dspy.LM creates the language model object DSPy will use.
    #
    # DSPY_MODEL example:
    #     "ollama_chat/qwen3.6:35b"
    #
    # DSPY_API_BASE example:
    #     "http://localhost:11434"
    lm = dspy.LM(
        DSPY_MODEL,
        api_base=DSPY_API_BASE,
        temperature=DSPY_TEMPERATURE,
        max_tokens=DSPY_MAX_TOKENS,
    )

    # This tells DSPy, "use this model for all DSPy calls."
    dspy.configure(lm=lm)


# =============================================================================
# Label cleanup helpers
# =============================================================================

def clean_labels(raw_labels: Any) -> List[str]:
    """
    Normalize model labels into your official label format.

    Why this is needed:
        LLMs may output labels in many formats:

            "RMCA"
            "RMCA, LMCA"
            ["RMCA", "LMCA"]
            "right MCA"
            "None"

        Your downstream code expects clean labels like:

            ["RMCA"]
            ["RMCA", "LMCA"]
            ["NONE"]

    This function:
        1. Uses your existing normalize_labels().
        2. Removes labels that are not in ALLOWED_LABELS.
        3. Ensures we always return at least ["NONE"].
        4. Removes NONE if another territory label is present.
    """

    # Let your existing project normalization do the first pass.
    labels = normalize_labels(raw_labels)

    cleaned: List[str] = []

    for label in labels:
        # Convert to uppercase and strip whitespace.
        label = str(label).strip().upper()

        # Keep only official allowed labels.
        if label in ALLOWED_LABELS and label not in cleaned:
            cleaned.append(label)

    # If the model returned nothing usable, default to NONE.
    if not cleaned:
        return ["NONE"]

    # NONE should never appear together with a positive territory.
    #
    # Bad:
    #     ["NONE", "RMCA"]
    #
    # Fixed:
    #     ["RMCA"]
    if "NONE" in cleaned and len(cleaned) > 1:
        cleaned = [label for label in cleaned if label != "NONE"]

    return cleaned


def prediction_to_result(pred: Any) -> StrokePrediction:
    """
    Convert a raw DSPy Prediction into StrokePrediction.

    DSPy returns objects where fields can be accessed like:

        pred.labels
        pred.reasoning

    This function extracts those fields and cleans them.
    """

    # Extract labels from the DSPy prediction.
    # If the model fails to return labels, default to NONE.
    labels = clean_labels(getattr(pred, "labels", "NONE"))

    # Extract reasoning.
    # If the model fails to return reasoning, store a placeholder.
    reasoning = str(getattr(pred, "reasoning", "") or "").strip()

    if not reasoning:
        reasoning = "No reasoning returned by DSPy program."

    return StrokePrediction(labels=labels, reasoning=reasoning)


# =============================================================================
# DSPy Signatures
# =============================================================================
# A Signature is where you define the task.
#
# Think of a Signature as the replacement for a huge prompt template.
#
# Each Signature has:
#   - InputField(s): what the model receives
#   - OutputField(s): what the model must return
#   - docstring: instructions/rules for the task
# =============================================================================

class CTStrokeSignature(dspy.Signature):
    """
    Label acute ischemic stroke territory from CT brain report text only.

    Allowed labels:
    NONE, RMCA, LMCA, RACA, LACA, RPCA, LPCA, RPICA, LPICA,
    BA, RVA, LVA, RICA, LICA, RCA, LCA.

    CT-specific rules:
    - Use NONE only when there is no qualifying acute ischemic stroke territory.
    - Do not include NONE with another label.
    - Do not use CTA or CTP findings when labeling CT.
    - CT-visible acute infarct signs should map to their arterial territory.
    - Basal ganglia, lentiform nucleus, putamen, caudate, internal capsule,
      corona radiata, centrum semiovale, insula, and operculum usually map to
      MCA territory.
    - Do not label chronic infarcts, old encephalomalacia, stable findings,
      artifacts, or nonspecific weak findings as acute stroke territories.
    """

    # The CT report text is the input.
    report_text: str = dspy.InputField(desc="CT brain report text.")

    # labels is the final output label list, returned as comma-separated text.
    labels: str = dspy.OutputField(desc="Comma-separated final labels.")

    # reasoning explains why those labels were chosen.
    reasoning: str = dspy.OutputField(desc="Brief explanation supporting the final labels.")


class CTAStrokeSignature(dspy.Signature):
    """
    Label acute stroke-related vascular territory from CTA report text.

    Allowed labels:
    NONE, RMCA, LMCA, RACA, LACA, RPCA, LPCA, RPICA, LPICA,
    BA, RVA, LVA, RICA, LICA, RCA, LCA.

    CTA-specific rules:
    - Use MCA/ACA/PCA labels for named branch occlusion or severe flow-limiting
      stenosis in that territory.
    - Use RICA/LICA for acute intracranial ICA, carotid terminus, terminal ICA,
      supraclinoid ICA, paraclinoid ICA, or intracranial carotid involvement.
    - Use RCA/LCA only for common carotid or cervical carotid involvement.
    - Do not use RCA/LCA for carotid terminus.
    - Prefer specific downstream territory labels when MCA/ACA/PCA involvement
      is clearly identified.
    - Use NONE when no qualifying acute occlusion or severe flow-limiting lesion
      is present.
    - Do not label mild stenosis, incidental atherosclerosis, chronic occlusion,
      or stable old findings unless the project rules say they qualify.
    """

    report_text: str = dspy.InputField(desc="CTA report text.")
    labels: str = dspy.OutputField(desc="Comma-separated final labels.")
    reasoning: str = dspy.OutputField(desc="Brief explanation supporting the final labels.")


class CTPStrokeSignature(dspy.Signature):
    """
    Label acute perfusion territory from CT perfusion report text.

    Allowed labels:
    NONE, RMCA, LMCA, RACA, LACA, RPCA, LPCA, RPICA, LPICA,
    BA, RVA, LVA, RICA, LICA, RCA, LCA.

    CTP-specific rules:
    - Label the territory of qualifying hypoperfusion, infarct core, mismatch,
      penumbra, or tissue at risk.
    - Use NONE when there is no qualifying perfusion deficit.
    - Do not label tiny nonspecific artifacts or clearly non-territorial findings.
    - Prefer the tissue/perfusion territory over an upstream mechanism.
    - If RAPID/perfusion values say core and hypoperfusion are 0 mL, usually use NONE.
    """

    report_text: str = dspy.InputField(desc="CT perfusion report text.")
    labels: str = dspy.OutputField(desc="Comma-separated final labels.")
    reasoning: str = dspy.OutputField(desc="Brief explanation supporting the final labels.")


class CombinedStrokeSignature(dspy.Signature):
    """
    Produce final combined acute stroke territory labels from modality labels and reports.

    Allowed labels:
    NONE, RMCA, LMCA, RACA, LACA, RPCA, LPCA, RPICA, LPICA,
    BA, RVA, LVA, RICA, LICA, RCA, LCA.

    Combined-specific rules:
    - Start from CT_GT, CTA_GT, and CTP_GT.
    - Add MRI acute/recent infarct territories when present.
    - Remove labels only for clear reasons, such as:
        - chronic/stable finding
        - artifact
        - weak nonspecific CT sign
        - isolated upstream ICA when a more specific downstream tissue territory
          is identified
        - non-qualifying mild stenosis
        - mismatch/perfusion finding that is too small or nonspecific
    - Prefer tissue/perfusion territory over upstream mechanism when they conflict.
    - Use NONE only when no final acute stroke territory remains.
    - Do not include NONE with any positive territory label.
    """

    # Combined gets the original report text so it can resolve conflicts.
    ct_report: str = dspy.InputField(desc="CT report text.")
    cta_report: str = dspy.InputField(desc="CTA report text.")
    ctp_report: str = dspy.InputField(desc="CTP report text.")
    mri_report: str = dspy.InputField(desc="MRI report text if available.")

    # Combined also gets the already predicted modality labels.
    # This mirrors your current manual pipeline where Combined uses CT/CTA/CTP results.
    ct_labels: str = dspy.InputField(desc="Already predicted CT labels.")
    cta_labels: str = dspy.InputField(desc="Already predicted CTA labels.")
    ctp_labels: str = dspy.InputField(desc="Already predicted CTP labels.")

    # Give Combined the modality reasoning too.
    # This can help it understand why a modality label was included.
    ct_reasoning: str = dspy.InputField(desc="CT reasoning.")
    cta_reasoning: str = dspy.InputField(desc="CTA reasoning.")
    ctp_reasoning: str = dspy.InputField(desc="CTP reasoning.")

    labels: str = dspy.OutputField(desc="Comma-separated final combined labels.")
    reasoning: str = dspy.OutputField(desc="Brief explanation for the final combined labels.")


# =============================================================================
# DSPy Modules
# =============================================================================
# A Module is the executable program that uses a Signature.
#
# dspy.ChainOfThought means:
#     Ask the model to reason before producing the final labels.
#
# If you want shorter/faster outputs, you can try dspy.Predict instead.
# =============================================================================

class CTLabeler(dspy.Module):
    """
    DSPy CT labeler.

    Input:
        report_text

    Output:
        labels
        reasoning
    """

    def __init__(self):
        super().__init__()

        # ChainOfThought usually improves reasoning tasks.
        self.predict = dspy.ChainOfThought(CTStrokeSignature)

    def forward(self, report_text: str):
        # DSPy modules use forward() as the main call.
        return self.predict(report_text=report_text)


class CTALabeler(dspy.Module):
    """
    DSPy CTA labeler.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(CTAStrokeSignature)

    def forward(self, report_text: str):
        return self.predict(report_text=report_text)


class CTPLabeler(dspy.Module):
    """
    DSPy CTP labeler.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(CTPStrokeSignature)

    def forward(self, report_text: str):
        return self.predict(report_text=report_text)


class CombinedLabeler(dspy.Module):
    """
    DSPy Combined labeler.

    This should be run after CT, CTA, and CTP have already been labeled.
    """

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


# =============================================================================
# Program loading and caching
# =============================================================================
# DSPy optimization can save compiled/optimized programs to disk.
#
# This section tries to load optimized versions if they exist.
# If they do not exist, it falls back to the basic unoptimized modules.
# =============================================================================

# Cache programs so they are created only once.
_PROGRAMS: Dict[str, dspy.Module] = {}


def _program_path(name: str) -> Path:
    """
    Return the path for a saved optimized DSPy program.

    Example:
        _program_path("ct_labeler")
        -> optimized_programs/ct_labeler.json
    """

    return Path(DSPY_PROGRAM_DIR) / f"{name}.json"


def _load_or_create(name: str, cls):
    """
    Load an optimized DSPy program if it exists.

    Args:
        name:
            Name of the saved program file without .json.

        cls:
            The class to instantiate if no optimized program exists.

    Returns:
        A DSPy module.
    """

    # If already loaded, reuse it.
    if name in _PROGRAMS:
        return _PROGRAMS[name]

    # Create the default/unoptimized module.
    program = cls()

    # Check whether an optimized version exists on disk.
    path = _program_path(name)

    if path.exists():
        try:
            # DSPy modules support load().
            program.load(str(path))
            print(f"Loaded optimized DSPy program: {path}")
        except Exception as e:
            # If loading fails, keep using the base version.
            print(f"⚠️ Could not load optimized DSPy program {path}: {e}")
            print("Using unoptimized DSPy program instead.")

    # Store in cache.
    _PROGRAMS[name] = program

    return program


def initialize_dspy_programs() -> None:
    """
    Configure DSPy and initialize all programs.

    Call this once before labeling cases.
    """

    configure_dspy()

    _load_or_create("ct_labeler", CTLabeler)
    _load_or_create("cta_labeler", CTALabeler)
    _load_or_create("ctp_labeler", CTPLabeler)
    _load_or_create("combined_labeler", CombinedLabeler)


# =============================================================================
# Public functions used by labeler_dspy.py
# =============================================================================
# These functions hide all DSPy internals from the rest of your pipeline.
#
# The labeler only needs:
#     label_ct(report_text)
#     label_cta(report_text)
#     label_ctp(report_text)
#     label_combined(case)
# =============================================================================

def label_ct(report_text: str) -> StrokePrediction:
    """
    Label one CT report.
    """

    program = _load_or_create("ct_labeler", CTLabeler)
    pred = program(report_text=report_text or "")
    return prediction_to_result(pred)


def label_cta(report_text: str) -> StrokePrediction:
    """
    Label one CTA report.
    """

    program = _load_or_create("cta_labeler", CTALabeler)
    pred = program(report_text=report_text or "")
    return prediction_to_result(pred)


def label_ctp(report_text: str) -> StrokePrediction:
    """
    Label one CTP report.
    """

    program = _load_or_create("ctp_labeler", CTPLabeler)
    pred = program(report_text=report_text or "")
    return prediction_to_result(pred)


def label_combined(case: Dict[str, Any]) -> StrokePrediction:
    """
    Label the final Combined_GT for one case.

    This function assumes CT_GT, CTA_GT, and CTP_GT have already been created.
    """

    program = _load_or_create("combined_labeler", CombinedLabeler)

    # Convert modality labels to comma-separated strings for DSPy input.
    ct_labels = ", ".join(clean_labels(case.get("CT_GT", ["NONE"])))
    cta_labels = ", ".join(clean_labels(case.get("CTA_GT", ["NONE"])))
    ctp_labels = ", ".join(clean_labels(case.get("CTP_GT", ["NONE"])))

    # Prefer sanitized CT report if available.
    # This preserves your previous CT contamination workflow.
    ct_report = str(case.get("New_CT_Report") or case.get("CT_Report") or "")

    pred = program(
        ct_report=ct_report,
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
