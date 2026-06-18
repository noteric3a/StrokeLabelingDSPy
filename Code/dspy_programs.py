"""DSPy programs and model helpers for automatic stroke-prompt optimization.

The production labeler can keep using its existing raw Ollama prompts, or it can
load the compiled DSPy programs created by ``dspy_optimize.py``.  The programs
below deliberately begin with small zero-shot instructions.  GEPA is expected
to evolve those instructions from report/ground-truth failures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypeAlias
from urllib.parse import urlsplit, urlunsplit

import dspy

import config as cfg

ALLOWED_LABELS = cfg.ALLOWED_LABELS


PROGRAM_NAMES = ("CT", "CTA", "CTP", "Combined")
MODALITY_PROGRAM_NAMES = ("CT", "CTA", "CTP")

# Keep the output type constrained at the DSPy adapter/parser layer.  This is
# intentionally written explicitly because Literal cannot be built from a list
# in a way that all supported Python/Pydantic versions type consistently.
AllowedLabel: TypeAlias = Literal[
    "NONE",
    "RMCA", "LMCA",
    "RACA", "LACA",
    "RPCA", "LPCA",
    "RPICA", "LPICA",
    "BA",
    "RVA", "LVA",
    "RICA", "LICA",
]


_DEFAULT_BASE_INSTRUCTIONS: dict[str, str] = {
    "CT": (
        "Label the acute or recent ischemic stroke territory in one non-contrast CT brain report. "
        "Use only the report, return the smallest directly supported label set, and return NONE when "
        "there is no qualifying acute/recent CT finding. Ignore chronic/old findings and any CTA/CTP "
        "content mixed into the report. Map side and named anatomy literally; do not infer extra territories."
    ),
    "CTA": (
        "Label qualifying acute vessel abnormalities in one CTA report. Definite occlusion, thrombus, "
        "filling defect, flow cutoff, non-opacification, or near-occlusion can qualify. Stenosis, plaque, "
        "variant anatomy, uncertainty, and stable/chronic disease alone do not. Use only the report and "
        "return the smallest exact label set, or NONE."
    ),
    "CTP": (
        "Label the acute stroke territory in one CT-perfusion report. Use localized core, hypoperfusion, "
        "Tmax delay, mismatch, penumbra, or tissue-at-risk evidence tied to a side and territory. Ignore "
        "artifact, chronic collateral/moyamoya patterns, and tiny nonspecific abnormalities. Use only the "
        "report and return the smallest exact label set, or NONE."
    ),
    "Combined": (
        "Reconcile preliminary CT, CTA, and CTP labels with the supplied reports and MRI. Keep only labels "
        "with direct acute/recent evidence, add an MRI label only for a direct acute/recent territorial "
        "finding, and do not invent labels from mechanism or broad anatomy. Return the smallest exact final "
        "label set, or NONE."
    ),
}


def _apply_instruction_mapping(
    output: dict[str, str],
    raw_mapping: Any,
    *,
    source_name: str,
) -> None:
    if raw_mapping is None:
        return
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"{source_name} must be a dictionary.")
    if isinstance(raw_mapping.get("programs"), dict):
        raw_mapping = raw_mapping["programs"]

    for raw_name, raw_instruction in raw_mapping.items():
        name = canonical_program_name(raw_name)
        if isinstance(raw_instruction, dict):
            raw_instruction = raw_instruction.get("instructions", "")
        instruction = str(raw_instruction or "").strip()
        if not instruction:
            raise ValueError(f"{source_name}[{name!r}] cannot be blank.")
        output[name] = instruction


def configured_base_instructions() -> dict[str, str]:
    """Return base instructions with config and optional JSON overrides."""
    output = dict(_DEFAULT_BASE_INSTRUCTIONS)
    _apply_instruction_mapping(
        output,
        getattr(cfg, "DSPY_BASE_INSTRUCTIONS", {}),
        source_name="config.DSPY_BASE_INSTRUCTIONS",
    )

    prompt_file_raw = getattr(cfg, "DSPY_BASE_PROMPTS_FILE", "")
    if str(prompt_file_raw).strip():
        prompt_file = Path(prompt_file_raw)
        if not prompt_file.exists() or not prompt_file.is_file():
            raise FileNotFoundError(
                f"config.DSPY_BASE_PROMPTS_FILE does not exist: {prompt_file}"
            )
        try:
            payload = json.loads(prompt_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"config.DSPY_BASE_PROMPTS_FILE is not valid JSON: {prompt_file}"
            ) from exc
        _apply_instruction_mapping(
            output,
            payload,
            source_name="config.DSPY_BASE_PROMPTS_FILE",
        )
    return output


_LABEL_DESCRIPTION = (
    "Exact stroke-territory label set. Allowed values: "
    + ", ".join(ALLOWED_LABELS)
    + ". Never mix NONE with another label."
)


class CTSignature(dspy.Signature):
    """Base CT instruction; GEPA may replace this text."""

    report: str = dspy.InputField(desc="One non-contrast CT head/brain report.")
    labels: list[AllowedLabel] = dspy.OutputField(desc=_LABEL_DESCRIPTION)
    reasoning: str = dspy.OutputField(
        desc="One to three concise sentences citing only the report evidence supporting the labels."
    )


class CTASignature(dspy.Signature):
    """Base CTA instruction; GEPA may replace this text."""

    report: str = dspy.InputField(desc="One CTA head/neck vessel report.")
    labels: list[AllowedLabel] = dspy.OutputField(desc=_LABEL_DESCRIPTION)
    reasoning: str = dspy.OutputField(
        desc="One to three concise sentences citing only the report evidence supporting the labels."
    )


class CTPSignature(dspy.Signature):
    """Base CTP instruction; GEPA may replace this text."""

    report: str = dspy.InputField(desc="One CT-perfusion report.")
    labels: list[AllowedLabel] = dspy.OutputField(desc=_LABEL_DESCRIPTION)
    reasoning: str = dspy.OutputField(
        desc="One to three concise sentences citing only the report evidence supporting the labels."
    )


class CombinedSignature(dspy.Signature):
    """Base multimodality reconciliation instruction; GEPA may replace it."""

    ct_report: str = dspy.InputField(desc="Non-contrast CT report.")
    cta_report: str = dspy.InputField(desc="CTA report.")
    ctp_report: str = dspy.InputField(desc="CT-perfusion report.")
    mri_report: str = dspy.InputField(desc="MRI report, possibly blank.")
    ct_labels: list[AllowedLabel] = dspy.InputField(desc="Preliminary CT labels.")
    cta_labels: list[AllowedLabel] = dspy.InputField(desc="Preliminary CTA labels.")
    ctp_labels: list[AllowedLabel] = dspy.InputField(desc="Preliminary CTP labels.")
    labels: list[AllowedLabel] = dspy.OutputField(desc=_LABEL_DESCRIPTION)
    reasoning: str = dspy.OutputField(
        desc="One to three concise sentences supporting only the final labels."
    )


_SIGNATURES: dict[str, type[dspy.Signature]] = {
    "CT": CTSignature,
    "CTA": CTASignature,
    "CTP": CTPSignature,
    "Combined": CombinedSignature,
}


class StrokeTerritoryProgram(dspy.Module):
    """One optimizable DSPy predictor for a modality or Combined reconciliation."""

    def __init__(self, program_name: str, instructions: str | None = None) -> None:
        super().__init__()
        name = canonical_program_name(program_name)
        signature = _SIGNATURES[name].with_instructions(
            instructions if instructions is not None else configured_base_instructions()[name]
        )
        self.program_name = name
        self.predict = dspy.Predict(signature)

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        return self.predict(**kwargs)


def canonical_program_name(program_name: str) -> str:
    """Return the canonical program name or raise a useful error."""
    normalized = str(program_name).strip().lower()
    mapping = {name.lower(): name for name in PROGRAM_NAMES}
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported DSPy program {program_name!r}. Expected one of {PROGRAM_NAMES}."
        )
    return mapping[normalized]


def base_instructions(program_name: str) -> str:
    return configured_base_instructions()[canonical_program_name(program_name)]


def create_program(program_name: str, instructions: str | None = None) -> StrokeTerritoryProgram:
    return StrokeTerritoryProgram(program_name, instructions=instructions)


def program_filename(program_name: str) -> str:
    return f"{canonical_program_name(program_name).lower()}_program.json"


def program_path(program_dir: str | Path, program_name: str) -> Path:
    return Path(program_dir) / program_filename(program_name)


def load_program(
    program_dir: str | Path,
    program_name: str,
    *,
    require_saved: bool = False,
) -> StrokeTerritoryProgram:
    """Instantiate a program and load its compiled DSPy state when available."""
    program = create_program(program_name)
    path = program_path(program_dir, program_name)
    if path.exists():
        program.load(path=str(path))
    elif require_saved:
        raise FileNotFoundError(
            f"No compiled DSPy program exists for {program_name}: {path}. "
            "Run dspy_optimize.py first or disable DSPY_REQUIRE_OPTIMIZED_PROGRAMS."
        )
    return program


def save_program(program: StrokeTerritoryProgram, path: str | Path) -> Path:
    """Atomically save a DSPy program state as readable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    program.save(path=str(temporary))
    temporary.replace(destination)
    return destination


def get_instructions(program: StrokeTerritoryProgram) -> str:
    predictors = program.predictors()
    if not predictors:
        return ""
    return str(predictors[0].signature.instructions)


def normalize_dspy_model_name(model: str) -> str:
    """Use DSPy's recommended Ollama chat provider prefix for local model names."""
    text = str(model).strip()
    if not text:
        raise ValueError("DSPy model name cannot be blank.")
    if "/" in text:
        return text
    return f"ollama_chat/{text}"


def ollama_api_base(ollama_url: str) -> str:
    """Convert an Ollama generate URL into the server root expected by DSPy."""
    raw = str(ollama_url).strip().rstrip("/")
    if not raw:
        return "http://localhost:11434"

    parsed = urlsplit(raw)
    path = parsed.path.rstrip("/")
    for suffix in ("/api/generate", "/api/chat"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def create_lm(
    *,
    model: str,
    api_base: str,
    temperature: float,
    max_tokens: int,
    num_ctx: int | None,
    timeout_seconds: int,
    cache: bool,
    num_retries: int = 2,
) -> dspy.LM:
    """Create a DSPy LM configured for an Ollama chat endpoint."""
    kwargs: dict[str, Any] = {
        "api_base": api_base,
        "api_key": "",
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "timeout": int(timeout_seconds),
        "cache": bool(cache),
        "num_retries": int(num_retries),
        # Match the existing raw Ollama client and keep reasoning-model scratch
        # text out of the structured JSON response.
        "think": False,
    }
    if num_ctx is not None:
        kwargs["num_ctx"] = int(num_ctx)
    return dspy.LM(normalize_dspy_model_name(model), **kwargs)


def create_adapter(adapter_name: str = "json") -> dspy.Adapter:
    name = str(adapter_name or "json").strip().lower()
    if name == "json":
        return dspy.JSONAdapter()
    if name == "chat":
        return dspy.ChatAdapter()
    raise ValueError("DSPy adapter must be 'json' or 'chat'.")
