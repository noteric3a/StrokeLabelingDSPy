"""DSPy-based prompt optimization for the stroke labeler.

This module is optional at runtime.  The normal Ollama pipeline does not import
DSPy.  Run it through `python main.py --optimize-prompts ...` to compile small
DSPy prompt programs against a ground-truth spreadsheet, extract the optimized
instructions / selected demos, and save them as JSON guidance that prompts.py
will automatically insert into the existing hand-written prompt templates.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

import config as cfg
from config import ALLOWED_LABELS
from utils import clean_report, normalize_labels


CASE_ID_COLUMN_CANDIDATES = (
    "Case Name",
    "case_id",
    "Case ID",
    "Case_Name",
    "case name",
    "CASE_ID",
    "Case",
)

INPUT_REPORT_COLUMNS = {
    "CT": ("CT", "CT_Report", "CT report", "Noncontrast CT", "NCCT"),
    "CTA": ("CTA", "CTA_Report", "CTA report", "CT Angiography"),
    "CTP": ("CTP", "CTP_Report", "CTP report", "CT Perfusion"),
    "MRI": ("MRI", "MRI_Report", "MRI report"),
}

GROUND_TRUTH_COLUMNS = {
    "CT": ("CT_GT", "CT GT", "CT.GT", "CTGT", "CT Ground Truth", "CT_Ground_Truth"),
    "CTA": ("CTA_GT", "CTA GT", "CTA.GT", "CTAGT", "CTA Ground Truth", "CTA_Ground_Truth"),
    "CTP": ("CTP_GT", "CTP GT", "CTP.GT", "CTPGT", "CTP Ground Truth", "CTP_Ground_Truth"),
    "COMBINED": (
        "Combined_GT",
        "Combined GT",
        "Combined.GT",
        "CombinedGT",
        "Combined Ground Truth",
        "Combined_Ground_Truth",
        "Combined",
    ),
}

LABEL_ORDER = [
    "NONE",
    "RMCA", "LMCA",
    "RACA", "LACA",
    "RPCA", "LPCA",
    "RPICA", "LPICA",
    "BA",
    "RVA", "LVA",
    "RICA", "LICA",
    "RCA", "LCA",
]


@dataclass
class OptimizationResult:
    modality: str
    train_count: int
    val_count: int
    baseline_score: float | None
    optimized_score: float | None
    program_path: str | None
    guidance_text: str
    instructions: str
    selected_examples: List[Dict[str, Any]]
    notes: List[str]


def _normalize_column_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    normalized_to_original = {_normalize_column_name(col): col for col in df.columns}
    for candidate in candidates:
        match = normalized_to_original.get(_normalize_column_name(candidate))
        if match is not None:
            return match
    return None


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    suffix = p.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported table format: {p.suffix}")


def _split_label_string(value: str) -> List[str]:
    text = value.strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            pass
    return re.split(r"[,;\n]+", text)


def _normalize_label_value(value: Any) -> List[str]:
    if value is None:
        return ["NONE"]
    if isinstance(value, float) and pd.isna(value):
        return ["NONE"]
    if isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "negative", "normal", "no acute"}:
            return ["NONE"]
        raw = _split_label_string(text)

    labels: List[str] = []
    for item in raw:
        label = str(item).strip().strip('"\'[](){}').upper().replace(" ", "")
        if label in {"", "NAN", "NULL", "NEGATIVE", "NORMAL"}:
            label = "NONE"
        if label in ALLOWED_LABELS and label not in labels:
            labels.append(label)

    if not labels:
        labels = ["NONE"]
    if "NONE" in labels and len(labels) > 1:
        labels = [label for label in labels if label != "NONE"]

    order = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    return sorted(labels, key=lambda label: order.get(label, 999))


def _labels_to_text(labels: Sequence[str]) -> str:
    return ", ".join(labels) if labels else "NONE"


def _label_key(labels: Any) -> Tuple[str, ...]:
    return tuple(_normalize_label_value(labels))


def _compact_report(text: str, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", clean_report(text))
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _metric_exact_labels(example: Any, prediction: Any, trace: Any | None = None) -> float:
    expected = _label_key(getattr(example, "labels", []))
    predicted = _label_key(getattr(prediction, "labels", []))
    return 1.0 if predicted == expected else 0.0


def _score_program(program: Any, examples: Sequence[Any]) -> float | None:
    if not examples:
        return None
    correct = 0.0
    total = 0
    for example in examples:
        try:
            pred = program(
                case_id=getattr(example, "case_id"),
                modality=getattr(example, "modality"),
                allowed_labels=getattr(example, "allowed_labels"),
                report=getattr(example, "report"),
                ct_labels=getattr(example, "ct_labels", "NONE"),
                cta_labels=getattr(example, "cta_labels", "NONE"),
                ctp_labels=getattr(example, "ctp_labels", "NONE"),
            )
            correct += _metric_exact_labels(example, pred)
            total += 1
        except Exception:
            total += 1
    return correct / total if total else None


def _load_dspy() -> Any:
    try:
        import dspy  # type: ignore
    except ImportError as e:  # pragma: no cover - depends on optional package.
        raise RuntimeError(
            "DSPy is not installed. Install dependencies with `pip install -r requirements.txt` "
            "or `pip install dspy`."
        ) from e
    return dspy


def _make_stroke_program(dspy: Any) -> Any:
    class StrokeTerritorySignature(dspy.Signature):
        """Label acute/recent stroke territory from radiology text.

        /no_think

        Return final answer only. Do not include hidden reasoning, scratchpad,
        chain-of-thought, analysis, self-correction, or explanation.

        Return only allowed labels. Use NONE only when the report has no
        qualifying acute/recent/current territory evidence. Prefer the smaller
        label set when evidence is ambiguous, and never include NONE with a real
        territory label.
        """

        case_id: str = dspy.InputField(desc="Case identifier")
        modality: str = dspy.InputField(desc="CT, CTA, CTP, or COMBINED")
        allowed_labels: str = dspy.InputField(desc="Comma-separated allowed output labels")
        report: str = dspy.InputField(desc="Relevant radiology report text")
        ct_labels: str = dspy.InputField(desc="Preliminary CT labels, or NONE")
        cta_labels: str = dspy.InputField(desc="Preliminary CTA labels, or NONE")
        ctp_labels: str = dspy.InputField(desc="Preliminary CTP labels, or NONE")
        labels: list[str] = dspy.OutputField(
            desc="A JSON-style list of labels chosen only from allowed_labels. Return labels only; no reasoning."
        )

    class StrokeTerritoryProgram(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            predictor_class = dspy.ChainOfThought if bool(getattr(cfg, "DSPY_USE_CHAIN_OF_THOUGHT", False)) else dspy.Predict
            self.predict = predictor_class(StrokeTerritorySignature)

        def forward(
            self,
            case_id: str,
            modality: str,
            allowed_labels: str,
            report: str,
            ct_labels: str = "NONE",
            cta_labels: str = "NONE",
            ctp_labels: str = "NONE",
        ) -> Any:
            pred = self.predict(
                case_id=case_id,
                modality=modality,
                allowed_labels=allowed_labels,
                report=report,
                ct_labels=ct_labels,
                cta_labels=cta_labels,
                ctp_labels=ctp_labels,
            )
            pred.labels = _normalize_label_value(getattr(pred, "labels", []))
            # Keep this attribute empty for compatibility with old helper code.
            pred.reasoning = ""
            return pred

    return StrokeTerritoryProgram()


def _dspy_example(dspy: Any, **kwargs: Any) -> Any:
    return dspy.Example(**kwargs).with_inputs(
        "case_id",
        "modality",
        "allowed_labels",
        "report",
        "ct_labels",
        "cta_labels",
        "ctp_labels",
    )


def _build_examples_for_modality(
    dspy: Any,
    input_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    modality: str,
) -> List[Any]:
    modality = modality.upper()
    input_case_col = _find_column(input_df, CASE_ID_COLUMN_CANDIDATES)
    gt_case_col = _find_column(gt_df, CASE_ID_COLUMN_CANDIDATES)
    if input_case_col is None:
        raise ValueError(f"Could not find case ID column in input file. Columns: {list(input_df.columns)}")
    if gt_case_col is None:
        raise ValueError(f"Could not find case ID column in ground-truth file. Columns: {list(gt_df.columns)}")

    gt_label_col = _find_column(gt_df, GROUND_TRUTH_COLUMNS[modality])
    if gt_label_col is None:
        raise ValueError(f"Could not find {modality} ground-truth column. Columns: {list(gt_df.columns)}")

    gt_by_case: Dict[str, pd.Series] = {}
    for _, row in gt_df.iterrows():
        case_id = str(row.get(gt_case_col, "")).strip()
        if case_id and case_id not in gt_by_case:
            gt_by_case[case_id] = row

    report_cols = {name: _find_column(input_df, candidates) for name, candidates in INPUT_REPORT_COLUMNS.items()}
    examples: List[Any] = []
    allowed_labels = ", ".join(ALLOWED_LABELS)

    for _, row in input_df.iterrows():
        case_id = str(row.get(input_case_col, "")).strip()
        if not case_id or case_id not in gt_by_case:
            continue

        gt_row = gt_by_case[case_id]
        labels = _normalize_label_value(gt_row.get(gt_label_col))

        if modality in {"CT", "CTA", "CTP"}:
            report_col = report_cols.get(modality)
            if report_col is None:
                raise ValueError(f"Could not find {modality} report column. Columns: {list(input_df.columns)}")
            report = clean_report(row.get(report_col, ""))
            if not report and labels == ["NONE"]:
                continue
            ct_labels = cta_labels = ctp_labels = "NONE"
        else:
            pieces = []
            for name in ("CT", "CTA", "CTP", "MRI"):
                col = report_cols.get(name)
                value = clean_report(row.get(col, "")) if col is not None else ""
                if value:
                    pieces.append(f"{name} Report:\n{value}")
            report = "\n\n".join(pieces)
            if not report and labels == ["NONE"]:
                continue
            ct_col = _find_column(gt_df, GROUND_TRUTH_COLUMNS["CT"])
            cta_col = _find_column(gt_df, GROUND_TRUTH_COLUMNS["CTA"])
            ctp_col = _find_column(gt_df, GROUND_TRUTH_COLUMNS["CTP"])
            ct_labels = _labels_to_text(_normalize_label_value(gt_row.get(ct_col))) if ct_col else "NONE"
            cta_labels = _labels_to_text(_normalize_label_value(gt_row.get(cta_col))) if cta_col else "NONE"
            ctp_labels = _labels_to_text(_normalize_label_value(gt_row.get(ctp_col))) if ctp_col else "NONE"

        examples.append(
            _dspy_example(
                dspy,
                case_id=case_id,
                modality=modality,
                allowed_labels=allowed_labels,
                report=report,
                ct_labels=ct_labels,
                cta_labels=cta_labels,
                ctp_labels=ctp_labels,
                labels=labels
            )
        )

    return examples


def _split_examples(
    examples: List[Any],
    train_limit: int,
    val_limit: int,
    seed: int,
) -> Tuple[List[Any], List[Any]]:
    shuffled = list(examples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    train_limit = max(1, int(train_limit))
    val_limit = max(0, int(val_limit))

    train = shuffled[:train_limit]
    val = shuffled[train_limit : train_limit + val_limit]
    if not val and len(shuffled) > len(train):
        val = shuffled[len(train) :]
    return train, val


def _configure_dspy_lm(dspy: Any, model: str, lm_spec: str | None = None, api_base: str | None = None) -> Any:
    lm_name = lm_spec or getattr(cfg, "DSPY_LM", None) or f"ollama_chat/{model}"
    base = api_base or getattr(cfg, "DSPY_API_BASE", "http://localhost:11434")
    kwargs: Dict[str, Any] = {
        "temperature": 0,
        "max_tokens": int(getattr(cfg, "NUM_PREDICT", 1670)),
    }
    if base:
        kwargs["api_base"] = base

    # Reasoning-capable local models sometimes return huge reasoning_content and
    # an empty final answer. These flags are harmless for providers that ignore
    # them and help Ollama/Qwen-style models stay in final-answer mode.
    if bool(getattr(cfg, "DSPY_DISABLE_THINKING", True)):
        kwargs["think"] = False
        kwargs["thinking"] = False
        kwargs["extra_body"] = {"think": False}

    lm = dspy.LM(lm_name, **kwargs)
    dspy.configure(lm=lm)
    return lm


def _make_optimizer(dspy: Any, optimizer_name: str, auto: str, log_dir: Path) -> Any:
    optimizer_name = optimizer_name.strip().lower()
    if optimizer_name == "miprov2":
        return dspy.MIPROv2(
            metric=_metric_exact_labels,
            auto=auto,
            max_labeled_demos=int(getattr(cfg, "DSPY_MAX_LABELED_DEMOS", 4)),
            max_bootstrapped_demos=int(getattr(cfg, "DSPY_MAX_BOOTSTRAPPED_DEMOS", 2)),
            num_threads=int(getattr(cfg, "DSPY_NUM_THREADS", 1)),
            seed=int(getattr(cfg, "DSPY_SEED", 9)),
            log_dir=str(log_dir),
            verbose=True,
        )
    if optimizer_name == "bootstrapfewshot":
        return dspy.BootstrapFewShot(
            metric=_metric_exact_labels,
            max_labeled_demos=int(getattr(cfg, "DSPY_MAX_LABELED_DEMOS", 4)),
            max_bootstrapped_demos=int(getattr(cfg, "DSPY_MAX_BOOTSTRAPPED_DEMOS", 2)),
        )
    if optimizer_name == "labeledfewshot":
        return dspy.LabeledFewShot(k=int(getattr(cfg, "DSPY_MAX_LABELED_DEMOS", 4)))
    raise ValueError("DSPY_OPTIMIZER must be MIPROv2, BootstrapFewShot, or LabeledFewShot")


def _iter_predictor_like_objects(obj: Any) -> Iterable[Any]:
    seen: set[int] = set()
    queue: List[Any] = [obj]
    while queue:
        current = queue.pop(0)
        ident = id(current)
        if ident in seen:
            continue
        seen.add(ident)

        if hasattr(current, "signature") or hasattr(current, "demos"):
            yield current

        named_predictors = getattr(current, "named_predictors", None)
        if callable(named_predictors):
            try:
                for _, predictor in named_predictors():
                    queue.append(predictor)
            except Exception:
                pass

        try:
            values = vars(current).values()
        except Exception:
            values = []
        for value in values:
            if id(value) not in seen and not isinstance(value, (str, bytes, int, float, bool, type(None))):
                if hasattr(value, "signature") or hasattr(value, "demos") or hasattr(value, "named_predictors"):
                    queue.append(value)


def _extract_instructions(program: Any) -> str:
    chunks: List[str] = []
    max_chars = int(getattr(cfg, "DSPY_EXTRACTED_INSTRUCTIONS_MAX_CHARS", 1200))
    suspicious_markers = (
        "here's a thinking process",
        "reasoning_content",
        "self-correction",
        "output generation",
        "adapterparseerror",
        "expected to find output fields",
    )
    for obj in _iter_predictor_like_objects(program):
        sig = getattr(obj, "signature", None)
        candidates = []
        if sig is not None:
            candidates.extend([
                getattr(sig, "instructions", None),
                getattr(sig, "__doc__", None),
                str(sig) if sig is not None else None,
            ])
        candidates.extend([
            getattr(obj, "instructions", None),
            getattr(obj, "__doc__", None),
        ])
        for candidate in candidates:
            text = str(candidate or "").strip()
            lower_text = text.lower()
            if not text or text in chunks:
                continue
            if any(marker in lower_text for marker in suspicious_markers):
                continue
            if len(text) > max_chars:
                text = text[: max_chars - 80].rstrip() + "\n[TRUNCATED extracted instruction to avoid prompt bloat]"
            chunks.append(text)
    return "\n\n".join(chunks[:3]).strip()


def _example_to_dict(example: Any) -> Dict[str, Any]:
    return {
        "case_id": str(getattr(example, "case_id", "")),
        "modality": str(getattr(example, "modality", "")),
        "report_excerpt": _compact_report(str(getattr(example, "report", "")), limit=600),
        "ct_labels": str(getattr(example, "ct_labels", "NONE")),
        "cta_labels": str(getattr(example, "cta_labels", "NONE")),
        "ctp_labels": str(getattr(example, "ctp_labels", "NONE")),
        "labels": _normalize_label_value(getattr(example, "labels", [])),
    }


def _extract_selected_examples(program: Any, fallback_train: Sequence[Any], limit: int = 4) -> List[Dict[str, Any]]:
    raw_demos: List[Any] = []
    for obj in _iter_predictor_like_objects(program):
        demos = getattr(obj, "demos", None)
        if demos:
            raw_demos.extend(list(demos))

    selected: List[Dict[str, Any]] = []
    for demo in raw_demos:
        try:
            selected.append(_example_to_dict(demo))
        except Exception:
            continue
        if len(selected) >= limit:
            break

    if not selected:
        for example in fallback_train[:limit]:
            selected.append(_example_to_dict(example))
    return selected


def _build_guidance_text(modality: str, instructions: str, selected_examples: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    if instructions:
        lines.append("Optimized instruction summary:")
        lines.append(instructions.strip())

    if selected_examples:
        lines.append("Selected DSPy few-shot examples to follow:")
        for index, example in enumerate(selected_examples, 1):
            lines.append(
                f"{index}. Report excerpt: {example['report_excerpt']}\n"
                f"   Expected {modality} labels: {_labels_to_text(example['labels'])}"
            )

    lines.append(
        "Use these optimized examples as calibration, but the current report text "
        "and the hard labeling rules above remain authoritative. Do not include "
        "reasoning or analysis in the final answer unless config.py enables it."
    )
    guidance = "\n".join(lines).strip()
    max_chars = int(getattr(cfg, "DSPY_GUIDANCE_MAX_CHARS", 2500))
    if max_chars > 0 and len(guidance) > max_chars:
        guidance = guidance[: max_chars - 80].rstrip() + "\n[TRUNCATED DSPy guidance to avoid prompt bloat]"
    return guidance


def optimize_one_modality(
    dspy: Any,
    input_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    modality: str,
    output_dir: Path,
    optimizer_name: str,
    auto: str,
    train_limit: int,
    val_limit: int,
    seed: int,
) -> OptimizationResult:
    modality = modality.upper()
    notes: List[str] = []
    all_examples = _build_examples_for_modality(dspy, input_df, gt_df, modality)
    if len(all_examples) < 2:
        raise ValueError(f"Not enough matched examples for {modality}; found {len(all_examples)}")

    train, val = _split_examples(all_examples, train_limit=train_limit, val_limit=val_limit, seed=seed)
    student = _make_stroke_program(dspy)
    baseline_score = _score_program(student, val) if val else None

    optimizer = _make_optimizer(dspy, optimizer_name=optimizer_name, auto=auto, log_dir=output_dir / "logs" / modality)
    compile_kwargs: Dict[str, Any] = {"trainset": train}
    if val:
        compile_kwargs["valset"] = val

    if optimizer_name.strip().lower() == "miprov2":
        compile_kwargs.update({
            "minibatch": True,
            "provide_traceback": True,
        })

    compiled = optimizer.compile(student, **compile_kwargs)
    optimized_score = _score_program(compiled, val) if val else None

    program_path = output_dir / f"{modality.lower()}_dspy_program.json"
    try:
        compiled.save(str(program_path), save_program=False)
    except Exception as e:
        notes.append(f"Could not save DSPy state JSON for {modality}: {e}")
        program_path = None  # type: ignore[assignment]

    instructions = _extract_instructions(compiled)
    selected_examples = _extract_selected_examples(compiled, train, limit=int(getattr(cfg, "DSPY_MAX_LABELED_DEMOS", 4)))
    guidance_text = _build_guidance_text(modality, instructions, selected_examples)

    return OptimizationResult(
        modality=modality,
        train_count=len(train),
        val_count=len(val),
        baseline_score=baseline_score,
        optimized_score=optimized_score,
        program_path=str(program_path) if program_path else None,
        guidance_text=guidance_text,
        instructions=instructions,
        selected_examples=selected_examples,
        notes=notes,
    )


def _normalize_modalities_arg(value: Any) -> List[str]:
    """Accept config.DSPY_MODALITIES as a list/tuple or comma-separated string."""
    if value is None:
        value = getattr(cfg, "DSPY_MODALITIES", ["CT", "CTA", "CTP", "COMBINED"])
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    return items or ["CT", "CTA", "CTP", "COMBINED"]


def optimize_prompts(
    input_file: str | Path | None = None,
    ground_truth_file: str | Path | None = None,
    output_file: str | Path | None = None,
    model: str | None = None,
    modalities: Sequence[str] | str | None = None,
    optimizer_name: str | None = None,
    auto: str | None = None,
    train_limit: int | None = None,
    val_limit: int | None = None,
    seed: int | None = None,
    lm_spec: str | None = None,
    api_base: str | None = None,
) -> Dict[str, Any]:
    dspy = _load_dspy()
    input_file = input_file or getattr(cfg, "INPUT_REPORTS_FILE", None)
    ground_truth_file = ground_truth_file or getattr(cfg, "GROUND_TRUTH_FILE", None)
    if input_file is None or str(input_file).strip() == "":
        raise ValueError("No input file configured. Set INPUT_REPORTS_FILE in config.py or pass --input.")
    if ground_truth_file is None or str(ground_truth_file).strip() == "":
        raise ValueError("No ground-truth file configured. Set GROUND_TRUTH_FILE in config.py or pass --ground-truth.")

    model = model or getattr(cfg, "MODEL_NAME", "")
    modalities = _normalize_modalities_arg(modalities)
    optimizer_name = optimizer_name or getattr(cfg, "DSPY_OPTIMIZER", "MIPROv2")
    auto = auto or getattr(cfg, "DSPY_AUTO", "light")
    train_limit = int(train_limit if train_limit is not None else getattr(cfg, "DSPY_TRAIN_LIMIT", 60))
    val_limit = int(val_limit if val_limit is not None else getattr(cfg, "DSPY_VAL_LIMIT", 30))
    seed = int(seed if seed is not None else getattr(cfg, "DSPY_SEED", 9))

    default_output = getattr(cfg, "DSPY_OPTIMIZED_PROMPTS_OUTPUT_FILE", getattr(cfg, "OPTIMIZED_PROMPTS_FILE", "optimized_prompts.json"))
    output_path = Path(output_file or default_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_path.parent / f"{output_path.stem}_dspy_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_dspy_lm(dspy, model=model, lm_spec=lm_spec, api_base=api_base)

    input_df = _read_table(input_file)
    gt_df = _read_table(ground_truth_file)

    payload: Dict[str, Any] = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "optimizer": optimizer_name,
        "auto": auto,
        "model": model,
        "train_limit": train_limit,
        "val_limit": val_limit,
        "seed": seed,
        "input_file": str(input_file),
        "ground_truth_file": str(ground_truth_file),
        "modalities": {},
    }

    for modality in modalities:
        normalized_modality = str(modality).strip().upper()
        if normalized_modality == "COMBINED_GT":
            normalized_modality = "COMBINED"
        if normalized_modality not in {"CT", "CTA", "CTP", "COMBINED"}:
            continue

        print(f"\n===== DSPy optimizing {normalized_modality} prompt guidance =====")
        try:
            result = optimize_one_modality(
                dspy=dspy,
                input_df=input_df,
                gt_df=gt_df,
                modality=normalized_modality,
                output_dir=output_dir,
                optimizer_name=optimizer_name,
                auto=auto,
                train_limit=train_limit,
                val_limit=val_limit,
                seed=seed,
            )
            payload["modalities"][normalized_modality] = {
                "enabled": True,
                "train_count": result.train_count,
                "val_count": result.val_count,
                "baseline_score": result.baseline_score,
                "optimized_score": result.optimized_score,
                "program_path": result.program_path,
                "instructions": result.instructions,
                "few_shot_examples": result.selected_examples,
                "guidance_text": result.guidance_text,
                "notes": result.notes,
            }
            print(
                f"Saved {normalized_modality} optimized guidance "
                f"(baseline={result.baseline_score}, optimized={result.optimized_score})"
            )
        except Exception as e:
            payload["modalities"][normalized_modality] = {
                "enabled": False,
                "error": repr(e),
                "guidance_text": "",
            }
            print(f"WARNING: {normalized_modality} optimization failed: {e}")

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(f"\nWrote DSPy optimized prompt guidance to: {output_path}")
    print(f"DSPy artifacts directory: {output_dir}")
    return payload


def optimize_prompts_from_args(args: Any) -> Dict[str, Any]:
    """Run optimization using config.py defaults plus any CLI overrides."""
    return optimize_prompts(
        input_file=getattr(args, "input", None),
        ground_truth_file=getattr(args, "ground_truth", None),
        output_file=getattr(args, "optimized_prompts_output", None),
        model=getattr(args, "model", None),
        modalities=getattr(args, "dspy_modalities", None),
        optimizer_name=getattr(args, "dspy_optimizer", None),
        auto=getattr(args, "dspy_auto", None),
        train_limit=getattr(args, "dspy_train_limit", None),
        val_limit=getattr(args, "dspy_val_limit", None),
        seed=getattr(args, "dspy_seed", None),
        lm_spec=getattr(args, "dspy_lm", None),
        api_base=getattr(args, "dspy_api_base", None),
    )
