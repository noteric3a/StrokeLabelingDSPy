"""
dspy_train.py

DSPy optimization script for CT / CTA / CTP stroke-territory labelers.

Safety and debugging design:
- The model-visible DSPy Example objects contain report_text only.
- Ground-truth labels are kept in Python-only maps and used only by metrics/logs.
- By default MIPRO runs without labeled or bootstrapped demos to prevent answer leakage.
- Prompt candidates are evaluated on the 28-case train split by default so the optimizer
  does not make decisions from only the 6-case dev split.
- CTA rule text is passed as a fixed input, so MIPRO can optimize guidance
  without deleting required CTA vessel/label rules.
- By default, training warm-starts from the saved optimized program when it exists.
- In --loop mode, each accepted candidate becomes the next iteration's starting prompt.
- Rejected candidates do not overwrite the saved best program.
- Candidate acceptance is based on the configured optimizer score, not just raw prompt length.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import dspy
import pandas as pd

import config as cfg
from dspy_programs import configure_dspy, CTLabeler, CTALabeler, CTPLabeler
from utils import normalize_labels

# Hide noisy internal DSPy/MIPRO warnings such as:
# "Input contains fields not in signature. These fields will be ignored..."
# Those warnings come from MIPRO's internal instruction-proposal modules, not from
# the stroke-labeling predictor itself.
logging.getLogger("dspy.predict.predict").setLevel(logging.ERROR)


def disable_training_caches() -> None:
    """Ensure DSPy training/evaluation does not reuse cached LM completions.

    The project ProcessingCache in cache.py is for main labeling runs and is not
    imported here.  This function only handles DSPy's optional LM cache layer.
    """
    configure_cache = getattr(dspy, "configure_cache", None)
    if callable(configure_cache):
        try:
            configure_cache(enable_memory_cache=False, enable_disk_cache=False)
        except TypeError:
            try:
                configure_cache(enable_disk_cache=False, enable_memory_cache=False)
            except Exception:
                pass
        except Exception:
            pass

    try:
        dspy.settings.configure(cache=False)
    except Exception:
        pass




# =============================================================================
# Python-only answer-key store
# =============================================================================
# These maps let the metric/evaluator use the answer key WITHOUT storing gold
# labels inside dspy.Example objects.  Only report_text is passed to the model.

_GOLD_BY_REPORT_KEY: Dict[str, str] = {}
_RAW_GOLD_BY_REPORT_KEY: Dict[str, Any] = {}
_CASE_ID_BY_REPORT_KEY: Dict[str, str] = {}


def _report_key(report_text: Any) -> str:
    """Stable key for matching an Example's report text to Python-only gold labels."""
    return " ".join(str(report_text or "").split())


def _example_value(example: Any, field: str, default: Any = "") -> Any:
    """Read a DSPy Example field without accidentally returning bound methods."""
    for reader in (
        lambda obj: obj[field],
        lambda obj: obj.get(field),
        lambda obj: getattr(obj, field),
    ):
        try:
            value = reader(example)
        except Exception:
            continue
        if not callable(value):
            return value
    return default


def _prediction_value(pred: Any, field: str, default: Any = "") -> Any:
    """Read a DSPy Prediction field safely.

    DSPy Prediction/Example classes may expose methods named labels/reasoning.
    Using getattr(pred, "labels") can therefore return a bound method instead of
    the predicted value.  Always prefer mapping-style access.
    """
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


def _report_text_for_example(example: Any) -> str:
    return str(_example_value(example, "report_text", "") or "")


def _case_id_for_example(example: Any) -> str:
    return _CASE_ID_BY_REPORT_KEY.get(_report_key(_report_text_for_example(example)), "")


def _raw_gold_for_example(example: Any) -> Any:
    return _RAW_GOLD_BY_REPORT_KEY.get(_report_key(_report_text_for_example(example)), "NONE")


def _gold_labels_for_example(example: Any) -> Set[str]:
    return normalize_gt(_GOLD_BY_REPORT_KEY.get(_report_key(_report_text_for_example(example)), "NONE"))


# =============================================================================
# Normalization helpers
# =============================================================================


def normalize_gt(value: Any) -> Set[str]:
    """Normalize a ground-truth label cell into a set of uppercase labels."""
    if pd.isna(value):
        return {"NONE"}
    text = str(value).strip()
    if not text or text.lower() in {"negative", "normal", "none"}:
        return {"NONE"}
    labels = {part.strip().upper() for part in text.split(",") if part.strip()}
    if not labels:
        return {"NONE"}
    if "NONE" in labels and len(labels) > 1:
        labels.remove("NONE")
    return labels


def normalize_pred(value: Any) -> Set[str]:
    labels = set(normalize_labels(value))
    if "NONE" in labels and len(labels) > 1:
        labels.remove("NONE")
    return labels


def normalize_case_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _normalize_col_name(name: Any) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _find_column(df: pd.DataFrame, candidates: Sequence[str], *, purpose: str) -> str:
    existing = list(df.columns)
    for col in candidates:
        if col in df.columns:
            return col
    normalized_existing = {_normalize_col_name(col): col for col in existing}
    for col in candidates:
        key = _normalize_col_name(col)
        if key in normalized_existing:
            return normalized_existing[key]
    raise ValueError(
        f"Could not find {purpose} column.\n"
        f"Tried: {list(candidates)}\n"
        f"Available columns: {existing}"
    )


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _report_column_candidates(report_type: str) -> List[str]:
    report_type = report_type.upper()
    training = cfg.TRAINING_COLUMN_CANDIDATES.get(report_type, {}).get("report", [])
    report_like = cfg.REPORT_COLUMN_CANDIDATES.get(f"{report_type}_Report", [])
    return _dedupe_keep_order([
        *training,
        *report_like,
        report_type,
        f"{report_type} Report",
        f"{report_type}_Report",
        f"{report_type} text",
        f"{report_type}_Text",
    ])


def _gt_column_candidates(report_type: str) -> List[str]:
    report_type = report_type.upper()
    training = cfg.TRAINING_COLUMN_CANDIDATES.get(report_type, {}).get("ground_truth", [])
    modality = []
    if hasattr(cfg, "MODALITY_COLUMN_CANDIDATES"):
        modality = list(cfg.MODALITY_COLUMN_CANDIDATES.get(report_type, {}).get("ground_truth", []))
    return _dedupe_keep_order([
        *training,
        *modality,
        f"{report_type} GT",
        f"{report_type}_GT",
        f"{report_type} Ground Truth",
        f"{report_type}_Ground_Truth",
        report_type,
    ])


def _signature_instructions(report_type: str) -> str:
    """Return the optimizable DSPy signature instruction for a modality."""
    report_type = report_type.upper()
    if report_type == "CT":
        return getattr(cfg, "CT_SIGNATURE_INSTRUCTIONS", "")
    if report_type == "CTA":
        return getattr(cfg, "CTA_SIGNATURE_INSTRUCTIONS", "")
    if report_type == "CTP":
        return getattr(cfg, "CTP_SIGNATURE_INSTRUCTIONS", "")
    return ""


def _fixed_rules_for_report_type(report_type: str) -> str:
    """Return non-optimizable rules supplied as normal model inputs."""
    report_type = report_type.upper()
    if report_type == "CTA":
        return getattr(cfg, "CTA_FIXED_RULES", "")
    return ""


def effective_prompt_text(report_type: str, signature_instructions: Optional[str] = None) -> str:
    """Human-readable prompt text actually seen by the model.

    For CTA, MIPRO only rewrites the short signature instruction. The CTA rule
    block is supplied by CTALabeler.forward as `cta_rules`, so prompt logs should
    show both pieces together. This prevents a short optimized signature from
    looking like the whole CTA prompt.
    """
    report_type = report_type.upper()
    instruction = _signature_instructions(report_type) if signature_instructions is None else str(signature_instructions or "")
    fixed_rules = _fixed_rules_for_report_type(report_type)
    if fixed_rules:
        return (
            f"{instruction.strip()}\n\n"
            f"Fixed {report_type} rules supplied as a non-optimizable input:\n"
            f"{fixed_rules.strip()}"
        ).strip()
    return instruction.strip()


# =============================================================================
# DSPy metric
# =============================================================================


def _raw_label_tokens(value: Any) -> List[str]:
    """Return the raw tokens the model placed in the labels field."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        raw_items = None
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    raw_items = list(parsed)
            except Exception:
                raw_items = None
        if raw_items is None:
            raw_items = re.split(r"[,;\n/]+", text)

    tokens: List[str] = []
    for item in raw_items:
        token = str(item).strip().upper()
        token = token.strip("\"'[](){}")
        token = re.sub(r"\s+", " ", token)
        token = re.sub(r"^(LABELS?|FINAL LABELS?|OUTPUT)\s*:\s*", "", token).strip()
        if token:
            tokens.append(token)
    return tokens


def raw_label_format_ok(value: Any) -> bool:
    """Require the labels field to contain only allowed labels/aliases.

    The old metric normalized away invalid outputs such as "MCA, NONE". That
    made unusable test outputs look correct when the normalized result matched
    gold. This check keeps optimization honest.
    """
    tokens = _raw_label_tokens(value)
    if not tokens:
        return False

    normalized_tokens: List[str] = []
    allowed = set(getattr(cfg, "ALLOWED_LABELS", []))
    aliases = getattr(cfg, "LABEL_ALIASES", {})
    for token in tokens:
        label = aliases.get(token, token)
        compact = label.replace(" ", "")
        if compact in allowed:
            label = compact
        if label not in allowed:
            return False
        normalized_tokens.append(label)

    return not ("NONE" in normalized_tokens and len(set(normalized_tokens)) > 1)


def exact_match_metric(example, pred, trace=None) -> float:
    """Return 1.0 only when predicted labels exactly match hidden gold labels."""
    gold = _gold_labels_for_example(example)
    predicted = normalize_pred(_prediction_value(pred, "labels", "NONE"))
    return 1.0 if gold == predicted else 0.0


def strict_exact_match_metric(example, pred, trace=None) -> float:
    """Exact match plus strict validation of the raw labels field."""
    raw_labels = _prediction_value(pred, "labels", "NONE")
    if not raw_label_format_ok(raw_labels):
        return 0.0
    return exact_match_metric(example, pred, trace=trace)


def label_f1_metric(example, pred, trace=None) -> float:
    """Dense label-overlap metric with strict raw-label validation."""
    raw_labels = _prediction_value(pred, "labels", "NONE")
    if not raw_label_format_ok(raw_labels):
        return 0.0

    gold = _gold_labels_for_example(example)
    predicted = normalize_pred(raw_labels)
    if not predicted:
        return 0.0
    if gold == predicted:
        return 1.0

    overlap = len(gold & predicted)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def optimizer_metric(example, pred, trace=None) -> float:
    metric_name = str(getattr(cfg, "DSPY_OPTIMIZER_METRIC", "strict_exact") or "strict_exact").lower().strip()
    if metric_name in {"exact", "exact_match", "accuracy"}:
        return exact_match_metric(example, pred, trace=trace)
    if metric_name in {"strict", "strict_exact", "strict_exact_match"}:
        return strict_exact_match_metric(example, pred, trace=trace)
    if metric_name in {"label_f1", "f1", "overlap"}:
        return label_f1_metric(example, pred, trace=trace)
    raise ValueError(f"Unknown DSPY_OPTIMIZER_METRIC: {metric_name}")


# =============================================================================
# DSPy optimizer helpers
# =============================================================================


def _mipro_v2_class():
    try:
        return dspy.MIPROv2
    except ImportError as exc:
        raise ImportError(
            "MIPROv2 requires optuna. Install it with: pip install \"dspy[optuna]\" "
            "or run this script with --baseline-only while debugging predictions."
        ) from exc
    except AttributeError:
        try:
            from dspy.teleprompt import MIPROv2
            return MIPROv2
        except ImportError as exc:
            raise ImportError(
                "MIPROv2 requires optuna. Install it with: pip install \"dspy[optuna]\" "
                "or run this script with --baseline-only while debugging predictions."
            ) from exc


def _optional_prompt_model():
    """Return a separate prompt-proposal LM when configured."""
    model_name = getattr(cfg, "DSPY_PROMPT_MODEL", None)
    if not model_name:
        return None

    lm_kwargs = dict(
        api_base=getattr(cfg, "DSPY_PROMPT_MODEL_API_BASE", cfg.DSPY_API_BASE),
        temperature=getattr(cfg, "DSPY_TEMPERATURE", 0.2),
        max_tokens=int(getattr(cfg, "DSPY_PROMPT_MODEL_MAX_TOKENS", 4000)),
        think=False,
    )
    if bool(getattr(cfg, "DSPY_DISABLE_CACHE", True)):
        lm_kwargs["cache"] = False
    try:
        return dspy.LM(model_name, **lm_kwargs)
    except TypeError:
        lm_kwargs.pop("cache", None)
        lm_kwargs.pop("think", None)
        return dspy.LM(model_name, **lm_kwargs)


def get_optimizer(metric, *, run_dir: Optional[Path] = None):
    MIPROv2 = _mipro_v2_class()
    kwargs = {
        "metric": metric,
        "auto": getattr(cfg, "DSPY_MIPRO_AUTO", "light"),
        "init_temperature": float(getattr(cfg, "DSPY_MIPRO_INIT_TEMPERATURE", 0.7)),
        "verbose": bool(getattr(cfg, "DSPY_MIPRO_VERBOSE", False)),
        "track_stats": True,
    }
    prompt_model = _optional_prompt_model()
    if prompt_model is not None:
        kwargs["prompt_model"] = prompt_model
    if run_dir is not None:
        kwargs["log_dir"] = str(run_dir / "mipro_logs")

    try:
        return MIPROv2(**kwargs)
    except TypeError:
        # Older DSPy versions may not accept every convenience kwarg.
        for optional_key in ("track_stats", "log_dir", "verbose", "init_temperature"):
            kwargs.pop(optional_key, None)
        return MIPROv2(**kwargs)


def choose_optimizer_valset(trainset, devset, testset, source: str):
    source = (source or "train").lower().strip()
    if source == "train":
        return trainset
    if source == "dev":
        return devset
    if source == "train_dev":
        return list(trainset) + list(devset)
    if source == "all":
        return list(trainset) + list(devset) + list(testset)
    raise ValueError("--mipro-valset-source must be one of: train, dev, train_dev, all")


def compile_with_no_answer_leak(optimizer, program, trainset, optimizer_valset, *, allow_demos: bool = False):
    """Compile while keeping answer-key labels out of all model prompts."""
    compile_kwargs = {
        "trainset": trainset,
        "valset": optimizer_valset,
        "program_aware_proposer": bool(getattr(cfg, "DSPY_MIPRO_PROGRAM_AWARE_PROPOSER", True)),
        "data_aware_proposer": bool(getattr(cfg, "DSPY_MIPRO_DATA_AWARE_PROPOSER", True)),
        "tip_aware_proposer": bool(getattr(cfg, "DSPY_MIPRO_TIP_AWARE_PROPOSER", False)),
        "fewshot_aware_proposer": bool(getattr(cfg, "DSPY_MIPRO_FEWSHOT_AWARE_PROPOSER", False)),
        "view_data_batch_size": int(getattr(cfg, "DSPY_MIPRO_VIEW_DATA_BATCH_SIZE", 20)),
    }
    if not allow_demos and bool(getattr(cfg, "DSPY_INSTRUCTION_ONLY_OPTIMIZATION", True)):
        compile_kwargs.update({
            "max_bootstrapped_demos": int(getattr(cfg, "DSPY_MAX_BOOTSTRAPPED_DEMOS", 0)),
            "max_labeled_demos": int(getattr(cfg, "DSPY_MAX_LABELED_DEMOS", 0)),
        })
    try:
        return optimizer.compile(program.deepcopy(), **compile_kwargs)
    except TypeError as exc:
        message = str(exc)
        if not allow_demos and any(k in message for k in ("max_bootstrapped_demos", "max_labeled_demos")):
            raise TypeError(
                "This DSPy/MIPROv2 version did not accept max_bootstrapped_demos=0 "
                "and max_labeled_demos=0. Stopping instead of falling back because "
                "fallback could create few-shot demos. Upgrade DSPy or rerun with "
                "--allow-demos if you accept model-generated demos."
            ) from exc
        proposer_keys = {
            "program_aware_proposer",
            "data_aware_proposer",
            "tip_aware_proposer",
            "fewshot_aware_proposer",
            "view_data_batch_size",
        }
        if any(key in message for key in proposer_keys):
            for key in proposer_keys:
                compile_kwargs.pop(key, None)
            return optimizer.compile(program.deepcopy(), **compile_kwargs)
        raise


# =============================================================================
# Loading examples from separate reports + ground-truth files
# =============================================================================


def load_examples(
    reports_file: str,
    ground_truth_file: str,
    report_type: str,
    max_cases: Optional[int] = None,
) -> List[dspy.Example]:
    """Load report-only DSPy Examples and store answer key in Python-only maps."""
    report_type = report_type.upper()
    if report_type not in {"CT", "CTA", "CTP"}:
        raise ValueError("report_type must be CT, CTA, or CTP")

    reports_path = Path(reports_file)
    gt_path = Path(ground_truth_file)
    if not reports_path.exists():
        raise FileNotFoundError(f"Reports file not found: {reports_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    reports_df = pd.read_excel(reports_path)
    gt_df = pd.read_excel(gt_path)

    reports_case_col = _find_column(reports_df, cfg.CASE_ID_COLUMNS, purpose="case ID in reports file")
    gt_case_col = _find_column(gt_df, cfg.CASE_ID_COLUMNS, purpose="case ID in ground-truth file")
    report_col = _find_column(reports_df, _report_column_candidates(report_type), purpose=f"{report_type} report text in reports file")
    gt_col = _find_column(gt_df, _gt_column_candidates(report_type), purpose=f"{report_type} ground-truth labels in ground-truth file")

    print("\nResolved training columns")
    print(f"  Reports file:      {reports_path}")
    print(f"  Ground-truth file: {gt_path}")
    print(f"  Reports case col:  {reports_case_col}")
    print(f"  Reports text col:  {report_col}")
    print(f"  GT case col:       {gt_case_col}")
    print(f"  GT label col:      {gt_col}")

    gt_by_case: Dict[str, Any] = {}
    for _, row in gt_df.iterrows():
        case_key = normalize_case_id(row.get(gt_case_col))
        if case_key:
            gt_by_case[case_key] = row.get(gt_col)

    _GOLD_BY_REPORT_KEY.clear()
    _RAW_GOLD_BY_REPORT_KEY.clear()
    _CASE_ID_BY_REPORT_KEY.clear()

    examples: List[dspy.Example] = []
    missing_gt = 0
    missing_report = 0

    for _, row in reports_df.iterrows():
        case_id = normalize_case_id(row.get(reports_case_col))
        report_text = "" if pd.isna(row.get(report_col)) else str(row.get(report_col)).strip()
        if not case_id:
            continue
        if not report_text:
            missing_report += 1
            continue
        if case_id not in gt_by_case:
            missing_gt += 1
            continue

        raw_gold = gt_by_case[case_id]
        normalized_gold = ", ".join(sorted(normalize_gt(raw_gold)))
        key = _report_key(report_text)
        _GOLD_BY_REPORT_KEY[key] = normalized_gold
        _RAW_GOLD_BY_REPORT_KEY[key] = raw_gold
        _CASE_ID_BY_REPORT_KEY[key] = case_id

        # Critical: the DSPy Example contains only the model input.
        # It does NOT contain labels, reasoning, or case_id.
        examples.append(dspy.Example(report_text=report_text).with_inputs("report_text"))

    if max_cases:
        examples = examples[:max_cases]

    print(f"Loaded {len(examples)} {report_type} examples")
    print(f"Skipped rows missing report text: {missing_report}")
    print(f"Skipped rows missing ground truth: {missing_gt}")
    if not examples:
        raise ValueError("No training examples were loaded. Check shared case IDs and column names.")
    return examples


# =============================================================================
# Splitting examples
# =============================================================================


def split_examples(examples: List[dspy.Example], seed: int = 42):
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    n = len(examples)
    if n < 3:
        raise ValueError(f"Only {n} examples were loaded. Use more cases so train/dev/test sets are not empty.")
    train_end = max(1, int(n * 0.70))
    dev_end = max(train_end + 1, int(n * 0.85)) if n >= 4 else train_end + 1
    dev_end = min(dev_end, n)
    trainset = examples[:train_end]
    devset = examples[train_end:dev_end] or examples[:train_end]
    testset = examples[dev_end:] or devset
    return trainset, devset, testset


# =============================================================================
# Evaluation and debugging
# =============================================================================


@dataclass
class EvalResult:
    name: str
    accuracy: float
    score: float
    correct: int
    total: int
    wrong: int
    errors: int
    rows: List[Dict[str, Any]]


def prediction_debug_row(*, example: dspy.Example, pred: Any | None = None, error: Exception | None = None) -> Dict[str, Any]:
    gold = _gold_labels_for_example(example)
    report_text = _report_text_for_example(example)
    row: Dict[str, Any] = {
        "case_id": _case_id_for_example(example),
        "gold_raw_for_metric_only": str(_raw_gold_for_example(example)),
        "gold_for_metric_only": sorted(gold),
        "gold_labels_stored_in_dspy_example": bool(_example_value(example, "labels", None) is not None),
        "report_text_preview": report_text[:1000],
    }
    if error is not None:
        row.update({
            "status": "error",
            "match": False,
            "error_type": type(error).__name__,
            "error": str(error)[:5000],
        })
        return row

    raw_labels = _prediction_value(pred, "labels", "NONE")
    raw_reasoning = _prediction_value(pred, "reasoning", "")
    predicted = normalize_pred(raw_labels)
    label_format_ok = raw_label_format_ok(raw_labels)
    row.update({
        "status": "ok",
        "match": label_format_ok and gold == predicted,
        "predicted": sorted(predicted),
        "label_format_ok": label_format_ok,
        "raw_labels": str(raw_labels),
        "raw_reasoning": str(raw_reasoning),
        "raw_prediction_repr": repr(pred)[:3000],
    })
    return row


def evaluate_split(
    program,
    examples: List[dspy.Example],
    name: str,
    history_on_error_path: Optional[Path] = None,
    history_size: int = 3,
) -> EvalResult:
    rows: List[Dict[str, Any]] = []
    correct = 0
    error_count = 0
    score_total = 0.0
    for ex in examples:
        try:
            pred = program(report_text=_report_text_for_example(ex))
            row = prediction_debug_row(example=ex, pred=pred)
            try:
                optimizer_score = float(optimizer_metric(ex, pred))
            except Exception as metric_exc:
                optimizer_score = 0.0
                row["metric_error"] = f"{type(metric_exc).__name__}: {metric_exc}"
            row["optimizer_score"] = optimizer_score
            score_total += optimizer_score
        except Exception as exc:
            row = prediction_debug_row(example=ex, error=exc)
            row["optimizer_score"] = 0.0
            error_count += 1
            if history_on_error_path and bool(getattr(cfg, "DSPY_SAVE_HISTORY_ON_ERROR", True)):
                append_dspy_history_on_error(history_on_error_path, case_id=row.get("case_id", ""), error=exc, n=history_size)
        rows.append(row)
        if row["match"]:
            correct += 1
    total = len(examples)
    acc = correct / total if total else 0.0
    score = score_total / total if total else 0.0
    wrong = total - correct - error_count
    print(
        f"{name}: {correct}/{total} = {acc:.2%}; "
        f"optimizer score = {score:.4f} ({wrong} wrong, {error_count} parse/runtime errors)"
    )
    return EvalResult(name=name, accuracy=acc, score=score, correct=correct, total=total, wrong=wrong, errors=error_count, rows=rows)

def evaluate_splits(
    program,
    splits: Dict[str, List[dspy.Example]],
    prefix: str,
    history_on_error_path: Optional[Path] = None,
    history_size: int = 3,
) -> Dict[str, EvalResult]:
    results: Dict[str, EvalResult] = {}
    for split_name, examples in splits.items():
        results[split_name] = evaluate_split(
            program,
            examples,
            f"{prefix} {split_name}",
            history_on_error_path=history_on_error_path,
            history_size=history_size,
        )
    return results


def summarize_eval(result: EvalResult) -> Dict[str, Any]:
    return {
        "accuracy": result.accuracy,
        "score": result.score,
        "correct": result.correct,
        "total": result.total,
        "wrong": result.wrong,
        "errors": result.errors,
    }


# =============================================================================
# Logging helpers
# =============================================================================


def make_optimization_run_dir(report_type: str, iteration: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("Files") / "Results" / "DSPy_Optimization_Runs" / report_type.upper() / f"{timestamp}_iter_{iteration:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_run_layout(run_dir: Path) -> Dict[str, Path]:
    layout = {"root": run_dir, "debug": run_dir / "debug", "prompts": run_dir / "prompts"}
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def capture_dspy_history(n: int = 30) -> str:
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            dspy.inspect_history(n=n)
        return buffer.getvalue()
    except Exception as exc:
        return f"Could not inspect DSPy history: {exc}\n"


def append_dspy_history_file(path: Path, heading: str, n: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(heading + "\n")
        f.write("=" * 100 + "\n")
        f.write(capture_dspy_history(n=n))
        f.write("\n")


def append_dspy_history_on_error(path: Path, *, case_id: str, error: Exception, n: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"Case ID: {case_id}\n")
        f.write(f"Error type: {type(error).__name__}\n")
        f.write(f"Error: {str(error)[:5000]}\n")
        f.write("\n--- dspy.inspect_history() ---\n")
        f.write(capture_dspy_history(n=n))
        f.write("\n")


def save_program(program, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        program.save(str(path))
    except Exception as exc:
        path.with_suffix(".error.txt").write_text(str(exc), encoding="utf-8")


def load_program_if_present(program, path: Path) -> Tuple[bool, str]:
    """Load an existing optimized DSPy program into `program` when possible."""
    if not path.exists():
        return False, f"no saved program found at {path}"
    try:
        program.load(str(path))
        return True, f"warm-started from saved optimized program: {path}"
    except Exception as exc:
        return False, f"could not warm-start from {path}: {type(exc).__name__}: {exc}"


def extract_program_instructions(program: Any) -> str:
    for attrs in (("predict", "signature", "instructions"), ("signature", "instructions")):
        value = program
        try:
            for attr in attrs:
                value = getattr(value, attr)
            if value:
                return str(value)
        except Exception:
            pass
    try:
        signature = getattr(getattr(program, "predict", None), "signature", None)
        if isinstance(signature, dict):
            return str(signature.get("instructions", ""))
    except Exception:
        pass
    return ""


def _normalized_prompt_text(text: str) -> str:
    text = str(text or "").lower()
    for ch in ('"', "'", "`", "“", "”", "‘", "’"):
        text = text.replace(ch, "")
    return " ".join(text.split())


def prompt_quality_ok(prompt: str, report_type: str = "") -> Tuple[bool, str]:
    """Reject obviously degraded effective prompts before saving them."""
    prompt = str(prompt or "").strip()
    min_chars = int(getattr(cfg, "DSPY_PROMPT_MIN_CHARS", 500))
    if len(prompt) < min_chars:
        return False, f"prompt too short ({len(prompt)} chars < {min_chars})"

    report_type = (report_type or "").upper().strip()
    if report_type == "CTA":
        required_terms = list(getattr(cfg, "DSPY_CTA_REQUIRED_PROMPT_TERMS", []))
    else:
        required_terms = ["Allowed labels", "NONE", "RMCA", "LMCA"]

    normalized = _normalized_prompt_text(prompt)
    missing = [term for term in required_terms if _normalized_prompt_text(term) not in normalized]
    if missing:
        return False, "prompt missing required safety terms: " + ", ".join(missing)
    return True, "ok"


def save_examples_debug(path: Path, *, trainset, devset, testset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for split_name, examples in (("train", trainset), ("dev", devset), ("test", testset)):
        for ex in examples:
            rows.append({
                "split": split_name,
                "case_id": _case_id_for_example(ex),
                "gold_labels_for_metric_only": sorted(_gold_labels_for_example(ex)),
                "gold_labels_stored_in_dspy_example": bool(_example_value(ex, "labels", None) is not None),
                "report_text_preview": _report_text_for_example(ex)[:500],
            })
    path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


def save_predictions_debug(path: Path, results_by_stage: Dict[str, Dict[str, EvalResult]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output: Dict[str, Any] = {}
    for stage, split_results in results_by_stage.items():
        output[stage] = {}
        for split_name, result in split_results.items():
            output[stage][split_name] = {
                "summary": summarize_eval(result),
                "rows": result.rows,
            }
    path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")


def save_accuracy_report(path: Path, summary: Dict[str, Any]) -> None:
    def pct(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.2%}"

    def line_for(stage: str, split: str) -> str:
        item = summary.get("accuracies", {}).get(stage, {}).get(split)
        if not item:
            return f"{stage} {split}: N/A"
        score = item.get("score")
        score_text = "N/A" if score is None else f"{float(score):.4f}"
        return f"{stage} {split}: {pct(item['accuracy'])} ({item['correct']}/{item['total']}), score={score_text}"

    lines = [
        f"Report type: {summary.get('report_type')}",
        f"Created at: {summary.get('created_at')}",
        f"Run folder: {summary.get('timestamped_run_dir')}",
        "",
        "Accuracy by split:",
        line_for("baseline", "train"),
        line_for("baseline", "dev"),
        line_for("baseline", "test"),
        line_for("candidate", "train"),
        line_for("candidate", "dev"),
        line_for("candidate", "test"),
        line_for("active_after", "train"),
        line_for("active_after", "dev"),
        line_for("active_after", "test"),
        "",
        f"Candidate accepted as after prompt: {summary.get('candidate_accepted')}",
        f"Acceptance reason: {summary.get('acceptance_reason')}",
        f"Prompt quality check: {summary.get('prompt_quality_reason')}",
        f"Saved program status: {summary.get('saved_program_status')}",
        f"Optimizer metric: {summary.get('optimizer_metric')}",
        f"Acceptance split: {summary.get('acceptance_split')}",
        f"Baseline accept score: {summary.get('baseline_accept_score')}",
        f"Candidate accept score: {summary.get('candidate_accept_score')}",
        f"Acceptance score delta: {summary.get('acceptance_score_delta')}",
        f"Minimum required score improvement: {summary.get('min_score_improvement')}",
        f"Candidate signature prompt chars: {summary.get('candidate_signature_prompt_chars')}",
        f"Candidate effective prompt chars: {summary.get('candidate_effective_prompt_chars')}",
        f"Fixed rules chars: {summary.get('fixed_rules_chars')}",
        "",
        f"Gold labels hidden from DSPy examples: {summary.get('gold_labels_hidden_from_dspy_examples')}",
        f"Warm-start enabled: {summary.get('warm_start_enabled')}",
        f"Warm-started this iteration: {summary.get('warm_started')}",
        f"Warm-start status: {summary.get('warm_start_status')}",
        f"Instruction-only optimization: {summary.get('instruction_only_optimization')}",
        f"MIPRO optimizer valset source: {summary.get('mipro_valset_source')}",
        f"MIPRO optimizer valset size: {summary.get('mipro_valset_size')}",
        f"Max bootstrapped demos: {summary.get('max_bootstrapped_demos')}",
        f"Max labeled demos: {summary.get('max_labeled_demos')}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Training one modality
# =============================================================================


def _program_for_report_type(report_type: str):
    report_type = report_type.upper()
    program_names = getattr(cfg, "DSPY_PROGRAM_NAMES", {})
    if report_type == "CT":
        return CTLabeler(), f"{program_names.get('CT', 'ct_labeler')}.json"
    if report_type == "CTA":
        return CTALabeler(), f"{program_names.get('CTA', 'cta_labeler_fixed_rules')}.json"
    if report_type == "CTP":
        return CTPLabeler(), f"{program_names.get('CTP', 'ctp_labeler')}.json"
    raise ValueError("report_type must be CT, CTA, or CTP")


def train_one(
    report_type: str,
    reports_file: str,
    ground_truth_file: str,
    max_cases: Optional[int] = None,
    iteration: int = 1,
    save_run_logs: bool = True,
    history_size: int = 30,
    baseline_only: bool = False,
    smoke_test: bool = False,
    allow_demos: bool = False,
    mipro_valset_source: Optional[str] = None,
    accept_equal: Optional[bool] = None,
    save_even_if_worse: bool = False,
    warm_start: bool = True,
) -> Dict[str, Any]:
    report_type = report_type.upper()
    disable_training_caches()
    configure_dspy()
    disable_training_caches()

    run_dir: Optional[Path] = None
    layout: Optional[Dict[str, Path]] = None
    if save_run_logs:
        run_dir = make_optimization_run_dir(report_type, iteration)
        layout = make_run_layout(run_dir)

    examples = load_examples(reports_file=reports_file, ground_truth_file=ground_truth_file, report_type=report_type, max_cases=max_cases)
    trainset, devset, testset = split_examples(examples)
    splits = {"train": trainset, "dev": devset, "test": testset}
    program, save_name = _program_for_report_type(report_type)

    active_save_path = Path(cfg.DSPY_PROGRAM_DIR) / save_name
    saved_program_existed_before = active_save_path.exists()
    warm_started = False
    if warm_start:
        warm_started, warm_start_status = load_program_if_present(program, active_save_path)
    else:
        warm_start_status = "warm-start disabled; starting from config signature instructions"

    before_signature_prompt = extract_program_instructions(program) or _signature_instructions(report_type)
    before_prompt = effective_prompt_text(report_type, before_signature_prompt)

    print(f"\nTraining {report_type}")
    print(f"Train: {len(trainset)} | Dev: {len(devset)} | Test: {len(testset)}")
    print(warm_start_status)

    if layout:
        (layout["prompts"] / "before_signature_prompt.txt").write_text(before_signature_prompt, encoding="utf-8")
        (layout["prompts"] / "before_prompt.txt").write_text(before_prompt, encoding="utf-8")
        save_program(program, layout["prompts"] / "before_program.json")
        save_examples_debug(layout["debug"] / "loaded_examples.json", trainset=trainset, devset=devset, testset=testset)

    if smoke_test:
        smoke_result = evaluate_split(program, [testset[0]], f"{report_type} smoke test")
        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "iteration": iteration,
            "report_type": report_type,
            "mode": "smoke_test",
            "smoke_test": summarize_eval(smoke_result),
            "timestamped_run_dir": str(run_dir) if run_dir else None,
            "gold_labels_hidden_from_dspy_examples": True,
            "warm_start_enabled": warm_start,
            "warm_started": warm_started,
            "warm_start_status": warm_start_status,
            "active_save_path": str(active_save_path),
        }
        if layout:
            save_predictions_debug(layout["debug"] / "predictions.json", {"smoke_test": {"test": smoke_result}})
            append_dspy_history_file(layout["debug"] / "dspy_history.txt", "after smoke test", n=history_size)
            (layout["root"] / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    baseline_results = evaluate_splits(
        program,
        splits,
        f"{report_type} baseline",
        history_on_error_path=(layout["debug"] / "dspy_history_on_errors.txt") if layout else None,
        history_size=getattr(cfg, "DSPY_ERROR_HISTORY_SIZE", 3),
    )
    if layout:
        append_dspy_history_file(layout["debug"] / "dspy_history.txt", "after baseline evaluation", n=history_size)

    base_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "report_type": report_type,
        "reports_file": str(reports_file),
        "ground_truth_file": str(ground_truth_file),
        "max_cases": max_cases,
        "total_examples": len(examples),
        "train_examples": len(trainset),
        "dev_examples": len(devset),
        "test_examples": len(testset),
        "gold_labels_hidden_from_dspy_examples": True,
        "dspy_cache_disabled": True,
        "instruction_only_optimization": not allow_demos,
        "max_bootstrapped_demos": 0 if not allow_demos else None,
        "max_labeled_demos": 0 if not allow_demos else None,
        "timestamped_run_dir": str(run_dir) if run_dir else None,
        "warm_start_enabled": warm_start,
        "warm_started": warm_started,
        "warm_start_status": warm_start_status,
        "saved_program_existed_before": saved_program_existed_before,
        "active_save_path": str(active_save_path),
        "before_signature_prompt_chars": len(before_signature_prompt),
        "before_effective_prompt_chars": len(before_prompt),
    }

    if baseline_only:
        summary = {
            **base_summary,
            "mode": "baseline_only",
            "candidate_accepted": False,
            "acceptance_reason": "baseline-only mode",
            "prompt_quality_reason": "not run",
            "accuracies": {"baseline": {k: summarize_eval(v) for k, v in baseline_results.items()}},
            "active_saved_program": str(active_save_path) if active_save_path.exists() else None,
            "saved_program_status": "baseline-only mode; saved program was not changed",
        }
        if layout:
            save_predictions_debug(layout["debug"] / "predictions.json", {"baseline": baseline_results})
            (layout["prompts"] / "after_prompt.txt").write_text(before_prompt, encoding="utf-8")
            save_program(program, layout["prompts"] / "after_program.json")
            (layout["root"] / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            save_accuracy_report(layout["root"] / "accuracy_report.txt", summary)
        print("Baseline-only mode: skipped MIPRO optimization.")
        return summary

    optimizer_valset_source = (mipro_valset_source or getattr(cfg, "DSPY_MIPRO_VALSET_SOURCE", "train"))
    optimizer_valset = choose_optimizer_valset(trainset, devset, testset, optimizer_valset_source)
    print(f"MIPRO candidate prompts will be evaluated on {len(optimizer_valset)} cases from split source: {optimizer_valset_source}")

    optimizer = get_optimizer(optimizer_metric, run_dir=run_dir)
    try:
        optimized_program = compile_with_no_answer_leak(optimizer, program, trainset, optimizer_valset, allow_demos=allow_demos)
    except Exception as exc:
        if layout:
            append_dspy_history_file(layout["debug"] / "dspy_history.txt", "after compile error", n=history_size)
            (layout["debug"] / "compile_error.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
            save_predictions_debug(layout["debug"] / "predictions.json", {"baseline": baseline_results})
        raise RuntimeError(
            "DSPy optimization failed during compile. Check debug/compile_error.txt and debug/dspy_history.txt."
        ) from exc

    if layout:
        append_dspy_history_file(layout["debug"] / "dspy_history.txt", "after MIPRO compile", n=history_size)

    candidate_results = evaluate_splits(
        optimized_program,
        splits,
        f"{report_type} candidate",
        history_on_error_path=(layout["debug"] / "dspy_history_on_errors.txt") if layout else None,
        history_size=getattr(cfg, "DSPY_ERROR_HISTORY_SIZE", 3),
    )
    if layout:
        append_dspy_history_file(layout["debug"] / "dspy_history.txt", "after candidate evaluation", n=history_size)

    # `before_signature_prompt` / `before_prompt` were captured after optional warm-start.
    candidate_signature_prompt = extract_program_instructions(optimized_program)
    candidate_prompt = effective_prompt_text(report_type, candidate_signature_prompt)
    quality_ok, quality_reason = prompt_quality_ok(candidate_prompt, report_type=report_type)

    if accept_equal is None:
        accept_equal = bool(getattr(cfg, "DSPY_ACCEPT_EQUAL_ACCURACY", False))

    acceptance_split = str(getattr(cfg, "DSPY_ACCEPTANCE_SPLIT", "dev") or "dev").lower().strip()
    if acceptance_split not in baseline_results:
        raise ValueError(f"DSPY_ACCEPTANCE_SPLIT must be one of {sorted(baseline_results)}")

    # Acceptance uses the same per-example reward metric given to MIPRO, averaged
    # over the configured acceptance split. With strict_exact this is identical to
    # strict accuracy; with label_f1 this can reward partial improvements.
    baseline_accept_score = baseline_results[acceptance_split].score
    candidate_accept_score = candidate_results[acceptance_split].score
    score_delta = candidate_accept_score - baseline_accept_score
    min_score_improvement = float(getattr(cfg, "DSPY_MIN_SCORE_IMPROVEMENT", 0.0))

    force_save_note = ""
    if save_even_if_worse:
        force_save_note = (
            " -- note: --save-even-if-worse no longer overwrites the active optimized program; "
            "rejected candidates are still saved in the run folder as candidate_program.json"
        )

    if not quality_ok:
        accepted = False
        acceptance_reason = f"candidate rejected: {quality_reason}{force_save_note}"
    elif score_delta > min_score_improvement:
        accepted = True
        acceptance_reason = (
            f"candidate improved {acceptance_split} optimizer score "
            f"({candidate_accept_score:.4f} > {baseline_accept_score:.4f}; "
            f"delta={score_delta:.4f})"
        )
    elif accept_equal and abs(score_delta) <= 1e-12:
        accepted = True
        acceptance_reason = (
            f"candidate tied {acceptance_split} optimizer score and equal scores are allowed "
            f"({candidate_accept_score:.4f})"
        )
    else:
        accepted = False
        acceptance_reason = (
            f"candidate not better on {acceptance_split} optimizer score "
            f"({candidate_accept_score:.4f} <= {baseline_accept_score:.4f}; "
            f"delta={score_delta:.4f}){force_save_note}"
        )

    active_program = optimized_program if accepted else program
    active_results = candidate_results if accepted else baseline_results

    Path(cfg.DSPY_PROGRAM_DIR).mkdir(parents=True, exist_ok=True)
    if accepted:
        active_program.save(str(active_save_path))
        saved_program_status = (
            f"accepted candidate saved to {active_save_path}; this prompt will be the next loop iteration's start"
        )
    elif not active_save_path.exists():
        program.save(str(active_save_path))
        saved_program_status = (
            f"candidate rejected; no previous saved program existed, so baseline was saved to {active_save_path}"
        )
    else:
        saved_program_status = (
            f"candidate rejected; existing saved program left unchanged at {active_save_path}"
        )

    summary = {
        **base_summary,
        "mipro_valset_source": optimizer_valset_source,
        "mipro_valset_size": len(optimizer_valset),
        "candidate_accepted": accepted,
        "acceptance_reason": acceptance_reason,
        "prompt_quality_reason": quality_reason,
        "saved_program_status": saved_program_status,
        "active_saved_program": str(active_save_path),
        "optimizer_metric": str(getattr(cfg, "DSPY_OPTIMIZER_METRIC", "strict_exact")),
        "acceptance_split": acceptance_split,
        "baseline_accept_score": baseline_accept_score,
        "candidate_accept_score": candidate_accept_score,
        "acceptance_score_delta": score_delta,
        "min_score_improvement": min_score_improvement,
        "candidate_signature_prompt_chars": len(candidate_signature_prompt),
        "candidate_effective_prompt_chars": len(candidate_prompt),
        "fixed_rules_chars": len(_fixed_rules_for_report_type(report_type)),
        "accuracies": {
            "baseline": {k: summarize_eval(v) for k, v in baseline_results.items()},
            "candidate": {k: summarize_eval(v) for k, v in candidate_results.items()},
            "active_after": {k: summarize_eval(v) for k, v in active_results.items()},
        },
        # Backward-compatible summary fields.
        "baseline_accuracy": baseline_results["test"].accuracy,
        "candidate_accuracy": candidate_results["test"].accuracy,
        "optimized_accuracy": active_results["test"].accuracy,
        "baseline_score": baseline_results["test"].score,
        "candidate_score": candidate_results["test"].score,
        "optimized_score": active_results["test"].score,
    }

    if layout:
        summary["folders"] = {"debug": str(layout["debug"]), "prompts": str(layout["prompts"])}
        summary["prompt_files"] = {
            "before_prompt": str(layout["prompts"] / "before_prompt.txt"),
            "after_prompt": str(layout["prompts"] / "after_prompt.txt"),
            "candidate_prompt": str(layout["prompts"] / "candidate_prompt.txt"),
            "before_signature_prompt": str(layout["prompts"] / "before_signature_prompt.txt"),
            "candidate_signature_prompt": str(layout["prompts"] / "candidate_signature_prompt.txt"),
            "fixed_rules_prompt": str(layout["prompts"] / "fixed_rules_prompt.txt"),
        }
        save_predictions_debug(layout["debug"] / "predictions.json", {
            "baseline": baseline_results,
            "candidate": candidate_results,
            "active_after": active_results,
        })
        (layout["prompts"] / "before_signature_prompt.txt").write_text(before_signature_prompt, encoding="utf-8")
        (layout["prompts"] / "candidate_signature_prompt.txt").write_text(candidate_signature_prompt, encoding="utf-8")
        (layout["prompts"] / "fixed_rules_prompt.txt").write_text(_fixed_rules_for_report_type(report_type), encoding="utf-8")
        (layout["prompts"] / "candidate_prompt.txt").write_text(candidate_prompt, encoding="utf-8")
        save_program(optimized_program, layout["prompts"] / "candidate_program.json")
        (layout["prompts"] / "after_prompt.txt").write_text(candidate_prompt if accepted else before_prompt, encoding="utf-8")
        save_program(active_program, layout["prompts"] / "after_program.json")
        (layout["root"] / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        save_accuracy_report(layout["root"] / "accuracy_report.txt", summary)

    print(f"Candidate accepted: {accepted} ({acceptance_reason})")
    print(saved_program_status)
    if layout:
        print(f"Saved this optimization run to {run_dir}")
        print(f"  Debug files:  {layout['debug']}")
        print(f"  Prompt files: {layout['prompts']}")
    return summary


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Optimize DSPy stroke labelers without leaking answer keys into prompts.")
    parser.add_argument("--reports", "--reports-file", dest="reports_file", default=getattr(cfg, "TRAINING_REPORTS_FILE", cfg.INPUT_REPORT_FILE), help="Path to reports Excel file containing CT/CTA/CTP report text.")
    parser.add_argument("--ground-truth", default=cfg.GROUND_TRUTH_FILE, help="Path to ground-truth answer key Excel file.")
    parser.add_argument("--report-type", choices=["CT", "CTA", "CTP"], required=True, help="Which modality to optimize.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional number of cases for faster testing.")
    parser.add_argument("--loop", action="store_true", help="Keep optimizing repeatedly until stopped with Ctrl+C. Each accepted prompt is saved and warm-started by the next iteration.")
    parser.add_argument("--fresh-start", action="store_true", help="Do not load an existing optimized program before optimizing. Use this only when you want to start over from config.py instructions.")
    parser.add_argument("--no-run-logs", action="store_true", help="Disable timestamped optimization folders.")
    parser.add_argument("--history-size", type=int, default=30, help="Number of visible DSPy history items to save per stage.")
    parser.add_argument("--baseline-only", action="store_true", help="Evaluate the baseline and save debug logs without running MIPRO.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one test prediction and stop.")
    parser.add_argument("--allow-demos", action="store_true", help="Allow MIPRO to build demos. Default is OFF to avoid answer leakage.")
    parser.add_argument("--mipro-valset-source", choices=["train", "dev", "train_dev", "all"], default=getattr(cfg, "DSPY_MIPRO_VALSET_SOURCE", "train"), help="Which split MIPRO uses to evaluate candidate prompts. Default train makes MIPRO score all 28 train cases instead of only 6 dev cases.")
    parser.add_argument("--accept-equal", action="store_true", help="Accept optimized prompt if the configured acceptance split ties baseline and prompt quality check passes.")
    parser.add_argument("--save-even-if-worse", action="store_true", help="Deprecated safety valve: candidate_program.json is still logged, but worse candidates will not overwrite the active optimized program.")
    args = parser.parse_args()

    iteration = 1
    try:
        while True:
            train_one(
                report_type=args.report_type,
                reports_file=args.reports_file,
                ground_truth_file=args.ground_truth,
                max_cases=args.max_cases,
                iteration=iteration,
                save_run_logs=not args.no_run_logs,
                history_size=args.history_size,
                baseline_only=args.baseline_only,
                smoke_test=args.smoke_test,
                allow_demos=args.allow_demos,
                mipro_valset_source=args.mipro_valset_source,
                accept_equal=args.accept_equal,
                save_even_if_worse=args.save_even_if_worse,
                warm_start=not args.fresh_start,
            )
            if not args.loop:
                break
            iteration += 1
    except KeyboardInterrupt:
        print("\nStopped DSPy optimization loop.")


if __name__ == "__main__":
    main()
