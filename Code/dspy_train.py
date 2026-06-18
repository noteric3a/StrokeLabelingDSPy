"""Optimize CT, CTA, or CTP DSPy programs using config.py settings.

The only command-line option is ``--loop``.  Every other setting is controlled
from config.py so repeated experiments are reproducible and easy to compare.

Accuracy-oriented design:
- supervised training examples expose reference labels as DSPy output fields,
  never as task-model inputs;
- a stratified four-way split separates prompt training, optimizer validation,
  promotion, and final testing;
- DSPy receives a dense exact/F1 search reward;
- a candidate replaces the saved best program only when its promotion score
  improves without regressing exact accuracy;
- accepted programs warm-start the next ``--loop`` iteration;
- the final test set is not used for prompt search or promotion.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import inspect
import io
import json
import logging
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import dspy
import pandas as pd

import config as cfg
from dspy_programs import CTALabeler, CTLabeler, CTPLabeler, configure_dspy
from utils import normalize_labels

logging.getLogger("dspy.predict.predict").setLevel(logging.ERROR)


# =============================================================================
# Reference-label lookup
# =============================================================================

_GOLD_BY_REPORT_KEY: Dict[str, str] = {}
_RAW_GOLD_BY_REPORT_KEY: Dict[str, Any] = {}
_CASE_ID_BY_REPORT_KEY: Dict[str, str] = {}


def _report_key(report_text: Any) -> str:
    return " ".join(str(report_text or "").split())


def _value(obj: Any, field: str, default: Any = "") -> Any:
    for reader in (
        lambda item: item[field],
        lambda item: item.get(field),
        lambda item: getattr(item, field),
    ):
        try:
            value = reader(obj)
        except Exception:
            continue
        if not callable(value):
            return value
    return default


def _report_text_for_example(example: Any) -> str:
    return str(_value(example, "report_text", "") or "")


def _case_id_for_example(example: Any) -> str:
    return _CASE_ID_BY_REPORT_KEY.get(_report_key(_report_text_for_example(example)), "")


def _raw_gold_for_example(example: Any) -> Any:
    direct = _value(example, "labels", None)
    if direct is not None and str(direct).strip():
        return direct
    return _RAW_GOLD_BY_REPORT_KEY.get(_report_key(_report_text_for_example(example)), "NONE")


def _gold_labels_for_example(example: Any) -> Set[str]:
    return normalize_gt(_raw_gold_for_example(example))


# =============================================================================
# Normalization and metrics
# =============================================================================


def normalize_gt(value: Any) -> Set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {"NONE"}
    labels = set(normalize_labels(value))
    if not labels:
        return {"NONE"}
    if "NONE" in labels and len(labels) > 1:
        labels.remove("NONE")
    return labels


def normalize_pred(value: Any) -> Set[str]:
    labels = set(normalize_labels(value))
    if "NONE" in labels and len(labels) > 1:
        labels.remove("NONE")
    return labels or {"NONE"}


def normalize_case_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _raw_label_tokens(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        raw_items: Any = None
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
        token = str(item).strip().upper().strip("\"'[](){}")
        token = re.sub(r"\s+", " ", token)
        token = re.sub(r"^(LABELS?|FINAL LABELS?|OUTPUT)\s*:\s*", "", token).strip()
        if token:
            tokens.append(token)
    return tokens


def raw_label_format_ok(value: Any) -> bool:
    tokens = _raw_label_tokens(value)
    if not tokens:
        return False

    normalized_tokens: List[str] = []
    allowed = set(cfg.ALLOWED_LABELS)
    for token in tokens:
        label = cfg.LABEL_ALIASES.get(token, token)
        compact = label.replace(" ", "")
        if compact in allowed:
            label = compact
        if label not in allowed:
            return False
        normalized_tokens.append(label)
    return not ("NONE" in normalized_tokens and len(set(normalized_tokens)) > 1)


def _label_f1(gold: Set[str], predicted: Set[str]) -> float:
    if gold == predicted:
        return 1.0
    overlap = len(gold & predicted)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2.0 * precision * recall / (precision + recall)


def _weighted_score(exact: float, f1: float, *, exact_weight: float, f1_weight: float) -> float:
    total = exact_weight + f1_weight
    if total <= 0:
        raise ValueError("Metric weights must sum to a positive value.")
    return (exact_weight * exact + f1_weight * f1) / total


@dataclass(frozen=True)
class MetricParts:
    format_ok: bool
    exact: float
    f1: float
    search: float
    promotion: float


def metric_parts(example: Any, pred: Any) -> MetricParts:
    raw_labels = _value(pred, "labels", "")
    format_ok = raw_label_format_ok(raw_labels)
    gold = _gold_labels_for_example(example)
    predicted = normalize_pred(raw_labels)

    exact = 1.0 if format_ok and gold == predicted else 0.0
    f1 = _label_f1(gold, predicted) if format_ok else 0.0
    search = _weighted_score(
        exact,
        f1,
        exact_weight=float(cfg.DSPY_SEARCH_EXACT_WEIGHT),
        f1_weight=float(cfg.DSPY_SEARCH_F1_WEIGHT),
    )
    promotion = _weighted_score(
        exact,
        f1,
        exact_weight=float(cfg.DSPY_PROMOTION_EXACT_WEIGHT),
        f1_weight=float(cfg.DSPY_PROMOTION_F1_WEIGHT),
    )
    return MetricParts(format_ok=format_ok, exact=exact, f1=f1, search=search, promotion=promotion)


def optimizer_metric(example: Any, pred: Any, trace: Any = None) -> float:
    """Dense scalar reward used by MIPROv2."""
    del trace
    return metric_parts(example, pred).search


def gepa_feedback_metric(
    gold: Any,
    pred: Any,
    trace: Any = None,
    pred_name: Optional[str] = None,
    pred_trace: Any = None,
) -> Any:
    """Dense reward plus actionable missing/extra-label feedback for GEPA."""
    del trace, pred_name, pred_trace
    parts = metric_parts(gold, pred)
    expected = _gold_labels_for_example(gold)
    raw_labels = _value(pred, "labels", "")
    predicted = normalize_pred(raw_labels)

    if not parts.format_ok:
        feedback = (
            f"Invalid labels field {raw_labels!r}. Use only allowed labels {cfg.ALLOWED_LABELS}; "
            "NONE cannot be combined with a positive label. "
            f"Expected labels for this report: {sorted(expected)}."
        )
    elif parts.exact == 1.0:
        feedback = "The labels exactly match the reference answer and the output format is valid."
    else:
        missing = sorted(expected - predicted)
        extra = sorted(predicted - expected)
        feedback = (
            f"Expected {sorted(expected)} but predicted {sorted(predicted)}. "
            f"Missing labels: {missing or 'none'}. Extra labels: {extra or 'none'}. "
            "Revise the instruction to correct this vessel/laterality/policy decision without breaking cases that already match."
        )

    prediction_cls = getattr(dspy, "Prediction", None)
    if prediction_cls is None:
        return {"score": parts.search, "feedback": feedback}
    return prediction_cls(score=parts.search, feedback=feedback)


# =============================================================================
# Configuration validation
# =============================================================================


def validate_training_config() -> None:
    report_type = str(cfg.TRAIN_REPORT_TYPE).upper().strip()
    if report_type not in {"CT", "CTA", "CTP"}:
        raise ValueError("TRAIN_REPORT_TYPE must be CT, CTA, or CTP.")

    ratios = dict(cfg.TRAIN_SPLIT_RATIOS)
    required = {"train", "optimizer_val", "dev", "test"}
    if set(ratios) != required:
        raise ValueError(f"TRAIN_SPLIT_RATIOS must contain exactly {sorted(required)}.")
    if any(float(value) <= 0 for value in ratios.values()):
        raise ValueError("Every TRAIN_SPLIT_RATIOS value must be positive.")
    if not math.isclose(sum(float(value) for value in ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("TRAIN_SPLIT_RATIOS values must sum to 1.0.")

    for pair in (
        (cfg.DSPY_SEARCH_EXACT_WEIGHT, cfg.DSPY_SEARCH_F1_WEIGHT),
        (cfg.DSPY_PROMOTION_EXACT_WEIGHT, cfg.DSPY_PROMOTION_F1_WEIGHT),
    ):
        if float(pair[0]) < 0 or float(pair[1]) < 0 or float(pair[0]) + float(pair[1]) <= 0:
            raise ValueError("Metric weights must be nonnegative and sum to more than zero.")

    optimizer_name = str(cfg.DSPY_OPTIMIZER).lower().strip()
    if optimizer_name not in {"mipro", "miprov2", "gepa"}:
        raise ValueError("DSPY_OPTIMIZER must be 'mipro' or 'gepa'.")

    if str(cfg.DSPY_ACCEPTANCE_SPLIT) not in ratios:
        raise ValueError("DSPY_ACCEPTANCE_SPLIT must name one of the configured splits.")
    if str(cfg.DSPY_ACCEPTANCE_SPLIT) == "test":
        raise ValueError("The final test split cannot be used as the promotion split.")


# =============================================================================
# Spreadsheet loading
# =============================================================================


def _normalize_col_name(name: Any) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _find_column(dataframe: pd.DataFrame, candidates: Sequence[str], *, purpose: str) -> str:
    for column in candidates:
        if column in dataframe.columns:
            return column
    normalized = {_normalize_col_name(column): column for column in dataframe.columns}
    for candidate in candidates:
        match = normalized.get(_normalize_col_name(candidate))
        if match is not None:
            return match
    raise ValueError(
        f"Could not find {purpose}. Tried {list(candidates)}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _report_column_candidates(report_type: str) -> List[str]:
    training = cfg.TRAINING_COLUMN_CANDIDATES[report_type]["report"]
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
    training = cfg.TRAINING_COLUMN_CANDIDATES[report_type]["ground_truth"]
    return _dedupe_keep_order([
        *training,
        f"{report_type} GT",
        f"{report_type}_GT",
        f"{report_type}.GT",
        f"{report_type}GT",
        f"{report_type} Ground Truth",
        report_type,
    ])


def load_examples(
    reports_file: str,
    ground_truth_file: str,
    report_type: str,
    max_cases: Optional[int] = None,
) -> List[Any]:
    """Load supervised DSPy examples with labels marked as output fields."""
    report_type = report_type.upper()
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
    report_col = _find_column(
        reports_df,
        _report_column_candidates(report_type),
        purpose=f"{report_type} report text in reports file",
    )
    gt_col = _find_column(
        gt_df,
        _gt_column_candidates(report_type),
        purpose=f"{report_type} ground-truth labels in ground-truth file",
    )

    print("\nResolved training columns")
    print(f"  Reports file:      {reports_path}")
    print(f"  Ground-truth file: {gt_path}")
    print(f"  Reports case col:  {reports_case_col}")
    print(f"  Reports text col:  {report_col}")
    print(f"  GT case col:       {gt_case_col}")
    print(f"  GT label col:      {gt_col}")

    gt_by_case: Dict[str, Any] = {}
    for _, row in gt_df.iterrows():
        case_id = normalize_case_id(row.get(gt_case_col))
        if case_id:
            gt_by_case[case_id] = row.get(gt_col)

    _GOLD_BY_REPORT_KEY.clear()
    _RAW_GOLD_BY_REPORT_KEY.clear()
    _CASE_ID_BY_REPORT_KEY.clear()

    examples: List[Any] = []
    missing_gt = 0
    missing_report = 0
    duplicate_conflicts: List[str] = []

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
        labels = sorted(normalize_gt(raw_gold), key=lambda label: cfg.LABEL_ORDER.index(label))
        normalized_gold = ", ".join(labels)
        key = _report_key(report_text)
        existing = _GOLD_BY_REPORT_KEY.get(key)
        if existing is not None and existing != normalized_gold:
            duplicate_conflicts.append(case_id)
            continue

        _GOLD_BY_REPORT_KEY[key] = normalized_gold
        _RAW_GOLD_BY_REPORT_KEY[key] = raw_gold
        _CASE_ID_BY_REPORT_KEY[key] = case_id

        # Labels are supervised outputs, not task inputs.  MIPRO/GEPA can learn
        # from training labels, while normal predictions still receive only
        # report_text.
        example = dspy.Example(report_text=report_text, labels=normalized_gold).with_inputs("report_text")
        examples.append(example)

    if max_cases is not None:
        examples = examples[: max(0, int(max_cases))]

    print(f"Loaded {len(examples)} {report_type} examples")
    print(f"Skipped rows missing report text: {missing_report}")
    print(f"Skipped rows missing ground truth: {missing_gt}")
    if duplicate_conflicts:
        print(f"Skipped duplicate report texts with conflicting labels: {len(duplicate_conflicts)}")
    if not examples:
        raise ValueError("No training examples were loaded. Check paths, case IDs, and column names.")
    return examples


# =============================================================================
# Multilabel-stratified splitting
# =============================================================================


def _target_split_sizes(total: int, ratios: Mapping[str, float]) -> Dict[str, int]:
    names = list(ratios)
    raw = {name: total * float(ratios[name]) for name in names}
    sizes = {name: int(math.floor(raw[name])) for name in names}

    if total >= len(names):
        for name in names:
            if sizes[name] == 0:
                sizes[name] = 1

    while sum(sizes.values()) > total:
        donors = [name for name in names if sizes[name] > 1]
        if not donors:
            break
        donor = max(donors, key=lambda name: sizes[name] - raw[name])
        sizes[donor] -= 1

    remaining = total - sum(sizes.values())
    for name in sorted(names, key=lambda item: raw[item] - math.floor(raw[item]), reverse=True):
        if remaining <= 0:
            break
        sizes[name] += 1
        remaining -= 1

    index = 0
    while remaining > 0:
        sizes[names[index % len(names)]] += 1
        index += 1
        remaining -= 1
    return sizes


def _stratification_features(example: Any) -> Set[str]:
    labels = _gold_labels_for_example(example)
    set_token = "SET=" + "+".join(sorted(labels))
    return set(labels) | {set_token}


def split_examples(examples: List[Any], seed: int) -> Dict[str, List[Any]]:
    """Greedily preserve labels and multilabel combinations across four splits."""
    examples = list(examples)
    ratios = {name: float(value) for name, value in cfg.TRAIN_SPLIT_RATIOS.items()}
    if len(examples) < len(ratios):
        raise ValueError(
            f"Only {len(examples)} examples were loaded; at least {len(ratios)} are needed "
            "for train/optimizer_val/dev/test."
        )

    targets = _target_split_sizes(len(examples), ratios)
    feature_counts: Counter[str] = Counter()
    records: List[Tuple[Any, Set[str], float]] = []
    rng = random.Random(seed)
    for example in examples:
        features = _stratification_features(example)
        feature_counts.update(features)
        records.append((example, features, rng.random()))

    # Rare labels and rare exact label sets are placed first.
    records.sort(
        key=lambda item: (
            min(feature_counts[feature] for feature in item[1]),
            -len(item[1]),
            item[2],
        )
    )

    desired_feature_counts: Dict[str, Dict[str, float]] = {
        split: {feature: feature_counts[feature] * ratios[split] for feature in feature_counts}
        for split in ratios
    }
    current_feature_counts: Dict[str, Counter[str]] = {split: Counter() for split in ratios}
    result: Dict[str, List[Any]] = {split: [] for split in ratios}

    for example, features, _ in records:
        candidates = [split for split in ratios if len(result[split]) < targets[split]]
        if not candidates:
            candidates = list(ratios)

        def placement_score(split: str) -> Tuple[float, float, float]:
            feature_need = 0.0
            for feature in features:
                desired = desired_feature_counts[split][feature]
                current = current_feature_counts[split][feature]
                rarity = 1.0 / max(1, feature_counts[feature])
                feature_need += max(0.0, desired - current) * (1.0 + rarity)
            size_need = (targets[split] - len(result[split])) / max(1, targets[split])
            return feature_need, size_need, rng.random()

        chosen = max(candidates, key=placement_score)
        result[chosen].append(example)
        current_feature_counts[chosen].update(features)

    for split, items in result.items():
        random.Random(seed + list(ratios).index(split) + 1).shuffle(items)
        if not items:
            raise ValueError(f"Stratification produced an empty {split} split.")
    return result


# =============================================================================
# Optimizer creation
# =============================================================================


def disable_training_caches() -> None:
    configure_cache = getattr(dspy, "configure_cache", None)
    if callable(configure_cache):
        try:
            configure_cache(enable_memory_cache=False, enable_disk_cache=False)
        except Exception:
            pass
    try:
        dspy.settings.configure(cache=False)
    except Exception:
        pass


@contextlib.contextmanager
def quiet_optimizer_prompt_logging():
    """Hide optimizer prompt bodies while preserving progress bars.

    MIPROv2 emits proposed instructions through INFO-level logging and can
    print complete candidate programs when its verbose flag is enabled.  The
    optimizer is configured with verbose=False below, and this context manager
    temporarily suppresses DEBUG/INFO log records during compilation.  It does
    not redirect stdout or stderr, so DSPy/tqdm progress bars remain visible.
    Warnings, errors, and tracebacks are still shown.
    """
    if bool(getattr(cfg, "TRAIN_SHOW_OPTIMIZER_PROMPTS", False)):
        yield
        return

    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(previous_disable_level)


def _make_lm(model: str, api_base: str, temperature: float, max_tokens: int) -> Any:
    kwargs: Dict[str, Any] = {
        "api_base": api_base,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "think": False,
    }
    if cfg.DSPY_DISABLE_CACHE:
        kwargs["cache"] = False
    try:
        return dspy.LM(model, **kwargs)
    except TypeError:
        kwargs.pop("cache", None)
        try:
            return dspy.LM(model, **kwargs)
        except TypeError:
            kwargs.pop("think", None)
            return dspy.LM(model, **kwargs)


def prompt_or_reflection_model() -> Any:
    model = cfg.DSPY_PROMPT_MODEL or cfg.DSPY_MODEL
    return _make_lm(
        model=model,
        api_base=cfg.DSPY_PROMPT_MODEL_API_BASE,
        temperature=float(cfg.DSPY_PROMPT_TEMPERATURE),
        max_tokens=int(cfg.DSPY_PROMPT_MAX_TOKENS),
    )


def _filter_supported_kwargs(callable_obj: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _mipro_class() -> Any:
    optimizer_class = getattr(dspy, "MIPROv2", None)
    if optimizer_class is not None:
        return optimizer_class
    try:
        from dspy.teleprompt import MIPROv2

        return MIPROv2
    except ImportError as exc:
        raise ImportError('MIPROv2 requires DSPy with Optuna support: pip install "dspy[optuna]"') from exc


def _gepa_class() -> Any:
    optimizer_class = getattr(dspy, "GEPA", None)
    if optimizer_class is not None:
        return optimizer_class
    try:
        from dspy.teleprompt import GEPA

        return GEPA
    except ImportError as exc:
        raise ImportError("GEPA is unavailable. Upgrade DSPy and install the GEPA dependency, or set DSPY_OPTIMIZER='mipro'.") from exc


def build_optimizer(run_dir: Optional[Path]) -> Tuple[str, Any]:
    optimizer_name = str(cfg.DSPY_OPTIMIZER).lower().strip()
    prompt_model = prompt_or_reflection_model()

    if optimizer_name in {"mipro", "miprov2"}:
        optimizer_class = _mipro_class()
        kwargs: Dict[str, Any] = {
            "metric": optimizer_metric,
            "prompt_model": prompt_model,
            "auto": cfg.DSPY_MIPRO_AUTO,
            "max_bootstrapped_demos": int(cfg.DSPY_MIPRO_MAX_BOOTSTRAPPED_DEMOS),
            "max_labeled_demos": int(cfg.DSPY_MIPRO_MAX_LABELED_DEMOS),
            "seed": int(cfg.DSPY_MIPRO_SEED),
            "init_temperature": float(cfg.DSPY_MIPRO_INIT_TEMPERATURE),
            # False by default: save prompt bodies to artifacts without
            # printing them. Progress bars are independent of this flag.
            "verbose": bool(cfg.TRAIN_SHOW_OPTIMIZER_PROMPTS),
            "track_stats": True,
            "metric_threshold": float(cfg.DSPY_MIPRO_METRIC_THRESHOLD),
        }
        if run_dir is not None:
            kwargs["log_dir"] = str(run_dir / "optimizer_logs")
        kwargs = _filter_supported_kwargs(optimizer_class, kwargs)
        return "mipro", optimizer_class(**kwargs)

    optimizer_class = _gepa_class()
    kwargs = {
        "metric": gepa_feedback_metric,
        "auto": cfg.DSPY_GEPA_AUTO,
        "reflection_lm": prompt_model,
        "reflection_minibatch_size": int(cfg.DSPY_GEPA_REFLECTION_MINIBATCH_SIZE),
        "candidate_selection_strategy": cfg.DSPY_GEPA_CANDIDATE_SELECTION,
        "add_format_failure_as_feedback": bool(cfg.DSPY_GEPA_ADD_FORMAT_FAILURE_AS_FEEDBACK),
        "use_merge": bool(cfg.DSPY_GEPA_USE_MERGE),
        "track_stats": True,
        "seed": int(cfg.DSPY_GEPA_SEED),
    }
    if run_dir is not None:
        kwargs["log_dir"] = str(run_dir / "optimizer_logs")
    kwargs = _filter_supported_kwargs(optimizer_class, kwargs)
    return "gepa", optimizer_class(**kwargs)


def compile_program(
    optimizer_name: str,
    optimizer: Any,
    program: Any,
    trainset: List[Any],
    optimizer_valset: List[Any],
) -> Any:
    compile_kwargs: Dict[str, Any] = {
        "trainset": trainset,
        "valset": optimizer_valset,
    }
    if optimizer_name == "mipro":
        compile_kwargs.update({
            "max_bootstrapped_demos": int(cfg.DSPY_MIPRO_MAX_BOOTSTRAPPED_DEMOS),
            "max_labeled_demos": int(cfg.DSPY_MIPRO_MAX_LABELED_DEMOS),
            "seed": int(cfg.DSPY_MIPRO_SEED),
            "minibatch_size": min(int(cfg.DSPY_MIPRO_MINIBATCH_SIZE), len(optimizer_valset)),
            "minibatch_full_eval_steps": int(cfg.DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS),
            "program_aware_proposer": bool(cfg.DSPY_MIPRO_PROGRAM_AWARE_PROPOSER),
            "data_aware_proposer": bool(cfg.DSPY_MIPRO_DATA_AWARE_PROPOSER),
            "tip_aware_proposer": bool(cfg.DSPY_MIPRO_TIP_AWARE_PROPOSER),
            "fewshot_aware_proposer": bool(cfg.DSPY_MIPRO_FEWSHOT_AWARE_PROPOSER),
            "view_data_batch_size": min(int(cfg.DSPY_MIPRO_VIEW_DATA_BATCH_SIZE), len(trainset)),
        })
    compile_kwargs = _filter_supported_kwargs(optimizer.compile, compile_kwargs)
    return optimizer.compile(program.deepcopy(), **compile_kwargs)


# =============================================================================
# Evaluation
# =============================================================================


@dataclass
class EvalResult:
    name: str
    accuracy: float
    label_f1: float
    search_score: float
    promotion_score: float
    correct: int
    total: int
    wrong: int
    errors: int
    rows: List[Dict[str, Any]]

    @property
    def score(self) -> float:
        """Backward-compatible alias for the optimizer/search score."""
        return self.search_score


def prediction_debug_row(example: Any, pred: Any = None, error: Optional[Exception] = None) -> Dict[str, Any]:
    gold = _gold_labels_for_example(example)
    row: Dict[str, Any] = {
        "case_id": _case_id_for_example(example),
        "gold": sorted(gold),
        "report_text_preview": _report_text_for_example(example)[:1000],
    }
    if error is not None:
        row.update({
            "status": "error",
            "match": False,
            "error_type": type(error).__name__,
            "error": str(error)[:5000],
            "exact_score": 0.0,
            "label_f1": 0.0,
            "search_score": 0.0,
            "promotion_score": 0.0,
        })
        return row

    raw_labels = _value(pred, "labels", "")
    predicted = normalize_pred(raw_labels)
    parts = metric_parts(example, pred)
    row.update({
        "status": "ok",
        "match": parts.exact == 1.0,
        "predicted": sorted(predicted),
        "label_format_ok": parts.format_ok,
        "raw_labels": str(raw_labels),
        "raw_reasoning": str(_value(pred, "reasoning", "")),
        "exact_score": parts.exact,
        "label_f1": parts.f1,
        "search_score": parts.search,
        "promotion_score": parts.promotion,
        "raw_prediction_repr": repr(pred)[:3000],
    })
    return row


def capture_dspy_history(count: int) -> str:
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            dspy.inspect_history(n=count)
        return buffer.getvalue()
    except Exception as exc:
        return f"Could not inspect DSPy history: {exc}\n"


def append_history(path: Path, heading: str, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "=" * 100 + "\n")
        handle.write(heading + "\n")
        handle.write("=" * 100 + "\n")
        handle.write(capture_dspy_history(count))


def evaluate_split(
    program: Any,
    examples: List[Any],
    name: str,
    history_on_error_path: Optional[Path] = None,
) -> EvalResult:
    rows: List[Dict[str, Any]] = []
    totals = defaultdict(float)
    errors = 0

    for example in examples:
        try:
            pred = program(report_text=_report_text_for_example(example))
            row = prediction_debug_row(example, pred=pred)
        except Exception as exc:
            row = prediction_debug_row(example, error=exc)
            errors += 1
            if history_on_error_path is not None and cfg.DSPY_SAVE_HISTORY_ON_ERROR:
                append_history(
                    history_on_error_path,
                    f"case {_case_id_for_example(example)} error: {type(exc).__name__}: {exc}",
                    int(cfg.DSPY_ERROR_HISTORY_SIZE),
                )
        rows.append(row)
        totals["exact"] += float(row["exact_score"])
        totals["f1"] += float(row["label_f1"])
        totals["search"] += float(row["search_score"])
        totals["promotion"] += float(row["promotion_score"])

    total = len(examples)
    divisor = total or 1
    correct = int(totals["exact"])
    wrong = total - correct - errors
    result = EvalResult(
        name=name,
        accuracy=totals["exact"] / divisor,
        label_f1=totals["f1"] / divisor,
        search_score=totals["search"] / divisor,
        promotion_score=totals["promotion"] / divisor,
        correct=correct,
        total=total,
        wrong=wrong,
        errors=errors,
        rows=rows,
    )
    print(
        f"{name}: exact={result.accuracy:.2%}, label_f1={result.label_f1:.4f}, "
        f"search={result.search_score:.4f}, promotion={result.promotion_score:.4f} "
        f"({correct}/{total}, {wrong} wrong, {errors} errors)"
    )
    return result


def evaluate_splits(
    program: Any,
    splits: Mapping[str, List[Any]],
    prefix: str,
    history_on_error_path: Optional[Path] = None,
) -> Dict[str, EvalResult]:
    return {
        split: evaluate_split(
            program,
            examples,
            f"{prefix} {split}",
            history_on_error_path=history_on_error_path,
        )
        for split, examples in splits.items()
    }


def summarize_eval(result: EvalResult) -> Dict[str, Any]:
    return {
        "accuracy": result.accuracy,
        "label_f1": result.label_f1,
        "search_score": result.search_score,
        "promotion_score": result.promotion_score,
        "correct": result.correct,
        "total": result.total,
        "wrong": result.wrong,
        "errors": result.errors,
    }


def decide_promotion(
    baseline: EvalResult,
    candidate: EvalResult,
    *,
    split_name: str,
    quality_ok: bool,
    quality_reason: str,
) -> Tuple[bool, str, float, bool]:
    """Return whether the candidate should replace the saved best program."""
    delta = candidate.promotion_score - baseline.promotion_score
    exact_non_regression = candidate.accuracy + 1e-12 >= baseline.accuracy

    if not quality_ok:
        return False, f"rejected by prompt guard: {quality_reason}", delta, exact_non_regression
    if cfg.DSPY_REQUIRE_EXACT_NON_REGRESSION and not exact_non_regression:
        return (
            False,
            f"rejected because exact accuracy regressed on {split_name}: "
            f"{candidate.accuracy:.4f} < {baseline.accuracy:.4f}",
            delta,
            exact_non_regression,
        )
    if delta > float(cfg.DSPY_MIN_PROMOTION_IMPROVEMENT):
        return (
            True,
            f"promotion score improved on {split_name}: "
            f"{candidate.promotion_score:.4f} > {baseline.promotion_score:.4f} "
            f"(delta={delta:.4f}); exact {baseline.accuracy:.4f} -> {candidate.accuracy:.4f}",
            delta,
            exact_non_regression,
        )
    return (
        False,
        f"promotion score did not improve enough on {split_name}: "
        f"delta={delta:.4f}, required>{cfg.DSPY_MIN_PROMOTION_IMPROVEMENT}",
        delta,
        exact_non_regression,
    )


# =============================================================================
# Prompt/program and logging helpers
# =============================================================================


def _signature_instructions(report_type: str) -> str:
    return {
        "CT": cfg.CT_SIGNATURE_INSTRUCTIONS,
        "CTA": cfg.CTA_SIGNATURE_INSTRUCTIONS,
        "CTP": cfg.CTP_SIGNATURE_INSTRUCTIONS,
    }[report_type]


def _fixed_rules(report_type: str) -> str:
    return cfg.CTA_FIXED_RULES if report_type == "CTA" else ""


def effective_prompt_text(report_type: str, signature_instructions: str) -> str:
    fixed = _fixed_rules(report_type)
    if not fixed:
        return signature_instructions.strip()
    return (
        f"{signature_instructions.strip()}\n\n"
        "Fixed CTA output constraints supplied as the `cta_rules` input:\n"
        f"{fixed.strip()}"
    )


def _program_for_report_type(report_type: str) -> Tuple[Any, str]:
    if report_type == "CT":
        return CTLabeler(), f"{cfg.DSPY_PROGRAM_NAMES['CT']}.json"
    if report_type == "CTA":
        return CTALabeler(), f"{cfg.DSPY_PROGRAM_NAMES['CTA']}.json"
    if report_type == "CTP":
        return CTPLabeler(), f"{cfg.DSPY_PROGRAM_NAMES['CTP']}.json"
    raise ValueError("report_type must be CT, CTA, or CTP")


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


def load_program_if_present(program: Any, path: Path) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"no saved program found at {path}"
    try:
        program.load(str(path))
        return True, f"warm-started from saved best program: {path}"
    except Exception as exc:
        return False, f"could not warm-start from {path}: {type(exc).__name__}: {exc}"


def save_program(program: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(path))


def _normalized_prompt(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = normalized.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return " ".join(normalized.split())


def prompt_quality_ok(signature_prompt: str, report_type: str) -> Tuple[bool, str]:
    """Validate only DSPy's generated signature instruction, not fixed rules."""
    prompt = str(signature_prompt or "").strip()
    if len(prompt) < int(cfg.DSPY_PROMPT_MIN_CHARS):
        return False, f"signature prompt too short ({len(prompt)} < {cfg.DSPY_PROMPT_MIN_CHARS})"
    if len(prompt) > int(cfg.DSPY_PROMPT_MAX_CHARS):
        return False, f"signature prompt too long ({len(prompt)} > {cfg.DSPY_PROMPT_MAX_CHARS})"

    normalized = " " + _normalized_prompt(prompt) + " "
    for forbidden in cfg.DSPY_PROMPT_FORBIDDEN_TERMS:
        if _normalized_prompt(forbidden) in normalized:
            return False, f"signature prompt contains forbidden wording: {forbidden!r}"

    if report_type == "CTA":
        missing_groups: List[Tuple[str, ...]] = []
        for group in cfg.DSPY_CTA_REQUIRED_TERM_GROUPS:
            if not any(_normalized_prompt(term) in normalized for term in group):
                missing_groups.append(tuple(group))
        if missing_groups:
            readable = ["/".join(group) for group in missing_groups]
            return False, "signature prompt is missing required policy concepts: " + ", ".join(readable)
    return True, "ok"


def make_run_dir(report_type: str, iteration: int, suffix: str = "") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_iter_{iteration:04d}{suffix}"
    path = Path(cfg.TRAIN_RUNS_ROOT) / report_type / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_layout(run_dir: Path) -> Dict[str, Path]:
    layout = {
        "root": run_dir,
        "debug": run_dir / "debug",
        "prompts": run_dir / "prompts",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def save_split_debug(
    path: Path,
    splits: Mapping[str, List[Any]],
    *,
    redact_splits: Set[str] | None = None,
) -> None:
    redacted = set(redact_splits or set())
    output: Dict[str, Any] = {"sizes": {}, "label_counts": {}, "examples": []}
    for split, examples in splits.items():
        output["sizes"][split] = len(examples)
        counts: Counter[str] = Counter()
        for example in examples:
            labels = sorted(_gold_labels_for_example(example))
            if split not in redacted:
                counts.update(labels)
            output["examples"].append({
                "split": split,
                "case_id": _case_id_for_example(example),
                "labels": labels if split not in redacted else "<redacted until final audit>",
                "report_text_preview": _report_text_for_example(example)[:500],
            })
        output["label_counts"][split] = (
            dict(sorted(counts.items())) if split not in redacted else "<redacted until final audit>"
        )
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def save_predictions(path: Path, stages: Mapping[str, Mapping[str, EvalResult]]) -> None:
    output: Dict[str, Any] = {}
    for stage, split_results in stages.items():
        output[stage] = {
            split: {"summary": summarize_eval(result), "rows": result.rows}
            for split, result in split_results.items()
        }
    path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")


def save_summary_text(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        f"Created: {summary.get('created_at')}",
        f"Iteration: {summary.get('iteration')}",
        f"Report type: {summary.get('report_type')}",
        f"Optimizer: {summary.get('optimizer')}",
        f"Candidate accepted: {summary.get('candidate_accepted')}",
        f"Acceptance reason: {summary.get('acceptance_reason')}",
        f"Saved program status: {summary.get('saved_program_status')}",
    ]
    for stage in ("baseline", "candidate", "active_after"):
        for split, values in summary.get("metrics", {}).get(stage, {}).items():
            lines.append(
                f"{stage} {split}: exact={values['accuracy']:.4f}, f1={values['label_f1']:.4f}, "
                f"search={values['search_score']:.4f}, promotion={values['promotion_score']:.4f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# One optimization iteration
# =============================================================================


def train_one(iteration: int, *, evaluate_test: bool = False) -> Dict[str, Any]:
    report_type = str(cfg.TRAIN_REPORT_TYPE).upper().strip()
    disable_training_caches()
    configure_dspy()
    disable_training_caches()

    run_dir: Optional[Path] = None
    layout: Optional[Dict[str, Path]] = None
    if cfg.TRAIN_SAVE_RUN_LOGS:
        run_dir = make_run_dir(report_type, iteration)
        layout = make_layout(run_dir)

    examples = load_examples(
        reports_file=cfg.TRAINING_REPORTS_FILE,
        ground_truth_file=cfg.GROUND_TRUTH_FILE,
        report_type=report_type,
        max_cases=cfg.TRAIN_MAX_CASES,
    )
    splits = split_examples(examples, seed=int(cfg.TRAIN_RANDOM_SEED))
    eval_splits = {
        name: values
        for name, values in splits.items()
        if name != "test" or evaluate_test
    }

    program, save_name = _program_for_report_type(report_type)
    active_save_path = Path(cfg.DSPY_PROGRAM_DIR) / save_name
    saved_program_existed = active_save_path.exists()
    if cfg.TRAIN_WARM_START:
        warm_started, warm_start_status = load_program_if_present(program, active_save_path)
    else:
        warm_started = False
        warm_start_status = "warm-start disabled in config.py"

    before_signature = extract_program_instructions(program) or _signature_instructions(report_type)
    before_effective = effective_prompt_text(report_type, before_signature)

    print(f"\nTraining {report_type}; iteration {iteration}")
    print(" | ".join(f"{name}: {len(values)}" for name, values in splits.items()))
    print(warm_start_status)

    if layout is not None:
        save_split_debug(layout["debug"] / "split_examples.json", splits, redact_splits={"test"})
        (layout["prompts"] / "before_signature_prompt.txt").write_text(before_signature, encoding="utf-8")
        (layout["prompts"] / "before_effective_prompt.txt").write_text(before_effective, encoding="utf-8")
        save_program(program, layout["prompts"] / "before_program.json")

    if cfg.TRAIN_SMOKE_TEST:
        smoke_split = splits["dev"]
        smoke = evaluate_split(program, smoke_split[:1], f"{report_type} smoke")
        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "iteration": iteration,
            "report_type": report_type,
            "mode": "smoke_test",
            "candidate_accepted": False,
            "smoke": summarize_eval(smoke),
        }
        if layout is not None:
            save_predictions(layout["debug"] / "predictions.json", {"smoke": {"dev": smoke}})
            (layout["root"] / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    error_history = layout["debug"] / "history_on_errors.txt" if layout is not None else None
    baseline_results = evaluate_splits(program, eval_splits, f"{report_type} baseline", error_history)
    if layout is not None:
        append_history(layout["debug"] / "dspy_history.txt", "after baseline evaluation", int(cfg.TRAIN_HISTORY_SIZE))

    if cfg.TRAIN_BASELINE_ONLY:
        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "iteration": iteration,
            "report_type": report_type,
            "mode": "baseline_only",
            "candidate_accepted": False,
            "acceptance_reason": "TRAIN_BASELINE_ONLY is enabled",
            "saved_program_status": "saved program unchanged",
            "metrics": {"baseline": {name: summarize_eval(result) for name, result in baseline_results.items()}},
        }
        if layout is not None:
            save_predictions(layout["debug"] / "predictions.json", {"baseline": baseline_results})
            (layout["root"] / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            save_summary_text(layout["root"] / "accuracy_report.txt", summary)
        return summary

    optimizer_name, optimizer = build_optimizer(run_dir)
    print(
        f"{optimizer_name.upper()} will train on {len(splits['train'])} examples and "
        f"search on {len(splits['optimizer_val'])} optimizer-validation examples."
    )
    try:
        with quiet_optimizer_prompt_logging():
            candidate_program = compile_program(
                optimizer_name,
                optimizer,
                program,
                splits["train"],
                splits["optimizer_val"],
            )
    except Exception as exc:
        if layout is not None:
            append_history(layout["debug"] / "dspy_history.txt", "after optimizer compile error", int(cfg.TRAIN_HISTORY_SIZE))
            (layout["debug"] / "compile_error.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
            save_predictions(layout["debug"] / "predictions.json", {"baseline": baseline_results})
        raise RuntimeError(
            f"{optimizer_name.upper()} compile failed. Inspect the run folder's compile_error.txt and DSPy history."
        ) from exc

    if layout is not None:
        append_history(layout["debug"] / "dspy_history.txt", "after optimizer compile", int(cfg.TRAIN_HISTORY_SIZE))

    candidate_results = evaluate_splits(candidate_program, eval_splits, f"{report_type} candidate", error_history)
    if layout is not None:
        append_history(layout["debug"] / "dspy_history.txt", "after candidate evaluation", int(cfg.TRAIN_HISTORY_SIZE))

    candidate_signature = extract_program_instructions(candidate_program)
    candidate_effective = effective_prompt_text(report_type, candidate_signature)
    quality_ok, quality_reason = prompt_quality_ok(candidate_signature, report_type)

    acceptance_split = str(cfg.DSPY_ACCEPTANCE_SPLIT)
    baseline_accept = baseline_results[acceptance_split]
    candidate_accept = candidate_results[acceptance_split]
    accepted, acceptance_reason, promotion_delta, exact_non_regression = decide_promotion(
        baseline_accept,
        candidate_accept,
        split_name=acceptance_split,
        quality_ok=quality_ok,
        quality_reason=quality_reason,
    )

    active_program = candidate_program if accepted else program
    active_results = candidate_results if accepted else baseline_results
    active_save_path.parent.mkdir(parents=True, exist_ok=True)
    if accepted:
        save_program(candidate_program, active_save_path)
        saved_program_status = f"accepted candidate saved to {active_save_path}; next loop iteration will warm-start from it"
    elif not active_save_path.exists():
        save_program(program, active_save_path)
        saved_program_status = f"candidate rejected; baseline saved because no prior program existed: {active_save_path}"
    else:
        saved_program_status = f"candidate rejected; existing best program preserved: {active_save_path}"

    summary: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "report_type": report_type,
        "optimizer": optimizer_name,
        "candidate_accepted": accepted,
        "acceptance_reason": acceptance_reason,
        "prompt_quality_reason": quality_reason,
        "saved_program_status": saved_program_status,
        "active_saved_program": str(active_save_path),
        "saved_program_existed_before": saved_program_existed,
        "warm_started": warm_started,
        "warm_start_status": warm_start_status,
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "test_evaluated_this_iteration": evaluate_test,
        "search_metric_weights": {
            "exact": cfg.DSPY_SEARCH_EXACT_WEIGHT,
            "label_f1": cfg.DSPY_SEARCH_F1_WEIGHT,
        },
        "promotion_metric_weights": {
            "exact": cfg.DSPY_PROMOTION_EXACT_WEIGHT,
            "label_f1": cfg.DSPY_PROMOTION_F1_WEIGHT,
        },
        "acceptance_split": acceptance_split,
        "baseline_acceptance_exact": baseline_accept.accuracy,
        "candidate_acceptance_exact": candidate_accept.accuracy,
        "baseline_acceptance_promotion": baseline_accept.promotion_score,
        "candidate_acceptance_promotion": candidate_accept.promotion_score,
        "promotion_delta": promotion_delta,
        "candidate_signature_prompt_chars": len(candidate_signature),
        "candidate_effective_prompt_chars": len(candidate_effective),
        "fixed_rules_chars": len(_fixed_rules(report_type)),
        "metrics": {
            "baseline": {name: summarize_eval(result) for name, result in baseline_results.items()},
            "candidate": {name: summarize_eval(result) for name, result in candidate_results.items()},
            "active_after": {name: summarize_eval(result) for name, result in active_results.items()},
        },
    }

    candidate_prompt_path: Optional[Path] = None
    if layout is not None:
        candidate_prompt_path = layout["prompts"] / "candidate_effective_prompt.txt"
        save_predictions(
            layout["debug"] / "predictions.json",
            {"baseline": baseline_results, "candidate": candidate_results, "active_after": active_results},
        )
        (layout["prompts"] / "candidate_signature_prompt.txt").write_text(candidate_signature, encoding="utf-8")
        candidate_prompt_path.write_text(candidate_effective, encoding="utf-8")
        (layout["prompts"] / "fixed_rules.txt").write_text(_fixed_rules(report_type), encoding="utf-8")
        (layout["prompts"] / "after_signature_prompt.txt").write_text(
            candidate_signature if accepted else before_signature,
            encoding="utf-8",
        )
        (layout["prompts"] / "after_effective_prompt.txt").write_text(
            candidate_effective if accepted else before_effective,
            encoding="utf-8",
        )
        save_program(candidate_program, layout["prompts"] / "candidate_program.json")
        save_program(active_program, layout["prompts"] / "after_program.json")
        (layout["root"] / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        save_summary_text(layout["root"] / "accuracy_report.txt", summary)

    print(f"Candidate accepted: {accepted}. {acceptance_reason}")
    print(saved_program_status)
    if candidate_prompt_path is not None:
        print(f"Candidate prompt saved to: {candidate_prompt_path}")
    else:
        print("Candidate prompt was not saved because TRAIN_SAVE_RUN_LOGS is disabled.")
    if run_dir is not None:
        print(f"Run artifacts: {run_dir}")
    return summary


# =============================================================================
# Final untouched-test audit
# =============================================================================


def run_final_test_audit(iteration: int) -> Dict[str, Any]:
    report_type = str(cfg.TRAIN_REPORT_TYPE).upper().strip()
    disable_training_caches()
    configure_dspy()
    examples = load_examples(
        cfg.TRAINING_REPORTS_FILE,
        cfg.GROUND_TRUTH_FILE,
        report_type,
        cfg.TRAIN_MAX_CASES,
    )
    splits = split_examples(examples, seed=int(cfg.TRAIN_RANDOM_SEED))
    program, save_name = _program_for_report_type(report_type)
    save_path = Path(cfg.DSPY_PROGRAM_DIR) / save_name
    loaded, status = load_program_if_present(program, save_path)
    print(f"\nFinal untouched-test audit: {status}")
    result = evaluate_split(program, splits["test"], f"{report_type} final test")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "iteration_after": iteration,
        "report_type": report_type,
        "program_path": str(save_path),
        "loaded_saved_program": loaded,
        "load_status": status,
        "test": summarize_eval(result),
    }
    if cfg.TRAIN_SAVE_RUN_LOGS:
        run_dir = make_run_dir(report_type, iteration, suffix="_FINAL_AUDIT")
        layout = make_layout(run_dir)
        save_split_debug(layout["debug"] / "split_examples.json", splits)
        save_predictions(layout["debug"] / "predictions.json", {"final_audit": {"test": result}})
        (layout["root"] / "final_test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Final audit artifacts: {run_dir}")
    return summary


# =============================================================================
# Entry point: only --loop remains on the command line
# =============================================================================


def _reset_saved_program_if_requested() -> None:
    if not cfg.TRAIN_RESET_SAVED_PROGRAM_BEFORE_RUN:
        return
    report_type = str(cfg.TRAIN_REPORT_TYPE).upper().strip()
    _, save_name = _program_for_report_type(report_type)
    path = Path(cfg.DSPY_PROGRAM_DIR) / save_name
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        if backup.exists():
            backup.unlink()
        path.replace(backup)
        print(f"Moved existing optimized program to {backup}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the config-selected DSPy stroke labeler.")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Repeat optimization; each accepted prompt becomes the next iteration's starting program.",
    )
    args = parser.parse_args()

    validate_training_config()
    _reset_saved_program_if_requested()

    iteration = 1
    consecutive_rejections = 0
    interrupted = False
    try:
        while True:
            summary = train_one(
                iteration,
                evaluate_test=bool(cfg.TRAIN_EVALUATE_TEST_EACH_ITERATION),
            )
            accepted = bool(summary.get("candidate_accepted", False))
            consecutive_rejections = 0 if accepted else consecutive_rejections + 1

            if not args.loop or cfg.TRAIN_BASELINE_ONLY or cfg.TRAIN_SMOKE_TEST:
                break

            max_iterations = cfg.TRAIN_LOOP_MAX_ITERATIONS
            if max_iterations is not None and iteration >= int(max_iterations):
                print(f"Stopping loop at TRAIN_LOOP_MAX_ITERATIONS={max_iterations}.")
                break

            patience = cfg.TRAIN_LOOP_PATIENCE
            if patience is not None and consecutive_rejections >= int(patience):
                print(
                    f"Stopping loop after {consecutive_rejections} consecutive rejected candidates "
                    f"(TRAIN_LOOP_PATIENCE={patience})."
                )
                break
            iteration += 1
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped DSPy optimization loop.")

    if not cfg.TRAIN_BASELINE_ONLY and not cfg.TRAIN_SMOKE_TEST:
        try:
            run_final_test_audit(iteration)
        except Exception as exc:
            print(f"Final test audit could not run: {type(exc).__name__}: {exc}")
            if not interrupted:
                raise


if __name__ == "__main__":
    main()
