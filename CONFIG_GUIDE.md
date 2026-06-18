# DSPy configuration guide

All run controls now live in `config.py`.

## Commands

Run one optimization pass:

```bash
python dspy_train.py
```

Run repeated improvement passes:

```bash
python dspy_train.py --loop
```

Run the labeling pipeline:

```bash
python main.py
```

Run validation directly with the configured paths:

```bash
python validate.py
```

`--loop` is the only retained command-line option in the repository.

## Most important training settings

```python
TRAIN_REPORT_TYPE = "CTA"
TRAINING_REPORTS_FILE = "Files/Report/New Reports.xlsx"
GROUND_TRUTH_FILE = "Files/GT/GroundTruthKeyNew.xlsx"
TRAIN_MAX_CASES = None
```

Use `TRAIN_MAX_CASES` only for quick debugging. Leave it as `None` for real optimization.

The split roles are independent:

```python
TRAIN_SPLIT_RATIOS = {
    "train": 0.50,
    "optimizer_val": 0.20,
    "dev": 0.15,
    "test": 0.15,
}
```

- `train` supplies supervised labels and bootstrapped-demo candidates.
- `optimizer_val` is where MIPROv2 or GEPA searches among prompt candidates.
- `dev` decides whether a candidate replaces the saved best program.
- `test` is evaluated only after optimization finishes unless `TRAIN_EVALUATE_TEST_EACH_ITERATION` is explicitly enabled.

## Reward and promotion

DSPy searches with a dense metric:

```python
DSPY_SEARCH_EXACT_WEIGHT = 0.25
DSPY_SEARCH_F1_WEIGHT = 0.75
```

This rewards partial progress on multi-label cases instead of giving every non-exact answer a zero.

Saved-program promotion remains exact-accuracy oriented:

```python
DSPY_PROMOTION_EXACT_WEIGHT = 0.85
DSPY_PROMOTION_F1_WEIGHT = 0.15
DSPY_REQUIRE_EXACT_NON_REGRESSION = True
DSPY_MIN_PROMOTION_IMPROVEMENT = 0.001
```

A candidate can advance when exact accuracy ties but label F1 improves. It cannot replace the current best program when exact accuracy falls.

## MIPROv2 and GEPA

Compatibility-first default:

```python
DSPY_OPTIMIZER = "mipro"
DSPY_MIPRO_AUTO = "medium"
DSPY_MIPRO_MAX_BOOTSTRAPPED_DEMOS = 3
DSPY_MIPRO_MAX_LABELED_DEMOS = 0
```

Bootstrapped demos are kept only when their search score is perfect:

```python
DSPY_MIPRO_METRIC_THRESHOLD = 1.0
```

To use textual missing/extra-label feedback instead:

```python
DSPY_OPTIMIZER = "gepa"
```

GEPA requires a DSPy installation that includes the GEPA dependency. The code reports an actionable installation error when it is unavailable.

## Separate task and prompt temperatures

```python
DSPY_TASK_TEMPERATURE = 0.0
DSPY_PROMPT_TEMPERATURE = 0.8
```

The classifier is deterministic. The prompt/reflection model remains exploratory.

Set a stronger prompt model without changing the deployed task model:

```python
DSPY_PROMPT_MODEL = "ollama_chat/<your-stronger-instruction-model>"
```

Leaving it as `None` reuses `DSPY_MODEL` with the separate prompt temperature.

## CTA policy switches

These switches build the initial optimizable CTA instruction:

```python
CTA_COUNT_SEVERE_STENOSIS = False
CTA_COUNT_CHRONIC_OR_STABLE_OCCLUSION = True
CTA_COUNT_POSSIBLE_OCCLUSION = True
CTA_INCLUDE_PARENT_ICA_WITH_DOWNSTREAM = False
CTA_INCLUDE_CERVICAL_CAROTID = True
```

The defaults reflect patterns inferred from the supplied optimization runs. They are annotation-policy settings, not universal clinical rules. Reconcile them with the adjudicated ground-truth protocol.

Only immutable output constraints remain in `CTA_FIXED_RULES`. Vessel mapping and annotation policy remain in `CTA_SIGNATURE_INSTRUCTIONS`, so DSPy can improve them.

## Loop behavior

```python
TRAIN_WARM_START = True
TRAIN_LOOP_PATIENCE = 20
TRAIN_LOOP_MAX_ITERATIONS = None
```

An accepted candidate becomes the next iteration's starting program. Rejected candidates never overwrite the saved best program. The loop stops after the configured number of consecutive rejections or the optional hard iteration cap.

To deliberately start over once:

```python
TRAIN_RESET_SAVED_PROGRAM_BEFORE_RUN = True
```

The old program is moved to a `.bak` file before optimization begins. Set this back to `False` after the reset.

## Debug modes

```python
TRAIN_SMOKE_TEST = False
TRAIN_BASELINE_ONLY = False
TRAIN_SAVE_RUN_LOGS = True
TRAIN_HISTORY_SIZE = 50
```

Only enable one debug mode at a time. Run folders include split composition, candidate predictions, prompt text, saved candidate programs, and score components.
