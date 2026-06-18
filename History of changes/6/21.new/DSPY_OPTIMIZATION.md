# DSPy prompt-optimization experiment

This project compares two paths on the same CT, CTA, and CTP report-labeling task:

- The established manually engineered prompts in `prompts.py`.
- Small DSPy programs whose instructions are repeatedly evolved by GEPA from exact report/ground-truth errors.

The optimization loop is:

```text
current best instruction
    -> run reports through the task model
    -> calculate exact label-set accuracy and mismatch feedback
    -> let GEPA propose a revised instruction
    -> evaluate the candidate independently
    -> promote only if it improves the configured metric
    -> repeat until the target or a stop condition
```

All experiment controls now live in `config.py`. No run arguments are needed.

## Install DSPy

From the code directory:

```bash
python -m pip install -r requirements-dspy.txt
```

The integration is pinned to DSPy 3.2.1. Ollama must be running and the configured local model must be available.

## Run the manual baseline

Set:

```python
RUN_MODE = "label"
LABELING_BACKEND = "ollama"
INPUT_REPORTS_FILE = REPORTS_DIR / "your_reports.xlsx"
GROUND_TRUTH_FILE = GROUND_TRUTH_DIR / "your_ground_truth.xlsx"
USE_PROCESSING_CACHE = False
SKIP_VALIDATION = False
```

Then run:

```bash
python main.py
```

Keep the generated `Files/finished_cases/<run>/report.json`. The optimizer can filter that report to the exact validation case IDs before reporting manual-versus-DSPy deltas.

## Configure the optimizer

Set:

```python
RUN_MODE = "optimize"

DSPY_OPTIMIZER_REPORTS_FILE = INPUT_REPORTS_FILE
DSPY_OPTIMIZER_GROUND_TRUTH_FILE = GROUND_TRUTH_FILE
DSPY_OPTIMIZER_OUTPUT_DIR = DSPY_OPTIMIZATION_DIR
DSPY_OPTIMIZER_PROGRAMS = ("CT", "CTA", "CTP")

DSPY_TASK_MODEL = MODEL_NAME
DSPY_REFLECTION_MODEL = ""  # blank uses the task model
DSPY_MANUAL_REPORT_FILE = FINISHED_CASES_ROOT / "<manual-run>" / "report.json"

DSPY_TARGET_ACCURACY = 1.0
DSPY_MAX_ROUNDS = 0
DSPY_METRIC_CALLS_PER_ROUND = 64
DSPY_REFLECTION_MINIBATCH_SIZE = 3
DSPY_SELECTION_STRATEGY = "worst"
```

`DSPY_MAX_ROUNDS = 0` means there is no predefined outer-round limit. The process ends when every selected program reaches the configured validation target, the user stops it, the STOP file is detected, or repeated runtime failures trigger a safe stop.

The simple starting prompts are the `DSPY_BASE_INSTRUCTIONS` dictionary in `config.py`. `DSPY_BASE_PROMPTS_FILE` may optionally name a JSON overlay, but it can remain blank.

Run:

```bash
python main.py
```

Direct `python dspy_optimize.py` reads the same settings.

## Data splitting and promotion

The defaults reserve approximately 70% of cases for GEPA training, 20% for validation, and 10% for held-out testing. Splitting is by case ID. Candidate promotion and the target stop condition use validation data; the test set does not select prompts.

For a deliberate fit-the-entire-sheet experiment:

```python
DSPY_VALIDATION_FRACTION = 0.0
DSPY_TEST_FRACTION = 0.0
```

That measures whether DSPy can fit the supplied sheet, not whether the evolved instruction generalizes.

Exact label-set accuracy is the primary metric. An exact-accuracy tie may be promoted when label-level F1 improves if:

```python
DSPY_PROMOTE_F1_TIES = True
```

## Stop, resume, reset, or evaluate only

Press `Ctrl+C`, or create:

```text
Files/dspy_optimization/STOP
```

The current best programs, fixed split, round history, and state are saved. To clear the STOP file at the next launch:

```python
DSPY_CLEAR_STOP_FILE_ON_START = True
```

To archive the current optimizer output and restart from the configured base prompts:

```python
DSPY_RESET_OPTIMIZATION = True
```

To score the saved/base programs without running GEPA:

```python
DSPY_EVALUATE_ONLY = True
```

Treat reset and clear-stop as one-launch switches and return them to `False` afterward.

## Use the optimized programs

Set:

```python
RUN_MODE = "label"
LABELING_BACKEND = "dspy"
DSPY_PROGRAM_DIR = DSPY_OPTIMIZATION_DIR / "best_programs"
DSPY_REQUIRE_OPTIMIZED_PROGRAMS = True
DSPY_USE_FOR_COMBINED = False
USE_PROCESSING_CACHE = False
```

Then run `python main.py`.

Keeping `DSPY_USE_FOR_COMBINED = False` isolates the requested comparison: optimized CT/CTA/CTP instructions with the existing manual Combined step. To optimize Combined too, add it to `DSPY_OPTIMIZER_PROGRAMS`; the ground-truth file must contain a Combined ground-truth column. Combined training uses modality ground truths as preliminary inputs, so a complete end-to-end labeling run remains the definitive Combined evaluation.

## Optimizer artifacts

`DSPY_OPTIMIZER_OUTPUT_DIR` contains:

```text
best_programs/
    ct_program.json
    cta_program.json
    ctp_program.json
    combined_program.json   # only when Combined is selected
best_prompts.json
state.json
latest_report.json
latest_report.txt
history.jsonl
rounds/
gepa_logs/
```

`state.json` preserves the fixed split and resume state. `best_prompts.json` provides plain-text instructions for direct comparison with the manual prompt. Every round records the candidate, mismatch summary, and promotion decision.

Mismatch artifacts contain report excerpts by default. Full reports can be included with `DSPY_INCLUDE_FULL_REPORTS = True`. Because optimizer traces may contain report text and ground-truth feedback, treat the entire output directory as potentially containing protected health information.

## Interpreting the experiment

A validation score of 100% means the evolved prompt fit that validation split. It does not by itself establish clinical reliability or generalization. Compare the untouched test results and then evaluate the complete pipeline on a separate external dataset.

By default, both the task and reflection models use local Ollama. Changing the API base or reflection model to a hosted service can transmit report text and ground-truth feedback outside the local environment.

## Verification

Run:

```bash
python -m unittest discover -s tests -v
```

The tests cover config resolution, program persistence, exact-match feedback, GEPA instruction mutation, strict ground-truth validation, case-level split integrity, manual-baseline filtering, a mock local Ollama structured-output exchange, backend routing, and manual Combined behavior.
