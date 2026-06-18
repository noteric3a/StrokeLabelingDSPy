# Config-only operation

All runnable behavior is controlled from `config.py`. The normal workflow is:

```bash
python main.py
```

Set `RUN_MODE` before launching:

```python
RUN_MODE = "label"      # normal report labeling
RUN_MODE = "optimize"   # run/resume DSPy prompt optimization
RUN_MODE = "validate"   # validate an existing generated JSON file
```

The executable scripts reject command-line options. This prevents a saved experiment from silently differing from `config.py`.

## 1. Shared file and model settings

At minimum, set the report and ground-truth paths:

```python
INPUT_REPORTS_FILE = REPORTS_DIR / "reports.xlsx"
GROUND_TRUTH_FILE = GROUND_TRUTH_DIR / "ground_truth.xlsx"
MODEL_NAME = "qwen3.6:latest"
```

The optimizer inherits those paths by default:

```python
DSPY_OPTIMIZER_REPORTS_FILE = INPUT_REPORTS_FILE
DSPY_OPTIMIZER_GROUND_TRUTH_FILE = GROUND_TRUTH_FILE
```

They can point to different files when needed.

## 2. Manual-prompt baseline

Use the existing manually engineered prompts first:

```python
RUN_MODE = "label"
LABELING_BACKEND = "ollama"
USE_PROCESSING_CACHE = False
SKIP_VALIDATION = False
```

Run:

```bash
python main.py
```

A timestamped folder under `Files/finished_cases/` contains `report.json`. Copy that path into `DSPY_MANUAL_REPORT_FILE` before optimization to obtain same-validation-case deltas.

## 3. DSPy optimization

A typical configuration is:

```python
RUN_MODE = "optimize"

DSPY_OPTIMIZER_PROGRAMS = ("CT", "CTA", "CTP")
DSPY_TASK_MODEL = MODEL_NAME
DSPY_REFLECTION_MODEL = ""       # blank reuses DSPY_TASK_MODEL
DSPY_MANUAL_REPORT_FILE = FINISHED_CASES_ROOT / "<manual-run>" / "report.json"

DSPY_TARGET_ACCURACY = 1.0       # 1.0 or 100 means 100%
DSPY_MAX_ROUNDS = 0              # no predefined outer-round limit
DSPY_METRIC_CALLS_PER_ROUND = 64
DSPY_VALIDATION_FRACTION = 0.20
DSPY_TEST_FRACTION = 0.10
```

Edit the starting instructions directly in:

```python
DSPY_BASE_INSTRUCTIONS = {
    "CT": "...",
    "CTA": "...",
    "CTP": "...",
    "Combined": "...",
}
```

`DSPY_BASE_PROMPTS_FILE` is an optional JSON overlay. Leave it blank to make `config.py` the only prompt source. When set, file entries override matching entries in `DSPY_BASE_INSTRUCTIONS`.

Run or resume:

```bash
python main.py
```

The optimizer automatically resumes from `DSPY_OPTIMIZER_OUTPUT_DIR`. It preserves the fixed case split and previously promoted programs.

### Stop and resume

Press `Ctrl+C`, or create:

```text
Files/dspy_optimization/STOP
```

The best programs and state are retained. To remove the STOP file on the next launch:

```python
DSPY_CLEAR_STOP_FILE_ON_START = True
```

Set it back to `False` after the run starts. To archive the current optimizer directory and restart from the base instructions:

```python
DSPY_RESET_OPTIMIZATION = True
```

This is also intended as a one-launch switch.

## 4. Use optimized prompts in the production pipeline

```python
RUN_MODE = "label"
LABELING_BACKEND = "dspy"
DSPY_PROGRAM_DIR = DSPY_OPTIMIZATION_DIR / "best_programs"
DSPY_REQUIRE_OPTIMIZED_PROGRAMS = True
DSPY_USE_FOR_COMBINED = False
USE_PROCESSING_CACHE = False
```

`DSPY_USE_FOR_COMBINED = False` keeps the established manual Combined reconciliation prompt while DSPy replaces CT, CTA, and CTP. Set it to `True` only after optimizing `Combined` and producing `combined_program.json`.

## 5. Standalone validation

```python
RUN_MODE = "validate"
VALIDATION_GENERATED_JSON_FILE = FILES_DIR / "labeled_cases.json"
VALIDATION_GROUND_TRUTH_FILE = GROUND_TRUTH_DIR / "ground_truth.xlsx"
VALIDATION_TEXT_REPORT_FILE = FILES_DIR / "report.txt"
VALIDATION_JSON_REPORT_FILE = FILES_DIR / "report.json"
```

Then run `python main.py`. Direct `python validate.py` uses the same settings.

## 6. Direct entry points

These are equivalent, config-driven entry points:

```bash
python main.py
python dspy_optimize.py
python validate.py
```

`main.py` is recommended because `RUN_MODE` makes the active workflow explicit. `dspy_optimize.py` always runs optimization, and `validate.py` always runs validation.
