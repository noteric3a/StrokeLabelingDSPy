# Modular Stroke Labeler

This version separates the original single script into smaller files and keeps runtime defaults in `config.py`.

## File layout

- `main.py` — entry point; command-line flags are optional overrides for `config.py`
- `config.py` — file paths, model settings, run controls, cache/validation controls, DSPy optimization settings, allowed labels
- `schemas.py` — Ollama JSON schemas
- `prompts.py` — CT, CTA, CTP, and Combined prompt builders
- `dspy_prompt_optimizer.py` — optional DSPy optimizer that writes `optimized_prompts.json`
- `ollama_client.py` — Ollama API calls
- `labeler.py` — async labeling workflow
- `validate.py` — scoring against ground truth
- `utils.py` — shared helper functions

## Configure first

Edit `config.py` for normal use. The main settings are:

```python
INPUT_REPORTS_FILE = REPORTS_DIR / "your_reports.xlsx"
GROUND_TRUTH_FILE = GROUND_TRUTH_DIR / "your_ground_truth.xlsx"
OUTPUT_JSON_FILE = FILES_DIR / "labeled_cases.json"

MODEL_NAME = "qwen3.6:latest"
MAX_CONCURRENT_REQUESTS = 4

RUN_PROMPT_OPTIMIZATION = False
RUN_VALIDATION_AFTER_LABELING = True
USE_PROCESSING_CACHE = True
INCLUDE_REASONING_IN_JSON = True

USE_OPTIMIZED_PROMPTS = True
OPTIMIZED_PROMPTS_FILE = FILES_DIR / "optimized_prompts.json"
DSPY_MODALITIES = ["CT", "CTA", "CTP", "COMBINED"]
DSPY_OPTIMIZER = "MIPROv2"
DSPY_AUTO = "light"
DSPY_TRAIN_LIMIT = 60
DSPY_VAL_LIMIT = 30
```

Then run from config defaults:

```bash
python main.py
```

## Optional one-run terminal overrides

All important settings have `config.py` defaults. CLI flags only override those defaults for a single run:

```bash
python main.py \
  --input Files/Reports/your_reports.xlsx \
  --ground-truth Files/GroundTruth/your_ground_truth.xlsx \
  --output labeled_cases.json \
  --model qwen3.6:latest \
  --concurrency 4
```

Useful boolean override pairs:

```bash
python main.py --run-validation
python main.py --skip-validation
python main.py --use-cache
python main.py --skip-cache
python main.py --include-reasoning
python main.py --no-reasoning
python main.py --use-optimized-prompts
python main.py --no-optimized-prompts
```

## DSPy prompt optimization

You can run DSPy optimization entirely from `config.py`:

```python
RUN_PROMPT_OPTIMIZATION = True
INPUT_REPORTS_FILE = REPORTS_DIR / "your_reports.xlsx"
GROUND_TRUTH_FILE = GROUND_TRUTH_DIR / "your_ground_truth.xlsx"
OPTIMIZED_PROMPTS_FILE = FILES_DIR / "optimized_prompts.json"
DSPY_MODALITIES = ["CT", "CTA", "CTP", "COMBINED"]
DSPY_OPTIMIZER = "MIPROv2"
DSPY_AUTO = "light"
DSPY_TRAIN_LIMIT = 60
DSPY_VAL_LIMIT = 30
```

Then run:

```bash
python main.py
```

Or override just that mode from the terminal:

```bash
python main.py --optimize-prompts
```

After optimization, set:

```python
RUN_PROMPT_OPTIMIZATION = False
USE_OPTIMIZED_PROMPTS = True
```

Then normal labeling runs will automatically insert the saved optimized guidance from `OPTIMIZED_PROMPTS_FILE`.

## Important

The prompt and JSON schema both use the labels from `config.py`, so they stay aligned.
