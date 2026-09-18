# Stroke Report Labeling with DSPy

A research pipeline for converting radiology report text into structured stroke-related labels. The project separates configuration, model calls, labeling, validation, and experiment utilities so prompt-development work can be inspected independently from application orchestration.

**Stack:** Python · DSPy · Ollama integration · pandas / Excel reporting

## Workflow

```text
Report workbook ──> case/column mapping ──> asynchronous labeling
                                               |
                                structured JSON + run metadata
                                               |
                             optional ground-truth validation
```

The current entry point uses settings from `Code/config.py`, reads report workbooks, labels cases asynchronously, writes structured outputs, and conditionally scores against ground truth. Timestamped output folders and model/concurrency metadata support run traceability.

## Repository map

| Path | Purpose |
| --- | --- |
| `Code/main.py` | Config-driven labeling entry point |
| `Code/config.py` | Input/output paths, model, labels, and run settings |
| `Code/labeler.py` | Labeling orchestration |
| `Code/dspy_programs.py`, `Code/dspy_train.py` | DSPy program and training utilities |
| `Code/ollama_client.py` | Model-service integration |
| `Code/validate.py` | Ground-truth comparison |
| `Code/confidence.py`, `Code/review_checks.py` | Additional output-review logic |
| `Files/` | Research input/workbook locations; review publication permission |

## Setup and execution

Create an isolated Python environment and install the dependencies:

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
```

Review `Code/config.py` before running. Set `INPUT_REPORT_FILE`, `GROUND_TRUTH_FILE`, output locations, model-service configuration, and concurrency to match authorized local data and the available host. Ensure the configured model service is running and the selected model is available.

From the repository root:

```bash
python Code/main.py
```

The current entry point is configuration-driven. Older examples showing a root `main.py` or CLI overrides are not the current interface. The previous README is retained unchanged in [LEGACY_USAGE.md](LEGACY_USAGE.md) for historical reference, not as an authoritative run guide.

## Evaluation and limitations

Prompt optimization and evaluation on the same cases measure development-set fit, not generalization. Keep final held-out evaluation separate, account for repeated patients, document label/scoring definitions, and record the model revision and prompt version with each reported result.

No model run, clinical evaluation, or performance benchmark was executed during this documentation cleanup. No accuracy improvement or clinical validity is claimed. Existing validation utilities are not a substitute for independent evaluation.

## Research-data handling

This is research software, not a validated clinical decision system. The current repository includes workbooks under `Files/GT/` and `Files/Report/`; their presence does not establish public-release authorization or de-identification. Review their contents, permission, derived outputs, and Git history before further distribution. Public demonstrations should use synthetic inputs and authorized aggregate results.

See [SECURITY.md](SECURITY.md). Source code and research workbooks are unchanged by this presentation update; only a generated OS metadata file is removed.
