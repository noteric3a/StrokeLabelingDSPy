# DSPy integration changes

## Config-only execution update

- `config.py` is now the single control surface for normal labeling, DSPy optimization, and standalone validation.
- `RUN_MODE` selects `label`, `optimize`, or `validate` when running `python main.py`.
- Former labeling flags now map to normal config fields such as `INPUT_REPORTS_FILE`, `LABELING_BACKEND`, `USE_PROCESSING_CACHE`, and `DSPY_REQUIRE_OPTIMIZED_PROGRAMS`.
- Former optimizer flags now map to the `DSPY_*` optimization section, including models, split fractions, target accuracy, GEPA budget, prompt selection, resume/reset controls, manual-baseline report, and artifact behavior.
- The simple CT/CTA/CTP/Combined starting instructions are editable directly in `DSPY_BASE_INSTRUCTIONS`.
- `main.py`, `dspy_optimize.py`, and `validate.py` reject legacy command-line options and read `config.py` instead.
- `CONFIGURATION.md` documents baseline, optimization, resume, production, and validation setups.

## DSPy experiment files

- `dspy_optimize.py`: resumable GEPA loop with exact-match feedback, train/validation/test splits, candidate rejection, STOP-file handling, prompt/history artifacts, and manual-baseline comparison.
- `dspy_programs.py`: small CT/CTA/CTP/Combined DSPy programs, config-driven starting instructions, Ollama setup, and program persistence.
- `dspy_runtime.py`: lazy DSPy runtime bridge used by the existing async labeler.
- `dspy_base_prompts.json`: optional JSON overlay/example for starting instructions.
- `requirements-dspy.txt`: pinned optional DSPy dependencies.
- `tests/test_dspy_integration.py`: offline integration and config-routing tests.

## Preserved behavior

The manual prompt backend remains the default. CT sanitization, async pipelining, caching, confidence voting, Combined safety fallback, review checks, JSON validation, and Excel conversion remain intact. Combined remains manual unless `DSPY_USE_FOR_COMBINED` is enabled.

## Accuracy-integrity safeguards

- Train/validation/test percentages are calculated against the full case set.
- Manual baseline deltas use identical validation case IDs when per-case details are available.
- Invalid ground-truth labels, `NONE` mixed with positive labels, and duplicate case IDs fail before optimization.
- Candidates replace the current best program only after independent evaluation.
- Nonzero-temperature confidence samples bypass DSPy caching.
- DSPy/Ollama calls disable model thinking to keep structured JSON clean.
- The DSPy production backend performs dependency/program preflight checks.
- Combined optimization requires explicit Combined ground truth.
