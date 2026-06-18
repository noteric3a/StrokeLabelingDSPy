# Refactor notes

## Accuracy changes

- Replaced random 70/15/15 splitting with deterministic multilabel-stratified `train`, `optimizer_val`, `dev`, and `test` splits.
- Added a dense search reward combining strict exact match and label F1.
- Added a separate exact-dominant promotion score.
- A candidate must improve promotion score and, by default, cannot reduce exact dev accuracy.
- Added optional GEPA feedback describing invalid formatting, missing labels, and extra labels.
- Enabled supervised reference labels in DSPy examples as output fields; the task model still receives only report text.
- Enabled up to three exact bootstrapped demonstrations for MIPROv2.
- Separated deterministic task-model temperature from exploratory prompt/reflection temperature.
- Changed the default MIPROv2 budget from `light` to `medium`.
- Added loop patience and a final untouched-test audit.

## CTA prompt changes

- Removed substantive CTA policy from the immutable rule input.
- `CTA_FIXED_RULES` now contains only allowed-label and output-format invariants.
- Vessel mapping and annotation-policy wording are optimizable.
- Added config switches for stenosis, chronic/stable occlusion, possible occlusion, parent ICA suppression, and cervical carotid handling.
- Candidate quality checks now inspect the generated signature itself rather than a long effective prompt that automatically included fixed rules.

## Interface and config cleanup

- `dspy_train.py` retains only `--loop`.
- `main.py` and `validate.py` are fully config-driven and have no argument parser.
- Removed compatibility aliases and duplicate config values.
- Moved deterministic review and converter implementation constants into their owning modules.
- Added `CONFIG_GUIDE.md`.
