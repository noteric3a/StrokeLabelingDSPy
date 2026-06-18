"""Central, config-only settings for the stroke labeler and DSPy experiment.

Edit this file, then run ``python main.py``. ``RUN_MODE`` selects normal
labeling, DSPy prompt optimization, or standalone validation. The runnable
scripts intentionally do not require command-line arguments, so one saved
configuration describes the entire experiment.

Expected layout::

    Project/
    ├── Code/
    │   ├── main.py
    │   ├── config.py
    │   └── ...
    └── Files/
        ├── Reports/
        ├── GroundTruth/
        ├── finished_cases/
        └── dspy_optimization/
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

# "label"    -> run the report-labeling pipeline
# "optimize" -> run/resume the DSPy GEPA prompt-optimization loop
# "validate" -> validate an existing generated JSON file
RUN_MODE = "label"


# ---------------------------------------------------------------------------
# Project folders
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
FILES_DIR = PROJECT_ROOT / "Files"
FILES_DIR_DEBUG = FILES_DIR / "Debug"
FINISHED_CASES_ROOT = FILES_DIR / "finished_cases"
GROUND_TRUTH_DIR = FILES_DIR / "GroundTruth"
REPORTS_DIR = FILES_DIR / "Reports"
DSPY_OPTIMIZATION_DIR = FILES_DIR / "dspy_optimization"
DSPY_PROGRAM_DIR = DSPY_OPTIMIZATION_DIR / "best_programs"

for directory in (
    FILES_DIR,
    FILES_DIR_DEBUG,
    FINISHED_CASES_ROOT,
    GROUND_TRUTH_DIR,
    REPORTS_DIR,
    DSPY_OPTIMIZATION_DIR,
    DSPY_PROGRAM_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared Ollama/model settings
# ---------------------------------------------------------------------------

MODEL_NAME = "qwen3.6:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_CONCURRENT_REQUESTS = 4
REQUEST_TIMEOUT_SECONDS = 600
NUM_PREDICT = 1670
NUM_CTX = 20000


# ---------------------------------------------------------------------------
# Normal labeling mode (RUN_MODE = "label")
# ---------------------------------------------------------------------------

INPUT_REPORTS_FILE: str | Path = REPORTS_DIR / "flagged_not_confident_cases_retest_input.xlsx"
# Example: GROUND_TRUTH_FILE = GROUND_TRUTH_DIR / "stroke_ground_truth.xlsx"
# Blank is allowed for ungraded labeling; optimization requires a real file.
GROUND_TRUTH_FILE: str | Path = ""
OUTPUT_JSON_FILE: str | Path = FILES_DIR / "labeled_cases.json"

# False runs answer validation after labeling. True skips it.
SKIP_VALIDATION = False

# True uses the resumable processing cache. False reprocesses every case.
USE_PROCESSING_CACHE = True

# "ollama" uses the manually engineered prompts in prompts.py.
# "dspy" uses saved programs from DSPY_PROGRAM_DIR.
LABELING_BACKEND = "ollama"

# DSPy runtime settings used when LABELING_BACKEND = "dspy".
DSPY_OLLAMA_API_BASE = ""  # Blank derives the server root from OLLAMA_URL.
DSPY_ADAPTER = "json"  # "json" or "chat"
DSPY_LM_CACHE = True  # Runtime labeling cache only. The optimizer always disables LM caching.
DSPY_REQUIRE_OPTIMIZED_PROGRAMS = False
# False keeps the established manual Combined prompt. True also loads
# combined_program.json from DSPY_PROGRAM_DIR.
DSPY_USE_FOR_COMBINED = False
DSPY_MAX_TOKENS = NUM_PREDICT
DSPY_NUM_CTX = NUM_CTX
DSPY_RUNTIME_LOG = FILES_DIR_DEBUG / "dspy_runtime_outputs.jsonl"

# Include or omit *_reasoning fields in labeled_cases.json and Excel output.
INCLUDE_REASONING_IN_JSON = True

# Repeated-sample confidence voting.
ENABLE_CONFIDENCE_CHECKING = False
CONFIDENCE_RUNS = 10
CONFIDENCE_TEMPERATURE = 0.2
# Blank uses confidence.py's default. Values below 51 are raised to 51.
MIN_CONFIDENCE_PERCENTAGE = "67"

# Prevent Combined_GT from erasing positive modality evidence by returning NONE.
ENFORCE_COMBINED_POSITIVE_FALLBACK = True


# ---------------------------------------------------------------------------
# DSPy optimization mode (RUN_MODE = "optimize")
# ---------------------------------------------------------------------------

# Reports and exact ground truths used by the optimizer.
DSPY_OPTIMIZER_REPORTS_FILE: str | Path = INPUT_REPORTS_FILE
DSPY_OPTIMIZER_GROUND_TRUTH_FILE: str | Path = GROUND_TRUTH_FILE
DSPY_OPTIMIZER_OUTPUT_DIR: str | Path = DSPY_OPTIMIZATION_DIR

# Simple starting instructions. GEPA rewrites these during optimization. These
# are also the runtime fallback when a compiled program is not required/found.
DSPY_BASE_INSTRUCTIONS = {
    "CT": (
        "Label the acute or recent ischemic stroke territory in one non-contrast CT brain report. "
        "Use only the report, return the smallest directly supported label set, and return NONE when "
        "there is no qualifying acute/recent CT finding. Ignore chronic/old findings and any CTA/CTP "
        "content mixed into the report. Map side and named anatomy literally; do not infer extra territories."
    ),
    "CTA": (
        "Label qualifying acute vessel abnormalities in one CTA report. Definite occlusion, thrombus, "
        "filling defect, flow cutoff, non-opacification, or near-occlusion can qualify. Stenosis, plaque, "
        "variant anatomy, uncertainty, and stable/chronic disease alone do not. Use only the report and "
        "return the smallest exact label set, or NONE."
    ),
    "CTP": (
        "Label the acute stroke territory in one CT-perfusion report. Use localized core, hypoperfusion, "
        "Tmax delay, mismatch, penumbra, or tissue-at-risk evidence tied to a side and territory. Ignore "
        "artifact, chronic collateral/moyamoya patterns, and tiny nonspecific abnormalities. Use only the "
        "report and return the smallest exact label set, or NONE."
    ),
    "Combined": (
        "Reconcile preliminary CT, CTA, and CTP labels with the supplied reports and MRI. Keep only labels "
        "with direct acute/recent evidence, add an MRI label only for a direct acute/recent territorial "
        "finding, and do not invent labels from mechanism or broad anatomy. Return the smallest exact final "
        "label set, or NONE."
    ),
}

# Optional JSON overlay for starting instructions. Leave blank to use only the
# dictionary above. A file entry overrides the matching dictionary entry.
DSPY_BASE_PROMPTS_FILE: str | Path = ""

# May be a tuple/list or a comma-separated string. Combined requires a Combined
# ground-truth column and uses modality ground truths as training inputs.
DSPY_OPTIMIZER_PROGRAMS = ("CT", "CTA", "CTP")

DSPY_TASK_MODEL = MODEL_NAME
# Blank reuses DSPY_TASK_MODEL.
DSPY_REFLECTION_MODEL = ""
DSPY_OPTIMIZER_API_BASE = DSPY_OLLAMA_API_BASE
DSPY_OPTIMIZER_ADAPTER = DSPY_ADAPTER
DSPY_OPTIMIZER_MAX_TOKENS = DSPY_MAX_TOKENS
DSPY_REFLECTION_MAX_TOKENS = max(4000, DSPY_MAX_TOKENS)
DSPY_OPTIMIZER_NUM_CTX = DSPY_NUM_CTX
DSPY_OPTIMIZER_TIMEOUT_SECONDS = REQUEST_TIMEOUT_SECONDS
DSPY_OPTIMIZER_NUM_RETRIES = 2
# Optimizer LM response caching is always disabled in code.

# Fixed case-level split. Set both fractions to 0 only for a deliberate
# fit-the-entire-sheet experiment; that does not measure generalization.
DSPY_VALIDATION_FRACTION = 0.20
DSPY_TEST_FRACTION = 0.10
DSPY_RANDOM_SEED = 42

# 1.0 and 100 both mean 100%. 0 rounds means no predefined outer-round limit.
DSPY_TARGET_ACCURACY = 1.0
DSPY_MAX_ROUNDS = 0

# GEPA search/evaluation controls.
DSPY_METRIC_CALLS_PER_ROUND = 64
DSPY_REFLECTION_MINIBATCH_SIZE = 3
DSPY_REFLECTION_TEMPERATURE = 1.0
DSPY_OPTIMIZER_NUM_THREADS = 1
DSPY_SELECTION_STRATEGY = "worst"  # "worst" or "round-robin"
DSPY_MAX_CONSECUTIVE_ERRORS = 3
# Promote exact-accuracy ties only when label-level F1 improves.
DSPY_PROMOTE_F1_TIES = True
DSPY_TRACK_GEPA_STATS = False

# Optional report.json from a manual-prompt run for same-validation-case deltas.
DSPY_MANUAL_REPORT_FILE: str | Path = ""

# Evaluation/artifact controls.
DSPY_EVALUATE_ONLY = False
DSPY_EVALUATE_TEST_EACH_ROUND = False
DSPY_INCLUDE_FULL_REPORTS = False
DSPY_REPORT_EXCERPT_CHARS = 600

# One-launch switches. RESET archives the current optimizer directory before
# starting. CLEAR_STOP removes an existing STOP file before resuming.
DSPY_RESET_OPTIMIZATION = False
DSPY_CLEAR_STOP_FILE_ON_START = False


# ---------------------------------------------------------------------------
# Standalone validation mode (RUN_MODE = "validate")
# ---------------------------------------------------------------------------

VALIDATION_GENERATED_JSON_FILE: str | Path = OUTPUT_JSON_FILE
VALIDATION_GROUND_TRUTH_FILE: str | Path = GROUND_TRUTH_FILE
VALIDATION_TEXT_REPORT_FILE: str | Path = FILES_DIR / "report.txt"
VALIDATION_JSON_REPORT_FILE: str | Path = FILES_DIR / "report.json"


# ---------------------------------------------------------------------------
# Log/generated files (main.py redirects these into each labeling run)
# ---------------------------------------------------------------------------

RAW_OUTPUT_LOG = FILES_DIR_DEBUG / "raw_ollama_outputs_async.txt"
OLLAMA_WRAPPER_LOG = FILES_DIR_DEBUG / "ollama_outputs_async.txt"
BAD_JSON_LOG = FILES_DIR_DEBUG / "bad_json_outputs_async.txt"
FAILED_CASES_LOG = FILES_DIR_DEBUG / "failed_cases_async.txt"


# ---------------------------------------------------------------------------
# Allowed output labels
# ---------------------------------------------------------------------------

ALLOWED_LABELS = [
    "NONE",
    "RMCA", "LMCA",
    "RACA", "LACA",
    "RPCA", "LPCA",
    "RPICA", "LPICA",
    "BA",
    "RVA", "LVA",
    "RICA", "LICA",
]
