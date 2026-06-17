"""
Central settings for the stroke labeler.
This assumes this folder structure:

Project/
├── Code/
│   ├── main.py
│   ├── config.py
│   └── ...
└── Files/
    ├── Filtered_Reports.xlsx
    ├── Radiology_Reports_GroundTruth_Subset.xlsx
    └── generated output files
"""

from pathlib import Path

# Folder paths
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
FILES_DIR = PROJECT_ROOT / "Files"
FILES_DIR_DEBUG = PROJECT_ROOT / "Files" / "Debug"
FINISHED_CASES_ROOT = FILES_DIR / "Finished Cases"
GROUND_TRUTH_DIR = FILES_DIR / "GT"
REPORTS_DIR = FILES_DIR / "Report"

# Make sure all directories exist
FILES_DIR.mkdir(exist_ok=True)
FILES_DIR_DEBUG.mkdir(exist_ok=True)
FINISHED_CASES_ROOT.mkdir(exist_ok=True)
GROUND_TRUTH_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "qwen3.6:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Input/output files
INPUT_REPORTS_FILE = REPORTS_DIR / "New Reports.xlsx"
GROUND_TRUTH_FILE = GROUND_TRUTH_DIR / "GroundTruthKeyNew.xlsx"
OUTPUT_JSON_FILE = FILES_DIR / "labeled_cases.json"

# ---------------------------------------------------------------------------
# Run controls
# ---------------------------------------------------------------------------
# These config.py values are the defaults for normal runs. CLI flags in main.py
# are only one-run overrides. You can run entirely from config.py by editing
# these values and calling `python main.py`.

# False: run the normal labeler pipeline.
# True: run DSPy prompt optimization using the INPUT_REPORTS_FILE and
# GROUND_TRUTH_FILE below, write OPTIMIZED_PROMPTS_FILE, then exit.
RUN_PROMPT_OPTIMIZATION = True

# True: compare the generated JSON to GROUND_TRUTH_FILE after labeling.
# False: label and convert output without validation.
RUN_VALIDATION_AFTER_LABELING = True

# True: use the resumable ProcessingCache.
# False: ignore the cache and reprocess all rows.
USE_PROCESSING_CACHE = False

# True: include *_reasoning fields in labeled_cases.json and the converted Excel file.
# False: omit reasoning text from the final output JSON/Excel while still keeping
# debug logs available in Files/finished_cases/<run>/Debug/.
INCLUDE_REASONING_IN_JSON = True


# ---------------------------------------------------------------------------
# DSPy prompt-optimization controls
# ---------------------------------------------------------------------------
# Normal labeling runs automatically insert the saved DSPy-optimized guidance
# into the hand-written prompt templates when this flag is True and the file
# exists.
USE_OPTIMIZED_PROMPTS = True
OPTIMIZED_PROMPTS_FILE = FILES_DIR / "optimized_prompts.json"
# Alias used by the CLI help text and optimizer; keep the same value unless you
# intentionally want a separate output path.
DSPY_OPTIMIZED_PROMPTS_OUTPUT_FILE = OPTIMIZED_PROMPTS_FILE

# Modalities to optimize when RUN_PROMPT_OPTIMIZATION=True or --optimize-prompts
# is used. Valid values: CT, CTA, CTP, COMBINED.
DSPY_MODALITIES = ["CT", "CTA", "CTP", "COMBINED"]

# DSPy uses LiteLLM provider/model strings. For local Ollama, the default
# `ollama_chat/<model>` works with the chat endpoint exposed by Ollama.
DSPY_LM = f"ollama_chat/{MODEL_NAME}"
DSPY_API_BASE = "http://localhost:11434"
DSPY_OPTIMIZER = "MIPROv2"  # MIPROv2, BootstrapFewShot, or LabeledFewShot
DSPY_AUTO = "light"          # MIPROv2 search budget: light, medium, heavy
DSPY_MAX_LABELED_DEMOS = 4
DSPY_MAX_BOOTSTRAPPED_DEMOS = 2
DSPY_NUM_THREADS = 1
DSPY_SEED = 9
DSPY_TRAIN_LIMIT = 60
DSPY_VAL_LIMIT = 30


# Confidence checking controls
# False: normal single deterministic run at temperature 0.
# True: run each label prompt CONFIDENCE_RUNS times at CONFIDENCE_TEMPERATURE,
# then use the most common answer as the final label and write possible answers
# with vote percentages into the JSON.
ENABLE_CONFIDENCE_CHECKING = False
CONFIDENCE_RUNS = 10
CONFIDENCE_TEMPERATURE = 0.2

# Minimum vote percentage required for *_is_confident = True.
# Leave blank ("") to use the default value from confidence.py.
# Values below 51 are automatically raised to 51 so the winner must be a majority.
MIN_CONFIDENCE_PERCENTAGE = "67"

# Combined safety controls
# True: if Combined_GT returns ["NONE"] even though CT_GT, CTA_GT, or CTP_GT
# has a positive territory label, replace Combined_GT with the union of positive
# modality labels and mark the case for review. This prevents the Combined step
# from accidentally erasing already-established modality evidence.
ENFORCE_COMBINED_POSITIVE_FALLBACK = True

# Log/generated files
RAW_OUTPUT_LOG = FILES_DIR_DEBUG / "raw_ollama_outputs_async.txt"
OLLAMA_WRAPPER_LOG = FILES_DIR_DEBUG / "ollama_outputs_async.txt"
BAD_JSON_LOG = FILES_DIR_DEBUG / "bad_json_outputs_async.txt"
FAILED_CASES_LOG = FILES_DIR_DEBUG / "failed_cases_async.txt"

# Ollama settings
MAX_CONCURRENT_REQUESTS = 4
REQUEST_TIMEOUT_SECONDS = 600
NUM_PREDICT = 1670
NUM_CTX = 20000

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