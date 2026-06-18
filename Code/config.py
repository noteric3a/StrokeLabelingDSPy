"""User-editable settings for the DSPy stroke-labeling project.

Only settings that change project behavior live here.  Implementation details,
review regular expressions, converter constants, and compatibility aliases live
in the modules that use them.

Training is config-driven.  The only command-line option retained by
``dspy_train.py`` is ``--loop``.
"""

from __future__ import annotations


# =============================================================================
# Model and Ollama settings
# =============================================================================

# Task model used to label reports and score prompt candidates.
DSPY_MODEL = "ollama_chat/qwen3.6:latest"
DSPY_API_BASE = "http://localhost:11434"
DSPY_TASK_TEMPERATURE = 0.0
DSPY_MAX_TOKENS = 1024
DSPY_CONTEXT_WINDOW = 16384
DSPY_DISABLE_CACHE = True

# Prompt/reflection model used by MIPROv2 or GEPA.  None reuses DSPY_MODEL but
# still creates a separate LM with the exploratory temperature below.
DSPY_PROMPT_MODEL = None
DSPY_PROMPT_MODEL_API_BASE = DSPY_API_BASE
DSPY_PROMPT_TEMPERATURE = 0.8
DSPY_PROMPT_MAX_TOKENS = 4096

DSPY_PROGRAM_DIR = "optimized_programs"
OLLAMA_REQUEST_TIMEOUT_SECONDS = 600
OLLAMA_WRAPPER_LOG = "Files/Logs/ollama_wrapper_log.jsonl"
BAD_JSON_LOG = "Files/Logs/bad_json_log.jsonl"


# =============================================================================
# Files and normal labeling runs
# =============================================================================

INPUT_REPORT_FILE = "Files/Report/New Reports.xlsx"
TRAINING_REPORTS_FILE = INPUT_REPORT_FILE
GROUND_TRUTH_FILE = "Files/GT/GroundTruthKeyNew.xlsx"
OUTPUT_JSON_FILE = "Files/Results/labeled_cases_dspy.json"
TEXT_REPORT_FILE = "Files/Results/report_dspy.txt"
JSON_REPORT_FILE = "Files/Results/report_dspy.json"
CACHE_FILE = "Files/.processing_cache.json"

MAX_CONCURRENT_CASES = 4
LAZY_EXCEL_CHUNK_SIZE = 50

USE_TIMESTAMPED_RUN_FOLDERS = True
RUN_OUTPUT_ROOT = "Files/Results/DSPy_Runs"
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


# =============================================================================
# Confidence sampling
# =============================================================================

ENABLE_CONFIDENCE_CHECKING = False
CONFIDENCE_ATTEMPTS = 10
CONFIDENCE_SAMPLE_TEMPERATURE = 0.2
CONFIDENCE_THRESHOLD_PERCENTAGE = 51.0
CONFIDENCE_REASONING_WINNING_LIMIT = 5
CONFIDENCE_REASONING_ALTERNATE_LIMIT = 5


# =============================================================================
# DSPy training run settings
# =============================================================================

# dspy_train.py reads these settings directly.  Run one pass with:
#     python dspy_train.py
# Run repeated improvement passes with:
#     python dspy_train.py --loop
TRAIN_REPORT_TYPE = "CTA"  # CT, CTA, or CTP
TRAIN_MAX_CASES = None       # Set an integer only for quick debugging.
TRAIN_RANDOM_SEED = 42

# Four independent roles:
# - train: creates instructions/demos
# - optimizer_val: MIPRO/GEPA candidate search
# - dev: promotion gate for replacing the saved best program
# - test: final audit only
TRAIN_SPLIT_RATIOS = {
    "train": 0.50,
    "optimizer_val": 0.20,
    "dev": 0.15,
    "test": 0.15,
}

TRAIN_SAVE_RUN_LOGS = True
TRAIN_RUNS_ROOT = "Files/Results/DSPy_Optimization_Runs"
TRAIN_HISTORY_SIZE = 50
TRAIN_BASELINE_ONLY = False
TRAIN_SMOKE_TEST = False
TRAIN_WARM_START = True
TRAIN_RESET_SAVED_PROGRAM_BEFORE_RUN = False
TRAIN_EVALUATE_TEST_EACH_ITERATION = False

# Loop stops after this many consecutive rejected candidates.  Use None for no
# patience stop.  TRAIN_LOOP_MAX_ITERATIONS is a separate hard cap.
TRAIN_LOOP_PATIENCE = 20
TRAIN_LOOP_MAX_ITERATIONS = None

DSPY_SAVE_HISTORY_ON_ERROR = True
DSPY_ERROR_HISTORY_SIZE = 3


# =============================================================================
# Optimizer and reward settings
# =============================================================================

# "mipro" is the compatibility-first default.  Set to "gepa" to use textual
# missing/extra-label feedback when your DSPy installation includes GEPA.
DSPY_OPTIMIZER = "mipro"

# Search reward given directly to DSPy.  Dense label F1 tells the optimizer that
# a partially corrected multi-label answer is better than an unrelated answer.
DSPY_SEARCH_EXACT_WEIGHT = 0.25
DSPY_SEARCH_F1_WEIGHT = 0.75

# Promotion reward used to replace the currently saved best program.  Exact
# accuracy remains dominant, while F1 breaks exact-score ties productively.
DSPY_PROMOTION_EXACT_WEIGHT = 0.85
DSPY_PROMOTION_F1_WEIGHT = 0.15
DSPY_MIN_PROMOTION_IMPROVEMENT = 0.001
DSPY_REQUIRE_EXACT_NON_REGRESSION = True
DSPY_ACCEPTANCE_SPLIT = "dev"

# MIPROv2 settings.
DSPY_MIPRO_AUTO = "medium"  # light, medium, or heavy
DSPY_MIPRO_SEED = 9
DSPY_MIPRO_INIT_TEMPERATURE = 0.9
# False keeps optimizer-generated prompt bodies out of the terminal while
# preserving DSPy/tqdm progress bars. Prompts are still saved in run artifacts.
TRAIN_SHOW_OPTIMIZER_PROMPTS = False
DSPY_MIPRO_MAX_BOOTSTRAPPED_DEMOS = 3
DSPY_MIPRO_MAX_LABELED_DEMOS = 0
DSPY_MIPRO_METRIC_THRESHOLD = 1.0  # Keep only exact bootstrapped demonstrations.
DSPY_MIPRO_PROGRAM_AWARE_PROPOSER = True
DSPY_MIPRO_DATA_AWARE_PROPOSER = True
DSPY_MIPRO_TIP_AWARE_PROPOSER = True
DSPY_MIPRO_FEWSHOT_AWARE_PROPOSER = True
DSPY_MIPRO_VIEW_DATA_BATCH_SIZE = 20
DSPY_MIPRO_MINIBATCH_SIZE = 8
DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS = 3

# Optional GEPA settings.
DSPY_GEPA_AUTO = "medium"
DSPY_GEPA_SEED = 9
DSPY_GEPA_REFLECTION_MINIBATCH_SIZE = 3
DSPY_GEPA_CANDIDATE_SELECTION = "pareto"
DSPY_GEPA_ADD_FORMAT_FAILURE_AS_FEEDBACK = True
DSPY_GEPA_USE_MERGE = True

# Candidate-signature guard.  This validates the instruction DSPy generated,
# not the fixed rules concatenated into a debug preview.
DSPY_PROMPT_MIN_CHARS = 250
DSPY_PROMPT_MAX_CHARS = 6000
DSPY_PROMPT_FORBIDDEN_TERMS = (
    "step-by-step",
    "chain of thought",
    "return only json",
    "respond with json",
    "output must be json",
    "{cta_rules}",
    "carotid transient ischemic attack",
)
DSPY_CTA_REQUIRED_TERM_GROUPS = (
    ("NONE",),
    ("RMCA",),
    ("LMCA",),
    ("MCA", "M1", "M2"),
    ("ACA", "A1", "A2"),
    ("PCA", "P1", "P2"),
    ("ICA", "carotid"),
    ("occlusion", "thrombus"),
    ("stenosis",),
    ("chronic", "stable"),
)


# =============================================================================
# Input spreadsheet columns
# =============================================================================

CASE_ID_COLUMNS = ["Case Name", "case_id", "Case ID", "ID", "Case_Name", "CASE_ID", "Case"]

REPORT_COLUMN_CANDIDATES = {
    "CT_Report": ["CT Report", "CT_Report", "CT text", "CT_Text", "CT"],
    "CTA_Report": ["CTA Report", "CTA_Report", "CTA text", "CTA_Text", "CTA"],
    "CTP_Report": ["CTP Report", "CTP_Report", "CTP text", "CTP_Text", "CTP"],
    "MRI_Report": ["MRI Report", "MRI_Report", "MRI text", "MRI_Text", "MRI"],
}

TRAINING_COLUMN_CANDIDATES = {
    "CT": {
        "report": ["CT Report", "CT_Report", "CT text", "CT_Text", "CT"],
        "ground_truth": ["CT GT", "CT_GT", "CT.GT", "CTGT", "CT Ground Truth", "CT"],
    },
    "CTA": {
        "report": ["CTA Report", "CTA_Report", "CTA text", "CTA_Text", "CTA"],
        "ground_truth": ["CTA GT", "CTA_GT", "CTA.GT", "CTAGT", "CTA Ground Truth", "CTA"],
    },
    "CTP": {
        "report": ["CTP Report", "CTP_Report", "CTP text", "CTP_Text", "CTP"],
        "ground_truth": ["CTP GT", "CTP_GT", "CTP.GT", "CTPGT", "CTP Ground Truth", "CTP"],
    },
}


# =============================================================================
# Labels and normalization
# =============================================================================

ALLOWED_LABELS = [
    "NONE",
    "RMCA", "LMCA",
    "RACA", "LACA",
    "RPCA", "LPCA",
    "RPICA", "LPICA",
    "BA",
    "RVA", "LVA",
    "RICA", "LICA",
    "RCA", "LCA",
]
LABEL_ORDER = list(ALLOWED_LABELS)

LABEL_ALIASES = {
    "RIGHT MCA": "RMCA",
    "LEFT MCA": "LMCA",
    "RIGHT MIDDLE CEREBRAL": "RMCA",
    "LEFT MIDDLE CEREBRAL": "LMCA",
    "RIGHT ACA": "RACA",
    "LEFT ACA": "LACA",
    "RIGHT ANTERIOR CEREBRAL": "RACA",
    "LEFT ANTERIOR CEREBRAL": "LACA",
    "RIGHT PCA": "RPCA",
    "LEFT PCA": "LPCA",
    "RIGHT POSTERIOR CEREBRAL": "RPCA",
    "LEFT POSTERIOR CEREBRAL": "LPCA",
    "RIGHT PICA": "RPICA",
    "LEFT PICA": "LPICA",
    "BASILAR": "BA",
    "BASILAR ARTERY": "BA",
    "RIGHT VERTEBRAL": "RVA",
    "LEFT VERTEBRAL": "LVA",
    "RIGHT VERTEBRAL ARTERY": "RVA",
    "LEFT VERTEBRAL ARTERY": "LVA",
    "RIGHT ICA": "RICA",
    "LEFT ICA": "LICA",
    "RIGHT INTERNAL CAROTID": "RICA",
    "LEFT INTERNAL CAROTID": "LICA",
    "RIGHT COMMON CAROTID": "RCA",
    "LEFT COMMON CAROTID": "LCA",
    "NEGATIVE": "NONE",
    "NORMAL": "NONE",
    "NO ACUTE STROKE": "NONE",
}


# =============================================================================
# CTA annotation-policy switches
# =============================================================================

# These defaults reflect the behavior inferred from the uploaded answer key and
# optimization runs.  They are annotation-policy choices, not universal clinical
# rules.  Review them against your adjudicated labeling protocol.
CTA_COUNT_SEVERE_STENOSIS = False
CTA_COUNT_CHRONIC_OR_STABLE_OCCLUSION = True
CTA_COUNT_POSSIBLE_OCCLUSION = True
CTA_INCLUDE_PARENT_ICA_WITH_DOWNSTREAM = False
CTA_INCLUDE_CERVICAL_CAROTID = True

def _build_cta_signature() -> str:
    stenosis_rule = (
        "Count severe flow-limiting stenosis as a positive territory."
        if CTA_COUNT_SEVERE_STENOSIS
        else "Do not assign a territory for stenosis alone, even when described as severe, unless an occlusion or thrombus is also present."
    )
    chronic_rule = (
        "Count a specifically named arterial occlusion even when it is described as chronic or stable."
        if CTA_COUNT_CHRONIC_OR_STABLE_OCCLUSION
        else "Do not assign labels for chronic or stable occlusions."
    )
    possible_rule = (
        "Count a specifically named possible or suspected occlusion when the report presents it as a vascular finding."
        if CTA_COUNT_POSSIBLE_OCCLUSION
        else "Do not assign a label for merely possible or suspected occlusion."
    )
    parent_rule = (
        "Include the parent RICA/LICA label together with downstream MCA/ACA labels when both are described."
        if CTA_INCLUDE_PARENT_ICA_WITH_DOWNSTREAM
        else "When an ICA/terminus lesion has explicit downstream MCA or ACA involvement, output the downstream territory labels and omit the parent RICA/LICA label."
    )
    cervical_rule = (
        "Use RCA/LCA for qualifying common or cervical carotid occlusion."
        if CTA_INCLUDE_CERVICAL_CAROTID
        else "Do not output RCA/LCA for common or cervical carotid disease."
    )

    return f"""
Label vascular territories from CTA report text using the dataset's annotation policy.

Vessel mapping:
- Right M1/M2/MCA occlusion or thrombus -> RMCA; left -> LMCA.
- Right A1/A2/ACA occlusion or thrombus -> RACA; left -> LACA.
- Right P1/P2/PCA occlusion or thrombus -> RPCA; left -> LPCA.
- Right/left PICA occlusion -> RPICA/LPICA; basilar occlusion -> BA; vertebral occlusion -> RVA/LVA.
- Intracranial ICA, carotid terminus, terminal ICA, supraclinoid ICA, or paraclinoid ICA -> RICA/LICA unless the parent-label policy below suppresses it.
- Common or cervical carotid -> RCA/LCA only when the cervical-carotid policy below permits it.

Annotation policy:
- Label named arterial occlusion or thrombus and preserve all supported territories in multi-vessel cases.
- {stenosis_rule}
- {chronic_rule}
- {possible_rule}
- {parent_rule}
- {cervical_rule}
- Use NONE only when no qualifying vascular label remains.
- Do not infer a territory from incidental atherosclerosis, hypoplasia, congenital variants, or nonvascular wording alone.
- Return only allowed comma-separated labels and exactly one short reasoning sentence.
""".strip()


# =============================================================================
# DSPy signature instructions
# =============================================================================

_ALLOWED_LABELS_TEXT = ", ".join(ALLOWED_LABELS)

CT_SIGNATURE_INSTRUCTIONS = f"""
Label acute ischemic stroke territory from CT brain report text only.

Allowed labels: {_ALLOWED_LABELS_TEXT}.

Rules:
- Use NONE only when there is no qualifying acute ischemic stroke territory.
- Never combine NONE with a positive label.
- Do not use CTA or CTP findings when labeling CT.
- Map CT-visible acute infarct signs to the corresponding arterial territory.
- Basal ganglia, lentiform nucleus, putamen, caudate, internal capsule,
  corona radiata, centrum semiovale, insula, and operculum usually map to MCA.
- Do not label chronic infarcts, old encephalomalacia, artifact, or weak nonspecific findings.
- Return only allowed comma-separated labels and exactly one short reasoning sentence.
""".strip()

CTA_BASE_PROMPT = f"""
Your goal is to label acute stroke-related vascular territory from CTA report text.

Allowed labels: {_ALLOWED_LABELS_TEXT}.

CTA-specific rules:
- Target only acute or newly/worsening large-vessel occlusion, named branch
  occlusion, or severe flow-limiting stenosis.
- Use MCA/ACA/PCA labels for named branch occlusion or severe flow-limiting stenosis.
- Right M1/M2/MCA occlusion or thrombus maps to RMCA only unless another acute
  territory is clearly stated.
- Left M1/M2/MCA occlusion or thrombus maps to LMCA only unless another acute
  territory is clearly stated.
- Right A1/A2/ACA occlusion or severe stenosis maps to RACA; left A1/A2/ACA
  maps to LACA.
- Right P1/P2/PCA severe stenosis or occlusion maps to RPCA; left P1/P2/PCA
  maps to LPCA.
- Use RICA/LICA for acute intracranial ICA, carotid terminus, terminal ICA,
  supraclinoid ICA, paraclinoid ICA, or intracranial carotid involvement.
- Use RCA/LCA only for common carotid or cervical carotid involvement.
  Do not use RCA/LCA for carotid terminus.
- Prefer specific downstream territory labels when MCA/ACA/PCA involvement
  is clearly identified.
- Use NONE when no qualifying acute occlusion or severe flow-limiting lesion
  is present.
- Do not include NONE with any positive label. If the answer is NONE,
  output only NONE.
- Never output every allowed label. Output only labels directly supported
  by the report.
- Do not label mild stenosis, incidental atherosclerosis, chronic occlusion,
  stable findings, congenital variants, or hypoplastic vessels.

Output rules:
- Do not explain step by step.
- Do not write hidden analysis.
- Labels must contain only comma-separated allowed labels.
- Reasoning must be exactly one short sentence summarizing the key report finding.
- Return no additional fields or commentary.
""".strip()

# Keep the base prompt in the fixed CTA input.
CTA_FIXED_RULES = CTA_BASE_PROMPT

CTA_SIGNATURE_INSTRUCTIONS = """
Apply the authoritative `cta_rules` to the CTA `report_text`.

Use this supplemental instruction to improve recognition of radiology wording,
named vessel segments, laterality, acuity, uncertainty, negation, and
parent-versus-downstream territory selection.

Do not summarize, replace, weaken, or contradict `cta_rules`.
Return only the required labels and one short reasoning sentence.
""".strip()

CTP_SIGNATURE_INSTRUCTIONS = f"""
Label acute perfusion territory from CT perfusion report text.

Allowed labels: {_ALLOWED_LABELS_TEXT}.

Rules:
- Label qualifying hypoperfusion, infarct core, mismatch, penumbra, or tissue at risk.
- Use NONE when there is no qualifying perfusion deficit.
- Do not label tiny nonspecific artifacts or clearly non-territorial findings.
- Prefer the tissue/perfusion territory over an upstream mechanism.
- If core and hypoperfusion volumes are both 0 mL, normally use NONE.
- Return only allowed comma-separated labels and exactly one short reasoning sentence.
""".strip()

COMBINED_SIGNATURE_INSTRUCTIONS = f"""
Produce final combined acute stroke territory labels from modality labels and reports.

Allowed labels: {_ALLOWED_LABELS_TEXT}.

Rules:
- Start from CT, CTA, and CTP labels and add MRI acute/recent infarct territories when present.
- Remove labels only for a clear modality-specific reason such as artifact, chronic tissue injury,
  weak nonspecific CT sign, nonqualifying vascular disease, or nonspecific perfusion change.
- Prefer tissue/perfusion territory over an upstream mechanism when they conflict.
- Use NONE only when no final acute stroke territory remains; never combine NONE with positives.
- Return only allowed comma-separated labels and exactly one short reasoning sentence.
""".strip()

DSPY_PROGRAM_NAMES = {
    "CT": "ct_labeler",
    "CTA": "cta_labeler_policy_optimized",
    "CTP": "ctp_labeler",
    "Combined": "combined_labeler",
}

# Include the active prompt text as columns in converted spreadsheets.
INCLUDE_CURRENT_PROMPT_COLUMNS = True
