"""
Central configuration for the DSPy stroke-labeling repository.

The goal of this file is to be the main place you edit project behavior:
- model settings
- file paths
- input spreadsheet column names
- allowed labels and aliases
- confidence settings
- DSPy program instructions
- review/checker terms
- converter/highlighting constants

The other files should import these values instead of hard-coding project rules.
"""

from __future__ import annotations

# =============================================================================
# DSPy / Ollama model settings
# =============================================================================

DSPY_MODEL = "ollama_chat/qwen3.6:latest"
DSPY_API_BASE = "http://localhost:11434"
DSPY_TEMPERATURE = 0.2
DSPY_MAX_TOKENS = 1200
DSPY_PROGRAM_DIR = "optimized_programs"

# Compatibility values for ollama_client.py for testing purposes
MODEL_NAME = DSPY_MODEL.replace("ollama_chat/", "").replace("ollama/", "")
OLLAMA_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT_SECONDS = 600
NUM_PREDICT = DSPY_MAX_TOKENS
NUM_CTX = 8192
OLLAMA_WRAPPER_LOG = "Files/Logs/ollama_wrapper_log.jsonl"


# =============================================================================
# File paths and run defaults
# =============================================================================

INPUT_REPORT_FILE = "Files/Report/New Reports.xlsx"
OUTPUT_JSON_FILE = "Files/Results/labeled_cases_dspy.json"
GROUND_TRUTH_FILE = "Files/GT/GroundTruthKeyNew.xlsx"
TEXT_REPORT_FILE = "Files/Results/report_dspy.txt"
JSON_REPORT_FILE = "Files/Results/report_dspy.json"
CACHE_FILE = "Files/.processing_cache.json"

MAX_CONCURRENT_CASES = 4
LAZY_EXCEL_CHUNK_SIZE = 50

# =============================================================================
# Timestamped run / logging defaults
# =============================================================================

# Keep each labeling experiment in its own timestamped folder so runs do not
# overwrite each other. main.py uses these values by default.
USE_TIMESTAMPED_RUN_FOLDERS = True
RUN_OUTPUT_ROOT = "Files/Results/DSPy_Runs"
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
BAD_JSON_LOG = "Files/Logs/bad_json_log.jsonl"

# DSPy optimization logging. dspy_train.py can repeatedly optimize until you
# stop it and will save each iteration in its own timestamped folder.
DSPY_OPTIMIZATION_LOG_DIR = "Files/Results/DSPy_Optimization_Runs"
DSPY_TRAIN_LOOP_SLEEP_SECONDS = 0
DSPY_INSPECT_HISTORY_N = 200


# =============================================================================
# Input spreadsheet column names
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
        "ground_truth": ["CT GT", "CT_GT", "CT"],
    },
    "CTA": {
        "report": ["CTA Report", "CTA_Report", "CTA text", "CTA_Text", "CTA"],
        "ground_truth": ["CTA GT", "CTA_GT", "CTA"],
    },
    "CTP": {
        "report": ["CTP Report", "CTP_Report", "CTP text", "CTP_Text", "CTP"],
        "ground_truth": ["CTP GT", "CTP_GT", "CTP"],
    },
}


# =============================================================================
# Confidence checking
# =============================================================================

ENABLE_CONFIDENCE_CHECKING = False
CONFIDENCE_ATTEMPTS = 10
CONFIDENCE_THRESHOLD_PERCENTAGE = 51.0
CONFIDENCE_REASONING_WINNING_LIMIT = 5
CONFIDENCE_REASONING_ALTERNATE_LIMIT = 5
ALTERNATE_REASONING_MARKER = "\n\nAlternate-label reasoning samples:"

# Compatibility aliases used by confidence.py / review_checks.py.
CONFIDENCE_RUNS = CONFIDENCE_ATTEMPTS
CONFIDENCE_TEMPERATURE = DSPY_TEMPERATURE
MIN_CONFIDENCE_PERCENTAGE = CONFIDENCE_THRESHOLD_PERCENTAGE


# =============================================================================
# Allowed labels and normalization aliases
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

LABEL_ORDER = [
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
# DSPy signature descriptions and instructions
# =============================================================================

DSPY_INPUT_DESCRIPTIONS = {
    "ct_report": "CT brain report text.",
    "cta_report": "CTA report text.",
    "ctp_report": "CT perfusion report text.",
    "mri_report": "MRI report text if available.",
    "ct_labels": "Already predicted CT labels.",
    "cta_labels": "Already predicted CTA labels.",
    "ctp_labels": "Already predicted CTP labels.",
    "ct_reasoning": "CT reasoning.",
    "cta_reasoning": "CTA reasoning.",
    "ctp_reasoning": "CTP reasoning.",
}

DSPY_OUTPUT_DESCRIPTIONS = {
    "labels": "Comma-separated final labels using only ALLOWED_LABELS.",
    "reasoning": "Brief explanation supporting the final labels.",
    "combined_reasoning": "Brief explanation for the final combined labels.",
}

_ALLOWED_LABELS_TEXT = ", ".join(ALLOWED_LABELS)

CT_SIGNATURE_INSTRUCTIONS = f"""
Label acute ischemic stroke territory from CT brain report text only.

Allowed labels: {_ALLOWED_LABELS_TEXT}.

CT-specific rules:
- Use NONE only when there is no qualifying acute ischemic stroke territory.
- Do not include NONE with another label.
- Do not use CTA or CTP findings when labeling CT.
- CT-visible acute infarct signs should map to their arterial territory.
- Basal ganglia, lentiform nucleus, putamen, caudate, internal capsule,
  corona radiata, centrum semiovale, insula, and operculum usually map to MCA territory.
- Do not label chronic infarcts, old encephalomalacia, stable findings,
  artifacts, or nonspecific weak findings as acute stroke territories.
"""

CTA_SIGNATURE_INSTRUCTIONS = f"""
Label acute stroke-related vascular territory from CTA report text.

Allowed labels: {_ALLOWED_LABELS_TEXT}.

CTA-specific rules:
- Use MCA/ACA/PCA labels for named branch occlusion or severe flow-limiting stenosis.
- Use RICA/LICA for acute intracranial ICA, carotid terminus, terminal ICA,
  supraclinoid ICA, paraclinoid ICA, or intracranial carotid involvement.
- Use RCA/LCA only for common carotid or cervical carotid involvement.
- Do not use RCA/LCA for carotid terminus.
- Prefer specific downstream territory labels when MCA/ACA/PCA involvement is clearly identified.
- Use NONE when no qualifying acute occlusion or severe flow-limiting lesion is present.
- Do not label mild stenosis, incidental atherosclerosis, chronic occlusion, or stable old findings
  unless your project rules say they qualify.
"""

CTP_SIGNATURE_INSTRUCTIONS = f"""
Label acute perfusion territory from CT perfusion report text.

Allowed labels: {_ALLOWED_LABELS_TEXT}.

CTP-specific rules:
- Label the territory of qualifying hypoperfusion, infarct core, mismatch, penumbra, or tissue at risk.
- Use NONE when there is no qualifying perfusion deficit.
- Do not label tiny nonspecific artifacts or clearly non-territorial findings.
- Prefer the tissue/perfusion territory over an upstream mechanism.
- If RAPID/perfusion values say core and hypoperfusion are 0 mL, usually use NONE.
"""

COMBINED_SIGNATURE_INSTRUCTIONS = f"""
Produce final combined acute stroke territory labels from modality labels and reports.

Allowed labels: {_ALLOWED_LABELS_TEXT}.

Combined-specific rules:
- Start from CT_GT, CTA_GT, and CTP_GT.
- Add MRI acute/recent infarct territories when present.
- Remove labels only for clear reasons such as chronic/stable finding, artifact, weak nonspecific CT sign,
  isolated upstream ICA when a more specific downstream tissue territory is identified, non-qualifying mild stenosis,
  or mismatch/perfusion finding that is too small or nonspecific.
- Prefer tissue/perfusion territory over upstream mechanism when they conflict.
- Use NONE only when no final acute stroke territory remains.
- Do not include NONE with any positive territory label.
"""

DSPY_PROGRAM_NAMES = {
    "CT": "ct_labeler",
    "CTA": "cta_labeler",
    "CTP": "ctp_labeler",
    "Combined": "combined_labeler",
}

# =============================================================================
# Excel prompt display settings
# =============================================================================
# When True, convert.py appends prompt/instruction columns to the final Excel
# sheet.  These are copied from the active config.py instructions at conversion
# time, so the spreadsheet records exactly what prompt rules were used for the
# run even if the JSON did not store prompt text per case.
INCLUDE_CURRENT_PROMPT_COLUMNS = True

# Each key becomes an Excel column header.  Each value is repeated on every row.
# This is intentional: filtering any case row still preserves the prompt context
# used for that output.  Set INCLUDE_CURRENT_PROMPT_COLUMNS = False if the sheet
# becomes too large.
CURRENT_PROMPT_COLUMNS = {
    "CT Current Prompt": CT_SIGNATURE_INSTRUCTIONS,
    "CTA Current Prompt": CTA_SIGNATURE_INSTRUCTIONS,
    "CTP Current Prompt": CTP_SIGNATURE_INSTRUCTIONS,
    "Combined Current Prompt": COMBINED_SIGNATURE_INSTRUCTIONS,
}


# =============================================================================
# Deterministic review/checker constants
# =============================================================================

REVIEW_VERSION = "reasoning-label-consistency-v9-bilateral-side-fix"

EXCLUSION_PHRASES = (
    "does not qualify", "do not qualify", "not qualify", "non-qualifying", "not enough",
    "insufficient", "excluded", "exclude", "omitted", "omit", "removed", "remove",
    "dropped", "drop", "should not be labeled", "should not be included", "not labeled",
    "not included", "ruled out",
)

POSITIVE_REASONING_PHRASES = (
    "qualifies for", "warrants", "is labeled", "are labeled", "should be labeled",
    "therefore label", "therefore, label", "output",
)

CT_CONTAMINATION_TERMS = (
    "cta", "ct angiogram", "ct angiography", "ctp", "ct perfusion", "tmax", "cbf", "cbv",
    "mismatch", "hypoperfusion", "penumbra", "tissue at risk", "rapid",
)

SUBDURAL_EXTRA_AXIAL_REVIEW_TERMS = (
    "subdural hematoma", "subdural haemorrhage", "subdural hemorrhage", "subdural blood",
    "subdural collection", "acute subdural", "sdh", "extra-axial hematoma",
    "extra-axial haemorrhage", "extra-axial hemorrhage", "extra-axial blood",
    "extraaxial hematoma", "extraaxial hemorrhage",
)

SUBARACHNOID_REVIEW_TERMS = (
    "subarachnoid hemorrhage", "subarachnoid haemorrhage", "subarachnoid blood", "sah",
)

SUBDURAL_SCAN_FIELDS = (
    ("CT_Report", "CT report"),
    ("New_CT_Report", "sanitized CT report"),
    ("MRI_Report", "MRI report"),
    ("CTA_Report", "CTA brain-window text"),
    ("CTP_Report", "CTP context text"),
)

SUBDURAL_SCAN_COLUMNS = SUBDURAL_SCAN_FIELDS

LABEL_VARIANTS = {
    "RMCA": ["rmca", "right mca", "right middle cerebral", "right m1", "right m2", "right m3", "right m4"],
    "LMCA": ["lmca", "left mca", "left middle cerebral", "left m1", "left m2", "left m3", "left m4"],
    "RACA": ["raca", "right aca", "right anterior cerebral", "right a1", "right a2", "right a3"],
    "LACA": ["laca", "left aca", "left anterior cerebral", "left a1", "left a2", "left a3"],
    "RPCA": ["rpca", "right pca", "right posterior cerebral", "right p1", "right p2", "right p3", "right p4"],
    "LPCA": ["lpca", "left pca", "left posterior cerebral", "left p1", "left p2", "left p3", "left p4"],
    "RPICA": ["rpica", "right pica", "right posterior inferior cerebellar", "right inferior cerebellar"],
    "LPICA": ["lpica", "left pica", "left posterior inferior cerebellar", "left inferior cerebellar"],
    "BA": ["ba", "basilar", "basilar artery", "basilar tip", "basilar trunk", "pons", "pontine", "central pons", "paramedian pons"],
    "RVA": ["rva", "right vertebral", "right vertebral artery", "right v1", "right v2", "right v3", "right v4", "right intradural vertebral"],
    "LVA": ["lva", "left vertebral", "left vertebral artery", "left v1", "left v2", "left v3", "left v4", "left intradural vertebral"],
    "RICA": ["rica", "right ica", "right internal carotid", "right carotid terminus", "right intracranial carotid"],
    "LICA": ["lica", "left ica", "left internal carotid", "left carotid terminus", "left intracranial carotid"],
    "RCA": ["rca", "right common carotid", "right cca", "right cervical carotid"],
    "LCA": ["lca", "left common carotid", "left cca", "left cervical carotid"],
    "NONE": ["none", "no qualifying", "no evidence", "negative"],
}

TERRITORY_TERMS = {
    "MCA": ["mca", "middle cerebral", "m1", "m2", "m3", "m4", "insula", "insular", "operculum", "basal ganglia", "lentiform", "putamen", "caudate", "internal capsule", "corona radiata", "centrum semiovale"],
    "ACA": ["aca", "anterior cerebral", "a1", "a2", "a3", "pericallosal", "callosomarginal", "medial frontal", "medial parietal", "parafalcine", "cingulate", "corpus callosum"],
    "PCA": ["pca", "posterior cerebral", "p1", "p2", "p3", "p4", "occipital", "calcarine", "posterior temporal", "thalamus", "thalamic"],
    "PICA": ["pica", "posterior inferior cerebellar", "inferior cerebellar", "cerebellar hemisphere", "cerebellum"],
    "BA": ["ba", "basilar", "basilar artery", "basilar tip", "basilar trunk", "pons", "pontine", "central pons", "paramedian pons", "brainstem"],
    "VA": ["vertebral", "vertebral artery", "v1", "v2", "v3", "v4", "intradural vertebral", "vertebrobasilar junction"],
    "ICA": ["ica", "internal carotid", "carotid terminus", "intracranial carotid", "petrous", "cavernous", "paraclinoid", "supraclinoid"],
    "CA": ["common carotid", "cca", "cervical carotid"],
}

LABEL_SIDE_TERRITORY = {
    "RMCA": ("right", "MCA"), "LMCA": ("left", "MCA"),
    "RACA": ("right", "ACA"), "LACA": ("left", "ACA"),
    "RPCA": ("right", "PCA"), "LPCA": ("left", "PCA"),
    "RPICA": ("right", "PICA"), "LPICA": ("left", "PICA"),
    "RVA": ("right", "VA"), "LVA": ("left", "VA"),
    "RICA": ("right", "ICA"), "LICA": ("left", "ICA"),
    "RCA": ("right", "CA"), "LCA": ("left", "CA"),
}

FIELD_REASONING_PAIRS = [
    ("CT_Original_GT", "CT_Original_GT_reasoning", "CT original"),
    ("CT_GT", "CT_GT_reasoning", "CT"),
    ("CTA_GT", "CTA_GT_reasoning", "CTA"),
    ("CTP_GT", "CTP_GT_reasoning", "CTP"),
    ("Combined_GT", "CT_Combined_GT_reasoning", "Combined"),
]

CONFIDENCE_FIELDS = [
    ("CT_Original_GT", "CT original"),
    ("CT_GT", "CT"),
    ("CTA_GT", "CTA"),
    ("CTP_GT", "CTP"),
    ("Combined_GT", "Combined"),
]


# =============================================================================
# Converter / spreadsheet constants
# =============================================================================

ANSWER_KEY_CONFIG_FIELD = "GROUND_TRUTH_FILE"
CASE_ID_COLUMN_CANDIDATES = tuple(CASE_ID_COLUMNS)

MODALITY_COLUMN_CANDIDATES = {
    "CT": {
        "prediction": ("CT_GT", "CT GT", "CT_label", "CT Label", "CT_result", "CT Result", "CT"),
        "ground_truth": ("CT GT", "CT_GT", "CT.GT", "CTGT", "CT Ground Truth", "CT_Ground_Truth", "CT"),
    },
    "CTA": {
        "prediction": ("CTA_GT", "CTA GT", "CTA_label", "CTA Label", "CTA_result", "CTA Result", "CTA"),
        "ground_truth": ("CTA GT", "CTA_GT", "CTA.GT", "CTAGT", "CTA Ground Truth", "CTA_Ground_Truth", "CTA"),
    },
    "CTP": {
        "prediction": ("CTP_GT", "CTP GT", "CTP_label", "CTP Label", "CTP_result", "CTP Result", "CTP"),
        "ground_truth": ("CTP GT", "CTP_GT", "CTP.GT", "CTPGT", "CTP Ground Truth", "CTP_Ground_Truth", "CTP"),
    },
    "Combined": {
        "prediction": ("Combined_GT", "Combined GT", "Combined_label", "Combined Label", "Combined_result", "Combined Result", "Combined"),
        "ground_truth": ("Combined GT", "Combined_GT", "Combined.GT", "CombinedGT", "Combined Ground Truth", "Combined_Ground_Truth", "Combined"),
    },
}

REPORT_LIKE_TERMS = (
    "EXAMINATION:", "EXAM:", "FINDINGS:", "IMPRESSION:", "TECHNIQUE:",
    "CLINICAL HISTORY", "COMPARISON:", "CT STROKE BRAIN", "CT ANGIO", "CT PERFUSION",
)

NONE_ALIASES = {
    "NONE", "NEGATIVE", "NORMAL", "NOACUTE", "NOACUTEFINDING", "NOACUTEFINDINGS",
    "NOACUTESTROKE", "NOACUTEINFARCT", "NOACUTEINFARCTION", "NOLABEL", "NOLABELS",
}
BLANK_ALIASES = {"", "NAN", "NULL", "NONE_"}
FALLBACK_ALLOWED_LABELS = set(ALLOWED_LABELS)
LABEL_COLUMNS = ("CT_Original_GT", "CT_GT", "CTA_GT", "CTP_GT", "Combined_GT")
RAW_REPORT_COLUMNS_TO_DROP_AFTER_MERGE = ("CT_Report", "CTA_Report", "CTP_Report", "MRI_Report")

CONFIDENCE_THRESHOLD_SUFFIX = "_confidence_threshold"
CONFIDENCE_VOTE_COUNT_SUFFIX = "_confidence_vote_count"
CONFIDENCE_TOTAL_VOTES_SUFFIX = "_confidence_total_votes"
CONFIDENCE_FINAL_LABEL_SUFFIX = "_final_label"
CONFIDENCE_VOTES_SUFFIX = "_confidence_votes"

REVIEW_TARGET_COLUMNS = {
    "CT_Original_GT": ("CT_Original_GT", "CT Original Report/Reasoning"),
    "CT_GT": ("CT_GT", "CT Report/Reasoning"),
    "CTA_GT": ("CTA_GT", "CTA Report/Reasoning"),
    "CTP_GT": ("CTP_GT", "CTP Report/Reasoning"),
    "Combined_GT": ("Combined_GT", "Combined Report/Reasoning"),
}
REVIEW_FLAG_COLUMNS = ("Review_Flags_Red", "Review_Flags_Yellow", "Review_Flags")
