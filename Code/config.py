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
# Forwarded to every DSPy/Ollama request as options.num_ctx.  The training LM
# refuses to send a request that would not fit without shortening the input.
DSPY_CONTEXT_WINDOW = 32768
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

# Three optimization roles are stratified from all non-test cases:
# - train: creates instruction proposals
# - optimizer_val: MIPRO/GEPA candidate search
# - dev: independent promotion gate for replacing the saved best program
# The final test cases are pinned below so experiment versions remain comparable.
TRAIN_SPLIT_RATIOS = {
    "train": 0.50,
    "optimizer_val": 0.25,
    "dev": 0.25,
}

# Preserve the same untouched six-case test set used by the v1 run.  These IDs
# are never shown to the optimizer and their labels remain redacted until the
# final audit.  Set this to () only when intentionally defining a new test set.
TRAIN_FIXED_TEST_CASE_IDS = (
    "101047983_20241218_19554800",
    "000935015_20220606_07274100",
    "023051824_20241130_13040900",
    "109509469_20241016_14211000",
    "029003746_20220922_14520000",
    "020034013_20230531_13443300",
)

# Keep representative hard phenotypes in optimizer validation and promotion.
# This prevents the small-data splitter from placing every rare multi-label or
# parent-ICA example in train.  The final test set remains untouched.
TRAIN_FIXED_NON_TEST_SPLIT_CASE_IDS = {
    "optimizer_val": (
        "018146324_20221108_20503800",  # uncertain P2 thrombus + definite M2 thrombus
        "041244179_20230521_12422300",  # new severe M1 stenosis, GT NONE
    ),
    "dev": (
        "021962121_20241220_18250500",  # ICA + MCA + ACA, downstream-only GT
        "041172966_20240202_02254100",  # stenosis-only GT NONE
    ),
}

TRAIN_SAVE_RUN_LOGS = True
TRAIN_RUNS_ROOT = "Files/Results/DSPy_Optimization_Runs"
TRAIN_HISTORY_SIZE = 50
TRAIN_BASELINE_ONLY = False
TRAIN_SMOKE_TEST = False
TRAIN_FINAL_AUDIT_ONLY = False
# Leave False while developing prompts.  After selecting the final saved program,
# set TRAIN_FINAL_AUDIT_ONLY=True for one explicit held-out test audit.
TRAIN_RUN_FINAL_TEST_AFTER_OPTIMIZATION = False
TRAIN_WARM_START = True
TRAIN_RESET_SAVED_PROGRAM_BEFORE_RUN = False
TRAIN_EVALUATE_TEST_EACH_ITERATION = False

# Each loop iteration explores a different optimizer seed.  An unchanged
# instruction-only candidate is rejected without re-running deterministic task
# evaluations, which saves substantial time and model calls.
TRAIN_ADVANCE_OPTIMIZER_SEED_PER_ITERATION = True
TRAIN_SKIP_UNCHANGED_CANDIDATE_EVALUATION = True

# Loop stops after this many consecutive rejected candidates.  Use None for no
# patience stop.  TRAIN_LOOP_MAX_ITERATIONS is a separate hard cap.
TRAIN_LOOP_PATIENCE = 8
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

# MIPROv2 settings.  DSPY_MIPRO_SEED is the first loop seed; when
# TRAIN_ADVANCE_OPTIMIZER_SEED_PER_ITERATION is True, later iterations use
# seed+1, seed+2, ... so the outer loop explores new prompt proposals.
DSPY_MIPRO_AUTO = "medium"  # light, medium, or heavy
DSPY_MIPRO_SEED = 9
DSPY_MIPRO_INIT_TEMPERATURE = 0.9
# Keep optimizer internals quiet so proposed prompt text is not dumped to the terminal.
# Candidate prompts are still saved in each run folder under prompts/.
DSPY_MIPRO_VERBOSE = False
DSPY_MIPRO_MAX_BOOTSTRAPPED_DEMOS = 0
DSPY_MIPRO_MAX_LABELED_DEMOS = 0
DSPY_MIPRO_METRIC_THRESHOLD = 1.0  # Keep only exact bootstrapped demonstrations.
DSPY_MIPRO_PROGRAM_AWARE_PROPOSER = True
DSPY_MIPRO_DATA_AWARE_PROPOSER = True
DSPY_MIPRO_TIP_AWARE_PROPOSER = True
DSPY_MIPRO_FEWSHOT_AWARE_PROPOSER = False
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

# Candidate-supplement guard. This validates only the instruction DSPy generated,
# not the immutable base prompt shown in the effective-prompt audit file.
DSPY_PROMPT_MIN_CHARS = 80
DSPY_PROMPT_MAX_CHARS = 3500
DSPY_PROMPT_FORBIDDEN_TERMS = (
    "step-by-step",
    "chain of thought",
    "return only json",
    "respond with json",
    "output must be json",
    "{cta_base_prompt}",
    "{cta_rules}",
    "ignore cta_base_prompt",
    "ignore the base prompt",
    "override cta_base_prompt",
    "override the base prompt",
    "replace cta_base_prompt",
    "replace the base prompt",
    "disregard cta_base_prompt",
    "disregard the base prompt",
    "base prompt is optional",
    "output every label",
    "carotid transient ischemic attack",
)

# Leave this empty for the discovery experiment: a candidate does not have to
# restate vessel names or even mention the base prompt. The base is enforced
# structurally as a separate immutable input.
DSPY_CTA_SUPPLEMENT_REQUIRED_TERM_GROUPS = ()

# Used once at startup to verify that the immutable base prompt still contains
# the intended clinical/output policy. Each tuple is an OR-group.
DSPY_CTA_BASE_REQUIRED_TERM_GROUPS = (
    ("Allowed labels",),
    ("RMCA",),
    ("LMCA",),
    ("RACA",),
    ("LACA",),
    ("RPCA",),
    ("LPCA",),
    ("RICA",),
    ("LICA",),
    ("RCA",),
    ("LCA",),
    ("M1", "M2"),
    ("A1", "A2"),
    ("P1", "P2"),
    ("stenosis alone", "severe flow-limiting stenosis"),
    ("chronic occlusion", "chronic or stable"),
    ("hypoplastic",),
    ("Do not include NONE",),
)

# Reject a candidate that copies the complete base prompt into the supplement.
# The base is already supplied separately on every CTA call.
DSPY_CTA_REJECT_FULL_BASE_COPY = True


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


# =============================================================================
# CTA base-prompt + DSPy supplement experiment
# =============================================================================

# Experimental invariant:
#   effective CTA guidance = immutable CTA_BASE_PROMPT
#                          + one DSPy-optimizable supplemental instruction.
# DSPy never edits CTA_BASE_PROMPT.  An accepted candidate replaces the previous
# supplement; supplements are never concatenated.
CTA_EXPERIMENT_VERSION = "v2"
CTA_EXPERIMENT_NAME = f"cta_base_plus_supplement_{DSPY_OPTIMIZER}_{CTA_EXPERIMENT_VERSION}"
CTA_BASE_PROMPT_VERSION = "manual_cta_base_v2_dataset_aligned"

# These four switches define the manual annotation policy before optimization.
# They are converted into plain-English rules by _build_cta_base_prompt(), then
# frozen for the entire training run.
CTA_COUNT_SEVERE_STENOSIS_WITHOUT_OCCLUSION = False
CTA_COUNT_POSSIBLE_OCCLUSION_OR_THROMBUS = True
CTA_COUNT_CHRONIC_OR_STABLE_OCCLUSION = False
CTA_OMIT_PARENT_ICA_WHEN_BOTH_MCA_AND_ACA_ARE_OCCLUDED = True

# Two supplied labels conflict directly with the acute-only policy above.  They
# are excluded from prompt search/promotion until the answer key is adjudicated;
# they remain listed here so the decision is explicit and reproducible.
TRAIN_EXCLUDE_CTA_POLICY_CONFLICTS = True
CTA_POLICY_CONFLICT_CASE_IDS = (
    "013938063_20250729_08000900",  # stable left M1 finding is labeled positive
    "034597583_20210124_14400000",  # likely chronic right M1/P2 findings are labeled positive
)


def _build_cta_base_prompt() -> str:
    stenosis_rule = (
        "Count severe flow-limiting stenosis in a named MCA/ACA/PCA segment as a positive territory."
        if CTA_COUNT_SEVERE_STENOSIS_WITHOUT_OCCLUSION
        else
        "Do not label stenosis alone, even when described as severe or flow-limiting, unless the same named vessel is also described as occluded, thrombosed, or near-occluded."
    )
    possible_rule = (
        "A specifically named possible or suspected occlusion/thrombus can qualify when the report presents it as an actual vascular finding; uncertainty caused only by technical limitation, poor visualization, or a conditional recommendation does not qualify."
        if CTA_COUNT_POSSIBLE_OCCLUSION_OR_THROMBUS
        else
        "Do not label merely possible or suspected occlusion/thrombus."
    )
    chronic_rule = (
        "Count a specifically named occlusion even when it is described as chronic or stable."
        if CTA_COUNT_CHRONIC_OR_STABLE_OCCLUSION
        else
        "Do not label chronic occlusion, stable findings, or unchanged vascular abnormalities."
    )
    parent_rule = (
        "When an acute intracranial ICA or carotid-terminus lesion is accompanied by explicit occlusion of both the MCA and ACA, output the downstream MCA and ACA labels and omit the parent RICA/LICA label; otherwise include RICA/LICA for a distinct qualifying intracranial ICA lesion."
        if CTA_OMIT_PARENT_ICA_WHEN_BOTH_MCA_AND_ACA_ARE_OCCLUDED
        else
        "Include RICA/LICA whenever a qualifying acute intracranial ICA or carotid-terminus lesion is present, including when downstream MCA/ACA labels are also present."
    )

    return f"""
Your goal is to label acute stroke-related vascular territory from CTA report text.

Allowed labels: {_ALLOWED_LABELS_TEXT}.

CTA-specific rules:
- Target only acute or newly/worsening named-vessel occlusion, thrombus, or another lesion explicitly qualifying under the rules below.
- Right M1/M2/MCA occlusion or thrombus maps to RMCA only unless another acute territory is clearly stated.
- Left M1/M2/MCA occlusion or thrombus maps to LMCA only unless another acute territory is clearly stated.
- Right A1/A2/ACA occlusion or thrombus maps to RACA; left A1/A2/ACA maps to LACA.
- Right P1/P2/PCA occlusion or thrombus maps to RPCA; left P1/P2/PCA maps to LPCA.
- Use RICA/LICA for qualifying acute intracranial ICA, carotid terminus, terminal ICA, supraclinoid ICA, paraclinoid ICA, or intracranial carotid involvement.
- Use RCA/LCA only for common carotid or cervical carotid involvement. Do not use RCA/LCA for carotid terminus.
- {parent_rule}
- {stenosis_rule}
- {possible_rule}
- {chronic_rule}
- Use NONE when no qualifying acute vascular lesion is present.
- Do not include NONE with any positive label. If the answer is NONE, output only NONE.
- Never output every allowed label. Output only labels directly supported by the report.
- Do not label mild or moderate stenosis, incidental atherosclerosis, congenital variants, hypoplastic vessels, or abnormalities explicitly negated as patent/normal.

Output rules:
- Do not explain step by step.
- Do not write hidden analysis.
- Labels must contain only comma-separated allowed labels.
- Reasoning must be exactly one short sentence summarizing the key report finding.
- Return no additional fields or commentary.
""".strip()


CTA_BASE_PROMPT = _build_cta_base_prompt()

# This is the only CTA instruction DSPy may rewrite.  It starts generic so the
# experiment can reveal whether MIPRO/GEPA discovers useful radiology terms,
# vessel segments, exclusion wording, or parent-versus-downstream distinctions.
CTA_INITIAL_SUPPLEMENT = """
Apply the immutable `cta_base_prompt` to the complete CTA `report_text`.

Use this supplemental instruction only to improve recognition of radiology wording,
named vessel segments, laterality, acuity, uncertainty, negation, and
parent-versus-downstream territory selection. Add generalizable distinctions when
they improve label accuracy. Do not summarize, replace, weaken, or contradict the
authoritative base prompt. Return only the required labels and one short reasoning
sentence.
""".strip()

# DSPy reads this as CTAStrokeSignature.instructions and optimizes it.
CTA_SIGNATURE_INSTRUCTIONS = CTA_INITIAL_SUPPLEMENT

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
    "CTA": CTA_EXPERIMENT_NAME,
    "CTP": "ctp_labeler",
    "Combined": "combined_labeler",
}

# Include the active prompt text as columns in converted spreadsheets.
INCLUDE_CURRENT_PROMPT_COLUMNS = True
