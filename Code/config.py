# ---------------------------------------------------------------------------
# DSPy model settings
# ---------------------------------------------------------------------------
# This model string assumes you are using Ollama through DSPy/LiteLLM.
#
# If this does not work on your machine, try changing it to:
#     "ollama/qwen3.6:35b"
#     "ollama_chat/qwen3.6:35b"
#     "ollama_chat/qwen3.6:27b"
#     "ollama_chat/llama3.1:8b"
#
# It must match a model you have already pulled with Ollama.
DSPY_MODEL = "ollama_chat/qwen3.6:latest"

# Default Ollama server URL.
DSPY_API_BASE = "http://localhost:11434"

# Lower temperature makes the model more stable.
# Since your task is labeling, stability matters more than creativity.
DSPY_TEMPERATURE = 0.2

# Maximum number of tokens DSPy allows the model to return.
# Increase this if reasoning is being cut off.
# Decrease this if the model is too verbose or slow.
DSPY_MAX_TOKENS = 1200


# ---------------------------------------------------------------------------
# Saved optimized DSPy programs
# ---------------------------------------------------------------------------
# DSPy can save optimized modules as JSON files.
# The DSPy runner will look in this directory.
DSPY_PROGRAM_DIR = "optimized_programs"

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
GROUND_TRUTH_FILE = "Files/Ground_truths.xlsx"


# ---------------------------------------------------------------------------
# Confidence checking
# ---------------------------------------------------------------------------
#
# When enabled:
#   - The DSPy labeler runs the same case multiple times.
#   - It counts votes.
#   - The majority label set becomes the final answer.
#   - The final reasoning becomes a union of the winning attempts.
#
# When disabled:
#   - Each report is labeled once.
# Confidence settings used by labeler.py
ENABLE_CONFIDENCE_CHECKING = False
CONFIDENCE_ATTEMPTS = 10
CONFIDENCE_THRESHOLD_PERCENTAGE = 51.0

# Compatibility aliases used by confidence.py / review_checks.py
CONFIDENCE_RUNS = CONFIDENCE_ATTEMPTS
CONFIDENCE_TEMPERATURE = DSPY_TEMPERATURE
MIN_CONFIDENCE_PERCENTAGE = CONFIDENCE_THRESHOLD_PERCENTAGE


# ---------------------------------------------------------------------------
# Allowed stroke labels
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
    "RCA", "LCA",
]
