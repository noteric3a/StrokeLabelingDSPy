"""
config.py

Copy these settings into your existing config.py.

This file is intentionally separate because your current config.py probably
already has many project-specific settings. The safest approach is to copy
only the settings you need from here into your real config.py.
"""

# ---------------------------------------------------------------------------
# Master toggle
# ---------------------------------------------------------------------------
# Set this to True when you want to use the DSPy pipeline.
# Set it to False when you want to keep using your manual prompts.py pipeline.
USE_DSPY = True


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
ENABLE_CONFIDENCE_CHECKING = False

# Number of repeated attempts per report when confidence mode is enabled.
CONFIDENCE_ATTEMPTS = 10

# Minimum vote percentage needed to be considered confident.
# Example:
#   51.0 means a label set must win at least 51% of attempts.
CONFIDENCE_THRESHOLD_PERCENTAGE = 67.0


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
