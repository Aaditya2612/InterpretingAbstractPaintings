"""Central configuration.

Everything is read from environment variables (with sensible defaults) so the
app runs anywhere — locally, in Docker, or on a PaaS — without code changes.
"""

from __future__ import annotations

import os
from pathlib import Path

# Suppress harmless symlink warnings on Windows (models still cache fine).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# --- Paths -----------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = STATIC_DIR / "samples"

# Local model cache so weights are only downloaded once and survive restarts.
HF_CACHE_DIR = Path(os.getenv("HF_CACHE_DIR", BASE_DIR / ".cache" / "huggingface")).resolve()

# --- Inference -------------------------------------------------------------

# Largest side we feed into the tagger / heatmap. Downscaling big uploads
# gives a large speed-up with no visible loss for this use-case.
MAX_INPUT_DIM = int(os.getenv("MAX_INPUT_DIM", "384"))

# Number of torch threads per process. We run several models in parallel
# threads, so each gets a small slice instead of all threads thrashing.
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "4"))

# --- Models ----------------------------------------------------------------

CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
HEATMAP_MODEL = os.getenv("HEATMAP_MODEL", "resnet50")

# --- Groq / LLM ------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Server ----------------------------------------------------------------

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Pre-warm models in a background thread on startup so the first upload is
# already fast. Disable with WARMUP=0 when you want the quickest cold start.
WARMUP = os.getenv("WARMUP", "1") not in {"0", "false", "False"}
