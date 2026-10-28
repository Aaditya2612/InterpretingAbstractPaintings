"""CLIP-based artistic-concept tagging.

Lazily loads a single shared CLIP model and scores an image against a curated
vocabulary of abstract-art concepts.
"""

from __future__ import annotations

import threading

import numpy as np
from PIL import Image

from . import config

_LOCK = threading.Lock()
_model = None
_processor = None
_device = None

CONCEPTS = [
    "chaotic abstract painting",
    "geometric abstraction",
    "fluid brush strokes",
    "organic shapes and forms",
    "dark emotional tone",
    "bright colorful abstraction",
    "minimalist composition",
    "dense complex textures",
    "energetic dynamic motion",
    "calm balanced composition",
    "watercolor wash",
    "oil impasto texture",
    "color field painting",
    "lyrical swirling lines",
    "fragmented cubist planes",
    "soothing pastel palette",
]


def _ensure_loaded():
    global _model, _processor, _device
    if _model is not None:
        return
    with _LOCK:
        if _model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        torch.set_num_threads(config.TORCH_THREADS)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = CLIPModel.from_pretrained(
            config.CLIP_MODEL, cache_dir=str(config.HF_CACHE_DIR)
        ).to(_device).eval()
        _processor = CLIPProcessor.from_pretrained(
            config.CLIP_MODEL, cache_dir=str(config.HF_CACHE_DIR)
        )


def tag(image_path: str, top_k: int = 5) -> list[dict]:
    """Score the image against every concept and return the top matches."""
    _ensure_loaded()
    import torch

    image = Image.open(image_path).convert("RGB")
    inputs = _processor(
        text=CONCEPTS, images=image, return_tensors="pt", padding=True
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits_per_image  # [1, n_concepts]
        probs = logits.softmax(dim=1).cpu().numpy()[0]

    ranked = sorted(zip(CONCEPTS, probs), key=lambda x: x[1], reverse=True)
    return [{"label": label, "score": round(float(score), 4)} for label, score in ranked[:top_k]]


def load_status() -> bool:
    return _model is not None
