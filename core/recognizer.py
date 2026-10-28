"""Zero-shot recognition via a single shared CLIP model.

One forward pass scores the image against four vocabularies — artistic concepts,
recognisable subjects, underlying themes, and emotional tone — each softmaxed
independently so the scores stay interpretable per category.
"""

from __future__ import annotations

import threading

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

SUBJECTS = [
    "ocean waves",
    "rolling hills",
    "flower petals",
    "swirling smoke",
    "drifting clouds",
    "flowing fabric",
    "flames and fire",
    "nebulae in space",
    "mountain ridges",
    "a city skyline",
    "a forest canopy",
    "storm clouds",
    "a rushing waterfall",
    "birds in flight",
    "tree branches",
    "melting ice",
    "a sunrise",
    "ocean currents",
    "sand dunes",
    "a blooming garden",
    "veins of marble",
    "rain on glass",
    "geometric patterns",
    "a desert vista",
]

THEMES = [
    "the passage of time",
    "chaos and order",
    "solitude and isolation",
    "growth and decay",
    "inner turmoil",
    "harmony and balance",
    "longing and nostalgia",
    "celebration of life",
    "the power of nature",
    "spiritual transcendence",
    "memory and loss",
    "freedom and movement",
    "the sublime",
    "rebirth and renewal",
    "quiet contemplation",
    "connection and unity",
]

EMOTIONS = [
    "joy and elation",
    "melancholy and sorrow",
    "serenity and calm",
    "anxiety and tension",
    "awe and wonder",
    "anger and frustration",
    "hope and optimism",
    "nostalgia",
    "peace",
    "excitement and energy",
    "loneliness",
    "warmth and love",
    "fear and unease",
    "curiosity",
    "contentment",
    "passion and intensity",
]

# (name, vocabulary, top_k)
_CATEGORIES = [
    ("concepts", CONCEPTS, 5),
    ("subjects", SUBJECTS, 4),
    ("themes", THEMES, 3),
    ("emotions", EMOTIONS, 3),
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


def recognize(image_path: str) -> dict[str, list[dict]]:
    """Score the image against every category and return the top matches."""
    _ensure_loaded()
    import torch

    image = Image.open(image_path).convert("RGB")
    if max(image.size) > config.MAX_INPUT_DIM:
        image.thumbnail((config.MAX_INPUT_DIM, config.MAX_INPUT_DIM), Image.LANCZOS)

    all_texts = [t for _, vocab, _ in _CATEGORIES for t in vocab]
    inputs = _processor(
        text=all_texts, images=image, return_tensors="pt", padding=True
    ).to(_device)

    with torch.no_grad():
        logits = _model(**inputs).logits_per_image[0]  # [N]

    result: dict[str, list[dict]] = {}
    offset = 0
    for name, vocab, top_k in _CATEGORIES:
        probs = logits[offset:offset + len(vocab)].softmax(dim=0).cpu().numpy()
        ranked = sorted(zip(vocab, probs), key=lambda x: x[1], reverse=True)
        result[name] = [
            {"label": label, "score": round(float(score), 4)}
            for label, score in ranked[:top_k]
        ]
        offset += len(vocab)

    return result


def load_status() -> bool:
    return _model is not None


def rebalance_emotions(emotions: list[dict], features: dict) -> list[dict]:
    """Nudge CLIP emotion scores with basic colour psychology.

    Colour is a strong, well-established emotional signal; blending it in keeps
    the emotion detector from misreading, e.g., a pale near-monochrome painting
    as "joy and elation".
    """
    b = features.get("brightness", 128)
    s = features.get("saturation", 64)
    c = features.get("colorfulness", 0)
    hues = [col.get("hue", 0) for col in features.get("dominant_colors", [])]

    dark = b < 90
    bright = b > 185
    muted = s < 50
    vivid = c > 100
    warm = any(h < 70 or h > 320 for h in hues)
    cool = any(160 <= h <= 270 for h in hues)

    boost = {
        "joy and elation": vivid and not dark,
        "excitement and energy": vivid,
        "passion and intensity": vivid and warm,
        "warmth and love": warm and not dark,
        "hope and optimism": bright and not muted,
        "serenity and calm": cool or (bright and muted),
        "peace": bright and muted,
        "contentment": muted and bright,
        "nostalgia": muted and warm,
        "melancholy and sorrow": dark or muted,
        "loneliness": dark,
        "fear and unease": dark,
        "anxiety and tension": vivid and dark,
        "anger and frustration": vivid and warm and not bright,
        "awe and wonder": vivid and cool,
        "curiosity": vivid and cool,
    }

    damp = {
        "joy and elation": muted or dark,
        "excitement and energy": muted or dark,
        "passion and intensity": muted or dark,
        "anger and frustration": muted,
        "fear and unease": bright,
        "loneliness": bright or warm,
        "melancholy and sorrow": bright and vivid,
        "anxiety and tension": muted,
        "hope and optimism": muted,
    }

    out = []
    for e in emotions:
        label, score = e["label"], e["score"]
        if boost.get(label):
            score *= 1.5
        if damp.get(label):
            score *= 0.6
        out.append({"label": label, "score": round(float(score), 4)})
    out.sort(key=lambda x: -x["score"])
    return out
