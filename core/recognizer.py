"""Zero-shot recognition via a single shared CLIP model.

Quality over speed here: each label is scored through *several* prompt
templates (ensembling), the vocabularies are large and granular, and the scores
are temperature-calibrated so CLIP no longer forces a confident-but-wrong winner.
Subjects, themes, emotions and concepts are all computed in a single forward
pass, then emotions are re-grounded in colour psychology.
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

# --- Vocabularies -----------------------------------------------------------

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
    "coral reefs",
    "a spider's web",
    "lichen on stone",
    "a topographic map",
    "weather patterns",
    "a plume of ink",
    "glass reflections",
    "a lunar surface",
    "constellations",
    "a river delta",
    "a field of wildflowers",
    "aurora borealis",
    "a winding canyon",
    "lava flows",
    "a whirlpool",
    "sea foam",
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
    "the cycle of seasons",
    "dreams and the subconscious",
    "the fragility of life",
    "the meeting of opposites",
    "eternity and the infinite",
    "childhood wonder",
    "the fleeting moment",
    "sacred geometry",
]

EMOTIONS = [
    "joy",
    "melancholy",
    "serenity",
    "anxiety",
    "awe",
    "anger",
    "hope",
    "nostalgia",
    "peace",
    "excitement",
    "loneliness",
    "warmth",
    "fear",
    "curiosity",
    "contentment",
    "passion",
    "restlessness",
    "tenderness",
    "playfulness",
    "longing",
    "wonder",
    "grief",
    "euphoria",
    "introspection",
]

# Prompt templates per category (ensembled). Concrete subjects use the standard
# "a photo of …" framing CLIP was trained on; themes/emotions use art framing.
_TEMPLATES = {
    "concepts": ["{}"],
    "subjects": ["a photo of {}", "a painting of {}", "{}"],
    "themes": ["a painting about {}", "artwork that explores {}", "{}"],
    "emotions": ["a painting that feels {}", "artwork expressing {}", "{}"],
}

# (name, vocabulary, top_k). Subjects and emotions return the full list so their
# rebalancing can promote/demote across the whole vocabulary before top-k is picked.
_CATEGORIES = [
    ("concepts", CONCEPTS, 5),
    ("subjects", SUBJECTS, len(SUBJECTS)),
    ("themes", THEMES, 3),
    ("emotions", EMOTIONS, len(EMOTIONS)),
]

# Cosine-space softmax temperature. Lower = softer/more even than CLIP's own
# (very sharp) logit scale, so weak matches don't masquerade as strong ones.
_TEMPERATURE = 28.0

# Mild priors that down-weight labels CLIP over-matches on abstract art (generic
# "smooth flow" / "space" subjects), so more specific matches get a fair chance.
_PRIORS = {
    "subjects": {
        "flowing fabric": 0.60,
        "swirling smoke": 0.70,
        "drifting clouds": 0.82,
        "ocean waves": 0.88,
        "weather patterns": 0.70,
        "a plume of ink": 0.70,
    },
    "themes": {
        "freedom and movement": 0.80,
    },
}


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


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def recognize(image_path: str) -> dict[str, list[dict]]:
    """Score the image against every category and return the top matches."""
    _ensure_loaded()
    import torch

    image = Image.open(image_path).convert("RGB")
    if max(image.size) > config.MAX_INPUT_DIM:
        image.thumbnail((config.MAX_INPUT_DIM, config.MAX_INPUT_DIM), Image.LANCZOS)

    # Build every templated prompt and remember which label each belongs to.
    texts: list[str] = []
    groups: dict[tuple[str, int], list[int]] = {}
    for name, vocab, _ in _CATEGORIES:
        templates = _TEMPLATES[name]
        for i, label in enumerate(vocab):
            idxs = []
            for tpl in templates:
                texts.append(tpl.format(label))
                idxs.append(len(texts) - 1)
            groups[(name, i)] = idxs

    inputs = _processor(
        text=texts, images=image, return_tensors="pt", padding=True
    ).to(_device)

    with torch.no_grad():
        logits = _model(**inputs).logits_per_image[0].cpu().numpy()  # [N]

    # Work in cosine-similarity space, not CLIP's over-sharp logit space.
    scale = float(_model.logit_scale.exp().item())
    cos = logits / scale

    result: dict[str, list[dict]] = {}
    for name, vocab, top_k in _CATEGORIES:
        scores = np.array([np.mean(cos[groups[(name, i)]]) for i in range(len(vocab))])
        prior = _PRIORS.get(name, {})
        for i, label in enumerate(vocab):
            scores[i] *= prior.get(label, 1.0)
        probs = _softmax(scores * _TEMPERATURE)
        ranked = sorted(zip(vocab, probs), key=lambda x: x[1], reverse=True)
        result[name] = [
            {"label": label, "score": round(float(p), 4)}
            for label, p in ranked[:top_k]
        ]

    return result


def load_status() -> bool:
    return _model is not None


# --- Colour-grounded emotion rebalancing ------------------------------------

def rebalance_emotions(emotions: list[dict], features: dict) -> list[dict]:
    """Re-weight CLIP emotion scores with colour psychology.

    CLIP's raw emotion scores are near-uniform for abstract art (essentially
    noise), so colour is treated as the decisive signal here: we compute the
    image's warm/cool, dark/bright, muted/vivid and energetic character, then
    strongly promote or suppress each emotion accordingly.
    """
    b = features.get("brightness", 128)
    s = features.get("saturation", 64)
    cf = features.get("colorfulness", 0)
    edge = features.get("edge_chaos", 0.05)

    palette = features.get("dominant_colors", [])
    warm_share = sum(c["share"] for c in palette if c.get("hue", 0) < 70 or c.get("hue", 0) > 320)
    cool_share = sum(c["share"] for c in palette if 160 <= c.get("hue", 0) <= 270)

    warm = warm_share >= 0.30 and warm_share > cool_share
    cool = cool_share >= 0.30 and cool_share > warm_share
    dark = b < 95
    bright = b > 180
    muted = s < 55
    vivid = cf > 70 or (s > 110 and warm)
    energetic = edge > 0.12

    # Each emotion: (boost_condition, damp_condition). Conditions are deliberately
    # discriminating (warm/cool × dark/bright × muted/vivid) so a single "catch-all"
    # emotion can't win for a whole broad class of images.
    rules = {
        "joy":         (vivid and warm and not dark, muted or dark),
        "excitement":  (vivid and (bright or energetic) and not dark, muted or dark),
        "euphoria":    (vivid and warm and bright, muted or dark),
        "passion":     (vivid and warm, muted or dark),
        "playfulness": (vivid and bright and warm, muted or dark),
        "warmth":      (warm and not dark, False),
        "tenderness":  (warm and muted, False),
        "hope":        (bright and not muted, muted or dark),
        "contentment": (warm and muted, dark or vivid),
        "peace":       (bright and muted, vivid or dark),
        "serenity":    (cool or (bright and muted), vivid and warm),
        "nostalgia":   (warm and muted, vivid),
        "longing":     (cool and muted, vivid),
        "melancholy":  ((cool and muted) or dark, vivid and warm),
        "loneliness":  (dark and not warm, warm or bright),
        "grief":       (dark, warm or bright),
        "fear":        (dark and cool, warm or bright),
        "anxiety":     (vivid and dark, muted),
        "restlessness": (energetic and not bright, muted),
        "anger":       (vivid and warm and (dark or energetic), muted),
        "awe":         (vivid and cool, muted),
        "wonder":      (vivid and cool, muted),
        "curiosity":   (vivid and cool and energetic, muted),
        "introspection": (dark and muted, vivid and bright),
    }

    out = []
    for e in emotions:
        label, score = e["label"], e["score"]
        boost, damp = rules.get(label, (False, False))
        if damp:
            score *= 0.25
        elif boost:
            score *= 3.0
        out.append({"label": label, "score": round(float(score), 4)})
    out.sort(key=lambda x: -x["score"])
    return out


# --- Geometry-grounded subject rebalancing ----------------------------------

# Subjects grouped by the low-level structure they imply. Blended with CLIP so
# subject detection uses two independent signals (semantic + geometric).
_HORIZONTAL_SUBJECTS = {
    "rolling hills", "ocean waves", "a desert vista", "ocean currents", "sand dunes",
    "mountain ridges", "a river delta", "sea foam", "a field of wildflowers",
    "weather patterns", "a winding canyon", "a sunrise", "a lunar surface",
}
_VERTICAL_SUBJECTS = {
    "a forest canopy", "tree branches", "rain on glass", "a city skyline",
    "a rushing waterfall", "lava flows", "aurora borealis",
}
_RADIAL_SUBJECTS = {
    "flower petals", "a blooming garden", "a whirlpool", "constellations",
    "a spider's web", "geometric patterns", "coral reefs",
    "nebulae in space", "lichen on stone", "a topographic map",
}


def rebalance_subjects(subjects: list[dict], features: dict) -> list[dict]:
    """Re-weight subject scores with the image's geometric structure.

    CLIP text matching is weakest for abstract art; geometry is a reliable,
    independent signal. Horizontal banding favours landscapes/seas/skies,
    vertical lines favour forests/cities/rain, and rotational symmetry favours
    flowers/mandalas/whirlpools.
    """
    structure = features.get("structure", {})
    horizontal = structure.get("horizontal", 0.5)   # >0.5 = horizontal bands
    rotational = structure.get("rotational", 0.0)   # 0..1

    out = []
    for s in subjects:
        label, score = s["label"], s["score"]
        m = 1.0
        if label in _RADIAL_SUBJECTS:
            m *= 0.7 + 1.6 * rotational
        if label in _HORIZONTAL_SUBJECTS:
            m *= 1.0 + 1.2 * (horizontal - 0.5)
        if label in _VERTICAL_SUBJECTS:
            m *= 1.0 + 1.2 * (0.5 - horizontal)
        out.append({"label": label, "score": round(float(score * m), 4)})
    out.sort(key=lambda x: -x["score"])
    return out
