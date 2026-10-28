"""Compose a natural, on-theme description of the artwork.

Instead of a generic BLIP caption (which is slow and frequently nonsensical for
abstract art), we derive a short, specific description from the CLIP-detected
concepts and the numeric visual features. This is instant and always relevant.
"""

from __future__ import annotations

# Concept label -> opening phrase.
_STYLE = {
    "geometric abstraction": "a geometric abstraction",
    "color field painting": "a color-field composition",
    "minimalist composition": "a minimalist composition",
    "chaotic abstract painting": "a chaotic, gestural abstraction",
    "fragmented cubist planes": "fragmented, cubist planes",
    "fluid brush strokes": "fluid brushwork",
    "lyrical swirling lines": "lyrical, swirling line-work",
    "watercolor wash": "a soft watercolor wash",
    "oil impasto texture": "thick, impasto oil texture",
    "organic shapes and forms": "organic, flowing shapes",
    "dense complex textures": "a dense, heavily-worked surface",
}


def _palette_phrase(features: dict) -> str:
    colorfulness = features.get("colorfulness", 0)
    if colorfulness > 100:
        return "vivid, saturated"
    if colorfulness < 35:
        return "restrained, near-monochrome"
    return "balanced, considered"


def _mood_phrase(features: dict) -> str:
    brightness = features.get("brightness", 128)
    saturation = features.get("saturation", 64)
    if brightness < 85:
        return "dark and brooding"
    if saturation < 45 and brightness > 180:
        return "airily contemplative"
    if saturation > 130:
        return "exuberant and celebratory"
    return "calm and atmospheric"


def _energy_phrase(features: dict) -> str:
    edge = features.get("edge_chaos", 0.05)
    symmetry = features.get("symmetry_score", 0.3)
    if edge > 0.15:
        return "restless, energetic motion"
    if symmetry > 0.55:
        return "a centered, meditative stillness"
    return "a gentle, flowing rhythm"


def describe(concepts: list[dict], features: dict) -> str:
    """Return a one-to-two sentence description of the artwork."""
    labels = [c["label"] for c in (concepts or [])]

    if not labels:
        return "an open, abstract composition with an ambiguous, non-literal presence."

    style = next((l for l in labels if l in _STYLE), labels[0])
    opener = _STYLE.get(style, style)

    secondary = next((l for l in labels if l != style and l in _STYLE), None)
    secondary_phrase = _STYLE.get(secondary) if secondary else None

    sentence = f"This reads as {opener}."
    if secondary_phrase:
        sentence += f" It carries echoes of {secondary_phrase}."
    sentence += (
        f" A {_palette_phrase(features)} palette gives it a {_mood_phrase(features)} mood, "
        f"and the composition moves with {_energy_phrase(features)}."
    )
    return sentence
