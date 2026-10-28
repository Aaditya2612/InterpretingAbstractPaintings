"""Art interpretation via Groq (LLaMA) with a graceful local fallback.

If a ``GROQ_API_KEY`` is not set, we produce a deterministic interpretation
directly from the computed features, caption and concepts — so the app is fully
functional out of the box, with the LLM layer as a drop-in upgrade.
"""

from __future__ import annotations

import random

from . import config


def _build_prompt(features: dict, caption: str, concepts: list[str]) -> str:
    concept_list = ", ".join(concepts) if concepts else "no strong concepts detected"
    return (
        "A machine-vision system analyzed an abstract painting and produced this:\n\n"
        f"IMAGE DESCRIPTION: {caption}\n"
        f"DETECTED VISUAL CONCEPTS: {concept_list}\n\n"
        "VISUAL METRICS:\n"
        f"- Color diversity: {features['color_diversity']}\n"
        f"- Colorfulness: {features['colorfulness']}\n"
        f"- Edge chaos: {features['edge_chaos']}\n"
        f"- Symmetry score: {features['symmetry_score']}\n"
        f"- Texture complexity: {features['texture_complexity']}\n\n"
        "Analyze the painting in three steps:\n"
        "STEP 1 — Visual Observations: describe the shapes, movement, textures and color.\n"
        "STEP 2 — Possible Subjects: what real things the abstract forms might resemble "
        "(flames, waves, fabric, smoke, growth, explosions...).\n"
        "STEP 3 — Artistic Interpretation: the emotional tone, technique and symbolic meaning.\n\n"
        "Write a thoughtful, evocative interpretation in 5–7 sentences."
    )


def _llm_interpret(features: dict, caption: str, concepts: list[str]) -> str:
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": "You are an expert art critic specializing in abstract paintings. "
                "You write vivid, accessible interpretations in plain English.",
            },
            {"role": "user", "content": _build_prompt(features, caption, concepts)},
        ],
    )
    return response.choices[0].message.content


# --- Local fallback ---------------------------------------------------------

_TONES = [
    ("dark emotional tone", "a brooding, introspective mood"),
    ("bright colorful abstraction", "an exuberant, celebratory energy"),
    ("calm balanced composition", "a serene, meditative stillness"),
    ("energetic dynamic motion", "a restless, kinetic dynamism"),
    ("minimalist composition", "a quiet, deliberate restraint"),
]

_SUBJECTS = [
    "swirling currents of water", "shifting curtains of smoke", "organic growth unfolding",
    "an explosion frozen in time", "fabric rippling in the wind", "a storm of pigment",
    "drifting nebulae of color", "the memory of a landscape",
]


def _local_interpret(features: dict, caption: str, concepts: list[str]) -> str:
    c = features
    tone, tone_desc = _TONES[0]
    if concepts:
        for label, phrase in _TONES:
            if label in " ".join(concepts):
                tone, tone_desc = label, phrase
                break

    subject = random.choice(_SUBJECTS)

    colorfulness = c["colorfulness"]
    palette_phrase = (
        "a restrained, almost monochrome palette" if colorfulness < 40
        else "a rich, saturated palette" if colorfulness > 90
        else "a balanced palette of carefully chosen hues"
    )

    symmetry = c["symmetry_score"]
    comp_phrase = (
        "strongly symmetrical and anchored composition" if symmetry > 0.5
        else "asymmetric, off-balance composition" if symmetry < 0.3
        else "gently asymmetric composition"
    )

    texture = c["texture_complexity"]
    texture_phrase = (
        "dense, agitated texture" if texture > 250
        else "smooth, quiet surface" if texture < 60
        else "moderately worked texture"
    )

    edge = c["edge_chaos"]
    motion_phrase = (
        "energetic, restless movement" if edge > 0.12
        else "soft, flowing movement" if edge > 0.05
        else "calm, unhurried movement"
    )

    return (
        f"The piece reads first as {subject}, its forms caught between recognition and pure "
        f"abstraction. The {palette_phrase} establishes {tone_desc}, while the "
        f"{comp_phrase} gives the composition its underlying structure. "
        f"A {texture_phrase} invites the eye to linger, and the {motion_phrase} pulls attention "
        f"across the canvas. The detected themes of {', '.join(concepts[:3]) if concepts else 'open-ended abstraction'} "
        f"suggest the artist is exploring memory and emotion rather than literal depiction. "
        f"The result is a work that rewards slow looking, revealing new layers with each viewing."
    )


def interpret(features: dict, caption: str, concepts: list[str]) -> tuple[str, str]:
    """Return ``(text, engine)`` where engine is "groq" or "local"."""
    if config.GROQ_API_KEY:
        try:
            return _llm_interpret(features, caption, concepts), "groq"
        except Exception:
            # Never fail the whole request because of an LLM outage.
            return _local_interpret(features, caption, concepts), "local"
    return _local_interpret(features, caption, concepts), "local"
