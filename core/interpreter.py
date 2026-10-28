"""Art interpretation via Groq (LLaMA) with a rich local fallback.

If a ``GROQ_API_KEY`` is set we use the LLM; otherwise a deterministic-but-varied
local generator writes the interpretation. The generator is seeded by the
detected subject/theme/emotion, so the *same* painting reads consistently while
*different* paintings get genuinely different takes — never the same template
with the words swapped.
"""

from __future__ import annotations

import hashlib
import random

from . import config


# --- Shared phrasing helpers ------------------------------------------------

def _palette_phrase(features: dict) -> str:
    names = []
    for c in features.get("dominant_colors", [])[:3]:
        n = c.get("name")
        if n and n not in names:
            names.append(n)
    if not names:
        colorfulness = features.get("colorfulness", 0)
        return "a restrained palette" if colorfulness < 40 else "a rich palette"
    if len(names) == 1:
        return f"a palette of {names[0]}"
    return "a palette of " + ", ".join(names[:-1]) + " and " + names[-1]


def _composition_phrase(features: dict) -> str:
    symmetry = features.get("symmetry_score", 0.3)
    if symmetry > 0.5:
        return "centred and balanced composition"
    if symmetry < 0.3:
        return "deliberately off-balance composition"
    return "gently asymmetric composition"


def _texture_phrase(features: dict) -> str:
    tex = features.get("texture_complexity", 50)
    if tex > 35:
        return "worked, agitated surface"
    if tex < 12:
        return "smooth, almost seamless surface"
    return "quietly modulated surface"


def _motion_phrase(features: dict) -> str:
    edge = features.get("edge_chaos", 0.05)
    if edge > 0.15:
        return "restless, energetic movement"
    if edge > 0.06:
        return "soft, drifting movement"
    return "calm, unhurried movement"


def _build_prompt(recognition: dict, features: dict, description: str) -> str:
    subjects = [s["label"] for s in recognition.get("subjects", [])]
    themes = [t["label"] for t in recognition.get("themes", [])]
    emotions = [e["label"] for e in recognition.get("emotions", [])]
    concepts = [c["label"] for c in recognition.get("concepts", [])]

    return (
        "A machine-vision system analysed an abstract painting and produced this:\n\n"
        f"WHAT IT RESEMBLES: {', '.join(subjects) or 'nothing recognisable'}\n"
        f"THEMES: {', '.join(themes) or 'none detected'}\n"
        f"EMOTIONAL TONE: {', '.join(emotions) or 'ambiguous'}\n"
        f"ARTISTIC CONCEPTS: {', '.join(concepts) or 'none'}\n"
        f"DESCRIPTION: {description}\n\n"
        "VISUAL METRICS:\n"
        f"- Color diversity: {features['color_diversity']}\n"
        f"- Colorfulness: {features['colorfulness']}\n"
        f"- Edge chaos: {features['edge_chaos']}\n"
        f"- Symmetry score: {features['symmetry_score']}\n"
        f"- Texture complexity: {features['texture_complexity']}\n\n"
        "Write a fresh, specific interpretation of 5-7 sentences that draws on the "
        "subject, themes and emotional tone above. Offer a genuinely original reading "
        "— avoid boilerplate — and don't simply list the inputs back."
    )


def _llm_interpret(recognition: dict, features: dict, description: str) -> str:
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=0.8,
        messages=[
            {
                "role": "system",
                "content": "You are an expert art critic who writes vivid, original "
                "interpretations of abstract paintings. Never repeat yourself and never "
                "fall back on generic phrases.",
            },
            {"role": "user", "content": _build_prompt(recognition, features, description)},
        ],
    )
    return response.choices[0].message.content


# --- Local generator --------------------------------------------------------

def _rng_for(recognition: dict) -> random.Random:
    seed = "|".join(
        (recognition.get("subjects") or [{}])[0].get("label", "")
        + (recognition.get("themes") or [{}])[0].get("label", "")
        + (recognition.get("emotions") or [{}])[0].get("label", "")
    )
    return random.Random(hashlib.md5(seed.encode("utf-8")).hexdigest())


def _openings(subj, subj2, emotion, palette, rng):
    return rng.choice([
        f"What first holds the eye are forms that read as {subj}, hovering just short of "
        f"naming themselves, and the whole canvas is suffused with {emotion}.",
        f"The painting keeps wanting to be {subj}"
        + (f", or perhaps {subj2}" if subj2 else "")
        + f"; then it dissolves back into pure pigment, leaving {emotion} in its wake.",
        f"There is something of {subj} in these shapes, but they never quite commit — "
        f"and it is in that hesitation, against {palette}, that {emotion} takes hold.",
    ])


def _bodies(theme, theme2, comp, texture, motion, palette, rng):
    return rng.choice([
        f"The work circles, quietly, around {theme}. Its {comp} gives it an armature, "
        f"while the {texture} and the {motion} keep it from ever settling still.",
        f"Beneath the surface there is a slow argument about {theme}"
        + (f" and {theme2}" if theme2 else "")
        + f". Much of the talking is done by {palette}, balanced against a {comp} that "
        f"feels almost architectural.",
        f"Look longer and the painting opens onto {theme}. The {texture} carries the "
        f"evidence of the artist's hand, and the {motion} threads through {palette} "
        f"like a current.",
    ])


def _closings(subj, subj2, theme, theme2, emotion, emotion2, rng):
    other = subj2 if subj2 else "something else entirely"
    theme_other = theme2 if theme2 else "its opposite"
    return rng.choice([
        f"What lingers is less an image than a memory of {subj} — the way a "
        f"half-forgotten place returns not as a picture but as a temperature and a light.",
        f"Heard as music, it is not a melody but a held chord: {emotion} sustained in "
        f"the air, with {theme} as its slow-moving bass line.",
        f"It is, in the end, a painting about becoming — {subj} caught in the instant "
        f"before it turns into {other}, {theme} suspended between two states.",
        f"Two impulses are locked in a quiet argument here: {theme} against "
        f"{theme_other}, the composition refusing to take sides. The result is a "
        f"tension the eye can feel but not resolve.",
        f"There is a private logic to this work, a rule the artist kept to themselves; "
        f"{subj} is its visible trace and {emotion} its after-effect.",
    ])


def _local_interpret(recognition: dict, features: dict, description: str) -> str:
    subjects = [s["label"] for s in recognition.get("subjects", [])]
    themes = [t["label"] for t in recognition.get("themes", [])]
    emotions = [e["label"] for e in recognition.get("emotions", [])]

    subj = subjects[0] if subjects else "abstract forms"
    subj2 = subjects[1] if len(subjects) > 1 else None
    theme = themes[0] if themes else "the tension between form and feeling"
    theme2 = themes[1] if len(themes) > 1 else None
    emotion = emotions[0] if emotions else "a quiet, unsettled feeling"
    emotion2 = emotions[1] if len(emotions) > 1 else None

    palette = _palette_phrase(features)
    comp = _composition_phrase(features)
    texture = _texture_phrase(features)
    motion = _motion_phrase(features)

    rng = _rng_for(recognition)
    opening = _openings(subj, subj2, emotion, palette, rng)
    body = _bodies(theme, theme2, comp, texture, motion, palette, rng)
    closing = _closings(subj, subj2, theme, theme2, emotion, emotion2, rng)

    return f"{opening} {body} {closing}"


def interpret(recognition: dict, features: dict, description: str) -> tuple[str, str]:
    """Return ``(text, engine)`` where engine is "groq" or "local"."""
    if config.GROQ_API_KEY:
        try:
            return _llm_interpret(recognition, features, description), "groq"
        except Exception:
            return _local_interpret(recognition, features, description), "local"
    return _local_interpret(recognition, features, description), "local"
