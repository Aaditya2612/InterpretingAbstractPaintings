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


def _light_phrase(features: dict) -> str:
    b = features.get("brightness", 128)
    if b < 90:
        return "low, shadowed light"
    if b > 185:
        return "clear, flooding light"
    return "an even, diffused light"


def _space_phrase(features: dict) -> str:
    edge = features.get("edge_chaos", 0.05)
    symmetry = features.get("symmetry_score", 0.3)
    if edge > 0.12 and symmetry < 0.3:
        return "a restless, cluttered space"
    if symmetry > 0.5:
        return "a still, centered space"
    return "a shallow, ambiguous space"


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


def _openings(subj, subj2, emotion, palette, texture, motion, rng):
    """Distinct frames for how the interpretation can begin."""
    return rng.choice([
        # the viewer's eye
        f"What the eye settles on first is something like {subj} — shapes that hover just "
        f"short of naming themselves, while the whole canvas breathes with {emotion}.",
        # quiet insistence
        f"There is a quiet insistence to these forms. They keep reaching toward {subj}, "
        f"then refusing to commit — and it is in that refusal that {emotion} takes root.",
        # temporal shift
        f"Look at it once and it is {subj}; look again and it has already slipped away "
        f"into {palette}, into {emotion}.",
        # memory / dream
        f"The painting does not so much depict {subj} as half-remember it, the way a dream "
        f"holds an image just out of reach — which is only another way of saying {emotion}.",
        # weather / sensory
        f"It arrives like weather: {emotion} gathering low over shapes that could be "
        f"{subj}, the whole field of it damp and luminous.",
        # the artist's hand
        f"You can feel the hand behind it — the confidence of the {texture}, the way the "
        f"{motion} never falters — building toward {subj} without ever quite arriving.",
        # a story half-told
        f"There is a story being half-told here, and it begins with {subj} caught "
        f"mid-becoming, its edges dissolving into {emotion}.",
        # the ground of colour first
        f"Before it is anything else, the painting is {palette}; only out of that ground of "
        f"colour do the shapes slowly begin to suggest {subj}.",
    ])


def _bodies(subj, theme, theme2, emotion, palette, comp, texture, motion, space, rng):
    """Distinct ways of weaving theme + form + colour together."""
    and2 = f" and {theme2}" if theme2 else ""
    return rng.choice([
        f"Beneath all of it the work circles {theme}{and2}. The {comp} gives it an armature, "
        f"while the {texture} stays alive under the eye and the {motion} refuses to let it rest.",
        f"There is a slow argument being made about {theme}{and2}. {palette} does much of the "
        f"talking — each colour pressing against the next — while the {comp} holds the whole "
        f"thing in a fragile order.",
        f"Look past the first impression and the painting opens onto {theme}{and2}. The "
        f"{texture} is the record of the artist's hand, and the {motion} threads through "
        f"{palette} like a current under ice.",
        f"The {comp} keeps the eye circling, never letting it rest, and in that circling "
        f"{theme}{and2} begins to surface. The {texture} complicates it; the {motion} "
        f"resolves it, briefly.",
        f"Colour does most of the emotional work here. {palette} sets a {emotion} key, and "
        f"the shapes — {subj} — seem to strain against it, which becomes its own small drama "
        f"about {theme}{and2}.",
        f"There is a feeling of {theme}{and2} woven into the very {texture} of it, the "
        f"{motion} rising and falling like breath, the {comp} holding the centre the way a "
        f"sentence holds its subject.",
        f"Space and surface pull in different directions within {space}: the {comp} suggesting "
        f"one kind of depth, the {texture} insisting on another. Between them {theme}{and2} "
        f"hangs, unresolved.",
        f"It reads almost as a diagram of {theme}{and2}: the {motion} mapped onto {palette}, "
        f"the {comp} standing in for structure, the {texture} for time itself.",
    ])


def _closings(subj, subj2, theme, theme2, emotion, emotion2, palette, comp, light, rng):
    """Distinct 'original readings' the interpretation can land on."""
    other = subj2 if subj2 else "something else entirely"
    theme_other = theme2 if theme2 else "its opposite"
    return rng.choice([
        # memory
        f"What lingers afterward is less an image than a memory of {subj} — the way a "
        f"half-forgotten place returns not as a picture but as a temperature and a light.",
        # music
        f"Heard as music, it is not a melody but a held chord — {emotion} sustained in the "
        f"air, with {theme} as its slow, patient bass line.",
        # becoming
        f"It is, in the end, a painting about becoming: {subj} caught in the instant before "
        f"it turns into {other}, {theme} held open between two states.",
        # unresolved tension
        f"Two impulses are locked in a quiet standoff here — {theme} against {theme_other} — "
        f"and the painting declines to take sides, leaving a tension the eye can feel but "
        f"never quite resolve.",
        # private logic
        f"There is a private logic to this work, a rule the artist kept to themselves; "
        f"{subj} is its visible trace, and {emotion} the thing it leaves behind in you.",
        # an open question
        f"It asks something of you without quite saying it — a question about {theme} that it "
        f"leaves open, so that every viewer answers differently.",
        # dream at noon
        f"The whole thing has the texture of a dream remembered at noon: {subj} recognisable "
        f"for a second, then gone, leaving only {emotion}.",
        # light
        f"In the end it is a study of {light} — the way {palette} holds it, the way {emotion} "
        f"moves through it, the way the {comp} lets it fall.",
        # contradiction
        f"It wants two opposite things at once — the order of {theme} and the drift of "
        f"{emotion} — and it is exactly that contradiction that keeps it alive.",
        # invitation
        f"The painting rewards the slow glance: return to it in a different mood and it will "
        f"have become a different {subj}, {theme} gone quiet under a different {emotion}.",
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
    light = _light_phrase(features)
    space = _space_phrase(features)

    rng = _rng_for(recognition)
    opening = _openings(subj, subj2, emotion, palette, texture, motion, rng)
    body = _bodies(subj, theme, theme2, emotion, palette, comp, texture, motion, space, rng)
    closing = _closings(subj, subj2, theme, theme2, emotion, emotion2, palette, comp, light, rng)

    return f"{opening} {body} {closing}"


def interpret(recognition: dict, features: dict, description: str) -> tuple[str, str]:
    """Return ``(text, engine)`` where engine is "groq" or "local"."""
    if config.GROQ_API_KEY:
        try:
            return _llm_interpret(recognition, features, description), "groq"
        except Exception:
            return _local_interpret(recognition, features, description), "local"
    return _local_interpret(recognition, features, description), "local"
