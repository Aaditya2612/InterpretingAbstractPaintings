"""Compose a short, concrete description of what the model "sees".

Derived from the CLIP recognition (subjects + emotions) and the numeric visual
features — instant and always on-theme, unlike a generic caption model.
"""

from __future__ import annotations


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


def describe(recognition: dict, features: dict) -> str:
    """Return a one-to-two sentence description of the artwork."""
    subjects = recognition.get("subjects", [])
    emotions = recognition.get("emotions", [])

    subj = subjects[0]["label"] if subjects else "abstract forms"
    subj2 = subjects[1]["label"] if len(subjects) > 1 else None
    emotion = emotions[0]["label"] if emotions else "an ambiguous, contemplative mood"

    if subj2:
        return (
            f"Shapes that call to mind {subj} and {subj2}, rendered in "
            f"{_palette_phrase(features)} with a feeling of {emotion}."
        )
    return (
        f"Shapes that call to mind {subj}, rendered in "
        f"{_palette_phrase(features)} with a feeling of {emotion}."
    )
