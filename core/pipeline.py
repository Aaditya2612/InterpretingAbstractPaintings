"""Analysis pipeline: orchestrates every stage and caches results.

The two heavy model stages (CLIP recognition and the ResNet-50 attention
heatmap) run in parallel on a thread pool, while the lightweight feature
extraction runs inline. The description and interpretation are then derived
instantly from those results. Results are cached by the SHA-256 of the raw image
bytes, so repeat uploads and the sample gallery resolve instantly.
"""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import config, describer, features, heatmap, interpreter, recognizer

# Fallback metrics used if feature extraction ever fails, so the interpreter
# always has something numeric to work with.
_DEFAULT_FEATURES = {
    "color_diversity": 0.0,
    "colorfulness": 0.0,
    "brightness": 128.0,
    "saturation": 64.0,
    "edge_chaos": 0.05,
    "symmetry_score": 0.3,
    "texture_complexity": 50.0,
    "contrast": 40.0,
    "dominant_colors": [],
    "resolution": {"width": 0, "height": 0},
}

# Small bounded cache. Enough for the sample gallery + recent uploads.
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()
_CACHE_MAX = 32

# One pool shared across requests; the heavy stages use it concurrently.
_executor = ThreadPoolExecutor(max_workers=3)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cache_get(key: str):
    with _cache_lock:
        return _cache.get(key)


def _cache_set(key: str, value: dict):
    with _cache_lock:
        _cache[key] = value
        while len(_cache) > _CACHE_MAX:
            _cache.pop(next(iter(_cache)))


def _stage(fn, *args):
    """Run a function and swallow errors into a friendly failure marker."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def run(image_bytes: bytes, original_name: str = "upload") -> dict:
    """Analyze an image. Returns a JSON-serialisable result dictionary."""
    key = _sha256(image_bytes)
    cached = _cache_get(key)
    if cached is not None:
        out = dict(cached)
        out["cached"] = True
        return out

    # Write to a temp file once; all stages read from it.
    tmp_dir = Path(config.BASE_DIR) / ".tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f"{key}.png"
    tmp_path.write_bytes(image_bytes)

    started = time.perf_counter()

    # Lightweight features first (blocking, ~ms).
    feats = _stage(features.analyze, str(tmp_path))

    # Heavy model stages in parallel (CLIP recognition + ResNet-50 Grad-CAM).
    recognition_f = _executor.submit(_stage, recognizer.recognize, str(tmp_path))
    heatmap_f = _executor.submit(_stage, heatmap.generate, str(tmp_path))
    recognition = recognition_f.result()
    heatmap_arr = heatmap_f.result()

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # Description + interpretation are derived from the above, instantly.
    if not isinstance(feats, dict) or "error" in feats:
        feats = _DEFAULT_FEATURES
    if not isinstance(recognition, dict):
        recognition = {"concepts": [], "subjects": [], "themes": [], "emotions": []}

    # Blend colour psychology into the emotion scores (over the full list, then
    # keep the top few).
    recognition["emotions"] = recognizer.rebalance_emotions(
        recognition.get("emotions", []), feats
    )[:3]

    # Blend geometric structure + art style into the subject scores.
    recognition["subjects"] = recognizer.rebalance_subjects(
        recognition.get("subjects", []), feats, recognition.get("concepts", [])
    )[:4]

    # Blend colour mood into the theme scores.
    recognition["themes"] = recognizer.rebalance_themes(
        recognition.get("themes", []), feats
    )[:3]

    description = describer.describe(recognition, feats)
    try:
        interpretation_text, engine = interpreter.interpret(recognition, feats, description)
    except Exception:  # noqa: BLE001
        interpretation_text, engine = "Unable to interpret this image.", "local"

    result = {
        "id": key,
        "name": original_name,
        "subjects": recognition.get("subjects", []),
        "themes": recognition.get("themes", []),
        "emotions": recognition.get("emotions", []),
        "concepts": recognition.get("concepts", []),
        "description": description,
        "features": feats,
        "interpretation": interpretation_text,
        "engine": engine,
        "heatmap": heatmap_arr if isinstance(heatmap_arr, np.ndarray) else None,
        "elapsed_ms": elapsed_ms,
        "cached": False,
    }

    # Drop the temp file to keep the workspace clean.
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        pass

    _cache_set(key, result)
    return result


def status() -> dict:
    """Report which models are loaded, for the /health endpoint."""
    return {
        "clip": recognizer.load_status(),
        "resnet50": heatmap.load_status(),
        "groq_configured": bool(config.GROQ_API_KEY),
    }


def warmup() -> None:
    """Pre-load all heavy models (call from a background thread)."""
    try:
        recognizer._ensure_loaded()
        heatmap._ensure_loaded()
    except Exception:  # noqa: BLE001
        pass
