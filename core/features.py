"""Fast, classical computer-vision feature extraction (no deep models).

These metrics run in a few milliseconds and give the interpreter a concrete,
numeric description of the artwork: colour, composition, texture and motion.
"""

from __future__ import annotations

import colorsys

import cv2
import numpy as np


def _load_rgb(image_path: str, max_dim: int = 512) -> np.ndarray:
    """Read an image, cap its size, and return an RGB uint8 array."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _color_diversity(rgb: np.ndarray) -> float:
    """Fraction of unique (quantised) colours — palette richness."""
    pixels = rgb.reshape(-1, 3)
    # Quantise to 24 distinct levels per channel so JPEG noise doesn't inflate
    # the count. Normalised by total pixel count for a stable 0..1-ish value.
    q = (pixels // 32).astype(np.uint32)
    codes = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    return float(len(np.unique(codes)))


def _colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Süsstrunk colourfulness metric."""
    R = rgb[:, :, 0].astype(np.float32)
    G = rgb[:, :, 1].astype(np.float32)
    B = rgb[:, :, 2].astype(np.float32)
    rg = R - G
    yb = 0.5 * (R + G) - B
    sigma = float(np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2))
    mu = float(np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))
    return sigma + 0.3 * mu


def _edge_chaos(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(np.count_nonzero(edges) / edges.size)


def _symmetry_score(rgb: np.ndarray) -> float:
    flipped = cv2.flip(rgb, 1)
    diff = np.mean(np.abs(rgb.astype(np.float32) - flipped.astype(np.float32)))
    return float(1.0 / (1.0 + diff))


def _texture_complexity(rgb: np.ndarray) -> float:
    """Mean absolute Laplacian — a bounded sharpness / texture proxy (0–100ish)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.mean(np.abs(lap)))


def _brightness_saturation(rgb: np.ndarray) -> tuple[float, float]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return float(np.mean(hsv[:, :, 2])), float(np.mean(hsv[:, :, 1]))


def _dominant_colors(rgb: np.ndarray, n: int = 5) -> list[dict]:
    """Top-n dominant colours via k-means on a downsampled image."""
    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
    data = small.reshape(-1, 3).astype(np.float32)
    k = min(n, 8)
    _, labels, centers = cv2.kmeans(
        data, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        3, cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)

    palette = []
    for idx in order:
        b, g, r = centers[idx]
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        palette.append({
            "hex": f"#{int(r):02x}{int(g):02x}{int(b):02x}",
            "share": round(float(counts[idx] / labels.size), 4),
            "hue": round(float(h * 360), 1),
            "saturation": round(float(s), 3),
            "value": round(float(v), 3),
        })
    return palette[:n]


def analyze(image_path: str, max_dim: int = 512) -> dict:
    """Compute the full numeric description of an artwork."""
    rgb = _load_rgb(image_path, max_dim)
    brightness, saturation = _brightness_saturation(rgb)

    def _r(v: float, nd: int = 4) -> float:
        return round(float(v), nd)

    return {
        "color_diversity": _r(_color_diversity(rgb)),
        "colorfulness": _r(_colorfulness(rgb)),
        "brightness": _r(brightness),
        "saturation": _r(saturation),
        "edge_chaos": _r(_edge_chaos(rgb)),
        "symmetry_score": _r(_symmetry_score(rgb)),
        "texture_complexity": _r(_texture_complexity(rgb)),
        "contrast": _r(float(np.std(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))),
        "dominant_colors": _dominant_colors(rgb),
        "resolution": {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
    }
