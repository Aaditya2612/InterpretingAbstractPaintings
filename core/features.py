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


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised correlation coefficient, clamped to 0..1."""
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a @ a) * (b @ b)))
    if denom < 1e-6:
        return 1.0
    return max(0.0, min(1.0, float((a @ b) / denom)))


def _symmetry_score(rgb: np.ndarray) -> float:
    """Left-vs-right mirror symmetry via correlation of the two halves.

    Compares the left half against the *mirrored* right half, so the result is a
    genuine 0..1 measure of bilateral symmetry (not the old 1/(1+diff) formula
    that collapsed everything to ~0.02).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    left = gray[:, : w // 2]
    right = cv2.flip(gray[:, w // 2 :], 1)
    n = min(left.shape[1], right.shape[1])
    return _corr(left[:, :n].ravel(), right[:, :n].ravel())


def _structure_metrics(rgb: np.ndarray) -> dict:
    """Geometric structure signals used to guide subject detection.

    - ``horizontal``: 0..1, proportion of edge energy in horizontal bands
      (vertical gradient) vs vertical lines (horizontal gradient).
    - ``rotational``: 0..1, correlation with a 90° rotation — high for
      radial / concentric forms (flowers, mandalas, whirlpools, sunbursts).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    h_energy = float(np.sum(np.abs(gy)))
    v_energy = float(np.sum(np.abs(gx)))
    total = h_energy + v_energy + 1e-6
    horizontal = h_energy / total

    h, w = gray.shape
    n = min(h, w)
    top, left = (h - n) // 2, (w - n) // 2
    sq = gray[top : top + n, left : left + n]
    rot = cv2.rotate(sq, cv2.ROTATE_90_CLOCKWISE)
    rotational = _corr(sq.ravel(), rot.ravel())

    return {
        "horizontal": round(horizontal, 3),
        "rotational": round(rotational, 3),
    }


def _texture_complexity(rgb: np.ndarray) -> float:
    """Mean absolute Laplacian — a bounded sharpness / texture proxy (0–100ish)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.mean(np.abs(lap)))


def _brightness_saturation(rgb: np.ndarray) -> tuple[float, float]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return float(np.mean(hsv[:, :, 2])), float(np.mean(hsv[:, :, 1]))


def _color_name(h: float, s: float, v: float) -> str:
    """A human name for an HSV colour (h in degrees, s/v in 0..1)."""
    if s < 0.12:
        if v < 0.22:
            return "near-black"
        if v > 0.86:
            return "near-white"
        return "grey"
    if h < 15 or h >= 345:
        base = "red"
    elif h < 45:
        base = "orange"
    elif h < 70:
        base = "yellow"
    elif h < 160:
        base = "green"
    elif h < 200:
        base = "cyan"
    elif h < 260:
        base = "blue"
    elif h < 300:
        base = "purple"
    else:
        base = "magenta"

    if v < 0.32:
        return f"deep {base}"
    if s < 0.35:
        return f"muted {base}"
    if v > 0.80 and s > 0.55:
        return f"bright {base}"
    if v > 0.74:
        return f"pale {base}"
    return base


def _dominant_colors(rgb: np.ndarray, n: int = 5) -> list[dict]:
    """Top-n *distinct* colours.

    K-means runs in RGB, but clusters are merged by perceptual (Lab) distance so
    we don't return several near-identical shades. Only genuinely different
    colours survive.
    """
    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
    data = small.reshape(-1, 3).astype(np.float32)

    k = 8
    _, labels, centers = cv2.kmeans(
        data, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0),
        5, cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.flatten(), minlength=k).astype(np.float32)

    # Perceptual (Lab) coordinates of each cluster centre, for distance only.
    centers_u8 = np.clip(np.round(centers), 0, 255).astype(np.uint8)
    centers_lab = cv2.cvtColor(
        centers_u8.reshape(k, 1, 3), cv2.COLOR_RGB2LAB
    ).reshape(k, 3).astype(np.float32)

    # Merge clusters closer than ~16 Lab units (near-identical shades).
    merged: list[list] = []  # [lab_centre, rgb_centre, count]
    for idx in np.argsort(-counts):
        c_lab, c_rgb, cnt = centers_lab[idx], centers[idx], counts[idx]
        if cnt <= 0:
            continue
        for m in merged:
            if float(np.linalg.norm(m[0] - c_lab)) < 16.0:
                total = m[2] + cnt
                m[0] = (m[0] * m[2] + c_lab * cnt) / total
                m[1] = (m[1] * m[2] + c_rgb * cnt) / total
                m[2] = total
                break
        else:
            merged.append([c_lab.copy(), c_rgb.copy(), cnt])

    merged.sort(key=lambda m: -m[2])

    total = small.shape[0] * small.shape[1]
    palette = []
    for _, c_rgb, cnt in merged[:n]:
        if cnt / total < 0.01:  # skip negligible (<1%) clusters
            continue
        r, g, b = (int(np.clip(round(v), 0, 255)) for v in c_rgb)
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        palette.append({
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "share": round(float(cnt / total), 4),
            "hue": round(float(h * 360), 1),
            "saturation": round(float(s), 3),
            "value": round(float(v), 3),
            "name": _color_name(h * 360, s, v),
        })
    return palette


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
        "structure": _structure_metrics(rgb),
        "resolution": {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
    }
