"""Rule-based photo-quality check for fundus photos.

It measures properties that make a photo hard to grade (most of the retina cut off, too dark or
too bright, glare, blur, low contrast) and compares them with limits taken from the training
photos (see calibrate_quality.py). A photo is flagged when it is more extreme than almost all
training photos on any measure. This catches clearly unusual photos; it is not a validated
gradability model. A model trained on graded quality labels (such as EyeQ) is the next step.
"""
from __future__ import annotations

import cv2
import numpy as np

from .preprocess import ben_graham

# (measure, which side is bad, message shown to the user)
RULES = [
    ("retina_fraction", "low", "most of the retina is cut off or very small"),
    ("brightness", "low", "too dark"),
    ("brightness", "high", "too bright or washed out"),
    ("glare_fraction", "high", "strong glare or reflections"),
    ("sharpness", "low", "blurry or out of focus"),
    ("contrast", "low", "low contrast or hazy"),
]


def quality_metrics(img: np.ndarray, tol: int = 10) -> dict:
    """Quality measures for a cropped RGB fundus photo (the output of crop_fundus)."""
    size = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    retina = gray > tol
    inner = cv2.erode(retina.astype(np.uint8), np.ones((3, 3), np.uint8),
                      iterations=max(1, size // 100)).astype(bool)
    metrics = {"retina_fraction": float(retina.mean())}
    if inner.sum() < 0.01 * size * size:
        nan = float("nan")
        return {**metrics, "brightness": nan, "glare_fraction": nan, "sharpness": nan, "contrast": nan}
    # Sharpness and contrast are measured relative to brightness, so a dark photo is reported
    # as dark rather than also as blurry or low-contrast.
    green = img[..., 1].astype(np.float32)
    level = max(float(green[inner].mean()), 1.0)
    laplacian = cv2.Laplacian(green / level, cv2.CV_32F, ksize=3)
    detail = (ben_graham(img)[..., 1].astype(np.float32) - 128.0) / 4.0  # green minus local average
    metrics.update(
        brightness=float(gray[inner].mean()),
        glare_fraction=float((gray[inner] > 240).mean()),
        sharpness=float(laplacian[inner].var()),
        contrast=float(detail[inner].std() / level),
    )
    return metrics


# Never flag less than this: a small bright reflection (e.g. at the optic disc) is normal.
MIN_LIMITS = {"glare_fraction_high": 0.02}
# Never flag more than this: wide-format cameras cut off the top and bottom of the retina as a
# matter of course, which is not a quality problem.
MAX_LIMITS = {"retina_fraction_low": 0.5}


def calibrate(rows: list[dict], low_pct: float = 1.0, high_pct: float = 99.0) -> dict:
    """Limits at the given percentiles of the training photos' measures."""
    limits = {}
    for metric, side, _ in RULES:
        values = np.array([row[metric] for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            key = f"{metric}_{side}"
            limit = float(np.percentile(values, low_pct if side == "low" else high_pct))
            limit = max(limit, MIN_LIMITS.get(key, -np.inf))
            limits[key] = float(min(limit, MAX_LIMITS.get(key, np.inf)))
    return limits


def assess(metrics: dict, limits: dict) -> list[str]:
    """Problems found in a photo, as short messages (an empty list means none were found)."""
    if not np.isfinite(metrics.get("brightness", float("nan"))):
        return ["no usable retina area"]
    issues = []
    for metric, side, message in RULES:
        limit, value = limits.get(f"{metric}_{side}"), metrics.get(metric)
        if limit is None or value is None or not np.isfinite(value):
            continue
        if (side == "low" and value < limit) or (side == "high" and value > limit):
            issues.append(message)
    return issues
