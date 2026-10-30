"""Fundus image preprocessing shared by data preparation, training and evaluation.

Every function takes an HxWx3 uint8 array. Channel order (RGB or BGR) makes no practical
difference here, so the same code serves OpenCV (BGR) and the training pipeline (RGB).
"""
from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def strip_image_ext(name: str) -> str:
    """'abc.png' -> 'abc'; names without an image extension are returned unchanged."""
    stem, ext = os.path.splitext(name)
    return stem if ext.lower() in IMAGE_EXTS else name


AMBIGUOUS = "<ambiguous>"


def index_images(image_dir: str) -> dict[str, str]:
    """Map lower-cased file names and stems to full paths for every image under image_dir.

    A name found in more than one folder (e.g. IDRiD's training and testing folders both
    contain IDRiD_001.jpg) is marked AMBIGUOUS rather than silently matched to one of them.
    """
    index: dict[str, str] = {}
    for root, dirs, files in os.walk(image_dir):
        dirs.sort()  # deterministic order
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                full = os.path.join(root, name)
                for key in {name.lower(), strip_image_ext(name).lower()}:
                    if key in index and index[key] != full:
                        index[key] = AMBIGUOUS
                    else:
                        index[key] = full
    return index


def lookup_image(index: dict[str, str], image_id) -> str | None:
    """Find an image by its id, with or without extension, ignoring case (None if ambiguous)."""
    key = str(image_id).strip().lower()
    hit = index.get(key) or index.get(strip_image_ext(key))
    return None if hit == AMBIGUOUS else hit


def count_ambiguous(index: dict[str, str], image_ids) -> int:
    """How many of these ids match files in more than one folder."""
    return sum(1 for i in image_ids
               if AMBIGUOUS in (index.get(str(i).strip().lower()), index.get(strip_image_ext(str(i).strip().lower()))))


@lru_cache(maxsize=16)
def circle_mask(size: int, radius_frac: float = 1.0) -> np.ndarray:
    """Read-only boolean mask of a centred disc (True inside)."""
    yy, xx = np.ogrid[:size, :size]
    centre = (size - 1) / 2.0
    radius = radius_frac * size / 2.0
    mask = (xx - centre) ** 2 + (yy - centre) ** 2 <= radius**2
    mask.setflags(write=False)
    return mask


def crop_fundus(img: np.ndarray | None, size: int = 512, tol: int = 10) -> np.ndarray | None:
    """Crop the dark border around the fundus, pad to a square, resize and black out the corners.

    The fundus is found as the largest bright region, so text or logos burnt into the border of
    a photo do not affect the crop. Returns None when no fundus is found (unreadable, black or
    badly exposed image); such images should be treated as ungradable.
    """
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        return None
    h0, w0 = img.shape[:2]
    scale = 256.0 / max(h0, w0)  # locate the fundus on a small copy, for speed
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                       (max(1, round(w0 * scale)), max(1, round(h0 * scale))), interpolation=cv2.INTER_AREA)
    n, _, stats, _ = cv2.connectedComponentsWithStats((small > tol).astype(np.uint8), connectivity=8)
    if n < 2:
        return None
    x, y, w, h, area = stats[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    if w < 0.25 * small.shape[1] or h < 0.25 * small.shape[0] or area < 0.3 * w * h:
        return None
    x0, y0 = max(0, int((x - 1) / scale)), max(0, int((y - 1) / scale))
    x1, y1 = min(w0, int(np.ceil((x + w + 1) / scale))), min(h0, int(np.ceil((y + h + 1) / scale)))
    img = img[y0:y1, x0:x1]
    h, w = img.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    img = cv2.copyMakeBorder(img, top, side - h - top, left, side - w - left,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    img = cv2.resize(img, (size, size), interpolation=interp)
    img[~circle_mask(size)] = 0
    return img


def ben_graham(img: np.ndarray, sigma_frac: float = 1 / 60, tol: int = 10) -> np.ndarray:
    """Ben Graham's local colour normalisation (winning entry of the 2015 Kaggle DR competition).

    Subtracts the local average colour so vessels and lesions stand out regardless of camera
    colour and lighting. Unlike the original, the local average is computed over retina pixels
    only, so the edge of the photo (including circles cut off at the top and bottom) does not
    create bright rings. Background pixels become neutral grey (128).
    """
    size = min(img.shape[:2])
    retina = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > tol
    retina = cv2.erode(retina.astype(np.uint8), np.ones((3, 3), np.uint8),
                       iterations=max(1, size // 200)).astype(bool)
    sigma = size * sigma_frac
    weight = retina.astype(np.float32)
    x = img.astype(np.float32)
    local = cv2.GaussianBlur(x * weight[..., None], (0, 0), sigma)
    local /= np.maximum(cv2.GaussianBlur(weight, (0, 0), sigma), 1e-3)[..., None]
    out = np.clip(4.0 * (x - local) + 128.0, 0, 255).astype(np.uint8)
    out[~retina] = 128
    return out


def dedup_vector(img: np.ndarray, size: int = 64) -> np.ndarray:
    """Unit-length fingerprint for finding near-duplicate photos.

    The dot product of two fingerprints is the correlation between the images: close to 1 for
    copies of the same photo (even at another resolution) and far lower for different eyes.
    It is computed on the Ben-Graham-normalised image, so it reflects vessels and lesions rather
    than overall colour and lighting, which are similar in every fundus photo.
    """
    gray = cv2.cvtColor(ben_graham(img), cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    v = small[circle_mask(size, 0.9)]
    v = v - v.mean()
    return (v / (np.linalg.norm(v) + 1e-6)).astype(np.float32)
