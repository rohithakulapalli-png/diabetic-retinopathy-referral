"""PyTorch dataset with fundus-specific augmentation."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocess import ben_graham, circle_mask

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_manifest(csv_path, splits=None, sources=None) -> pd.DataFrame:
    """Read a CSV written by prepare_data.py, optionally filtered by split and source."""
    df = pd.read_csv(csv_path)
    root = Path(csv_path).resolve().parent
    df["abs_path"] = [str(root / p) for p in df["path"]]
    if splits:
        df = df[df["split"].isin(splits)]
    if sources:
        df = df[df["source"].isin(sources)]
    return df.reset_index(drop=True)


def random_augment(img: np.ndarray, rng: np.random.Generator, strong: bool = False,
                   fov_frac: float = 1.0) -> np.ndarray:
    """Geometric and photometric changes that mimic differences between cameras and visits.

    strong=True widens the colour and exposure ranges and adds lower resolution, JPEG
    compression and a narrower field of view, so the model depends less on one camera's look.
    """
    size = img.shape[0]
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1]
    # Any rotation (fundus photos have no fixed orientation), small zoom and shift.
    m = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(-180, 180), rng.uniform(0.9, 1.1))
    m[:, 2] += rng.uniform(-0.03, 0.03, size=2) * size
    img = cv2.warpAffine(np.ascontiguousarray(img), m, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    fov = cv2.warpAffine(circle_mask(size, fov_frac).astype(np.uint8), m, (size, size),
                         flags=cv2.INTER_NEAREST) > 0
    # Exposure, contrast, white balance and gamma.
    contrast, brightness, tint, gamma = ((0.7, 1.3), 30, 0.15, (0.7, 1.4)) if strong else \
                                        ((0.8, 1.2), 20, 0.08, (0.8, 1.25))
    x = img.astype(np.float32) * rng.uniform(*contrast) + rng.uniform(-brightness, brightness)
    x = x * rng.uniform(1 - tint, 1 + tint, size=3).astype(np.float32)
    x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** rng.uniform(*gamma)
    if rng.random() < (0.3 if strong else 0.2):  # defocus
        x = cv2.GaussianBlur(x, (0, 0), sigmaX=rng.uniform(0.3, 2.0 if strong else 1.2))
    if strong:
        if rng.random() < 0.3:  # a lower-resolution camera
            small = max(32, int(size * rng.uniform(0.5, 0.85)))  # keep tiny lesions visible
            x = cv2.resize(cv2.resize(x, (small, small), interpolation=cv2.INTER_AREA), (size, size),
                           interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.3:  # JPEG compression
            ok, buf = cv2.imencode(".jpg", np.clip(x, 0, 255).astype(np.uint8),
                                   [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 90))])
            if ok:
                x = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if rng.random() < 0.3:  # a narrower field of view
            fov &= circle_mask(size, float(rng.uniform(0.82, 1.0)))
    x[~fov] = 0
    return np.clip(x, 0, 255).astype(np.uint8)


class FundusDataset(Dataset):
    """Yields (image tensor, grade 0-4, referable 0/1) for each row of a manifest."""

    def __init__(self, df: pd.DataFrame, img_size: int = 512, train: bool = False,
                 ben: bool = True, seed: int = 0, strong_aug: bool = False, fov_frac: float = 1.0):
        self.paths = df["abs_path"].tolist()
        self.grades = df["grade"].to_numpy(np.float32)
        self.referable = df["referable"].to_numpy(np.float32)
        self.img_size, self.train, self.ben, self.strong_aug = img_size, train, ben, strong_aug
        self.fov_frac = fov_frac  # <1 hides the photo's outer rim, where camera signatures live
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def load_rgb(self, i: int) -> np.ndarray:
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read {self.paths[i]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (self.img_size, self.img_size):
            interp = cv2.INTER_AREA if img.shape[0] > self.img_size else cv2.INTER_LINEAR
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=interp)
        return img

    def __getitem__(self, i: int):
        img = self.load_rgb(i)
        if self.fov_frac < 1.0:
            img[~circle_mask(self.img_size, self.fov_frac)] = 0
        if self.train:
            img = random_augment(img, self.rng, strong=self.strong_aug, fov_frac=self.fov_frac)
        if self.ben:
            img = ben_graham(img)
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        return x, torch.tensor(self.grades[i]), torch.tensor(self.referable[i])


def worker_init_fn(_worker_id: int) -> None:
    """Each DataLoader worker gets a copy of the dataset, so give each its own random stream."""
    cv2.setNumThreads(0)
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2**32)


def sampler_weights(grades, power: float = 0.5) -> torch.Tensor:
    """Per-image sampling weights proportional to 1 / count(grade)**power.

    power=0.5 (square-root balancing) shows rare grades more often without fully equalising
    them, which would make the model over-call disease.
    """
    g = np.asarray(grades).astype(int)
    counts = np.bincount(g, minlength=5).astype(np.float64)
    w = 1.0 / np.power(np.maximum(counts[g], 1.0), power)
    return torch.as_tensor(w / w.sum(), dtype=torch.double)
