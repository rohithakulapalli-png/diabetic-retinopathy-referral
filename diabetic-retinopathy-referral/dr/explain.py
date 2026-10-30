"""Single-image prediction and Grad-CAM heatmaps, used by the demo app and the example sheets."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .preprocess import ben_graham, circle_mask


def to_tensor(img: np.ndarray, ben: bool = True, fov_frac: float = 1.0) -> torch.Tensor:
    """Cropped RGB uint8 photo -> normalised 1x3xHxW tensor, exactly as during training."""
    if fov_frac < 1.0:
        img = img.copy()
        img[~circle_mask(img.shape[0], fov_frac)] = 0
    if ben:
        img = ben_graham(img)
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


@torch.no_grad()
def predict_one(model, x: torch.Tensor, tta: bool = True) -> tuple[float, float]:
    """(grade score, referable-DR probability) for one photo, averaged over flips when tta=True."""
    model.eval()
    views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)] if tta else [x]
    scores, probs = [], []
    for view in views:
        score, logit = model(view)
        scores.append(float(score[0]))
        probs.append(float(torch.sigmoid(logit[0])))
    return float(np.mean(scores)), float(np.mean(probs))


def gradcam(model, x: torch.Tensor, output: int = 1) -> np.ndarray:
    """Grad-CAM map (HxW, scaled 0-1) for one photo.

    output=1 explains the referable-DR score, output=0 the grade score. For this architecture
    (global average pooling followed by a linear head) Grad-CAM equals the classic class
    activation map, so it reflects exactly which regions raised the score.
    """
    model.eval()
    with torch.enable_grad():
        feats = model.backbone.forward_features(x)
        feats.retain_grad()
        pooled = model.backbone.forward_head(feats, pre_logits=True)
        target = model.head(model.dropout(pooled))[0, output]
        model.zero_grad(set_to_none=True)
        target.backward()
        weights = feats.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feats).sum(dim=1))[0].detach().float().cpu().numpy()
    model.zero_grad(set_to_none=True)
    cam = np.clip(cv2.resize(cam, (x.shape[3], x.shape[2]), interpolation=cv2.INTER_CUBIC), 0, None)
    return cam / cam.max() if cam.max() > 0 else cam


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a Grad-CAM map onto the cropped RGB photo; the black background stays black.

    The map is rescaled using retina pixels only, because its strongest values can lie in the
    black border, which would otherwise make everything inside the retina look faint.
    """
    retina = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 10
    peak = float(np.percentile(cam[retina], 99.5)) if retina.any() else 0.0
    if peak <= 0 and retina.any():
        peak = float(cam[retina].max())  # very small hotspots
    if peak > 0:
        cam = np.clip(np.where(retina, cam, 0) / peak, 0, 1)
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1]
    out = img.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha
    out[~retina] = img[~retina]
    return np.clip(out, 0, 255).astype(np.uint8)


def caption(img: np.ndarray, text: str, height: int = 26) -> np.ndarray:
    """Add a dark caption bar above an RGB image (for example sheets)."""
    bar = np.full((height, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text, (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([bar, img])
