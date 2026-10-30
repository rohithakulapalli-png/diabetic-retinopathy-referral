"""Model definition, mixed-precision helper and batched prediction with optional TTA."""
from __future__ import annotations

import contextlib

import timm
import torch
import torch.nn as nn
from tqdm.auto import tqdm


class DRNet(nn.Module):
    """timm backbone plus a linear head with two outputs.

    output 0: continuous DR grade (regression target 0-4)
    output 1: logit for referable DR (grade >= 2, i.e. moderate NPDR or worse)
    """

    def __init__(self, backbone: str, pretrained: bool = True, dropout: float = 0.3,
                 drop_path: float = 0.2):
        super().__init__()
        try:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                              drop_path_rate=drop_path)
        except TypeError:  # architectures without stochastic depth
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.num_features, 2)

    def forward(self, x: torch.Tensor):
        out = self.head(self.dropout(self.backbone(x)))
        return out[:, 0], out[:, 1]


def autocast(device: torch.device, enabled: bool):
    """float16 autocast on CUDA, a no-op everywhere else."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict(model, loader, device, tta: bool = False, amp: bool = True, desc: str = "predict"):
    """Return (grade scores, referable probabilities) as numpy arrays, in loader order.

    With tta=True the prediction is averaged over the image and three flipped copies.
    """
    model.eval()
    scores, probs = [], []
    for x, *_ in tqdm(loader, desc=desc, leave=False, mininterval=2.0):
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)] if tta else [x]
        s_sum = torch.zeros(x.shape[0], device=device)
        p_sum = torch.zeros(x.shape[0], device=device)
        for view in views:
            with autocast(device, amp):
                s, logit = model(view)
            s_sum += s.float()
            p_sum += torch.sigmoid(logit.float())
        scores.append((s_sum / len(views)).cpu())
        probs.append((p_sum / len(views)).cpu())
    return torch.cat(scores).numpy(), torch.cat(probs).numpy()


def load_checkpoint(path, device):
    """Rebuild a DRNet from a checkpoint written by train.py. Returns (model, checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = DRNet(ckpt["args"]["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).to(memory_format=torch.channels_last).eval()
    return model, ckpt
