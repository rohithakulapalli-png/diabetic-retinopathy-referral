"""Does the model rely on the rim of the photo? A masking test for shortcut learning.

Grad-CAM showed strong responses along the edge of the retina on some confident cases. This
script blanks parts of each photo and measures how the referral predictions change:

  no rim 5/10/20%   blank the outer 5, 10 or 20% of the retina's radius
  random patches    blank random interior patches covering the same area as the 10% rim
  rim only 10%      keep only the outer 10% ring and blank everything inside it

If removing the rim hurts much more than removing the same area of interior, or the rim alone
still predicts well, the model leans on the rim. That can be real lesions in the periphery or
artefacts of the camera's edge, which is a shortcut that may not carry over to other cameras.

    python rim_test.py --run-dir runs/b4_512 --csv data/processed/manifest.csv --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from dr.dataset import load_manifest
from dr.explain import caption, to_tensor
from dr.metrics import binary_metrics, roc_auc
from dr.model import autocast, load_checkpoint


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


def random_patches(inner: np.ndarray, target: int, radius: int, rng) -> np.ndarray:
    """Random disks inside `inner` that together cover about `target` pixels."""
    holes = np.zeros(inner.shape, np.uint8)
    ys, xs = np.nonzero(inner)
    while ys.size and (holes.astype(bool) & inner).sum() < target:
        i = rng.integers(ys.size)
        cv2.circle(holes, (int(xs[i]), int(ys[i])), radius, 1, -1)
    return holes.astype(bool) & inner


def masked_versions(img: np.ndarray, rng):
    """Masked copies of one cropped RGB photo, and the share of retina each one removes."""
    size = img.shape[0]
    retina = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 10
    r = size / 2
    keep = {
        "original": retina,
        "no rim 5%": erode(retina, round(0.05 * r)),
        "no rim 10%": erode(retina, round(0.10 * r)),
        "no rim 20%": erode(retina, round(0.20 * r)),
    }
    rim_area = int(retina.sum() - keep["no rim 10%"].sum())
    keep["random patches"] = retina & ~random_patches(keep["no rim 10%"], rim_area, round(0.06 * size), rng)
    keep["rim only 10%"] = retina & ~keep["no rim 10%"]
    versions, removed = {}, {}
    for name, mask in keep.items():
        version = img.copy()
        version[~mask] = 0
        versions[name] = version
        removed[name] = 1 - mask.sum() / max(int(retina.sum()), 1)
    return versions, removed


@torch.no_grad()
def referral_score(model, img: np.ndarray, ben: bool, device, fov: float = 1.0) -> float:
    x = to_tensor(img, ben, fov).to(device)
    views = torch.cat([x, x.flip(3), x.flip(2), x.flip(2).flip(3)])
    with autocast(device, True):
        _, logit = model(views)
    return float(torch.sigmoid(logit.float()).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--csv", required=True, help="manifest.csv from prepare_data.py")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="rim_test", help="folder for results and the example sheet")
    p.add_argument("--limit", type=int, default=0, help="only use this many photos (quick runs)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(args.run_dir)
    model, ckpt = load_checkpoint(run / "best.pt", device)
    ben = not ckpt["args"]["no_ben"]
    fov = ckpt["args"].get("fov_frac", 1.0)  # the model's own rim mask is applied on top of each test mask
    threshold = json.loads((run / "thresholds.json").read_text())["referable_threshold"]
    df = load_manifest(args.csv, [args.split])
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    rows, removed, examples = [], [], {}
    for r in tqdm(df.itertuples(), total=len(df), desc="masking", mininterval=5.0):
        img = cv2.cvtColor(cv2.imread(r.abs_path), cv2.COLOR_BGR2RGB)
        versions, frac = masked_versions(img, rng)
        scores = {name: referral_score(model, v, ben, device, fov) for name, v in versions.items()}
        rows.append({"image_id": r.image_id, "referable": int(r.referable), **scores})
        removed.append(frac)
        kind = "referable" if r.referable else "not referable"
        if kind not in examples:
            examples[kind] = (versions, scores)

    res = pd.DataFrame(rows)
    names = list(removed[0])
    y = res["referable"].to_numpy()
    base = res["original"].to_numpy()
    summary = []
    for name in names:
        prob = res[name].to_numpy()
        m = binary_metrics(y, prob, threshold)
        summary.append({
            "condition": name,
            "retina removed": float(np.mean([f[name] for f in removed])),
            "AUC": roc_auc(y, prob),
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "mean score change": float(np.mean(np.abs(prob - base))),
            "decisions flipped": float(np.mean((prob >= threshold) != (base >= threshold))),
        })
    table = pd.DataFrame(summary).set_index("condition")
    print(f"\nRim test on {len(res)} {args.split} photos (referral threshold {threshold:.2f}):")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "scores.csv", index=False)
    table.to_csv(out / "summary.csv")
    tiles = []
    for kind, (versions, scores) in examples.items():
        row = [caption(cv2.resize(v, (192, 192), interpolation=cv2.INTER_AREA), f"{n}: {scores[n]:.2f}", 22)
               for n, v in versions.items()]
        tiles.append(caption(np.hstack(row), f"{kind} photo (referral score per version)", 26))
    sheet = np.vstack(tiles)
    cv2.imwrite(str(out / "examples.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Saved scores, summary and examples.jpg to {out}/")


if __name__ == "__main__":
    main()
