"""Save a sheet of Grad-CAM examples from an evaluated split: hits, misses and false alarms.

    python explain_examples.py --run-dir runs/b4_512 --csv data/processed/manifest.csv \\
        --predictions runs/b4_512/eval_manifest_test/predictions.csv --out gradcam_examples.jpg

Each row shows one kind of outcome; each tile shows the photo and its heatmap side by side.
Looking at the misses and false alarms is the quickest way to learn what the model gets wrong.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from dr.dataset import load_manifest
from dr.explain import caption, gradcam, overlay, to_tensor
from dr.model import load_checkpoint

GROUPS = [  # (title, true referable, predicted referable, sort most confident first)
    ("caught: referable, flagged", 1, 1, False),
    ("missed: referable, not flagged", 1, 0, True),
    ("false alarm: not referable, flagged", 0, 1, False),
    ("cleared: not referable, not flagged", 0, 0, True),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--csv", required=True, help="manifest the predictions were made on")
    p.add_argument("--predictions", required=True, help="predictions.csv from evaluate.py")
    p.add_argument("--out", default="gradcam_examples.jpg")
    p.add_argument("--per-group", type=int, default=3)
    p.add_argument("--tile", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(Path(args.run_dir) / "best.pt", device)
    ben = not ckpt["args"]["no_ben"]
    fov = ckpt["args"].get("fov_frac", 1.0)
    preds = pd.read_csv(args.predictions).merge(
        load_manifest(args.csv)[["source", "image_id", "abs_path"]], on=["source", "image_id"])

    rows = []
    for title, truth, flagged, ascending in GROUPS:
        part = preds[(preds["referable"] == truth) & (preds["pred_referable"] == flagged)]
        part = part.sort_values("prob_referable", ascending=ascending).head(args.per_group)
        tiles = []
        for _, r in part.iterrows():
            img = cv2.cvtColor(cv2.imread(r["abs_path"]), cv2.COLOR_BGR2RGB)
            heat = overlay(img, gradcam(model, to_tensor(img, ben, fov).to(device)))
            pair = np.hstack([cv2.resize(im, (args.tile, args.tile), interpolation=cv2.INTER_AREA)
                              for im in (img, heat)])
            tiles.append(caption(pair, f"true grade {int(r['grade'])} | score {r['prob_referable']:.2f}"))
        while len(tiles) < args.per_group:
            tiles.append(caption(np.zeros((args.tile, 2 * args.tile, 3), np.uint8), "(no examples)"))
        rows.append(caption(np.hstack(tiles), f"{title} ({len(preds[(preds['referable'] == truth) & (preds['pred_referable'] == flagged)])} in total)", 30))
    sheet = np.vstack(rows)
    cv2.imwrite(args.out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Saved {args.out} ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
