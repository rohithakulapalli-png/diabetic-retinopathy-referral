"""Calibrate the photo-quality check, and test whether the photos it flags are harder for the model.

    python calibrate_quality.py --csv data/processed/manifest.csv --out quality_thresholds.json \\
        --predictions runs/b4_512/eval_manifest_test/predictions.csv --examples quality_examples.jpg

Limits are set on the training split only. If --predictions is given (from evaluate.py), the
script compares the model's accuracy on test photos that pass the check with those it flags:
a useful check should flag photos on which the model does worse.
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from dr.dataset import load_manifest
from dr.explain import caption
from dr.metrics import binary_metrics, roc_auc
from dr.quality import RULES, assess, calibrate, quality_metrics


def _measure(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if img is None else quality_metrics(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def measure(df, workers):
    with Pool(max(1, workers)) as pool:
        rows = pool.map(_measure, df["abs_path"].tolist(), chunksize=16)
    keep = [r is not None for r in rows]
    out = df[keep].reset_index(drop=True)
    return pd.concat([out, pd.DataFrame([r for r in rows if r is not None])], axis=1)


def flag(df, limits):
    df = df.copy()
    df["issues"] = [assess(row, limits) for row in df.to_dict("records")]
    df["flagged"] = df["issues"].map(bool)
    return df


def compare(df):
    """Model accuracy on photos that pass the check vs photos it flags."""
    print("\nModel performance on evaluation photos, by photo quality:")
    print(f"  {'':14}{'photos':>7}{'referable':>11}{'AUC':>8}{'sensitivity':>13}{'specificity':>13}")
    for name, part in (("passed check", df[~df["flagged"]]), ("flagged", df[df["flagged"]])):
        if part.empty:
            print(f"  {name:14}{0:>7}")
            continue
        y, p = part["referable"].to_numpy(), part["prob_referable"].to_numpy()
        m = binary_metrics(y, part["pred_referable"].to_numpy(), 0.5)
        print(f"  {name:14}{len(part):>7}{y.mean():>11.0%}{roc_auc(y, p):>8.3f}"
              f"{m['sensitivity']:>13.3f}{m['specificity']:>13.3f}")
    if df["flagged"].sum() < 20:
        print("  (Few photos were flagged, so the 'flagged' row is very noisy.)")


def example_sheet(df, limits, path, per_rule=3, tile=200):
    """The most extreme flagged training photos for each rule, with the measured value."""
    rows = []
    for metric, side, message in RULES:
        key = f"{metric}_{side}"
        if key not in limits:
            continue
        bad = df[(df[metric] < limits[key]) if side == "low" else (df[metric] > limits[key])]
        bad = bad.sort_values(metric, ascending=(side == "low")).head(per_rule)
        tiles = []
        for _, r in bad.iterrows():
            img = cv2.cvtColor(cv2.imread(r["abs_path"]), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
            tiles.append(caption(img, f"{r[metric]:.3g} (limit {limits[key]:.3g})", 22))
        while len(tiles) < per_rule:
            tiles.append(caption(np.zeros((tile, tile, 3), np.uint8), "(none)", 22))
        label = caption(np.full((tile, 150, 3), 30, np.uint8), "", 22)
        for k, line in enumerate(textwrap.wrap(message, 18)):
            cv2.putText(label, line, (6, 110 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        rows.append(np.hstack([label] + tiles))
    if rows:
        cv2.imwrite(path, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"Saved examples of flagged photos to {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="manifest.csv from prepare_data.py")
    p.add_argument("--split", default="train", help="split used to set the limits (default: train)")
    p.add_argument("--out", default="quality_thresholds.json")
    p.add_argument("--low-pct", type=float, default=1.0, help="percentile for 'too low' limits")
    p.add_argument("--high-pct", type=float, default=99.0, help="percentile for 'too high' limits")
    p.add_argument("--predictions", help="predictions.csv from evaluate.py, to test the check")
    p.add_argument("--examples", help="save a sheet of flagged training photos to this .jpg")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = p.parse_args()

    calib = measure(load_manifest(args.csv, [args.split]), args.workers)
    limits = calibrate(calib.to_dict("records"), args.low_pct, args.high_pct)
    with open(args.out, "w") as f:
        json.dump({"limits": limits, "low_pct": args.low_pct, "high_pct": args.high_pct,
                   "n_images": int(len(calib)), "calibrated_on": f"{args.csv} (split {args.split})"}, f, indent=2)
    calib = flag(calib, limits)
    print(f"Limits from {len(calib)} {args.split} photos (saved to {args.out}):")
    for key, value in limits.items():
        print(f"  {key:22} {value:.4g}")
    counts = pd.Series([i for issues in calib["issues"] for i in issues]).value_counts()
    print(f"{calib['flagged'].mean():.1%} of {args.split} photos are flagged:")
    for message, n in counts.items():
        print(f"  {message}: {n}")

    if args.predictions:
        preds = pd.read_csv(args.predictions)
        manifest = load_manifest(args.csv)
        merged = preds.merge(manifest[["source", "image_id", "abs_path"]], on=["source", "image_id"])
        compare(flag(measure(merged, args.workers), limits))
    if args.examples:
        example_sheet(calib, limits, args.examples)


if __name__ == "__main__":
    main()
