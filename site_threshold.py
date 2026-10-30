"""Set the referral threshold for a new clinic or camera from a small set of its labelled photos.

The external test on IDRiD showed that the model ranks photos from a new camera well but scores
them lower, so the threshold set on APTOS misses referable eyes. This script picks the threshold
that reaches the target sensitivity on the new site's TUNING photos and saves a copy of
thresholds.json with it. Always measure the result on DIFFERENT photos from the same site.

    python evaluate.py --run-dir runs/b4_512 --csv data/processed/external_site_tune.csv --n-boot 0
    python site_threshold.py --predictions runs/b4_512/eval_external_site_tune/predictions.csv \
        --thresholds runs/b4_512/thresholds.json --out runs/b4_512/thresholds_site.json
    python evaluate.py --run-dir runs/b4_512 --csv data/processed/external_site_test.csv \
        --thresholds runs/b4_512/thresholds_site.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dr.metrics import binary_metrics, threshold_for_sensitivity


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True, help="predictions.csv from evaluate.py on the site's tuning photos")
    p.add_argument("--thresholds", required=True, help="the model's thresholds.json")
    p.add_argument("--out", required=True, help="where to write the site's thresholds file")
    p.add_argument("--target-sensitivity", type=float, default=0.90)
    p.add_argument("--min-positives", type=int, default=30,
                   help="refuse to tune on fewer referable photos than this (the threshold would be unreliable)")
    args = p.parse_args()

    preds = pd.read_csv(args.predictions)
    n_pos = int(preds["referable"].sum())
    if n_pos < args.min_positives:
        raise SystemExit(f"Only {n_pos} referable photos in the tuning set; need at least {args.min_positives}.")
    cfg = json.loads(Path(args.thresholds).read_text())
    old = cfg["referable_threshold"]
    new = threshold_for_sensitivity(preds["referable"], preds["prob_referable"], args.target_sensitivity)
    before = binary_metrics(preds["referable"], preds["prob_referable"], old)
    after = binary_metrics(preds["referable"], preds["prob_referable"], new)
    cfg.update(referable_threshold=new, target_sensitivity=args.target_sensitivity,
               site_tuning={"predictions": args.predictions, "photos": int(len(preds)), "referable": n_pos,
                            "original_threshold": old})
    Path(args.out).write_text(json.dumps(cfg, indent=2))
    print(f"Tuning photos: {len(preds)} ({n_pos} referable)")
    print(f"  original threshold {old:.3f}: sensitivity {before['sensitivity']:.3f}, specificity {before['specificity']:.3f}")
    print(f"  site threshold     {new:.3f}: sensitivity {after['sensitivity']:.3f}, specificity {after['specificity']:.3f}")
    print(f"Saved {args.out}. Measure it on different photos from the same site before using it.")


if __name__ == "__main__":
    main()
