"""Evaluate a trained run on the internal test split or on an external dataset.

Uses the thresholds locked on validation (thresholds.json) and reports 95% bootstrap confidence
intervals that resample whole patients.

    python evaluate.py --run-dir runs/b4_512 --csv data/processed/manifest.csv --split test
    python evaluate.py --run-dir runs/b4_512 --csv data/processed/external_messidor2.csv
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dr.dataset import FundusDataset, load_manifest, worker_init_fn
from dr.metrics import (binary_metrics, confusion, expected_calibration_error, group_bootstrap, qwk,
                        roc_auc, scores_to_grades)
from dr.model import load_checkpoint, predict

MAIN_METRICS = ("auc", "sensitivity", "specificity", "qwk")


def compute(df, scores, probs, cfg, n_boot):
    y_grade = df["grade"].to_numpy(int)
    y_ref = df["referable"].to_numpy(int)
    pred_grade = scores_to_grades(scores, cfg["grade_thresholds"])
    threshold = cfg["referable_threshold"]

    def stat(idx):
        m = binary_metrics(y_ref[idx], probs[idx], threshold)
        return {"auc": roc_auc(y_ref[idx], probs[idx]), "sensitivity": m["sensitivity"],
                "specificity": m["specificity"], "qwk": qwk(y_grade[idx], pred_grade[idx])}

    groups = df["group"].astype(str).to_numpy()
    point = stat(np.arange(len(df)))
    ci = group_bootstrap(stat, groups, n_boot=n_boot)
    ops = binary_metrics(y_ref, probs, threshold)
    return {
        "n_images": int(len(df)),
        "n_groups": int(len(np.unique(groups))),
        "referable_prevalence": float(y_ref.mean()),
        **{k: {"value": point[k], "ci95": ci.get(k)} for k in MAIN_METRICS},
        **{k: ops[k] for k in ("ppv", "npv", "tp", "fp", "tn", "fn")},
        "ece_referable": expected_calibration_error(y_ref, probs),
        "confusion_matrix": confusion(y_grade, pred_grade).tolist(),
    }


def fmt(m, key):
    value, ci = m[key]["value"], m[key]["ci95"]
    text = f"{value:.3f}" if np.isfinite(value) else "n/a"
    return text + (f"  (95% CI {ci[0]:.3f}-{ci[1]:.3f})" if ci else "")


def report(title, m, cfg, includes_dme, full=True):
    definition = "grade >= 2 or DME" if includes_dme else "grade >= 2 (moderate NPDR or worse)"
    print(f"\n=== {title}: {m['n_images']} images, {m['n_groups']} patients ===")
    print(f"Referable DR = {definition}; prevalence {m['referable_prevalence']:.1%}")
    print(f"  AUC          {fmt(m, 'auc')}")
    print(f"  Sensitivity  {fmt(m, 'sensitivity')}")
    print(f"  Specificity  {fmt(m, 'specificity')}")
    if full:
        where = "this site's tuning photos" if "site_tuning" in cfg else "validation"
        print(f"  (locked threshold {cfg['referable_threshold']:.3f}, set on {where} for "
              f"{cfg['target_sensitivity']:.0%} sensitivity)")
        print(f"  PPV {m['ppv']:.3f} | NPV {m['npv']:.3f} | calibration error (ECE) {m['ece_referable']:.3f}")
    print(f"5-class grading QWK {fmt(m, 'qwk')}")
    if full:
        print("Confusion matrix (rows = true grade 0-4, columns = predicted grade 0-4):")
        for g, row in enumerate(m["confusion_matrix"]):
            print(f"  {g}: " + " ".join(f"{c:6d}" for c in row))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="folder written by train.py")
    p.add_argument("--csv", required=True, help="manifest.csv or external_<name>.csv")
    p.add_argument("--split", nargs="*", help="only these splits, e.g. --split test (default: all rows)")
    p.add_argument("--sources", nargs="*", help="only these sources")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--thresholds", help="thresholds file to use (default: thresholds.json in --run-dir), "
                                        "e.g. one made by site_threshold.py")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for the 95%% CIs")
    p.add_argument("--name", help="name of the output folder (default: from --csv and --split)")
    args = p.parse_args()

    if args.split and "val" in args.split:
        print("NOTE: thresholds were tuned on 'val', so results on it are optimistic by design.")
    run = Path(args.run_dir)
    cfg = json.loads(Path(args.thresholds).read_text() if args.thresholds else (run / "thresholds.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(run / args.checkpoint, device)
    train_args = ckpt["args"]

    df = load_manifest(args.csv, args.split, args.sources)
    if df.empty:
        raise SystemExit("No rows to evaluate - check --split and --sources.")
    dataset = FundusDataset(df, train_args["img_size"], train=False, ben=not train_args["no_ben"],
                            fov_frac=train_args.get("fov_frac", 1.0))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda",
                        worker_init_fn=worker_init_fn if args.workers > 0 else None)
    scores, probs = predict(model, loader, device, tta=cfg.get("tta", True),
                            amp=device.type == "cuda", desc="evaluate")

    name = args.name or Path(args.csv).stem + ("_" + "-".join(args.split) if args.split else "")
    includes_dme = "dme" in df and df["dme"].notna().any()
    results = {"overall": compute(df, scores, probs, cfg, args.n_boot)}
    report(name, results["overall"], cfg, includes_dme)
    if df["source"].nunique() > 1:
        for source in sorted(df["source"].unique()):
            mask = (df["source"] == source).to_numpy()
            part = df[mask].reset_index(drop=True)
            results[source] = compute(part, scores[mask], probs[mask], cfg, args.n_boot)
            report(f"{name} / {source}", results[source], cfg, includes_dme, full=False)

    out = run / f"eval_{name}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "image_id": df["image_id"], "source": df["source"], "grade": df["grade"],
        "referable": df["referable"], "score": scores,
        "pred_grade": scores_to_grades(scores, cfg["grade_thresholds"]),
        "prob_referable": probs, "pred_referable": (probs >= cfg["referable_threshold"]).astype(int),
    }).to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved predictions and metrics to {out}/")


if __name__ == "__main__":
    main()
