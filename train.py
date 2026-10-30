"""Train the baseline diabetic retinopathy model.

The network predicts a continuous 0-4 grade (regression) and a referable-DR probability
(moderate NPDR or worse). The best epoch is picked on the validation split by referable-DR AUC.
Afterwards the grade cut-offs and the referable-DR operating threshold are tuned on validation
and locked in thresholds.json, which evaluate.py uses unchanged on test and external data.

Example (Kaggle or Colab with a T4 GPU):
    python train.py --manifest data/processed/manifest.csv --out runs/b4_512
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from dr.dataset import FundusDataset, load_manifest, sampler_weights, worker_init_fn
from dr.metrics import (binary_metrics, optimize_thresholds, qwk, roc_auc, scores_to_grades,
                        threshold_for_sensitivity)
from dr.model import DRNet, autocast, predict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="manifest.csv from prepare_data.py")
    p.add_argument("--out", required=True, help="run folder for checkpoints, logs and thresholds")
    p.add_argument("--backbone", default="tf_efficientnet_b4.ns_jft_in1k", help="any timm model name")
    p.add_argument("--no-pretrained", action="store_true", help="random initialisation (smoke tests only)")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--no-ben", action="store_true", help="turn off Ben Graham normalisation")
    p.add_argument("--fov-frac", type=float, default=1.0,
                   help="keep only this fraction of the retina's radius (e.g. 0.9 hides the outer rim, "
                        "where camera and hospital signatures live); used again automatically at test time")
    p.add_argument("--aug", choices=["standard", "strong"], default="standard",
                   help="'strong' adds wider colour/exposure changes, low resolution, JPEG and field-of-view "
                        "changes, to cope better with other cameras")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum-steps", type=int, default=2,
                   help="gradient accumulation; effective batch = batch-size x accum-steps")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--bce-weight", type=float, default=1.0, help="weight of the referable-DR loss")
    p.add_argument("--sampler", choices=["sqrt", "none"], default="sqrt",
                   help="'sqrt' shows rare grades more often during training")
    p.add_argument("--target-sensitivity", type=float, default=0.90,
                   help="the referable-DR threshold is set on validation to reach this sensitivity")
    p.add_argument("--sources", nargs="*", help="only use these sources, e.g. --sources aptos")
    p.add_argument("--max-train", type=int, default=0, help="random subset of training images (quick tests)")
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument("--no-amp", action="store_true", help="turn off mixed precision")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # older PyTorch


def build_optimizer(model, lr, weight_decay):
    """AdamW without weight decay on biases and normalisation layers."""
    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (no_decay if param.ndim <= 1 else decay).append(param)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def warmup_cosine(warmup_steps: int, total_steps: int, floor: float = 0.01):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    return factor


def save_checkpoint(path, model, args, epoch, metrics):
    torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                "metrics": {k: float(v) for k, v in metrics.items()}}, path)


def describe(name, df):
    grades = df["grade"].value_counts().reindex(range(5), fill_value=0).tolist()
    print(f"  {name:<5} {len(df):6d} images | grades 0-4: {grades} | referable {df['referable'].mean():.1%}")


def main():
    args = parse_args()
    seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)} | mixed precision: {amp}")
    else:
        print("WARNING: no GPU found. Training will be very slow (fine for the smoke test only).")

    train_df = load_manifest(args.manifest, ["train"], args.sources)
    val_df = load_manifest(args.manifest, ["val"], args.sources)
    if args.max_train:
        train_df = train_df.sample(min(len(train_df), args.max_train), random_state=args.seed).reset_index(drop=True)
    if args.max_val:
        val_df = val_df.sample(min(len(val_df), args.max_val), random_state=args.seed).reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise SystemExit("Empty train or val split - check --manifest and --sources.")
    print("Data:")
    describe("train", train_df)
    describe("val", val_df)

    ben = not args.no_ben
    train_ds = FundusDataset(train_df, args.img_size, train=True, ben=ben, seed=args.seed,
                             strong_aug=args.aug == "strong", fov_frac=args.fov_frac)
    val_ds = FundusDataset(val_df, args.img_size, train=False, ben=ben, fov_frac=args.fov_frac)
    loader_kw = {"num_workers": args.workers, "pin_memory": device.type == "cuda"}
    if args.workers > 0:
        loader_kw.update(persistent_workers=True, worker_init_fn=worker_init_fn)
    sampler = None
    if args.sampler == "sqrt":
        sampler = WeightedRandomSampler(sampler_weights(train_df["grade"]), num_samples=len(train_ds),
                                        replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              shuffle=sampler is None, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, **loader_kw)

    model = DRNet(args.backbone, pretrained=not args.no_pretrained)
    model = model.to(device).to(memory_format=torch.channels_last)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, int(args.warmup_epochs * steps_per_epoch))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(warmup_steps, total_steps))
    scaler = make_scaler(amp)
    print(f"Model: {args.backbone} at {args.img_size}px | effective batch {args.batch_size * args.accum_steps} "
          f"| {args.epochs} epochs x {steps_per_epoch} steps")

    y_grade = val_df["grade"].to_numpy(int)
    y_ref = val_df["referable"].to_numpy(int)
    best_metric, best_epoch, log_rows = -float("inf"), 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n_batches = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False, mininterval=2.0)
        for step, (x, grade, referable) in enumerate(bar):
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            grade = grade.to(device, non_blocking=True)
            referable = referable.to(device, non_blocking=True)
            with autocast(device, amp):
                score, logit = model(x)
            loss = (F.smooth_l1_loss(score.float(), grade)
                    + args.bce_weight * F.binary_cross_entropy_with_logits(logit.float(), referable))
            scaler.scale(loss / args.accum_steps).backward()
            if (step + 1) % args.accum_steps == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running += loss.item()
            n_batches += 1
            bar.set_postfix(loss=f"{running / n_batches:.3f}")

        scores, probs = predict(model, val_loader, device, tta=False, amp=amp, desc="validate")
        row = {"epoch": epoch, "train_loss": running / max(1, n_batches),
               "val_auc_referable": roc_auc(y_ref, probs),
               "val_qwk": qwk(y_grade, scores_to_grades(scores)),
               "val_mae": float(np.mean(np.abs(scores - y_grade))),
               "lr": optimizer.param_groups[0]["lr"], "minutes": (time.time() - t0) / 60}
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(out / "log.csv", index=False)
        metric = row["val_auc_referable"] if np.isfinite(row["val_auc_referable"]) else row["val_qwk"]
        improved = metric > best_metric
        print(f"epoch {epoch:2d} | loss {row['train_loss']:.3f} | val referable AUC {row['val_auc_referable']:.4f} "
              f"| val QWK {row['val_qwk']:.4f} | {row['minutes']:.1f} min" + ("  <- best" if improved else ""))
        if improved:
            best_metric, best_epoch = metric, epoch
            save_checkpoint(out / "best.pt", model, args, epoch, row)
    save_checkpoint(out / "last.pt", model, args, args.epochs, log_rows[-1])

    # Tune and lock the thresholds on validation, using the best epoch and flip TTA.
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True)["model"])
    scores, probs = predict(model, val_loader, device, tta=True, amp=amp, desc="validate (TTA)")
    thresholds, qwk_tuned = optimize_thresholds(scores, y_grade)
    ref_threshold = threshold_for_sensitivity(y_ref, probs, args.target_sensitivity)
    ops = binary_metrics(y_ref, probs, ref_threshold)
    locked = {
        "grade_thresholds": thresholds,
        "referable_threshold": ref_threshold,
        "target_sensitivity": args.target_sensitivity,
        "tta": True,
        "best_epoch": best_epoch,
        "validation": {"auc_referable": roc_auc(y_ref, probs),
                       "qwk_default_thresholds": qwk(y_grade, scores_to_grades(scores)),
                       "qwk_tuned_thresholds": qwk_tuned,
                       **{f"referable_{k}": v for k, v in ops.items()}},
    }
    (out / "thresholds.json").write_text(json.dumps(locked, indent=2))
    pd.DataFrame({"image_id": val_df["image_id"], "source": val_df["source"], "grade": y_grade,
                  "referable": y_ref, "score": scores, "prob_referable": probs}
                 ).to_csv(out / "val_predictions.csv", index=False)

    v = locked["validation"]
    print(f"\nBest epoch {best_epoch}. Validation with TTA:")
    print(f"  referable AUC {v['auc_referable']:.4f} | at threshold {ref_threshold:.3f}: "
          f"sensitivity {ops['sensitivity']:.3f}, specificity {ops['specificity']:.3f}")
    print(f"  QWK {qwk_tuned:.4f} with tuned cut-offs {np.round(thresholds, 2).tolist()} "
          f"({v['qwk_default_thresholds']:.4f} with the defaults)")
    print(f"Locked thresholds saved to {out / 'thresholds.json'}. Do not change them after looking at test results.")
    print(f"Next: python evaluate.py --run-dir {out} --csv {args.manifest} --split test")


if __name__ == "__main__":
    main()
