"""Prepare fundus images and patient-level splits.

Training sets (APTOS 2019 and/or EyePACS 2015) -> manifest.csv with train/val/test splits:
    python prepare_data.py train-sets --aptos-dir /kaggle/input/aptos2019-blindness-detection \\
        --out-dir data/processed

External test set (e.g. Messidor-2, IDRiD) -> external_<name>.csv, never used for training or tuning:
    python prepare_data.py external --name messidor2 --csv messidor_data.csv \\
        --image-dir messidor2_images --image-col image_id --label-col adjudicated_dr_grade \\
        --dme-col adjudicated_dme --gradable-col adjudicated_gradable --out-dir data/processed
"""
from __future__ import annotations

import argparse
import json
import os
import re
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

from dr.preprocess import count_ambiguous, crop_fundus, dedup_vector, index_images, lookup_image, strip_image_ext

REFERABLE_GRADE = 2  # moderate NPDR or worse
MANIFEST_COLS = ["path", "source", "image_id", "grade", "referable", "dme", "group", "split"]


# ----------------------------------------------------------------------------- image processing
def _init_worker() -> None:
    cv2.setNumThreads(0)


def _process(task):
    """Crop and resize one image into dst (reused if already done); optionally fingerprint it."""
    src, dst, size, quality, want_vec = task
    out = cv2.imread(dst, cv2.IMREAD_COLOR) if os.path.exists(dst) else None
    if out is None or out.shape[0] != size:
        out = crop_fundus(cv2.imread(src, cv2.IMREAD_COLOR), size)
        if out is None:
            return dst, None, "no_fundus_found"
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if dst.endswith(".jpg") else []
        if not cv2.imwrite(dst, out, params):
            return dst, None, "write_failed"
    return dst, (dedup_vector(out) if want_vec else None), "ok"


def process_images(df, out_dir, size, fmt, quality, workers, dedup_sources=()):
    """Write processed images to out_dir/images/<source>/ and add path, fingerprint and status columns."""
    tasks = []
    for src, source, image_id in zip(df["src_path"], df["source"], df["image_id"]):
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", strip_image_ext(str(image_id)))
        dst = os.path.join(out_dir, "images", source, f"{safe}.{fmt}")
        tasks.append((src, dst, size, quality, source in dedup_sources))
    dsts = [t[1] for t in tasks]
    if len(set(dsts)) != len(dsts):
        raise SystemExit("Two images map to the same output file name; check for repeated image ids.")
    for source in df["source"].unique():
        os.makedirs(os.path.join(out_dir, "images", source), exist_ok=True)

    results = {}
    if workers > 1:
        with Pool(workers, initializer=_init_worker) as pool:
            for dst, vec, status in tqdm(pool.imap_unordered(_process, tasks, chunksize=8),
                                         total=len(tasks), desc="images", mininterval=2.0):
                results[dst] = (vec, status)
    else:
        for task in tqdm(tasks, desc="images", mininterval=2.0):
            dst, vec, status = _process(task)
            results[dst] = (vec, status)

    df = df.copy()
    df["path"] = [os.path.relpath(d, out_dir) for d in dsts]
    df["fingerprint"] = [results[d][0] for d in dsts]
    df["status"] = [results[d][1] for d in dsts]
    return df


def drop_missing(df, out_dir, label, report):
    """Remove label rows whose image file was not found, and say so."""
    missing = df["src_path"].isna()
    report["missing_files"] = int(missing.sum())
    if missing.all():
        raise SystemExit("None of the labelled images were found on disk - check the image folder path.")
    if missing.any():
        path = os.path.join(out_dir, f"missing_{label}.csv")
        df.loc[missing, ["source", "image_id"]].to_csv(path, index=False)
        print(f"WARNING: {int(missing.sum())} labelled images were not found on disk (list: {path})")
    return df[~missing].copy()


def drop_failed(df, out_dir, label, report):
    """Remove images that could not be cropped (e.g. black or unreadable files), and say so."""
    failed = df["status"] != "ok"
    report["failed_images"] = int(failed.sum())
    if failed.any():
        path = os.path.join(out_dir, f"failed_{label}.csv")
        df.loc[failed, ["source", "image_id", "src_path", "status"]].to_csv(path, index=False)
        print(f"WARNING: {int(failed.sum())} images could not be processed (list: {path})")
    return df[~failed].copy()


# ----------------------------------------------------------------------------- de-duplication
def near_duplicate_roots(vectors, threshold):
    """Union-find over all pairs whose fingerprint correlation is >= threshold.

    Returns (group root per image, correlation of each image with its most similar other image).
    """
    n = len(vectors)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nearest = np.full(n, -1.0, dtype=np.float32)
    for start in range(0, n, 1024):
        sim = vectors[start:start + 1024] @ vectors.T
        rows = np.arange(len(sim))
        sim[rows, start + rows] = -1.0  # ignore self-matches
        nearest[start:start + len(sim)] = sim.max(1)
        for i, j in zip(*np.nonzero(sim >= threshold)):
            a, b = find(start + int(i)), find(int(j))
            if a != b:
                parent[b] = a
    return np.array([find(i) for i in range(n)]), nearest


def deduplicate(part, threshold, drop_conflicts):
    """Keep one image per duplicate group; drop groups whose grades disagree (unreliable labels)."""
    roots, nearest = near_duplicate_roots(np.stack(part["fingerprint"].to_list()), threshold)
    part = part.copy()
    part["dup_root"] = roots
    group_size = part.groupby("dup_root")["image_id"].transform("size")
    n_grades = part.groupby("dup_root")["grade"].transform("nunique")
    in_dup = group_size > 1
    conflicting = in_dup & (n_grades > 1)
    keep = ~part.duplicated("dup_root", keep="first")
    if drop_conflicts:
        keep &= ~conflicting
    stats = {
        "threshold": float(threshold),
        "duplicate_groups": int(part.loc[in_dup, "dup_root"].nunique()),
        "images_in_duplicate_groups": int(in_dup.sum()),
        "conflicting_groups": int(part.loc[conflicting, "dup_root"].nunique()),
        "images_removed": int((~keep).sum()),
        "median_nearest_correlation": round(float(np.median(nearest)), 3),
    }
    dup_rows = part.loc[in_dup, ["dup_root", "image_id", "grade", "path"]].sort_values(["dup_root", "image_id"])
    return part[keep].drop(columns="dup_root"), dup_rows, stats, nearest


def print_dup_summary(stats, nearest):
    levels = (0.99, 0.97, 0.95, 0.90, 0.80, 0.70)
    within = ", ".join(f">={t:.2f}: {int((nearest >= t).sum())}" for t in levels)
    print(f"\nAPTOS duplicate check (image correlation, threshold {stats['threshold']:.2f}):")
    print(f"  images whose most similar other image has correlation {within}")
    print(f"  median correlation with the most similar other image: {stats['median_nearest_correlation']}")
    print(f"  {stats['duplicate_groups']} duplicate groups ({stats['images_in_duplicate_groups']} images); "
          f"{stats['conflicting_groups']} with conflicting grades; {stats['images_removed']} images removed")
    print("  Real duplicates sit near 1.0, well apart from everything else. Spot-check a few groups "
          "in duplicates_aptos.csv; if different eyes were merged, raise --dup-threshold.")


# ----------------------------------------------------------------------------- sources and splits
def load_aptos(aptos_dir):
    labels = pd.read_csv(os.path.join(aptos_dir, "train.csv"))
    labels.columns = [str(c).strip() for c in labels.columns]
    image_dir = os.path.join(aptos_dir, "train_images")
    index = index_images(image_dir if os.path.isdir(image_dir) else aptos_dir)
    ids = labels["id_code"].astype(str).str.strip()
    df = pd.DataFrame({"source": "aptos", "image_id": ids, "grade": labels["diagnosis"].astype(int)})
    df["src_path"] = [lookup_image(index, i) for i in ids]
    df["group"] = "aptos_" + ids  # APTOS has no patient ids; duplicates are merged later
    return df


def load_eyepacs(labels_csv, image_dir):
    labels = pd.read_csv(labels_csv)
    labels.columns = [str(c).strip() for c in labels.columns]
    ids = labels["image"].astype(str).str.strip()
    index = index_images(image_dir)
    df = pd.DataFrame({"source": "eyepacs", "image_id": ids, "grade": labels["level"].astype(int)})
    df["src_path"] = [lookup_image(index, i) for i in ids]
    df["group"] = "eyepacs_" + ids.str.split("_").str[0]  # '10_left' and '10_right' = patient 10
    return df


def load_ddr(ddr_dir):
    """DDR (Li et al., 2019): DR_grading/{train,valid,test}/ plus {train,valid,test}.txt ("image grade").

    DDR's official splits are kept. Grade 5 means "ungradable" and is left out.
    """
    if os.path.isdir(os.path.join(ddr_dir, "DR_grading")):
        root = os.path.join(ddr_dir, "DR_grading")
    else:
        found = [r for r, ds, _ in os.walk(ddr_dir) if os.path.basename(r) == "DR_grading"]
        if not found:
            raise SystemExit(f"No DR_grading folder found under {ddr_dir}")
        root = found[0]
    frames = []
    for name, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        labels = pd.read_csv(os.path.join(root, f"{name}.txt"), sep=r"\s+", header=None,
                             names=["image", "grade"], engine="python")
        index = index_images(os.path.join(root, name))
        ids = labels["image"].astype(str).str.strip()
        df = pd.DataFrame({"source": "ddr", "image_id": ids, "grade": labels["grade"].astype(int), "split": split})
        df["src_path"] = [lookup_image(index, i) for i in ids]
        df["group"] = "ddr_" + ids
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    ungradable = df["grade"] == 5
    print(f"DDR: {len(df)} photos; leaving out {int(ungradable.sum())} graded ungradable (5)")
    return df[~ungradable].copy()


def assign_splits(df, val_frac, test_frac, seed):
    """Stratified by source and grade; whole groups (patients) go to exactly one split."""
    df = df.reset_index(drop=True).copy()
    df["split"] = "train"
    strata = (df["source"].astype(str) + "_" + df["grade"].astype(str)).to_numpy()
    groups = df["group"].astype(str).to_numpy()
    if test_frac > 0:
        k = max(2, round(1 / test_frac))
        _, idx = next(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(strata, strata, groups))
        df.loc[idx, "split"] = "test"
    rest = np.flatnonzero(df["split"].to_numpy() == "train")
    if val_frac > 0:
        k = max(2, round((1 - test_frac) / val_frac))
        _, idx = next(StratifiedGroupKFold(k, shuffle=True, random_state=seed + 1)
                      .split(rest, strata[rest], groups[rest]))
        df.loc[rest[idx], "split"] = "val"
    assert df.groupby("group")["split"].nunique().max() == 1, "a patient ended up in two splits"
    return df


def print_split_summary(df):
    order = {"train": 0, "val": 1, "test": 2, "external": 3}
    table = pd.crosstab([df["split"], df["source"]], df["grade"]).reindex(columns=range(5), fill_value=0)
    table.columns = [f"grade {g}" for g in table.columns]
    table["total"] = table.sum(axis=1)
    table["referable"] = df.groupby(["split", "source"])["referable"].mean().map("{:.0%}".format)
    table = table.loc[sorted(table.index, key=lambda k: (order.get(k[0], 9), k[1]))]
    print("\n" + table.to_string())


def counts(df):
    out = {}
    for (split, source), part in df.groupby(["split", "source"]):
        vc = part["grade"].value_counts().sort_index()
        out[f"{split}/{source}"] = {int(k): int(v) for k, v in vc.items()}
    return out


# ----------------------------------------------------------------------------- commands
def cmd_train_sets(args):
    frames = []
    if args.aptos_dir:
        frames.append(load_aptos(args.aptos_dir))
    if args.eyepacs_labels or args.eyepacs_images:
        if not (args.eyepacs_labels and args.eyepacs_images):
            raise SystemExit("Give both --eyepacs-labels and --eyepacs-images.")
        frames.append(load_eyepacs(args.eyepacs_labels, args.eyepacs_images))
    if args.ddr_dir:
        frames.append(load_ddr(args.ddr_dir))
    if not frames:
        raise SystemExit("Nothing to do: give --aptos-dir, --ddr-dir and/or --eyepacs-labels with --eyepacs-images.")
    os.makedirs(args.out_dir, exist_ok=True)
    report = {"command": "train-sets", "args": vars(args)}

    df = pd.concat(frames, ignore_index=True)
    bad = ~df["grade"].between(0, 4)
    if bad.any():
        print(f"WARNING: dropping {int(bad.sum())} rows with grades outside 0-4")
        df = df[~bad]
    df = drop_missing(df, args.out_dir, "train_sets", report)
    if args.limit:
        df = df.sample(frac=1.0, random_state=args.seed).groupby("source").head(args.limit)
    df = process_images(df, args.out_dir, args.size, args.format, args.jpeg_quality, args.workers,
                        dedup_sources={"aptos"})
    df = drop_failed(df, args.out_dir, "train_sets", report)

    parts = []
    for source, part in df.groupby("source", sort=False):
        if source == "aptos" and len(part) > 1:
            part, dup_rows, stats, nearest = deduplicate(part, args.dup_threshold,
                                                         not args.keep_conflicting_duplicates)
            dup_rows.to_csv(os.path.join(args.out_dir, "duplicates_aptos.csv"), index=False)
            report["aptos_duplicates"] = stats
            print_dup_summary(stats, nearest)
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df["grade"] = df["grade"].astype(int)
    df["referable"] = (df["grade"] >= REFERABLE_GRADE).astype(int)
    df["dme"] = np.nan  # APTOS/EyePACS have no DME labels
    preset = df["split"].notna() if "split" in df else pd.Series(False, index=df.index)
    parts = [df[preset]]
    if (~preset).any():
        parts.append(assign_splits(df[~preset].drop(columns="split", errors="ignore"), args.val_frac,
                                   args.test_frac, args.seed))
    df = pd.concat(parts, ignore_index=True)
    assert df.groupby("group")["split"].nunique().max() == 1, "a patient ended up in two splits"

    out = os.path.join(args.out_dir, "manifest.csv")
    df[MANIFEST_COLS].to_csv(out, index=False)
    report["counts"] = counts(df)
    with open(os.path.join(args.out_dir, "prep_report_train_sets.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print_split_summary(df)
    print(f"\nWrote {out} ({len(df)} images).")
    print(f"Next: python train.py --manifest {out} --out runs/b4_512")


def cmd_external(args):
    os.makedirs(args.out_dir, exist_ok=True)
    labels = pd.read_csv(args.csv, sep=None, engine="python")
    labels.columns = [str(c).strip() for c in labels.columns]
    for col in (args.image_col, args.label_col, args.dme_col, args.gradable_col, args.group_col):
        if col and col.strip() not in labels.columns:
            raise SystemExit(f"Column '{col}' not found. Columns in {args.csv}: {list(labels.columns)}")
    report = {"command": "external", "args": vars(args), "rows_in_csv": int(len(labels))}

    if args.gradable_col:
        gradable = labels[args.gradable_col.strip()].astype(str).str.strip().str.lower()
        ok = gradable.isin({"1", "1.0", "true", "yes", "y"})
        report["dropped_ungradable"] = int((~ok).sum())
        labels = labels[ok]
    grade = pd.to_numeric(labels[args.label_col.strip()], errors="coerce")
    valid = grade.between(0, 4) & labels[args.image_col.strip()].notna()
    report["dropped_missing_or_invalid_grade"] = int((~valid).sum())
    labels, grade = labels[valid], grade[valid].astype(int)

    ids = labels[args.image_col.strip()].astype(str).str.strip()
    df = pd.DataFrame({"source": args.name, "image_id": ids, "grade": grade})
    if args.dme_col:
        dme = pd.to_numeric(labels[args.dme_col.strip()], errors="coerce")
        df["dme"] = (dme > 0).astype(float).where(dme.notna())
    else:
        df["dme"] = np.nan
    df["referable"] = ((df["grade"] >= REFERABLE_GRADE) | (df["dme"] == 1)).astype(int)
    patient = labels[args.group_col.strip()].astype(str) if args.group_col else ids
    df["group"] = args.name + "_" + patient
    repeated = df["image_id"].duplicated(keep=False)
    if repeated.any():
        print(f"WARNING: {int(repeated.sum())} rows share an image id with another row, so those ids are "
              "skipped: their labels cannot be matched to one photo reliably. If the label file combines "
              "several folders that reuse file names, prepare each folder separately.")
        df = df[~repeated]

    index = index_images(args.image_dir)
    n_ambiguous = count_ambiguous(index, df["image_id"])
    if n_ambiguous:
        print(f"WARNING: {n_ambiguous} image ids match files in more than one folder under {args.image_dir}, "
              "so they are skipped rather than guessed. Point --image-dir at one folder at a time "
              "(e.g. IDRiD's training and testing folders separately).")
    df["src_path"] = [lookup_image(index, i) for i in df["image_id"]]
    df = drop_missing(df, args.out_dir, f"external_{args.name}", report)
    df = process_images(df, args.out_dir, args.size, args.format, args.jpeg_quality, args.workers)
    df = drop_failed(df, args.out_dir, f"external_{args.name}", report)
    df["split"] = "external"

    out = os.path.join(args.out_dir, f"external_{args.name}.csv")
    df[MANIFEST_COLS].to_csv(out, index=False)
    report["counts"] = counts(df)
    with open(os.path.join(args.out_dir, f"prep_report_external_{args.name}.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print_split_summary(df)
    if args.dme_col:
        dme_only = int(((df["dme"] == 1) & (df["grade"] < REFERABLE_GRADE)).sum())
        if dme_only:
            print(f"{dme_only} images are referable only because of DME. The model never saw DME "
                  "labels, so it may miss some of these.")
    print(f"\nWrote {out} ({len(df)} images).")
    print(f"Next: python evaluate.py --run-dir runs/b4_512 --csv {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--out-dir", default="data/processed")
        sp.add_argument("--size", type=int, default=512, help="output image size in pixels (default 512)")
        sp.add_argument("--format", choices=["jpg", "png"], default="jpg")
        sp.add_argument("--jpeg-quality", type=int, default=95)
        sp.add_argument("--workers", type=int, default=os.cpu_count() or 1)

    t = sub.add_parser("train-sets", help="APTOS/EyePACS -> manifest.csv with train/val/test splits")
    add_common(t)
    t.add_argument("--aptos-dir", help="folder with train.csv and train_images/")
    t.add_argument("--eyepacs-labels", help="EyePACS trainLabels.csv (columns: image, level)")
    t.add_argument("--eyepacs-images", help="folder with the EyePACS training images")
    t.add_argument("--ddr-dir", help="DDR dataset folder (containing DR_grading/); its own splits are kept")
    t.add_argument("--val-frac", type=float, default=0.1)
    t.add_argument("--test-frac", type=float, default=0.1)
    t.add_argument("--dup-threshold", type=float, default=0.95,
                   help="minimum image correlation (0-1) for two APTOS images to count as duplicates")
    t.add_argument("--keep-conflicting-duplicates", action="store_true",
                   help="keep duplicate groups whose grades disagree (default: drop them)")
    t.add_argument("--limit", type=int, default=0, help="max images per source, for quick experiments")
    t.add_argument("--seed", type=int, default=42)

    e = sub.add_parser("external", help="external test set -> external_<name>.csv")
    add_common(e)
    e.add_argument("--name", required=True, help="short name, e.g. messidor2")
    e.add_argument("--csv", required=True, help="label file")
    e.add_argument("--image-dir", required=True)
    e.add_argument("--image-col", required=True, help="column with the image file name or id")
    e.add_argument("--label-col", required=True, help="column with the 0-4 ICDR grade")
    e.add_argument("--dme-col", help="optional DME column; values > 0 make the eye referable")
    e.add_argument("--gradable-col", help="optional; only rows where this is 1/true are kept")
    e.add_argument("--group-col", help="optional patient id column, for patient-level confidence intervals")

    args = p.parse_args()
    if args.command == "train-sets":
        cmd_train_sets(args)
    else:
        cmd_external(args)


if __name__ == "__main__":
    main()
