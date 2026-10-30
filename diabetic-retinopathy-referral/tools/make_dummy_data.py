"""Create synthetic 'fundus' images in APTOS-, EyePACS- and IDRiD-style folder layouts.

Only for checking that the pipeline runs end to end. The images are cartoons whose number of
dark-red dots and yellow spots grows with the grade, so a model can learn *something*, but the
resulting scores mean nothing clinically.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd

GRADE_P = [0.45, 0.12, 0.25, 0.08, 0.10]
LESIONS = {0: (0, 0), 1: (5, 0), 2: (15, 4), 3: (35, 10), 4: (50, 15)}  # mean red dots, yellow spots


def fake_fundus(rng: np.random.Generator, grade: int, w: int = 640, h: int = 480) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    cx, cy, r = w // 2, h // 2, int(min(w, h) * 0.48)
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    inside = dist <= 1.0
    colour = np.array([40, 80, 180], np.float32) * rng.uniform(0.8, 1.2)  # BGR orange-red
    shaded = colour * np.clip(1.15 - 0.55 * dist, 0, 1)[..., None]
    img[inside] = shaded[inside].astype(np.uint8)

    def point_inside(max_frac=0.9):
        rad, ang = r * max_frac * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
        return int(cx + rad * np.cos(ang)), int(cy + rad * np.sin(ang))

    disc = (int(cx + rng.choice([-1, 1]) * r * 0.45), int(cy + rng.uniform(-0.1, 0.1) * r))
    for _ in range(8):  # vessels radiating from the optic disc
        pts, ang = [disc], rng.uniform(0, 2 * np.pi)
        for _ in range(6):
            ang += rng.uniform(-0.5, 0.5)
            pts.append((int(pts[-1][0] + np.cos(ang) * r * 0.2), int(pts[-1][1] + np.sin(ang) * r * 0.2)))
        cv2.polylines(img, [np.array(pts, np.int32)], False, (25, 35, 120), int(rng.integers(2, 5)))
    cv2.circle(img, disc, int(r * 0.12), (130, 205, 240), -1)
    n_red, n_yellow = LESIONS[grade]
    for _ in range(rng.poisson(n_red)):
        cv2.circle(img, point_inside(), int(rng.integers(2, 6)), (15, 15, 95), -1)
    for _ in range(rng.poisson(n_yellow)):
        cv2.circle(img, point_inside(), int(rng.integers(3, 8)), (90, 215, 235), -1)
    img[~inside] = 0
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="smoke/raw")
    p.add_argument("--n-aptos", type=int, default=200)
    p.add_argument("--n-eyepacs-patients", type=int, default=100)
    p.add_argument("--n-external", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    # APTOS layout: train.csv (id_code, diagnosis) + train_images/<id>.png
    aptos_dir = os.path.join(a.out, "aptos", "train_images")
    os.makedirs(aptos_dir, exist_ok=True)
    rows, images = [], []
    for _ in range(a.n_aptos):
        grade, code = int(rng.choice(5, p=GRADE_P)), rng.bytes(6).hex()
        img = fake_fundus(rng, grade, 800, 600)
        cv2.imwrite(os.path.join(aptos_dir, f"{code}.png"), img)
        rows.append((code, grade))
        images.append(img)
    # Duplicates like those in the real APTOS data: exact copies, smaller copies, conflicting labels.
    for k in range(6):
        code, grade = rows[k]
        copy = cv2.resize(images[k], (600, 450), interpolation=cv2.INTER_AREA) if k in (2, 3) else images[k]
        label = (grade + 2) % 5 if k >= 4 else grade
        new = rng.bytes(6).hex()
        cv2.imwrite(os.path.join(aptos_dir, f"{new}.png"), copy)
        rows.append((new, label))
    black = rng.bytes(6).hex()  # an unusable image, which prepare_data.py should reject
    cv2.imwrite(os.path.join(aptos_dir, f"{black}.png"), np.zeros((600, 800, 3), np.uint8))
    rows.append((black, 0))
    pd.DataFrame(rows, columns=["id_code", "diagnosis"]).to_csv(
        os.path.join(a.out, "aptos", "train.csv"), index=False)

    # EyePACS layout: trainLabels.csv (image, level) + train/<patient>_<eye>.jpeg
    eye_dir = os.path.join(a.out, "eyepacs", "train")
    os.makedirs(eye_dir, exist_ok=True)
    rows = []
    for pid in range(1, a.n_eyepacs_patients + 1):
        grade = int(rng.choice(5, p=GRADE_P))
        for eye in ("left", "right"):
            g = int(np.clip(grade + rng.integers(-1, 2), 0, 4)) if rng.random() < 0.3 else grade
            cv2.imwrite(os.path.join(eye_dir, f"{pid}_{eye}.jpeg"), fake_fundus(rng, g, 900, 600))
            rows.append((f"{pid}_{eye}", g))
    pd.DataFrame(rows, columns=["image", "level"]).to_csv(
        os.path.join(a.out, "eyepacs", "trainLabels.csv"), index=False)

    # IDRiD-style external set; the header with a trailing space tests that it gets stripped.
    ext_dir = os.path.join(a.out, "external", "images")
    os.makedirs(ext_dir, exist_ok=True)
    rows = []
    for i in range(1, a.n_external + 1):
        grade = int(rng.choice(5, p=GRADE_P))
        dme = int(rng.integers(1, 3)) if grade >= 2 and rng.random() < 0.5 else 0
        name = f"IDRiD_{i:03d}"
        cv2.imwrite(os.path.join(ext_dir, f"{name}.jpg"), fake_fundus(rng, grade, 700, 500))
        rows.append((name, grade, dme))
    pd.DataFrame(rows, columns=["Image name", "Retinopathy grade", "Risk of macular edema "]).to_csv(
        os.path.join(a.out, "external", "labels.csv"), index=False)
    print(f"Synthetic data written to {a.out}/")


if __name__ == "__main__":
    main()
