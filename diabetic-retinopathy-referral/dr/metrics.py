"""Metrics: QWK, threshold tuning, referable-DR operating point, calibration, bootstrap CIs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

N_GRADES = 5
DEFAULT_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]


def scores_to_grades(scores, thresholds=DEFAULT_THRESHOLDS) -> np.ndarray:
    """Turn continuous grade scores into grades 0-4 using four cut-offs."""
    return np.digitize(np.asarray(scores, dtype=float), np.sort(np.asarray(thresholds, dtype=float)))


def confusion(y_true, y_pred, n: int = N_GRADES) -> np.ndarray:
    """Confusion matrix, rows = true grade, columns = predicted grade."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return np.bincount(y_true * n + y_pred, minlength=n * n).reshape(n, n)


def qwk(y_true, y_pred, n: int = N_GRADES) -> float:
    """Quadratic weighted kappa (same value as sklearn's cohen_kappa_score(weights='quadratic'))."""
    observed = confusion(y_true, y_pred, n).astype(float)
    weights = (np.subtract.outer(np.arange(n), np.arange(n)) ** 2).astype(float)
    expected = np.outer(observed.sum(1), observed.sum(0)) / max(observed.sum(), 1.0)
    denom = (weights * expected).sum()
    return float(1.0 - (weights * observed).sum() / denom) if denom > 0 else float("nan")


def optimize_thresholds(scores, y_true, init=DEFAULT_THRESHOLDS, passes: int = 3, steps: int = 80):
    """Coordinate search for the grade cut-offs that maximise QWK. Use on validation data only."""
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    t = [float(v) for v in init]
    best = qwk(y_true, scores_to_grades(scores, t))
    for _ in range(passes):
        for i in range(len(t)):
            lo = t[i - 1] if i > 0 else min(t[0], scores.min()) - 0.5
            hi = t[i + 1] if i + 1 < len(t) else max(t[-1], scores.max()) + 0.5
            for c in np.linspace(lo, hi, steps)[1:-1]:
                cand = t.copy()
                cand[i] = float(c)
                k = qwk(y_true, scores_to_grades(scores, cand))
                if k > best + 1e-9:
                    best, t = k, cand
    return t, best


def roc_auc(y_true, prob) -> float:
    y = np.asarray(y_true).astype(int)
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, prob))


def threshold_for_sensitivity(y_true, prob, target: float = 0.90) -> float:
    """Highest probability cut-off that still reaches the target sensitivity on this data."""
    pos = np.sort(np.asarray(prob, dtype=float)[np.asarray(y_true).astype(bool)])[::-1]
    if pos.size == 0:
        return 0.5
    k = min(pos.size, max(1, int(np.ceil(target * pos.size))))
    return float(pos[k - 1])


def binary_metrics(y_true, prob, threshold: float) -> dict:
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(prob, dtype=float) >= threshold
    tp, fn = int(np.sum(pred & y)), int(np.sum(~pred & y))
    tn, fp = int(np.sum(~pred & ~y)), int(np.sum(pred & ~y))

    def div(a, b):
        return a / b if b else float("nan")

    return {"sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def expected_calibration_error(y_true, prob, n_bins: int = 10) -> float:
    """Average gap between predicted probability and observed frequency (0 = perfectly calibrated)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def group_bootstrap(stat_fn, groups, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile confidence intervals that resample whole groups (patients) with replacement.

    stat_fn(idx) must return {metric name: value} computed on the rows idx. Resampling patients
    rather than images keeps both eyes of a person together, so the intervals are not too narrow.
    """
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(inv))[:-1])
    samples: dict[str, list] = {}
    for _ in range(n_boot):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        for key, value in stat_fn(idx).items():
            if value is not None and np.isfinite(value):
                samples.setdefault(key, []).append(value)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in samples.items()}
