"""
DeLong's test for two correlated ROC curves (DeLong et al., Biometrics 44, 1988).

The paper this project benchmarks against uses DeLong's test for its AUC
comparisons, so using the same test keeps the comparison on equal terms.

WHY THIS MATTERS HERE: comparing fold-mean AUROCs is not a significance test.
Per-fold AUROC in this project swings 0.72-0.91 on a single model, which is far
wider than the effect sizes being claimed (~0.01-0.03). DeLong's test compares
two models' predictions on the SAME samples, so the paired structure removes
that between-fold variance and answers the actual question: on identical
patients, is model A ranking better than model B?

Requires both models' out-of-fold predictions on the SAME samples in the SAME
order -- i.e. both trained with the same fold partition. Use
--fold_mode patient_level on train_ctg_crossformer.py to guarantee that against
Model 8.

Usage (as a library):
    from scripts.delong_test import delong_roc_test
    p, auc_a, auc_b = delong_roc_test(y_true, scores_a, scores_b)
"""

from typing import Tuple

import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks, handling ties (the T-vector in DeLong's formulation)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, m: int):
    """
    Fast DeLong covariance (Sun & Xu, IEEE SPL 21(11), 2014).

    Args:
        predictions_sorted_transposed: (k, n) array, positives first (m of them).
        m: number of positive samples.
    Returns:
        (aucs (k,), covariance (k, k))
    """
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_roc_test(
    y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray
) -> Tuple[float, float, float]:
    """
    Two-sided DeLong test that AUC(A) == AUC(B) on the same samples.

    Args:
        y_true:   (n,) binary labels.
        scores_a: (n,) model A scores (higher = more positive).
        scores_b: (n,) model B scores, same sample order as A.

    Returns:
        (p_value, auc_a, auc_b)
    """
    y_true = np.asarray(y_true).astype(int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    assert y_true.shape == scores_a.shape == scores_b.shape, "shape mismatch"
    assert set(np.unique(y_true).tolist()) == {0, 1}, "y_true must be binary with both classes"

    order = np.argsort(-y_true, kind="mergesort")  # positives first, stable
    label_sorted = y_true[order]
    m = int(label_sorted.sum())
    preds = np.vstack([scores_a[order], scores_b[order]])

    aucs, cov = _fast_delong(preds, m)
    l = np.array([[1, -1]], dtype=float)
    var = float((l @ cov @ l.T).item())
    if var <= 0:
        return 1.0, float(aucs[0]), float(aucs[1])
    z = float(aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(p), float(aucs[0]), float(aucs[1])


def report(y_true, scores_a, scores_b, name_a="Model A", name_b="Model B", alpha=0.05):
    """Prints a formatted comparison. Returns the p-value."""
    p, auc_a, auc_b = delong_roc_test(y_true, scores_a, scores_b)
    print(f"  {name_a:<32}: AUROC {auc_a:.4f}")
    print(f"  {name_b:<32}: AUROC {auc_b:.4f}")
    print(f"  difference                      : {auc_a - auc_b:+.4f}")
    print(f"  DeLong p-value                  : {p:.4g}  "
          f"({'significant' if p < alpha else 'NOT significant'} at alpha={alpha})")
    print(f"  n = {len(y_true)} samples, {int(np.sum(y_true))} positive")
    return p


if __name__ == "__main__":
    # Self-check: a clearly better model should be significant; identical scores
    # should be non-significant with a difference of exactly zero.
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.binomial(1, 0.15, n)
    good = y * rng.normal(1.2, 1.0, n) + (1 - y) * rng.normal(0.0, 1.0, n)
    weak = y * rng.normal(0.5, 1.0, n) + (1 - y) * rng.normal(0.0, 1.0, n)
    print("Self-check 1 -- clearly better vs weaker model:")
    report(y, good, weak, "strong", "weak")
    print("\nSelf-check 2 -- identical scores (must be p=1.0, diff=0):")
    report(y, good, good.copy(), "model", "same model")
