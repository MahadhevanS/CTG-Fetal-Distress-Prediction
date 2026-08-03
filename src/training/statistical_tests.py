"""
Statistical Significance Testing Module
========================================
Provides tests for comparing Model 8 (Knowledge-Infused Framework) against
standalone temporal baseline models (principally PatchTST standalone).

Tests Implemented:
  1. Wilcoxon Signed-Rank Test  — non-parametric paired comparison over 5-fold CV metrics.
  2. DeLong AUC Comparison Test — statistical comparison of two ROC curves.
  3. run_all_significance_tests()  — convenience wrapper for a full comparison report.

Usage:
    # After generating fold_results for baseline (PatchTST) and Model 8:
    from src.training.statistical_tests import run_all_significance_tests
    report = run_all_significance_tests(baseline_fold_results, model8_fold_results)
    print(report)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


# =============================================================================
# Wilcoxon Signed-Rank Test
# =============================================================================

def wilcoxon_signed_rank_test(
    baseline_scores: List[float],
    model8_scores: List[float],
    metric_name: str = "AUROC",
    alpha: float = 0.05,
) -> Dict:
    """
    Performs a two-sided Wilcoxon Signed-Rank test on paired per-fold metric scores.
    Appropriate for non-Gaussian distributions with n=5 pairs.

    Args:
        baseline_scores: Per-fold metric scores from the baseline model (e.g., PatchTST standalone).
        model8_scores:   Per-fold metric scores from Model 8 (Knowledge-Infused Framework).
        metric_name:     Human-readable metric label for the report.
        alpha:           Significance threshold (default 0.05).

    Returns:
        Dict containing: metric, n_folds, W_statistic, p_value, significant, delta_mean, verdict.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        raise ImportError("scipy is required for statistical tests. Install via: pip install scipy")

    assert len(baseline_scores) == len(model8_scores), (
        f"Fold count mismatch: baseline={len(baseline_scores)}, model8={len(model8_scores)}"
    )

    baseline = np.array(baseline_scores, dtype=float)
    model8 = np.array(model8_scores, dtype=float)
    delta = model8 - baseline

    if np.all(delta == 0):
        return {
            "metric": metric_name,
            "n_folds": len(baseline_scores),
            "W_statistic": float("nan"),
            "p_value": 1.0,
            "significant": False,
            "delta_mean": 0.0,
            "verdict": f"No difference between models on {metric_name}.",
        }

    stat, p_val = wilcoxon(model8, baseline, alternative="two-sided", zero_method="wilcox")
    significant = bool(p_val < alpha)
    delta_mean = float(delta.mean())
    direction = "higher" if delta_mean > 0 else "lower"

    verdict = (
        f"Model 8 is {'statistically significantly ' if significant else 'NOT significantly '}"
        f"{direction} than baseline on {metric_name} "
        f"(W={stat:.3f}, p={p_val:.4f}, Δmean={delta_mean:+.4f})."
    )

    return {
        "metric": metric_name,
        "n_folds": len(baseline_scores),
        "W_statistic": float(stat),
        "p_value": float(p_val),
        "significant": significant,
        "delta_mean": delta_mean,
        "verdict": verdict,
    }


# =============================================================================
# DeLong AUC Comparison Test
# =============================================================================

def _compute_auc_variance(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Computes AUC and placement values for DeLong test."""
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    pos_probs = y_prob[pos_mask]
    neg_probs = y_prob[neg_mask]
    n_pos = len(pos_probs)
    n_neg = len(neg_probs)

    # Placement values (DeLong et al., 1988)
    V_pos = np.array([np.mean(p > neg_probs) + 0.5 * np.mean(p == neg_probs) for p in pos_probs])
    V_neg = np.array([np.mean(n < pos_probs) + 0.5 * np.mean(n == pos_probs) for n in neg_probs])
    theta = np.mean(V_pos)

    return theta, V_pos, V_neg


def delong_roc_test(
    y_true: np.ndarray,
    y_prob_baseline: np.ndarray,
    y_prob_model8: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    DeLong's test for comparing two correlated ROC curves.
    Reference: DeLong et al. (1988) — "Comparing the Areas under Two or More
    Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach."

    Appropriate when both models were evaluated on the same data (paired).

    Args:
        y_true:           True binary labels (0/1), shape (N,).
        y_prob_baseline:  Probability predictions from baseline model, shape (N,).
        y_prob_model8:    Probability predictions from Model 8, shape (N,).
        alpha:            Significance threshold (default 0.05).

    Returns:
        Dict containing: auc_baseline, auc_model8, delta_auc, z_score, p_value, significant, verdict.
    """
    try:
        from scipy.stats import norm
    except ImportError:
        raise ImportError("scipy is required. Install via: pip install scipy")

    y_true = np.array(y_true, dtype=int)
    y_prob_b = np.array(y_prob_baseline, dtype=float)
    y_prob_m = np.array(y_prob_model8, dtype=float)

    auc_b, V_pos_b, V_neg_b = _compute_auc_variance(y_true, y_prob_b)
    auc_m, V_pos_m, V_neg_m = _compute_auc_variance(y_true, y_prob_m)

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    # Covariance matrix of the two AUCs
    S10 = np.cov(V_pos_b, V_pos_m)[0, 1] / n_pos
    S01 = np.cov(V_neg_b, V_neg_m)[0, 1] / n_neg
    S = np.array([
        [np.var(V_pos_b) / n_pos + np.var(V_neg_b) / n_neg,
         S10 + S01],
        [S10 + S01,
         np.var(V_pos_m) / n_pos + np.var(V_neg_m) / n_neg],
    ])

    # Difference vector [1, -1] selects AUC_model8 - AUC_baseline
    L = np.array([1.0, -1.0])
    var_delta = L @ S @ L.T
    se = np.sqrt(max(var_delta, 1e-12))
    delta_auc = auc_m - auc_b
    z = delta_auc / se
    p_val = float(2.0 * (1.0 - norm.cdf(abs(z))))
    significant = p_val < alpha

    direction = "higher" if delta_auc > 0 else "lower"
    verdict = (
        f"Model 8 AUC ({auc_m:.4f}) is {'significantly ' if significant else 'NOT significantly '}"
        f"{direction} than baseline AUC ({auc_b:.4f}) "
        f"(ΔAUC={delta_auc:+.4f}, Z={z:.3f}, p={p_val:.4f})."
    )

    return {
        "auc_baseline": float(auc_b),
        "auc_model8": float(auc_m),
        "delta_auc": float(delta_auc),
        "z_score": float(z),
        "p_value": p_val,
        "significant": significant,
        "verdict": verdict,
    }


# =============================================================================
# Convenience Wrapper
# =============================================================================

def run_all_significance_tests(
    baseline_fold_results: Dict[str, List[float]],
    model8_fold_results: Dict[str, List[float]],
    alpha: float = 0.05,
    delong_data: Optional[Dict] = None,
) -> str:
    """
    Runs Wilcoxon tests on all shared metric keys and optionally the DeLong test.
    Produces a formatted report string.

    Args:
        baseline_fold_results: Dict mapping metric_name → List of per-fold values (baseline).
        model8_fold_results:   Dict mapping metric_name → List of per-fold values (Model 8).
        alpha:                 Significance level.
        delong_data:           Optional dict with keys "y_true", "y_prob_baseline", "y_prob_model8"
                               for running the DeLong ROC comparison test.

    Returns:
        str: Formatted significance testing report.

    Example:
        baseline = {"auroc": [0.74, 0.76, 0.72, 0.75, 0.73], "sens_at_90spec": [34, 36, 33, 35, 35]}
        model8   = {"auroc": [0.76, 0.78, 0.75, 0.77, 0.75], "sens_at_90spec": [37, 39, 36, 38, 38]}
        print(run_all_significance_tests(baseline, model8))
    """
    # Metric labels with display names
    METRIC_DISPLAY = {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "f1": "F1 Score",
        "sens_at_90spec": "Sens@90%Spec",
        "recall": "Recall/Sensitivity",
        "specificity": "Specificity",
    }

    lines = []
    lines.append("=" * 65)
    lines.append(" Statistical Significance Tests: Model 8 vs. Baseline")
    lines.append(f" Significance Threshold: α = {alpha}")
    lines.append("=" * 65)

    # Wilcoxon tests for all available metrics
    for key, display in METRIC_DISPLAY.items():
        if key not in baseline_fold_results or key not in model8_fold_results:
            continue
        b_scores = baseline_fold_results[key]
        m_scores = model8_fold_results[key]
        if len(b_scores) != len(m_scores) or len(b_scores) == 0:
            continue

        result = wilcoxon_signed_rank_test(b_scores, m_scores, metric_name=display, alpha=alpha)
        sig_flag = "✓ SIGNIFICANT" if result["significant"] else "✗ Not significant"
        lines.append(f"\n[{display}]")
        lines.append(f"  Baseline: {np.mean(b_scores):.4f} ± {np.std(b_scores):.4f}")
        lines.append(f"  Model 8:  {np.mean(m_scores):.4f} ± {np.std(m_scores):.4f}")
        lines.append(f"  Δ mean:   {result['delta_mean']:+.4f}")
        lines.append(f"  W = {result['W_statistic']:.3f}, p = {result['p_value']:.4f}  [{sig_flag}]")

    # DeLong test (optional)
    if delong_data is not None:
        lines.append("\n" + "-" * 65)
        lines.append("[DeLong ROC Comparison Test]")
        try:
            dl_result = delong_roc_test(
                y_true=np.array(delong_data["y_true"]),
                y_prob_baseline=np.array(delong_data["y_prob_baseline"]),
                y_prob_model8=np.array(delong_data["y_prob_model8"]),
                alpha=alpha,
            )
            lines.append(f"  Baseline AUC: {dl_result['auc_baseline']:.4f}")
            lines.append(f"  Model 8 AUC:  {dl_result['auc_model8']:.4f}")
            lines.append(f"  ΔAUC:         {dl_result['delta_auc']:+.4f}")
            lines.append(f"  Z = {dl_result['z_score']:.3f}, p = {dl_result['p_value']:.4f}")
            flag = "✓ SIGNIFICANT" if dl_result["significant"] else "✗ Not significant"
            lines.append(f"  [{flag}]")
        except Exception as e:
            lines.append(f"  DeLong test failed: {e}")

    lines.append("\n" + "=" * 65)
    report = "\n".join(lines)
    print(report)
    return report
