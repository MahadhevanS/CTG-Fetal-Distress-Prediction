"""
Decision Threshold Optimization & Probability Calibration for Model 8
======================================================================
Provides two complementary post-training improvements to the binary
distress classifier:

  1. ThresholdOptimizer:
     The default sigmoid threshold of 0.5 is almost certainly sub-optimal for
     an imbalanced clinical classification problem. This class searches the
     [0.05, 0.95] range during validation to find the threshold maximizing a
     chosen objective (F1, Sens@90%Spec, or balanced F1 with clinical weighting).
     One optimal threshold is stored per fold.

  2. TemperatureScaler:
     Post-hoc calibration that learns a single scalar temperature T such that
     sigmoid(logit / T) is well-calibrated (ECE ≈ 0). Reports ECE and Brier Score.
     Important if clinicians rely on the outputted probability estimate.

Usage:
    # Threshold optimization (run at end of each fold's validation)
    optimizer = ThresholdOptimizer(objective="sens_at_90spec")
    best_thresh, best_score = optimizer.fit(val_probs, val_targets)
    metrics_at_best = optimizer.evaluate(val_probs, val_targets, best_thresh)

    # Temperature scaling (run once after all folds, on combined val probs)
    scaler = TemperatureScaler()
    T = scaler.fit(val_logits, val_targets)
    calibrated_probs = scaler.calibrate(val_logits)
    ece = scaler.expected_calibration_error(calibrated_probs, val_targets)
"""

from typing import Dict, Optional, Tuple

import numpy as np


# =============================================================================
# 1. Threshold Optimizer
# =============================================================================

class ThresholdOptimizer:
    """
    Finds the optimal decision threshold on the validation set for a given metric.

    Searches the grid [threshold_min, threshold_max] at `n_steps` points and
    selects the threshold that maximizes the chosen objective.

    Args:
        objective:       One of 'f1', 'sens_at_90spec', 'balanced_f1', 'recall_at_precision'.
                         Default: 'f1'.
        threshold_min:   Lower bound of search grid (default 0.05).
        threshold_max:   Upper bound of search grid (default 0.95).
        n_steps:         Number of grid points to evaluate (default 91 → step 0.01).
        min_precision:   For 'recall_at_precision' objective — minimum required PPV.
    """

    def __init__(
        self,
        objective: str = "f1",
        threshold_min: float = 0.05,
        threshold_max: float = 0.95,
        n_steps: int = 91,
        min_precision: float = 0.20,
    ):
        self.objective = objective.lower()
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.n_steps = n_steps
        self.min_precision = min_precision
        self.best_threshold: float = 0.5
        self.best_score: float = 0.0
        self._grid = np.linspace(threshold_min, threshold_max, n_steps)

    def _compute_objective(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        threshold: float,
    ) -> float:
        """Compute the selected objective metric at a given threshold."""
        y_pred = (y_probs >= threshold).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

        if self.objective == "f1":
            return 2 * precision * recall / (precision + recall + 1e-8)

        elif self.objective == "sens_at_90spec":
            # Only valid if specificity >= 0.90
            return float(recall) if specificity >= 0.90 else 0.0

        elif self.objective == "balanced_f1":
            # Harmonic mean of sensitivity and specificity (like geometric mean of
            # sensitivity/specificity — balances both clinical concerns)
            return 2 * recall * specificity / (recall + specificity + 1e-8)

        elif self.objective == "recall_at_precision":
            # Maximize recall subject to precision >= min_precision
            return float(recall) if precision >= self.min_precision else 0.0

        else:
            raise ValueError(f"Unknown objective: '{self.objective}'")

    def fit(
        self,
        y_probs: np.ndarray,
        y_true: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Search for the optimal threshold on validation data.

        Args:
            y_probs: Predicted probabilities (N,), values in [0, 1].
            y_true:  Binary labels (N,), values in {0, 1}.

        Returns:
            (best_threshold, best_objective_score)
        """
        best_thresh = 0.5
        best_score = -1.0

        for thresh in self._grid:
            score = self._compute_objective(y_true, y_probs, thresh)
            if score > best_score:
                best_score = score
                best_thresh = thresh

        self.best_threshold = float(best_thresh)
        self.best_score = float(best_score)
        return self.best_threshold, self.best_score

    def evaluate(
        self,
        y_probs: np.ndarray,
        y_true: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute full metric set at a given (or best-found) threshold.

        Args:
            y_probs:   Predicted probabilities (N,).
            y_true:    Binary labels (N,).
            threshold: Threshold to use. Defaults to self.best_threshold.

        Returns:
            Dict with accuracy, precision, recall, specificity, f1, sens_at_90spec.
        """
        if threshold is None:
            threshold = self.best_threshold

        y_pred = (y_probs >= threshold).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        total = len(y_true)

        accuracy = (tp + tn) / (total + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Sens @ 90% Spec from ROC curve
        try:
            from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
            auroc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
            p_vals, r_vals, _ = precision_recall_curve(y_true, y_probs)
            auprc = auc(r_vals, p_vals) if len(np.unique(y_true)) > 1 else 0.0
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            spec_arr = 1.0 - fpr
            idx = np.where(spec_arr >= 0.90)[0]
            sens_at_90spec = float(tpr[idx[-1]]) * 100.0 if len(idx) > 0 else 0.0
        except ImportError:
            auroc, auprc, sens_at_90spec = 0.5, 0.0, 0.0

        return {
            "accuracy": accuracy * 100.0,
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1,
            "precision": precision * 100.0,
            "recall": recall * 100.0,
            "specificity": specificity * 100.0,
            "sens_at_90spec": sens_at_90spec,
            "threshold_used": threshold,
        }


# =============================================================================
# 2. Temperature Scaler
# =============================================================================

class TemperatureScaler:
    """
    Post-hoc probability calibration via Temperature Scaling.

    Learns a single scalar temperature T > 0 that minimizes NLL on a held-out
    validation set. Calibrated probability: p = sigmoid(logit / T).

    T > 1: Model is over-confident → probabilities pushed toward 0.5.
    T < 1: Model is under-confident → probabilities pushed toward extremes.
    T = 1: No change.

    Reports:
      - Expected Calibration Error (ECE): reliability diagram metric
      - Brier Score: mean squared error between probability and true label

    Usage:
        scaler = TemperatureScaler()
        T = scaler.fit(val_logits, val_targets)
        cal_probs = scaler.calibrate(test_logits)
        ece = scaler.expected_calibration_error(cal_probs, test_targets)
    """

    def __init__(self, n_bins: int = 10):
        self.temperature: float = 1.0
        self.n_bins = n_bins

    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Fit temperature parameter on validation set by minimizing NLL.

        Args:
            logits:   Raw sigmoid logits (before sigmoid) of shape (N,).
            y_true:   Binary labels of shape (N,).
            lr:       Learning rate for optimization (default 0.01).
            max_iter: Maximum optimization iterations (default 100).

        Returns:
            Optimized temperature scalar T.
        """
        try:
            import torch
            import torch.optim as optim

            logits_t = torch.tensor(logits, dtype=torch.float32)
            y_t = torch.tensor(y_true, dtype=torch.float32)
            T = torch.nn.Parameter(torch.ones(1))
            optimizer = optim.LBFGS([T], lr=lr, max_iter=max_iter)

            def _nll():
                optimizer.zero_grad()
                cal_logits = logits_t / T.clamp(min=1e-3)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(cal_logits, y_t)
                loss.backward()
                return loss

            optimizer.step(_nll)
            self.temperature = float(T.item())

        except ImportError:
            # Fallback: grid search temperature
            best_nll = float("inf")
            for T_cand in np.linspace(0.1, 5.0, 100):
                cal_logits = logits / T_cand
                probs = 1.0 / (1.0 + np.exp(-cal_logits))
                nll = -np.mean(
                    y_true * np.log(probs + 1e-8) + (1 - y_true) * np.log(1 - probs + 1e-8)
                )
                if nll < best_nll:
                    best_nll = nll
                    self.temperature = T_cand

        return self.temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to convert raw logits to calibrated probabilities.

        Args:
            logits: Raw model logits (N,).

        Returns:
            Calibrated probabilities (N,) in [0, 1].
        """
        T = max(self.temperature, 1e-3)
        cal_logits = logits / T
        return 1.0 / (1.0 + np.exp(-cal_logits))

    def expected_calibration_error(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE) using equal-width probability bins.

        ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

        Args:
            probs:  Calibrated probabilities (N,) in [0, 1].
            y_true: Binary labels (N,).

        Returns:
            ECE scalar (lower is better; 0 = perfectly calibrated).
        """
        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece = 0.0
        n_total = len(probs)

        for i in range(self.n_bins):
            in_bin = (probs >= bins[i]) & (probs < bins[i + 1])
            n_bin = in_bin.sum()
            if n_bin == 0:
                continue
            acc = y_true[in_bin].mean()
            conf = probs[in_bin].mean()
            ece += (n_bin / n_total) * abs(acc - conf)

        return float(ece)

    def brier_score(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """
        Compute Brier Score: mean squared error between probability and true label.

        Brier Score = (1/N) Σ (p_i - y_i)²

        Args:
            probs:  Probabilities (N,) in [0, 1].
            y_true: Binary labels (N,).

        Returns:
            Brier Score scalar (lower is better; 0.25 = random classifier).
        """
        return float(np.mean((probs - y_true) ** 2))
