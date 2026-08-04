"""
Structured Error Analysis Module for Model 8 (KI-MTF)
=======================================================
Categorizes all false predictions from a validation fold to identify
systematic failure modes. This analysis directly informs the Discussion
section of the research paper.

Error categories examined:
  - False Negatives (FN): Missed distress — clinically the most dangerous
    * Categorized by predicted FIGO class, LTV level, deceleration counts
  - False Positives (FP): False alarms — clinically costly (unnecessary surgery)
    * Categorized by baseline FHR, LTV variability, UC pattern

Output:
  - Per-fold markdown/JSON report
  - Aggregated error pattern summary across folds

Usage:
    from src.training.error_analysis import ErrorAnalyzer

    analyzer = ErrorAnalyzer()
    report = analyzer.analyze(
        y_true=val_targets,
        y_probs=val_probs,
        y_figo_pred=figo_class_predictions,
        y_features_pred=feature_predictions,  # (N, 8) — clinical features
        patient_ids=val_patient_ids,
        threshold=0.5,
    )
    analyzer.print_report(report)
    analyzer.save_report(report, "checkpoints/model8/results/fold1_error_analysis.json")
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Feature index constants (matches ClinicalFeatureHead output mapping)
# =============================================================================
IDX_BASELINE_FHR = 0
IDX_STV = 1
IDX_LTV = 2
IDX_ACCELS = 3
IDX_EARLY_DECELS = 4
IDX_LATE_DECELS = 5
IDX_VAR_DECELS = 6
IDX_PROLONGED_DECELS = 7

FIGO_CLASS_NAMES = {0: "Normal", 1: "Suspicious", 2: "Pathological"}

# LTV clinical ranges (FIGO 2015)
LTV_REDUCED_THRESH = 5.0   # bpm — below this is clinically concerning
LTV_NORMAL_MIN = 5.0
LTV_NORMAL_MAX = 25.0

# Baseline clinical range
BASELINE_NORMAL_MIN = 110.0
BASELINE_NORMAL_MAX = 160.0


# =============================================================================
# ErrorAnalyzer
# =============================================================================

class ErrorAnalyzer:
    """
    Analyzes and categorizes false predictions from a validation fold.

    Args:
        feature_names: Names of the 8 clinical features (default: standard CTG names).
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or [
            "Baseline FHR (bpm)",
            "STV (bpm)",
            "LTV (bpm)",
            "Acceleration Count",
            "Early Decel Count",
            "Late Decel Count",
            "Variable Decel Count",
            "Prolonged Decel Count",
        ]

    def analyze(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        y_figo_pred: Optional[np.ndarray] = None,
        y_features_pred: Optional[np.ndarray] = None,
        patient_ids: Optional[List[str]] = None,
        threshold: float = 0.5,
        fold_idx: Optional[int] = None,
    ) -> Dict:
        """
        Categorize all false predictions and return a structured report.

        Args:
            y_true:         Binary ground-truth labels (N,).
            y_probs:        Predicted probabilities (N,), values in [0, 1].
            y_figo_pred:    Predicted FIGO class indices (N,) — optional.
            y_features_pred: Predicted clinical features (N, 8) — optional.
            patient_ids:    List of patient ID strings (N,) — optional.
            threshold:      Decision threshold (default 0.5).
            fold_idx:       Fold number for report labeling.

        Returns:
            Dict containing:
              - summary: Overall TP/TN/FP/FN counts
              - false_negatives: Detailed FN analysis
              - false_positives: Detailed FP analysis
              - patterns: Identified systematic failure patterns
        """
        y_pred = (y_probs >= threshold).astype(int)

        # --- Binary classification masks ---
        tp_mask = (y_pred == 1) & (y_true == 1)
        tn_mask = (y_pred == 0) & (y_true == 0)
        fp_mask = (y_pred == 1) & (y_true == 0)
        fn_mask = (y_pred == 0) & (y_true == 1)

        n_tp = int(tp_mask.sum())
        n_tn = int(tn_mask.sum())
        n_fp = int(fp_mask.sum())
        n_fn = int(fn_mask.sum())
        n_total = len(y_true)

        # Confidence statistics for each error type
        report = {
            "fold": fold_idx,
            "threshold": threshold,
            "summary": {
                "n_total": n_total,
                "n_positive": int(y_true.sum()),
                "n_negative": int((1 - y_true).sum()),
                "TP": n_tp,
                "TN": n_tn,
                "FP": n_fp,
                "FN": n_fn,
                "sensitivity": float(n_tp / (n_tp + n_fn + 1e-8) * 100.0),
                "specificity": float(n_tn / (n_tn + n_fp + 1e-8) * 100.0),
                "FN_rate": float(n_fn / (n_fn + n_tp + 1e-8) * 100.0),
                "FP_rate": float(n_fp / (n_fp + n_tn + 1e-8) * 100.0),
            },
            "false_negatives": self._analyze_errors(
                mask=fn_mask,
                y_probs=y_probs,
                y_figo_pred=y_figo_pred,
                y_features_pred=y_features_pred,
                patient_ids=patient_ids,
                error_type="FN",
            ),
            "false_positives": self._analyze_errors(
                mask=fp_mask,
                y_probs=y_probs,
                y_figo_pred=y_figo_pred,
                y_features_pred=y_features_pred,
                patient_ids=patient_ids,
                error_type="FP",
            ),
            "patterns": self._identify_patterns(
                fn_mask=fn_mask,
                fp_mask=fp_mask,
                y_figo_pred=y_figo_pred,
                y_features_pred=y_features_pred,
            ),
        }

        return report

    def _analyze_errors(
        self,
        mask: np.ndarray,
        y_probs: np.ndarray,
        y_figo_pred: Optional[np.ndarray],
        y_features_pred: Optional[np.ndarray],
        patient_ids: Optional[List[str]],
        error_type: str,
    ) -> Dict:
        """Detailed analysis of one error category (FN or FP)."""
        if mask.sum() == 0:
            return {"count": 0, "note": f"No {error_type}s in this fold."}

        analysis = {
            "count": int(mask.sum()),
            "mean_confidence": float(y_probs[mask].mean()),
            "std_confidence": float(y_probs[mask].std()),
        }

        # FIGO class distribution among errors
        if y_figo_pred is not None:
            figo_among_errors = y_figo_pred[mask]
            figo_dist = {}
            for cls_idx, cls_name in FIGO_CLASS_NAMES.items():
                n_cls = int((figo_among_errors == cls_idx).sum())
                figo_dist[cls_name] = {
                    "count": n_cls,
                    "pct": float(n_cls / mask.sum() * 100.0),
                }
            analysis["figo_distribution"] = figo_dist

        # Clinical feature statistics among errors
        if y_features_pred is not None and y_features_pred.ndim == 2:
            features_among_errors = y_features_pred[mask]  # (n_errors, 8)
            feat_stats = {}
            for i, name in enumerate(self.feature_names):
                if i < features_among_errors.shape[1]:
                    feat_stats[name] = {
                        "mean": float(features_among_errors[:, i].mean()),
                        "std": float(features_among_errors[:, i].std()),
                        "min": float(features_among_errors[:, i].min()),
                        "max": float(features_among_errors[:, i].max()),
                    }
            analysis["feature_statistics"] = feat_stats

        # Unique patients in this error category
        if patient_ids is not None:
            pids = np.array(patient_ids)
            analysis["unique_patients"] = int(len(set(pids[mask])))

        return analysis

    def _identify_patterns(
        self,
        fn_mask: np.ndarray,
        fp_mask: np.ndarray,
        y_figo_pred: Optional[np.ndarray],
        y_features_pred: Optional[np.ndarray],
    ) -> Dict:
        """
        Identify systematic failure patterns that may be reported in the paper.
        """
        patterns = []

        if y_features_pred is not None and y_features_pred.ndim == 2:
            # Pattern 1: FN with predicted Normal FIGO class
            if y_figo_pred is not None and fn_mask.sum() > 0:
                fn_figo = y_figo_pred[fn_mask]
                pct_normal_fn = float((fn_figo == 0).sum() / len(fn_figo) * 100.0)
                if pct_normal_fn > 40:
                    patterns.append({
                        "type": "FN_predicted_normal",
                        "description": (
                            f"{pct_normal_fn:.1f}% of false negatives were predicted as "
                            "FIGO Normal — model missed pathological cases without "
                            "sufficient FIGO signal."
                        ),
                        "pct": pct_normal_fn,
                    })

            # Pattern 2: FN with low predicted LTV
            if fn_mask.sum() > 0:
                fn_ltv = y_features_pred[fn_mask, IDX_LTV]
                pct_reduced_ltv = float((fn_ltv < LTV_REDUCED_THRESH).sum() / len(fn_ltv) * 100.0)
                if pct_reduced_ltv > 30:
                    patterns.append({
                        "type": "FN_reduced_ltv",
                        "description": (
                            f"{pct_reduced_ltv:.1f}% of false negatives have predicted "
                            f"LTV < {LTV_REDUCED_THRESH} bpm — reduced variability not "
                            "triggering distress prediction."
                        ),
                        "pct": pct_reduced_ltv,
                    })

            # Pattern 3: FN with late decelerations predicted > 0
            if fn_mask.sum() > 0:
                fn_late = y_features_pred[fn_mask, IDX_LATE_DECELS]
                pct_late = float((fn_late > 0.5).sum() / len(fn_late) * 100.0)
                if pct_late > 20:
                    patterns.append({
                        "type": "FN_with_late_decels",
                        "description": (
                            f"{pct_late:.1f}% of false negatives have predicted late "
                            "decelerations — FIGO rule loss may need higher lambda."
                        ),
                        "pct": pct_late,
                    })

            # Pattern 4: FP with normal baseline FHR range
            if fp_mask.sum() > 0:
                fp_baseline = y_features_pred[fp_mask, IDX_BASELINE_FHR]
                pct_normal_baseline = float(
                    ((fp_baseline >= BASELINE_NORMAL_MIN) & (fp_baseline <= BASELINE_NORMAL_MAX)).sum()
                    / len(fp_baseline) * 100.0
                )
                if pct_normal_baseline > 60:
                    patterns.append({
                        "type": "FP_normal_baseline",
                        "description": (
                            f"{pct_normal_baseline:.1f}% of false positives have predicted "
                            "normal baseline FHR (110–160 bpm) — false alarms occurring "
                            "despite normal baseline."
                        ),
                        "pct": pct_normal_baseline,
                    })

        if not patterns:
            patterns.append({
                "type": "no_dominant_pattern",
                "description": "No dominant failure pattern identified above thresholds.",
            })

        return {"identified_patterns": patterns}

    def print_report(self, report: Dict) -> None:
        """Print a formatted summary of the error analysis report."""
        fold_label = f"Fold {report['fold']}" if report.get("fold") else "Validation Set"
        s = report["summary"]

        print(f"\n{'='*65}")
        print(f" ERROR ANALYSIS REPORT — {fold_label} (threshold={report['threshold']:.3f})")
        print(f"{'='*65}")
        print(f" Total samples: {s['n_total']} | Positive: {s['n_positive']} | Negative: {s['n_negative']}")
        print(f" TP: {s['TP']:4d} | TN: {s['TN']:4d} | FP: {s['FP']:4d} | FN: {s['FN']:4d}")
        print(f" Sensitivity: {s['sensitivity']:.2f}% | Specificity: {s['specificity']:.2f}%")
        print(f" FN Rate: {s['FN_rate']:.2f}% | FP Rate: {s['FP_rate']:.2f}%")

        fn = report.get("false_negatives", {})
        print(f"\n --- FALSE NEGATIVES (Missed Distress: {fn.get('count', 0)}) ---")
        if "figo_distribution" in fn:
            for cls, data in fn["figo_distribution"].items():
                print(f"  FIGO {cls:12s}: {data['count']:3d} ({data['pct']:.1f}%)")

        fp = report.get("false_positives", {})
        print(f"\n --- FALSE POSITIVES (False Alarms: {fp.get('count', 0)}) ---")
        print(f"  Mean confidence: {fp.get('mean_confidence', 0.0):.3f}")

        patterns = report.get("patterns", {}).get("identified_patterns", [])
        if patterns:
            print(f"\n --- IDENTIFIED FAILURE PATTERNS ---")
            for p in patterns:
                print(f"  [{p['type']}] {p['description']}")

        print(f"{'='*65}\n")

    def save_report(self, report: Dict, output_path: str) -> None:
        """Save the full error analysis report as JSON."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
        print(f"[error_analysis] Report saved → {output_path}")
