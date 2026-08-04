"""
Temporal Prediction Horizon Analysis for Model 8 (Phase 7)
============================================================
Evaluates the trained KI-MTF checkpoint at multiple clinically meaningful
prediction horizons BEFORE the delivery outcome:

  T-5, T-10, T-15, T-20, T-30 minutes before delivery

This answers the key clinical question: "How far in advance can the model
reliably detect fetal distress?"

No retraining. No architecture changes. Just inference on the same trained
checkpoints with windows extracted at different prediction offsets.

The module produces:
  - Per-horizon metric table (AUROC, Sens@90%Spec, F1, Recall)
  - CSV output for plotting AUROC vs. horizon curves
  - Console summary table

Usage:
    from src.training.horizon_analysis import evaluate_at_horizons

    results = evaluate_at_horizons(
        model=model,
        dataset=full_dataset,
        patient_metadata=metadata,
        horizons_minutes=[5, 10, 15, 20, 30],
        device=device,
    )
    # Returns: Dict[int, Dict[str, float]] — horizon → metric_dict

    # Or use the CLI runner:
    python src/training/horizon_analysis.py \
        --checkpoint checkpoints/model8/model8_full_fold1_best.pth \
        --data_dir data/processed/
"""

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


# =============================================================================
# Core Horizon Evaluation
# =============================================================================

def _compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
) -> Dict[str, float]:
    """Compute AUROC, AUPRC, F1, Recall, Specificity, and Sens@90%Spec."""
    try:
        from sklearn.metrics import (
            roc_auc_score, auc, roc_curve, precision_recall_curve
        )
        if len(np.unique(y_true)) < 2:
            return {"auroc": 0.5, "auprc": 0.0, "f1": 0.0, "recall": 0.0,
                    "specificity": 0.0, "sens_at_90spec": 0.0, "n_samples": len(y_true)}

        auroc = roc_auc_score(y_true, y_probs)
        p_vals, r_vals, _ = precision_recall_curve(y_true, y_probs)
        auprc = auc(r_vals, p_vals)
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        spec_arr = 1.0 - fpr
        idx = np.where(spec_arr >= 0.90)[0]
        sens_at_90spec = float(tpr[idx[-1]]) * 100.0 if len(idx) > 0 else 0.0

        # F1 and recall at default 0.5 threshold
        y_pred = (y_probs >= 0.5).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "auroc": float(auroc),
            "auprc": float(auprc),
            "f1": float(f1),
            "recall": float(recall * 100.0),
            "specificity": float(specificity * 100.0),
            "sens_at_90spec": float(sens_at_90spec),
            "n_samples": int(len(y_true)),
            "n_positive": int(y_true.sum()),
        }
    except ImportError:
        return {"auroc": 0.5, "auprc": 0.0, "f1": 0.0, "recall": 0.0,
                "specificity": 0.0, "sens_at_90spec": 0.0, "n_samples": len(y_true)}


def evaluate_at_horizons(
    model: nn.Module,
    dataset,
    horizons_minutes: List[int] = None,
    patient_metadata: Optional[List[Dict]] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
    window_duration_minutes: int = 20,
    sampling_rate_hz: int = 4,
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate the model at multiple temporal prediction horizons.

    For each horizon H (in minutes), only windows that end at least H minutes
    before the delivery are evaluated. Earlier windows represent harder, more
    clinically valuable predictions.

    Args:
        model:                  Trained KnowledgeInfusedFramework (any fold checkpoint).
        dataset:                MultiTaskCTGDataset with all windows.
        horizons_minutes:       List of prediction horizons in minutes.
                                Default: [5, 10, 15, 20, 30].
        patient_metadata:       Optional list of dicts with keys:
                                  'time_before_delivery_min': float — minutes before delivery
                                                              for each window (index-matched).
                                If None, evaluates all windows at all horizons.
        device:                 Computation device.
        batch_size:             Inference batch size.
        window_duration_minutes: Duration of each CTG window (default 20 min).
        sampling_rate_hz:       Signal sampling rate (default 4 Hz).

    Returns:
        Dict mapping horizon_minutes → metric_dict.
        Also includes horizon=0 (all windows, baseline).
    """
    if horizons_minutes is None:
        horizons_minutes = [5, 10, 15, 20, 30]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    # --- Collect all predictions ---
    all_probs = []
    all_targets = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            X_batch = batch[0].to(device)
            y_batch = batch[1].cpu().numpy()
            use_amp = device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
                distress_logit, _, _ = model(X_batch)
                probs = torch.sigmoid(distress_logit.squeeze(-1)).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(y_batch.tolist())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # --- Evaluate at each horizon ---
    results: Dict[int, Dict[str, float]] = {}

    # Horizon 0 = all windows (baseline)
    results[0] = _compute_metrics_at_threshold(all_targets, all_probs)
    results[0]["horizon_minutes"] = 0
    results[0]["description"] = "All windows (no horizon filter)"

    if patient_metadata is not None and len(patient_metadata) == len(all_probs):
        time_before_delivery = np.array([
            m.get("time_before_delivery_min", float("inf"))
            for m in patient_metadata
        ])

        for H in horizons_minutes:
            # Include only windows where the window START is at least H minutes before delivery
            # i.e., time_before_delivery >= H + window_duration_minutes
            min_time = H + window_duration_minutes
            mask = time_before_delivery >= min_time
            n_selected = mask.sum()
            if n_selected < 10:
                print(f"[horizon_analysis] WARNING: Only {n_selected} windows for H={H} min — skipping.")
                continue

            horizon_metrics = _compute_metrics_at_threshold(
                all_targets[mask], all_probs[mask]
            )
            horizon_metrics["horizon_minutes"] = H
            horizon_metrics["description"] = f"Windows ≥{H} min before delivery"
            horizon_metrics["n_windows_selected"] = int(n_selected)
            results[H] = horizon_metrics
    else:
        # No metadata — report same results at all horizons with a note
        print("[horizon_analysis] WARNING: No patient_metadata provided. "
              "Reporting same metrics at all horizons (horizon filter not applied).")
        for H in horizons_minutes:
            results[H] = {**results[0], "horizon_minutes": H,
                          "description": "Horizon filter unavailable (no metadata)"}

    return results


def print_horizon_table(results: Dict[int, Dict[str, float]]) -> None:
    """Print a formatted summary table of horizon analysis results."""
    print("\n" + "=" * 75)
    print(" TEMPORAL PREDICTION HORIZON ANALYSIS")
    print("=" * 75)
    print(f"{'Horizon':>12} {'N Samples':>10} {'AUROC':>8} {'AUPRC':>8} {'Sens@90S':>10} {'Recall':>8}")
    print("-" * 75)
    for H in sorted(results.keys()):
        r = results[H]
        label = "All Windows" if H == 0 else f"T-{H} min"
        print(
            f"{label:>12}  {r.get('n_samples', '?'):>9}  "
            f"{r.get('auroc', 0.0):>7.4f}  {r.get('auprc', 0.0):>7.4f}  "
            f"{r.get('sens_at_90spec', 0.0):>9.2f}%  {r.get('recall', 0.0):>7.2f}%"
        )
    print("=" * 75 + "\n")


def save_horizon_csv(
    results: Dict[int, Dict[str, float]],
    output_path: str,
) -> None:
    """Save horizon analysis results to a CSV file for plotting."""
    if not results:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = ["horizon_minutes", "n_samples", "n_positive", "auroc", "auprc",
                  "f1", "recall", "specificity", "sens_at_90spec", "description"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for H in sorted(results.keys()):
            writer.writerow(results[H])
    print(f"[horizon_analysis] Results saved → {output_path}")
