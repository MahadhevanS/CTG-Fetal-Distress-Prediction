"""
FIGO 2015 Clinical Knowledge Module
====================================
Provides:
  1. classify_figo()             — scalar rule-based FIGO classification (preprocessing utility).
  2. vectorized_classify_figo()  — batched numpy variant for bulk preprocessing.
  3. figo_rule_loss()            — original knowledge loss (for reference / testing only).
                                   Assumes pred_features are in RAW CLINICAL UNITS.
  4. figo_rule_loss_normalized() — CANONICAL TRAINING LOSS. Accepts Z-normalized model
                                   outputs + scaler stats and un-normalizes internally before
                                   applying FIGO clinical thresholds. Use this in train loops.
"""

from typing import Dict

import numpy as np


def classify_figo(baseline: float, variability_ltv: float,
                  accelerations: int, decelerations: Dict[str, int]) -> int:
    """
    Classifies a CTG window into FIGO categories (2015 guidelines simplified).
    0 = Normal, 1 = Suspicious, 2 = Pathological

    IMPORTANT: Scalar-only utility for preprocessing. Use vectorized_classify_figo()
    for batch numpy arrays, or rely on pre-computed y_figo targets in .pt files.

    Args:
        baseline: Baseline FHR in bpm.
        variability_ltv: Long-term variability amplitude in bpm.
        accelerations: Number of accelerations.
        decelerations: Dictionary of deceleration counts ('early', 'late', 'variable', 'prolonged').

    Returns:
        int: FIGO class (0=Normal, 1=Suspicious, 2=Pathological).
    """
    # 1. Baseline Rules
    normal_baseline = 110 <= baseline <= 160

    # 2. Variability Rules
    normal_var = 5 <= variability_ltv <= 25
    reduced_var = variability_ltv < 5

    # 3. Deceleration Rules
    has_prolonged = decelerations.get('prolonged', 0) > 0
    has_late = decelerations.get('late', 0) > 0
    has_variable = decelerations.get('variable', 0) > 0

    # Pathological: baseline < 100, reduced var (simplified), repeated late/prolonged decels.
    if baseline < 100 or has_prolonged or (has_late and reduced_var):
        return 2  # Pathological

    # Suspicious: lacking one normal characteristic, but not pathological
    if not normal_baseline or not normal_var or has_late or has_variable:
        return 1  # Suspicious

    # Normal: baseline 110-160, var 5-25, no late/prolonged decels
    return 0  # Normal


def vectorized_classify_figo(
    baselines: np.ndarray,
    variability_ltvs: np.ndarray,
    late_decels: np.ndarray,
    variable_decels: np.ndarray,
    prolonged_decels: np.ndarray,
) -> np.ndarray:
    """
    Vectorized batch FIGO classification over numpy arrays.
    All inputs must have the same shape (N,).

    Args:
        baselines:        Baseline FHR in bpm, shape (N,).
        variability_ltvs: LTV amplitude in bpm, shape (N,).
        late_decels:      Late deceleration counts, shape (N,).
        variable_decels:  Variable deceleration counts, shape (N,).
        prolonged_decels: Prolonged deceleration counts, shape (N,).

    Returns:
        np.ndarray: FIGO class labels, shape (N,), dtype int (0=Normal, 1=Suspicious, 2=Pathological).
    """
    labels = np.zeros(len(baselines), dtype=np.int64)

    normal_baseline = (baselines >= 110) & (baselines <= 160)
    normal_var = (variability_ltvs >= 5) & (variability_ltvs <= 25)
    reduced_var = variability_ltvs < 5
    has_late = late_decels > 0
    has_variable = variable_decels > 0
    has_prolonged = prolonged_decels > 0

    # Pathological mask
    pathological = (baselines < 100) | has_prolonged | (has_late & reduced_var)
    # Suspicious mask (not already pathological)
    suspicious = (~pathological) & (~normal_baseline | ~normal_var | has_late | has_variable)

    labels[suspicious] = 1
    labels[pathological] = 2
    return labels


import torch
import torch.nn as nn
import torch.nn.functional as F


def figo_rule_loss(
    pred_features: torch.Tensor,
    pred_figo_logits: torch.Tensor,
    target_figo: torch.Tensor = None,
    lambda_consistency: float = 0.5,
) -> torch.Tensor:
    """
    [LEGACY — FOR TESTING ONLY]
    Computes FIGO knowledge loss assuming pred_features is in RAW CLINICAL UNITS.

    WARNING: Do NOT call this directly in a training loop. Model outputs are
    Z-normalized. Use figo_rule_loss_normalized() instead.

    Args:
        pred_features:    Tensor (Batch, 8) in RAW clinical units:
                          [Baseline(bpm), STV(bpm), LTV(bpm), Accels,
                           EarlyDecels, LateDecels, VarDecels, ProlongedDecels]
        pred_figo_logits: Tensor (Batch, 3) — FIGO 3-class logits.
        target_figo:      Optional Tensor (Batch,) with ground-truth FIGO labels (0,1,2).
        lambda_consistency: Weight for the soft clinical consistency penalty.

    Returns:
        torch.Tensor: Scalar composite loss.
    """
    return _compute_figo_penalties(
        pred_features=pred_features,
        pred_figo_logits=pred_figo_logits,
        target_figo=target_figo,
        lambda_consistency=lambda_consistency,
    )


def figo_rule_loss_normalized(
    pred_features_norm: torch.Tensor,
    pred_figo_logits: torch.Tensor,
    feature_means: torch.Tensor,
    feature_stds: torch.Tensor,
    target_figo: torch.Tensor = None,
    lambda_consistency: float = 0.5,
) -> torch.Tensor:
    """
    [CANONICAL TRAINING LOSS — USE THIS IN TRAINING LOOPS]

    Computes the FIGO knowledge loss from Z-normalized model feature predictions.
    Internally un-normalizes predictions to clinical units before applying
    FIGO 2015 thresholds, eliminating the unit mismatch between network outputs
    and rule-based penalty constants.

    Args:
        pred_features_norm: Tensor (Batch, 8) — Z-normalized feature predictions
                            (direct output of ClinicalFeatureHead, before activation):
                            [Baseline, STV, LTV, Accels, EarlyDecels, LateDecels,
                             VarDecels, ProlongedDecels]
        pred_figo_logits:   Tensor (Batch, 3) — FIGO 3-class logits.
        feature_means:      Tensor (8,) — per-feature means from ctu_signal_scaler.npz.
                            Cast to same device as pred_features_norm before calling.
        feature_stds:       Tensor (8,) — per-feature stds from ctu_signal_scaler.npz.
                            Cast to same device as pred_features_norm before calling.
        target_figo:        Optional Tensor (Batch,) — ground-truth FIGO labels (0,1,2).
        lambda_consistency: Weight for soft FIGO clinical consistency penalties.

    Returns:
        torch.Tensor: Scalar composite loss (Cross-Entropy + consistency penalties).

    Example:
        >>> import numpy as np, torch
        >>> scaler = np.load("data/processed/ctu_signal_scaler.npz")
        >>> # Extend scaler to 8 features (signal scaler has 2 channel stats; features have 8)
        >>> # In practice, load pre-computed feature_means/stds from the dataset.
        >>> means = torch.tensor(feature_means, dtype=torch.float32).to(device)
        >>> stds  = torch.tensor(feature_stds,  dtype=torch.float32).to(device)
        >>> loss = figo_rule_loss_normalized(pred_feat_norm, pred_figo_logits, means, stds, y_figo)
    """
    # Un-normalize: x_clinical = x_norm * std + mean
    # Clamp stds to avoid division by zero (should never be zero from scaler, but be safe)
    safe_stds = feature_stds.clamp(min=1e-6)
    pred_features = pred_features_norm * safe_stds + feature_means

    return _compute_figo_penalties(
        pred_features=pred_features,
        pred_figo_logits=pred_figo_logits,
        target_figo=target_figo,
        lambda_consistency=lambda_consistency,
    )


def _compute_figo_penalties(
    pred_features: torch.Tensor,
    pred_figo_logits: torch.Tensor,
    target_figo: torch.Tensor = None,
    lambda_consistency: float = 0.5,
) -> torch.Tensor:
    """
    Internal: Computes FIGO rule-consistency penalties.
    pred_features must be in RAW CLINICAL UNITS at this point.

    All continuous count-type features (decel counts) use F.softplus instead
    of F.relu to avoid dead-gradient zones when network outputs negative values.
    F.softplus is smooth everywhere and equals F.relu asymptotically.
    """
    total_loss = torch.tensor(0.0, device=pred_figo_logits.device)

    # 1. Primary FIGO Classification Loss (if targets provided)
    if target_figo is not None:
        ce_loss = F.cross_entropy(pred_figo_logits, target_figo.long())
        total_loss = total_loss + ce_loss

    # Softmax probabilities for FIGO classes
    probs = F.softmax(pred_figo_logits, dim=-1)
    p_normal = probs[:, 0]          # P(Normal)
    p_pathological = probs[:, 2]    # P(Pathological)

    # Extract predicted clinical feature columns (8-column mapping):
    # [0]=Baseline, [1]=STV, [2]=LTV, [3]=Accels,
    # [4]=EarlyDecels, [5]=LateDecels, [6]=VarDecels, [7]=ProlongedDecels
    baseline = pred_features[:, 0]
    ltv = pred_features[:, 2]
    late_decels = pred_features[:, 5]
    prolonged_decels = pred_features[:, 7]

    # -------------------------------------------------------------------------
    # Rule A: Baseline Deviation Penalty
    # If baseline < 110 or > 160, penalize confident Normal predictions.
    # Normalized by a typical clinical range (~50 bpm) to keep penalty on 0-1 scale.
    # Uses smooth linear (relu) rather than quadratic to prevent runaway gradients
    # during cold-start when un-normalized predictions are far from clinical ranges.
    # -------------------------------------------------------------------------
    BASELINE_RANGE_NORM = 50.0  # normalization constant (bpm)
    baseline_under = F.relu(110.0 - baseline) / BASELINE_RANGE_NORM
    baseline_over = F.relu(baseline - 160.0) / BASELINE_RANGE_NORM
    penalty_baseline = (baseline_under + baseline_over) * p_normal

    # -------------------------------------------------------------------------
    # Rule B: LTV Variability Deviation Penalty
    # If LTV < 5 or > 25, penalize confident Normal predictions.
    # Normalized by typical LTV range (~20 bpm).
    # -------------------------------------------------------------------------
    LTV_RANGE_NORM = 20.0  # normalization constant (bpm)
    ltv_under = F.relu(5.0 - ltv) / LTV_RANGE_NORM
    ltv_over = F.relu(ltv - 25.0) / LTV_RANGE_NORM
    penalty_ltv = (ltv_under + ltv_over) * p_normal

    # -------------------------------------------------------------------------
    # Rule C: Pathological Deceleration Penalty
    # Late or prolonged decelerations should suppress Normal predictions.
    # FIX (Flaw 5): Use F.softplus instead of F.relu to ensure smooth gradients
    # even when the network predicts negative decel counts. F.relu(negative) = 0
    # creates a dead gradient zone; F.softplus is always > 0 and differentiable.
    # Clamped to max=5.0 to prevent dominant penalty from rare extreme predictions.
    # -------------------------------------------------------------------------
    penalty_decels = (
        torch.clamp(F.softplus(late_decels), max=5.0)
        + 2.0 * torch.clamp(F.softplus(prolonged_decels), max=5.0)
    ) * p_normal

    # -------------------------------------------------------------------------
    # Rule D: Pathological Baseline Penalty
    # Baseline < 100 or prolonged decels should suppress non-Pathological predictions.
    # -------------------------------------------------------------------------
    patho_baseline = F.relu(100.0 - baseline) / BASELINE_RANGE_NORM
    penalty_patho = (
        patho_baseline + torch.clamp(F.softplus(prolonged_decels), max=5.0)
    ) * (1.0 - p_pathological)

    # Sum consistency loss across batch (all penalties already on ~0-1 scale)
    consistency_loss = (
        penalty_baseline + penalty_ltv + penalty_decels + penalty_patho
    ).mean()

    total_loss = total_loss + lambda_consistency * consistency_loss
    return total_loss
