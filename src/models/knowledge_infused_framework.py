"""
Knowledge-Infused Multi-Task Framework (Model 8)
=================================================
Phase 4 Architecture: PatchTSTEncoder backbone + 3 clinical task heads.

Conforms to the universal encoding pipeline:
  Backbone Input:  (Batch, 2, 4800)  — Baseline-corrected FHR + UC at 4 Hz
  Shared Latent:   (Batch, 128)      — Universal latent z from PatchTST
  Head Outputs:
    - DistressHead:       (Batch, 1)  — Binary logit (fetal acidemia pH ≤ 7.15)
    - FIGOHead:           (Batch, 3)  — 3-class logits (Normal / Suspicious / Pathological)
    - ClinicalFeatureHead:(Batch, 8)  — Physiological feature predictions
        [0] Baseline FHR (bpm)
        [1] STV (bpm)
        [2] LTV (bpm)
        [3] Accel Count        ← F.softplus activation (non-negative count)
        [4] Early Decel Count  ← F.softplus activation
        [5] Late Decel Count   ← F.softplus activation
        [6] Var Decel Count    ← F.softplus activation
        [7] Prolonged Decel    ← F.softplus activation

Patent Compliance (GE US12094611B2):
  Continuous end-to-end signal encoding (B, 2, 4800) → latent z ∈ ℝ¹²⁸ via
  channel-independent patch transformer. No bounding boxes, no shape-matching loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DistressHead(nn.Module):
    """
    Primary Task Head: Binary Fetal Distress Prediction.

    Predicts a single logit for BCEWithLogitsLoss.
    Clinical target: pH ≤ 7.15 within 30-minute intrapartum prediction horizon.

    Architecture: Linear(128→64) → LayerNorm → GELU → Dropout → Linear(64→1)
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent representation (Batch, 128)
        Returns:
            logit: Tensor of shape (Batch, 1) — raw binary logit
        """
        return self.head(z)


class FIGOHead(nn.Module):
    """
    Auxiliary Task Head: FIGO 2015 3-Class Classification.

    Predicts categorical FIGO class logits for CrossEntropyLoss.
    Classes: 0=Normal, 1=Suspicious, 2=Pathological.

    Architecture: Linear(128→64) → LayerNorm → GELU → Dropout → Linear(64→3)
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent representation (Batch, 128)
        Returns:
            logits: Tensor of shape (Batch, 3) — FIGO class logits (unnormalized)
        """
        return self.head(z)


class ClinicalFeatureHead(nn.Module):
    """
    Auxiliary Task Head: Physiological Feature Regression.

    Predicts 8 clinical features from the shared latent representation.
    For count-type outputs (indices 3–7: Accels, Early/Late/Var/Prolonged Decels),
    a F.softplus activation is applied to enforce non-negativity while maintaining
    smooth, non-zero gradients (avoiding the dead-gradient zone of F.relu).
    Continuous outputs (indices 0–2: Baseline, STV, LTV) are left as linear.

    Architecture: Linear(128→64) → LayerNorm → GELU → Dropout → Linear(64→8)
                  + selective F.softplus on count outputs

    Feature Mapping (N, 8):
        [0]: Baseline FHR (bpm)       — linear output
        [1]: STV (bpm)                — linear output
        [2]: LTV (bpm)                — linear output
        [3]: Acceleration Count       — softplus output
        [4]: Early Deceleration Count — softplus output
        [5]: Late Deceleration Count  — softplus output
        [6]: Variable Decel Count     — softplus output
        [7]: Prolonged Decel Count    — softplus output
    """

    # Indices that represent physiological counts (must be non-negative)
    COUNT_FEATURE_INDICES = [3, 4, 5, 6, 7]

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(hidden_dim, 8)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent representation (Batch, 128)
        Returns:
            features: Tensor of shape (Batch, 8) — mixed linear + softplus outputs.
                      In Z-normalized space (before un-normalization by figo_rule_loss_normalized).
        """
        h = self.backbone(z)
        raw_out = self.output_layer(h)  # (Batch, 8)

        # Apply softplus selectively to count features (indices 3–7)
        # Continuous features (indices 0–2) remain linear (can be negative in Z-space)
        continuous = raw_out[:, :3]                              # (Batch, 3)
        counts = F.softplus(raw_out[:, 3:])                     # (Batch, 5) — always ≥ 0
        return torch.cat([continuous, counts], dim=1)            # (Batch, 8)


class FIGOCriteriaHead(nn.Module):
    """
    Prototype auxiliary head (2026-08-14): predicts the INTERMEDIATE binary clinical
    judgments that classify_figo() computes internally on the way to its single
    collapsed 3-class FIGO label -- e.g. "is baseline in the normal band?", "is
    variability increased?", "are late decelerations present?" -- rather than only
    the final aggregate class. Motivation: plus_figo/distress_figo_only have been
    the weakest-performing auxiliary variants across every Model 8 run so far
    (PatchTST and CrossFormer alike); a single 3-way softmax dominated ~75% by one
    class is a blunt training signal, and decomposing it into its constituent rule
    criteria gives the network more to actually learn from, using information the
    pipeline already computes (see src.knowledge.figo.derive_figo_criteria_flags)
    -- no new raw-signal work required.

    Wired into KnowledgeInfusedFramework's loss computation for the
    plus_criteria / features_criteria ablation variants (see Auxiliary Task 4
    in train_knowledge_infused.py's compute_multitask_loss()), each opt-in via
    include_criteria_head, so all other ablation variants and configs remain
    unaffected.

    Target ordering (7 binary flags, (N, 7), see derive_figo_criteria_flags()):
        [0] baseline_low       (<100 bpm)
        [1] baseline_normal    (110-160 bpm)
        [2] baseline_high      (>160 bpm)
        [3] variability_normal (LTV 5-25 bpm)
        [4] has_late_decel
        [5] has_variable_decel
        [6] has_prolonged_decel

    Two variability flags are deliberately excluded (see derive_figo_criteria_flags()
    for full rationale):
      - variability_reduced (LTV<5): 0% prevalence on the corrected training data
        (0 positive examples out of 6266 windows) -- unlearnable/unevaluable.
      - variability_increased (LTV>25): dropped 2026-08-15 after a literature check
        found FIGO 2015 defines "saltatory" as >25bpm sustained for >30 minutes --
        longer than this pipeline's entire 20-minute assessment window can ever
        show, regardless of how accurately LTV itself is computed. A structural
        window-length mismatch, not a fixable calibration bug.

    Architecture: Linear(128→64) → LayerNorm → GELU → Dropout → Linear(64→7)
                  (raw logits out -- apply BCEWithLogitsLoss per-flag, not softmax)
    """

    N_CRITERIA = 7

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.N_CRITERIA),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent representation (Batch, 128)
        Returns:
            logits: Tensor of shape (Batch, 7) — raw per-flag logits (unnormalized).
        """
        return self.head(z)


class KnowledgeInfusedFramework(nn.Module):
    """
    Knowledge-Infused Multi-Task Framework (Model 8) for CTG Fetal Distress Prediction.

    Combines the winning PatchTST temporal encoder backbone with three clinically-
    grounded task heads trained simultaneously under a composite multi-task loss:

        L_total = L_distress + λ₁·L_FIGO + λ₂·L_features + λ₃·L_knowledge

    where L_knowledge = figo_rule_loss_normalized() — a differentiable penalty
    enforcing FIGO 2015 clinical consistency constraints on the predicted features.

    Args:
        encoder (nn.Module): Any temporal encoder conforming to the universal signature
                             (Batch, 2, 4800) → (Batch, 128). Intended: PatchTSTEncoder
                             with tuned config (n_layers=4, n_heads=4, patch_len=16).
        latent_dim (int):    Encoder output dimension. Must be 128.
        head_hidden_dim (int): Hidden units in each task head (default 64).
        head_dropout (float):  Dropout rate in each task head (default 0.2).
        include_criteria_head (bool): Prototype opt-in (2026-08-14, default False).
            Adds FIGOCriteriaHead and makes forward() return a 4-tuple instead of
            the usual 3-tuple. Defaults to False so every existing config and
            training script -- and anything already running -- is completely
            unaffected; this is not wired into the loss computation yet, it only
            adds the head so it can be inspected/prototyped in isolation.
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int = 128,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.2,
        include_criteria_head: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.include_criteria_head = include_criteria_head
        self.distress_head = DistressHead(
            latent_dim=latent_dim, hidden_dim=head_hidden_dim, dropout=head_dropout
        )
        self.figo_head = FIGOHead(
            latent_dim=latent_dim, hidden_dim=head_hidden_dim, dropout=head_dropout
        )
        self.feature_head = ClinicalFeatureHead(
            latent_dim=latent_dim, hidden_dim=head_hidden_dim, dropout=head_dropout
        )
        if include_criteria_head:
            self.criteria_head = FIGOCriteriaHead(
                latent_dim=latent_dim, hidden_dim=head_hidden_dim, dropout=head_dropout
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input CTG signal of shape (Batch, 2, 4800).
               Channel 0: Baseline-corrected FHR, Channel 1: UC.
        Returns:
            Tuple of:
              - distress_logit:  (Batch, 1)  — binary distress logit
              - figo_logits:     (Batch, 3)  — FIGO 3-class logits
              - feature_preds:   (Batch, 8)  — physiological feature predictions (Z-normalized space)
              - criteria_logits: (Batch, 7)  — ONLY present if include_criteria_head=True
                                  (prototype FIGOCriteriaHead output; omitted entirely,
                                  not None, when the flag is off, so the return arity
                                  matches every pre-existing call site exactly).
        """
        z = self.encoder(x)                         # (Batch, 128)
        distress_logit = self.distress_head(z)       # (Batch, 1)
        figo_logits = self.figo_head(z)              # (Batch, 3)
        feature_preds = self.feature_head(z)         # (Batch, 8)
        if self.include_criteria_head:
            criteria_logits = self.criteria_head(z)  # (Batch, 7)
            return distress_logit, figo_logits, feature_preds, criteria_logits
        return distress_logit, figo_logits, feature_preds

    @property
    def param_count(self) -> int:
        """Total trainable parameter count (backbone + all heads)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
