"""
Clinical-relational pretraining -- knowledge-structured representation learning.

IDEA. Every knowledge-infusion mechanism tried in this project injected clinical
knowledge as information the model was asked to reproduce or consume, and all
failed for one measured reason: the network already encodes what the FIGO rules
encode (7 flags reach AUROC 0.762 vs the network's 0.777). This objective does
something different -- it uses clinical knowledge to shape the GEOMETRY of the
latent space before any outcome label is seen.

    If two windows sit close together in clinical feature space,
    their embeddings should sit close together too.

Why this is not just another auxiliary head:
  * It consumes NO outcome labels. Our binding constraint is 226 positive
    windows; this objective trains on all 5,286. Knowledge-derived objectives
    are the only ones able to exploit CTG data without pH -- which is most CTG
    data in existence.
  * It supervises RELATIONS, not values. Predicting features (the DeepCTG
    approach) asks the encoder to reproduce the rule engine's outputs, which
    risks binding the representation to that engine's 0.762 ceiling. Matching
    pairwise structure constrains the space more loosely.
  * It preserves MAGNITUDE. Contrastive grouping by binary FIGO signature was
    considered and rejected: thresholded flags discard deceleration depth,
    area and duration, which mechanism 10 showed is the one part of clinical
    knowledge carrying complementary signal.

Related to Relational Knowledge Distillation (Park et al., 2019), but the
teacher here is clinical feature geometry rather than a teacher network.

CAVEAT held throughout: shaping a representation with knowledge the model
already possesses may bias it toward the rule ceiling. With 226 positives,
fine-tuning may lack the signal to escape that bias. This is exactly why the
objective is tested before anything is built on top of it.
"""

import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.models.ctg_crossformer import CTGCrossformerEncoder


class ProjectionHead(nn.Module):
    """Maps the encoder latent to the space where relations are matched.
    Discarded after pretraining -- only the encoder is the deliverable."""

    def __init__(self, latent_dim: int = 128, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_dim)
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """(B, D) -> (B, B) euclidean distances."""
    return torch.cdist(x, x, p=2)


def relational_loss(z_proj: torch.Tensor, clin: torch.Tensor,
                    huber_delta: float = 1.0) -> torch.Tensor:
    """
    Match the latent distance structure to the clinical distance structure.

    Both matrices are scaled by their own mean off-diagonal distance before
    comparison, so the objective constrains RELATIVE geometry (which windows are
    near which) and not absolute scale -- the encoder is free to choose its own
    embedding magnitude, which keeps the constraint loose enough to leave room
    for the downstream task.
    """
    d_lat = pairwise_distances(z_proj)
    d_clin = pairwise_distances(clin)

    n = d_lat.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=d_lat.device)
    if off.sum() == 0:
        return d_lat.sum() * 0.0

    d_lat = d_lat / d_lat[off].mean().clamp(min=1e-8)
    d_clin = d_clin / d_clin[off].mean().clamp(min=1e-8)

    return F.huber_loss(d_lat[off], d_clin[off], delta=huber_delta)


class ClinicalRelationalPretrainer(nn.Module):
    def __init__(self, encoder: Optional[nn.Module] = None, latent_dim: int = 128):
        super().__init__()
        self.encoder = encoder if encoder is not None else CTGCrossformerEncoder(
            in_channels=2, seq_len=4800, cnn_channels=128, n_heads_cross=4,
            n_heads_tf=8, n_tf_layers=4, d_ff=512, dropout=0.1, latent_dim=latent_dim,
        )
        self.proj = ProjectionHead(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if isinstance(z, tuple):
            z = z[0]
        return self.proj(z)


def build_clinical_space(y_features: np.ndarray, extended: np.ndarray,
                         drop_constant: bool = True) -> Tuple[np.ndarray, list]:
    """
    Assembles the 19-dim clinical feature space that acts as the teacher, and
    standardises it so no single feature dominates the distance metric purely
    through its units (deceleration counts are O(1), baseline is O(100)).
    """
    from src.knowledge.extended_features import EXTENDED_FEATURE_NAMES
    base_names = ["baseline_fhr", "stv", "ltv", "accel_count",
                  "early_decel", "late_decel", "variable_decel", "prolonged_decel"]
    X = np.column_stack([y_features, extended]).astype(np.float64)
    names = base_names + list(EXTENDED_FEATURE_NAMES)

    if drop_constant:
        keep = X.std(axis=0) > 1e-8
        X, names = X[:, keep], [n for n, k in zip(names, keep) if k]

    # Rank-normalise before standardising: several features (deceleration
    # counts, burden) are heavily skewed, and raw euclidean distance on skewed
    # features is dominated by a handful of extreme windows.
    from scipy.stats import rankdata
    Xr = np.column_stack([rankdata(X[:, j]) / len(X) for j in range(X.shape[1])])
    Xr = (Xr - Xr.mean(axis=0)) / Xr.std(axis=0).clip(min=1e-8)
    return Xr.astype(np.float32), names
