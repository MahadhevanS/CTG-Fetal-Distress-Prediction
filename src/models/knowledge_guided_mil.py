"""
Knowledge-Guided Multiple-Instance Learning (KG-MIL) -- Model 9.

Restructures knowledge infusion around the objective the clinical task actually
poses: umbilical pH <= 7.15 is a PATIENT-level outcome, so a patient's bag of
20-minute windows should produce one prediction. Every prior model in this
project (Models 1-8) trained and scored per-window, a proxy for that.

Why the previous knowledge infusion plateaued: FIGOHead / ClinicalFeatureHead /
FIGOCriteriaHead / figo_rule_loss all inject clinical knowledge as an auxiliary
OUTPUT the encoder must re-derive, and feature-fusion injected it as a late
concatenation at the final head. Across 11 tracked runs those delivered ~0 to
+0.01 AUROC. Here, knowledge instead determines WHERE THE MODEL LOOKS -- it
biases the MIL attention distribution over a patient's windows. That is a
mechanism with no analogue in the window-level framing, and it makes the model
explainable by construction (the attention weights ARE the explanation).

Components (each independently ablatable, so the knowledge contribution is
measured separately from the patient-level reframing gain):
  1. Shared CTGCrossformerEncoder over every window in the bag.
  2. Per-window auxiliary distress head (keeps window-level supervision alive).
  3. Gated attention pooling (ABMIL, Ilse et al. 2018).
  4. use_knowledge_attention: a clinical risk prior from FIGO criteria flags +
     features biases the attention logits. lambda is LEARNABLE and starts at 0,
     so the model can fall back to plain ABMIL if the prior does not help --
     and lambda_init=0 with use_knowledge_attention=False is exactly the
     Stage 1 plain-MIL baseline.
  5. use_trajectory: patient-level summaries of how clinical features EVOLVE
     across the recording (slopes, cumulative/consecutive pathology). No single
     20-minute window contains this; it is what a clinician reads off the trace
     over time.

Padded bag slots are masked everywhere -- attention (-inf pre-softmax),
per-window losses, and trajectory statistics.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.ctg_crossformer import CTGCrossformerEncoder
from src.knowledge.figo import derive_figo_criteria_flags_torch

N_FIGO_CRITERIA = 7
N_CLINICAL_FEATURES = 8
N_TRAJECTORY_FEATURES = 12


class GatedAttentionPooling(nn.Module):
    """
    Gated attention MIL pooling (Ilse et al., 2018):
        e_i = w^T (tanh(V h_i) * sigmoid(U h_i))
    Optionally biased by a per-window clinical risk prior:
        a = softmax(e + lambda * r)

    lambda is a learnable scalar initialized at lambda_init (default 0.0), so
    the knowledge prior starts inert and the model must actively choose to use
    it -- this keeps the ablation honest rather than forcing the prior in.
    """

    def __init__(self, latent_dim: int = 128, attn_dim: int = 128, lambda_init: float = 0.0):
        super().__init__()
        self.V = nn.Linear(latent_dim, attn_dim)
        self.U = nn.Linear(latent_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.knowledge_lambda = nn.Parameter(torch.tensor(float(lambda_init)))

    def forward(
        self,
        h: torch.Tensor,                       # (B, n, latent)
        mask: torch.Tensor,                    # (B, n) bool, True = real
        risk_prior: Optional[torch.Tensor] = None,  # (B, n) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)  # (B, n)

        if risk_prior is not None:
            e = e + self.knowledge_lambda * risk_prior

        # Exclude padding before softmax so padded slots get exactly zero weight
        e = e.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(e, dim=1)
        # A bag of all-padding cannot occur (every bag has >=1 window), but guard
        # against NaN propagation regardless.
        attn = torch.nan_to_num(attn, nan=0.0)

        pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1)  # (B, latent)
        return pooled, attn


class ClinicalRiskPrior(nn.Module):
    """
    Maps a window's clinical evidence (7 FIGO criteria flags + 8 normalized
    clinical features) to a scalar risk score used to bias attention.

    This is the knowledge-infusion mechanism: rather than asking the encoder to
    reproduce these values as auxiliary outputs, they are used directly to steer
    which windows the patient-level decision attends to.
    """

    def __init__(self, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FIGO_CRITERIA + N_CLINICAL_FEATURES, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, criteria: torch.Tensor, features_norm: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([criteria, features_norm], dim=-1)).squeeze(-1)


def compute_trajectory_features(
    features_norm: torch.Tensor,   # (B, n, 8) normalized clinical features
    criteria: torch.Tensor,        # (B, n, 7) FIGO criteria flags
    mask: torch.Tensor,            # (B, n) bool
) -> torch.Tensor:
    """
    Patient-level summaries of how clinical evidence EVOLVES across the
    recording -- information no single window contains.

    Returns (B, 12):
        [0:3]  temporal slope of STV, LTV, baseline (normalized units per window)
        [3:6]  mean / max / last value of a per-window pathology count
        [6:9]  cumulative late, variable, prolonged deceleration burden
        [9]    max run of consecutive pathological windows
        [10]   fraction of windows flagged pathological
        [11]   bag length (log1p, normalized) -- retained deliberately: on a
               uniform-stride dataset this reflects recording length only, and
               the standing bag-size audit keeps it that way. It is a genuine
               clinical covariate (labour duration), not a label proxy.
    """
    B, n, _ = features_norm.shape
    m = mask.float()                      # (B, n)
    counts = m.sum(dim=1).clamp(min=1.0)  # (B,)

    # Position index within the bag, mean-centred over real windows only
    pos = torch.arange(n, device=features_norm.device, dtype=torch.float32)
    pos = pos.unsqueeze(0).expand(B, n)
    pos_mean = (pos * m).sum(dim=1, keepdim=True) / counts.unsqueeze(1)
    pos_c = (pos - pos_mean) * m
    denom = (pos_c ** 2).sum(dim=1).clamp(min=1e-6)  # (B,)

    def slope(series: torch.Tensor) -> torch.Tensor:
        s_mean = (series * m).sum(dim=1, keepdim=True) / counts.unsqueeze(1)
        s_c = (series - s_mean) * m
        return (pos_c * s_c).sum(dim=1) / denom

    stv, ltv, base = features_norm[..., 1], features_norm[..., 2], features_norm[..., 0]
    slopes = torch.stack([slope(stv), slope(ltv), slope(base)], dim=1)  # (B,3)

    # Per-window pathology load from the FIGO criteria flags
    patho = criteria.sum(dim=-1) * m                                     # (B, n)
    patho_mean = patho.sum(dim=1) / counts
    patho_max = patho.masked_fill(~mask, float("-inf")).max(dim=1).values
    patho_max = torch.nan_to_num(patho_max, neginf=0.0)
    last_idx = (counts.long() - 1).clamp(min=0)
    patho_last = patho.gather(1, last_idx.unsqueeze(1)).squeeze(1)
    patho_stats = torch.stack([patho_mean, patho_max, patho_last], dim=1)  # (B,3)

    # Cumulative deceleration burden (features 5,6,7 = late, variable, prolonged)
    decel_burden = (features_norm[..., 5:8] * m.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(1)

    # Longest run of consecutive pathological windows
    is_patho = ((criteria.sum(dim=-1) > 0) & mask).float()
    run, best = torch.zeros(B, device=features_norm.device), torch.zeros(B, device=features_norm.device)
    for t in range(n):
        run = (run + is_patho[:, t]) * is_patho[:, t]
        best = torch.maximum(best, run)
    frac_patho = is_patho.sum(dim=1) / counts

    bag_len = torch.log1p(counts) / 5.0

    return torch.cat([
        slopes, patho_stats, decel_burden,
        best.unsqueeze(1), frac_patho.unsqueeze(1), bag_len.unsqueeze(1),
    ], dim=1)


class KnowledgeGuidedMIL(nn.Module):
    """
    Patient-level CTG classifier over a bag of windows.

    Ablation flags map directly onto the planned ladder:
        use_knowledge_attention=False, use_trajectory=False -> plain MIL baseline
        use_knowledge_attention=True,  use_trajectory=False -> + knowledge attention
        use_knowledge_attention=False, use_trajectory=True  -> + trajectory
        both True                                           -> full KG-MIL
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        latent_dim: int = 128,
        attn_dim: int = 128,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.3,
        use_knowledge_attention: bool = True,
        use_trajectory: bool = True,
        lambda_init: float = 0.0,
        feature_means: Optional[torch.Tensor] = None,
        feature_stds: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else CTGCrossformerEncoder(
            in_channels=2, seq_len=4800, cnn_channels=128,
            n_heads_cross=4, n_heads_tf=8, n_tf_layers=4, d_ff=512,
            dropout=0.1, latent_dim=latent_dim,
        )
        self.use_knowledge_attention = use_knowledge_attention
        self.use_trajectory = use_trajectory

        self.attention = GatedAttentionPooling(latent_dim, attn_dim, lambda_init)
        if use_knowledge_attention:
            self.risk_prior = ClinicalRiskPrior()

        # Per-window auxiliary head -- without it the encoder receives gradient
        # only through the attention-weighted sum, which starves low-attention
        # windows of learning signal.
        self.window_head = nn.Sequential(
            nn.Linear(latent_dim, head_hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(head_dropout), nn.Linear(head_hidden_dim, 1),
        )

        clf_in = latent_dim + (N_TRAJECTORY_FEATURES if use_trajectory else 0)
        self.patient_head = nn.Sequential(
            nn.Linear(clf_in, head_hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(head_dropout), nn.Linear(head_hidden_dim, 1),
        )

        # Feature normalization stats (buffers so they move with .to(device)
        # and are saved in the state_dict alongside the weights)
        if feature_means is None:
            feature_means = torch.zeros(N_CLINICAL_FEATURES)
        if feature_stds is None:
            feature_stds = torch.ones(N_CLINICAL_FEATURES)
        self.register_buffer("feature_means", feature_means.float())
        self.register_buffer("feature_stds", feature_stds.float().clamp(min=1e-6))

    def forward(
        self,
        X: torch.Tensor,          # (B, n, 2, 4800)
        mask: torch.Tensor,       # (B, n) bool
        y_features: torch.Tensor,  # (B, n, 8) RAW clinical units
    ) -> Dict[str, torch.Tensor]:
        B, n = X.shape[0], X.shape[1]

        # Encode every window in the batch in one pass
        h = self.encoder(X.reshape(B * n, X.shape[2], X.shape[3]))
        if isinstance(h, tuple):      # dual-latent encoder variants return (z, pooled)
            h = h[0]
        h = h.reshape(B, n, -1)       # (B, n, latent)

        window_logits = self.window_head(h).squeeze(-1)  # (B, n)

        features_norm = (y_features - self.feature_means) / self.feature_stds
        criteria = derive_figo_criteria_flags_torch(y_features.reshape(B * n, -1)).reshape(B, n, -1)

        risk = None
        if self.use_knowledge_attention:
            risk = self.risk_prior(criteria, features_norm)      # (B, n)
            risk = risk.masked_fill(~mask, 0.0)

        pooled, attn = self.attention(h, mask, risk)

        if self.use_trajectory:
            traj = compute_trajectory_features(features_norm, criteria, mask)
            pooled = torch.cat([pooled, traj], dim=1)

        patient_logit = self.patient_head(pooled).squeeze(-1)     # (B,)

        return {
            "patient_logit": patient_logit,
            "window_logits": window_logits,
            "attention": attn,
            "risk_prior": risk,
            "criteria": criteria,
        }

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
