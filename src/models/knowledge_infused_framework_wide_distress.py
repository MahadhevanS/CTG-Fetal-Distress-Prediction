"""
Knowledge-Infused Multi-Task Framework -- Wide-Distress Variant (prototype, 2026-08-16)
=========================================================================================
Gives DistressHead a private path to CrossFormer's richer 256-dim pre-adapter
pooled representation, instead of the shared 128-dim latent every other head
uses. Mirrors exactly what CTGCrossformerForClassification (the standalone
benchmark) already does: Linear(256->128)->ReLU->Dropout(0.3)->Linear(128->1).

WHY: across every experiment this session, Model 8's DistressHead AUROC (best:
0.7893, `full`, CrossFormer) has trailed the standalone CrossFormer classifier's
own AUROC (0.8565) by ~0.067 -- the single largest, most consistently-observed
gap of anything tried. The best-diagnosed cause: DistressHead only ever sees the
128-dim latent `z` AFTER it has been compressed by latent_adapter (Linear(256->128),
shared across all 3-4 task heads), while the standalone classifier bypasses that
adapter and classifies directly from the un-compressed 256-dim pooled
representation. FIGOHead/ClinicalFeatureHead/FIGOCriteriaHead still share the
128-dim `z` unchanged -- only DistressHead's input changes.

PATENT NOTE: initially treated the 128-dim shared bottleneck as an untouchable
patent-differentiation constraint -- re-checked docs/patent_risk_analysis.md
directly and that's an overstatement. The documented differentiation (continuous
end-to-end signal->latent mapping, no bounding boxes, no shape-matching/
correlation loops, knowledge-infused loss, biochemical target, explainability)
does not depend on a specific latent width or on every head sharing one
bottleneck -- this variant is still a single continuous differentiable mapping
from raw signal to prediction, so it doesn't reintroduce anything GE's patent
actually claims.

SCOPE: kept as a separate file/class rather than modifying knowledge_infused_
framework.py in place, so every existing config, ablation result, and
comparison from this session remains exactly reproducible against the
original KnowledgeInfusedFramework. This is a prototype to A/B against that
baseline, not a replacement.
"""

import torch
import torch.nn as nn
from typing import Tuple

from src.models.ctg_crossformer import CTGCrossformerEncoder
from src.models.knowledge_infused_framework import (
    FIGOHead,
    ClinicalFeatureHead,
    FIGOCriteriaHead,
)


class CTGCrossformerDualLatentEncoder(CTGCrossformerEncoder):
    """
    Same weights/architecture as CTGCrossformerEncoder (subclass, not a fork --
    identical __init__, so state_dicts are interchangeable with the original
    encoder). Only forward() changes: computes the shared CNN/cross-attention/
    transformer stages ONCE and returns BOTH representations instead of just
    the final compressed one:

        z:      (Batch, 128) -- the usual compressed universal latent, via
                latent_adapter. Same tensor FIGOHead/ClinicalFeatureHead/
                FIGOCriteriaHead already consume.
        pooled: (Batch, 256) -- the pre-adapter globally-pooled representation,
                before latent_adapter's compression. What DistressHead consumes
                in this variant.

    Deliberately does NOT duplicate CTGCrossformerForClassification's pattern
    of re-running the encoder's internal stages a second time from scratch --
    that class recomputes fhr_branch/uc_branch/cross_attn/transformer_encoder
    a second time to reach `pooled`, which is correct but wasteful when both
    representations are needed from the same forward pass (this class's actual
    use case).
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, L = x.shape
        assert C == 2, f"Expected 2 input channels (FHR and UC), got {C}"
        assert L == self.seq_len, f"Expected sequence length {self.seq_len}, got {L}"

        x_fhr = x[:, 0:1, :]
        x_uc = x[:, 1:2, :]

        f_fhr = self.fhr_branch(x_fhr)
        f_uc = self.uc_branch(x_uc)
        f_fused = self.cross_attn(f_fhr, f_uc)

        f_fused = f_fused + self.pos_embed[:, :f_fused.size(1), :]
        f_fused = self.pos_dropout(f_fused)

        tf_out = self.transformer_encoder(f_fused)
        pooled = tf_out.mean(dim=1)  # (Batch, 256)

        z = self.latent_adapter(pooled)  # (Batch, 128)
        return z, pooled


class WideDistressHead(nn.Module):
    """
    DistressHead variant that classifies from the 256-dim pre-adapter pooled
    representation instead of the shared 128-dim latent. Architecture matches
    CTGCrossformerForClassification's own classifier exactly (same hidden_dim/
    dropout defaults as configs/model8_crossformer_config.yaml's heads section,
    which already matched this same classifier's capacity for the OTHER heads).
    """

    def __init__(self, pooled_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.head(pooled)


class KnowledgeInfusedFrameworkWideDistress(nn.Module):
    """
    Drop-in-compatible variant of KnowledgeInfusedFramework: same constructor
    shape, same forward() output arity (3-tuple, or 4-tuple with
    include_criteria_head=True) -- existing call sites (_forward_model() in
    train_knowledge_infused.py) work unchanged. The only functional difference
    is DistressHead's input: 256-dim pooled representation instead of the
    128-dim shared latent `z`.

    Requires a dual-output encoder (CTGCrossformerDualLatentEncoder or
    equivalent) whose forward() returns (z, pooled) -- NOT a drop-in for the
    universal single-latent-output encoders (PatchTSTEncoder,
    MultiScaleLSTMEncoder, CNN1DEncoder) used elsewhere in this project.
    CrossFormer-specific by construction, mirroring how the head-capacity fix
    in configs/model8_crossformer_config.yaml was already scoped to CrossFormer
    only.
    """

    def __init__(
        self,
        encoder: CTGCrossformerDualLatentEncoder,
        latent_dim: int = 128,
        pooled_dim: int = 256,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.2,
        distress_hidden_dim: int = 128,
        distress_dropout: float = 0.3,
        include_criteria_head: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.include_criteria_head = include_criteria_head

        self.distress_head = WideDistressHead(
            pooled_dim=pooled_dim, hidden_dim=distress_hidden_dim, dropout=distress_dropout
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
        Returns:
            Tuple of:
              - distress_logit:  (Batch, 1)  -- from the 256-dim pooled representation
              - figo_logits:     (Batch, 3)  -- from the shared 128-dim z, unchanged
              - feature_preds:   (Batch, 8)  -- from the shared 128-dim z, unchanged
              - criteria_logits: (Batch, 7)  -- ONLY present if include_criteria_head=True
        """
        z, pooled = self.encoder(x)
        distress_logit = self.distress_head(pooled)
        figo_logits = self.figo_head(z)
        feature_preds = self.feature_head(z)
        if self.include_criteria_head:
            criteria_logits = self.criteria_head(z)
            return distress_logit, figo_logits, feature_preds, criteria_logits
        return distress_logit, figo_logits, feature_preds

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
