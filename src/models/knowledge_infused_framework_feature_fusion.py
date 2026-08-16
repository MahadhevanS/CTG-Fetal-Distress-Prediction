"""
Knowledge-Infused Multi-Task Framework -- Feature-Fusion Variant (prototype, 2026-08-16)
==========================================================================================
Gives DistressHead direct access to the 8 hand-crafted clinical features
(baseline, STV, LTV, accel/decel counts) as INPUT, concatenated with the
shared latent `z`, instead of only ever seeing them as an auxiliary
*prediction target* (ClinicalFeatureHead) or rule-loss constraint.

WHY: every knowledge-infusion mechanism tried this session (FIGOHead,
ClinicalFeatureHead, FIGOCriteriaHead, figo_rule_loss) infuses knowledge as an
auxiliary OUTPUT the encoder has to learn to reproduce from the raw signal --
a weak, indirect regularization signal. A quick standalone check (2026-08-16,
GradientBoostingClassifier trained directly on the 8 features alone, patient-
level 5-fold CV, no raw signal at all) got AUROC 0.7297 +/- 0.0347 -- close to,
but below, the deep model's ~0.79-0.80. That means these 8 numbers carry real,
substantial signal that isn't redundant with what the deep model already
extracts, but nothing in the framework ever hands them to the network directly
as input -- it only asks the network to re-derive them from scratch through a
weak auxiliary loss. Direct fusion is a much stronger, more direct form of
infusion than an auxiliary loss can be.

WHY THIS IS NOT LEAKAGE: y_features are not expert/clinical ground-truth
labels -- they're deterministically computable from the raw FHR/UC signal via
the same signal-processing pipeline (calculate_iterative_baseline,
calculate_variability, detect_accelerations, detect_decelerations) that
already runs on every window before any model ever sees it. A real deployed
system would compute them the same way at inference time, exactly like this
model expects them. This mirrors real clinical decision support: combining
"what the raw trace shows" with "what the summary numbers say" is standard
practice, not information the model wouldn't have in production.

WHY ONLY DistressHead, NOT the other heads: FIGOHead's target (FIGO class) and
FIGOCriteriaHead's targets are themselves DERIVED from y_features via
classify_figo()/derive_figo_criteria_flags() -- deterministic functions of
these exact 8 numbers. Feeding y_features into those heads as input would make
their auxiliary tasks trivial (implement the lookup, not learn anything),
destroying their value as a signal that pushes the shared encoder toward
clinically-relevant representations. ClinicalFeatureHead's job is literally to
predict y_features FROM z alone -- feeding y_features into it as input would
mean predicting its own input back, a degenerate shortcut. DistressHead's
target (pH-based distress) is NOT derived from y_features by any known rule,
so fusing them there is legitimate feature engineering, not task-trivializing.

SCOPE: backbone-agnostic (works with any single-output encoder -- PatchTST,
standard CrossFormer, MultiScaleLSTM, CNN1D -- unlike the wide-distress
variant, which is CrossFormer-specific by construction). Kept as a separate
file, opt-in via heads.feature_fusion: true, so every existing config/result
remains exactly reproducible.
"""

import torch
import torch.nn as nn

from src.models.knowledge_infused_framework import (
    FIGOHead,
    ClinicalFeatureHead,
    FIGOCriteriaHead,
)


class FeatureFusionDistressHead(nn.Module):
    """
    DistressHead variant that classifies from [z ; normalized y_features]
    instead of z alone. Same depth/capacity as the original DistressHead
    (Linear->LayerNorm->GELU->Dropout->Linear), just a wider first layer to
    accommodate the concatenated input.
    """

    def __init__(self, latent_dim: int = 128, n_features: int = 8,
                 hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim + n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, y_features_norm: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([z, y_features_norm], dim=-1)
        return self.head(fused)


class KnowledgeInfusedFrameworkFeatureFusion(nn.Module):
    """
    Drop-in-compatible variant of KnowledgeInfusedFramework, EXCEPT forward()
    takes an additional y_features argument -- call sites must use
    _forward_model() in train_knowledge_infused.py, which checks this class's
    `requires_features_input = True` marker and passes y_features
    automatically wherever it's already in scope (every training/validation/
    SWA loop already unpacks yfeat_batch from the DataLoader).

    feature_means/feature_stds are registered as buffers (not plain
    attributes) so they move to the correct device automatically with
    model.to(device), and the model is self-contained -- callers pass raw
    (un-normalized) y_features straight from the dataset, normalization
    happens inside forward().
    """

    requires_features_input = True

    def __init__(
        self,
        encoder: nn.Module,
        feature_means: torch.Tensor,
        feature_stds: torch.Tensor,
        latent_dim: int = 128,
        n_features: int = 8,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.2,
        include_criteria_head: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.include_criteria_head = include_criteria_head

        self.register_buffer("feature_means", feature_means.clone().detach())
        self.register_buffer("feature_stds", feature_stds.clone().detach().clamp(min=1e-6))

        self.distress_head = FeatureFusionDistressHead(
            latent_dim=latent_dim, n_features=n_features,
            hidden_dim=head_hidden_dim, dropout=head_dropout,
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

    def forward(self, x: torch.Tensor, y_features: torch.Tensor):
        """
        Args:
            x:          Input CTG signal (Batch, 2, 4800).
            y_features: RAW (un-normalized) clinical features (Batch, 8), same
                        column order/units as *_dataset.pt's y_features
                        [Baseline, STV, LTV, Accels, Early, Late, Variable, Prolonged].
        Returns:
            Same tuple shape as KnowledgeInfusedFramework (3-tuple, or 4-tuple
            with include_criteria_head=True). Only distress_logit's derivation
            changes -- figo_logits/feature_preds/criteria_logits are computed
            from z alone, exactly as in the original framework.
        """
        z = self.encoder(x)
        y_features_norm = (y_features - self.feature_means) / self.feature_stds
        distress_logit = self.distress_head(z, y_features_norm)
        figo_logits = self.figo_head(z)
        feature_preds = self.feature_head(z)
        if self.include_criteria_head:
            criteria_logits = self.criteria_head(z)
            return distress_logit, figo_logits, feature_preds, criteria_logits
        return distress_logit, figo_logits, feature_preds

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
