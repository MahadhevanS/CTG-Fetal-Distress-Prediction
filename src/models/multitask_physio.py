"""
Multi-task physiological supervision over a shared CTG representation.

IDEA. The binding constraint on this dataset is 110 positive patients, not model
capacity (8 architectures across a 17.4x parameter range all land 0.617-0.718
patient-level, at or below a 19-feature logistic regression at 0.7268). Extra
INPUTS do not help -- the 19 clinical features are mutually redundant, and
fusing a signal branch with a feature branch measured NEGATIVE twice. Extra
SUPERVISION might, because it adds gradient signal the binary label cannot.

    CTG window -> encoder -> z (128-d)
                              |-- distress   (binary, the only head used at inference)
                              |-- pH         (regression)
                              |-- BDecf      (regression)
                              |-- Apgar5     (regression)
                              +-- descriptors (19-d regression: 8 FIGO + 11 extended)

HONESTY NOTE -- READ BEFORE WRITING THIS UP
-------------------------------------------
y_primary is DEFINED as pH <= 7.15, so pH ranks the label at AUROC 1.0000 and
BDecf at 0.9386 (they correlate r=-0.727). The pH head is therefore NOT
independent physiological knowledge -- it is the same label at higher
resolution, 547 continuous values in place of 110 binary ones. That is a
legitimate and standard use of auxiliary supervision (the head exists only at
training time; inference uses `distress` alone) but it must be described
accurately. Do not claim the pH head injects new clinical information.

Apgar5 is genuinely different information (r=+0.426 with pH) but CTG predicts it
barely above chance (Apgar5<7 -> 0.5592 patient AUROC), so that head may add
noise rather than signal. The ladder in scripts/run_multitask_ladder.py exists
to find out one target at a time.

The descriptor head is the one with the strongest prior evidence: Model 8's
`plus_features` ablation measured +0.048 (docs/model_inferences_log.md), the
largest knowledge-infusion gain this project has recorded. Note this is
regression ONTO the descriptors, which shapes the representation -- not
concatenation of the descriptors as inputs, which is a different mechanism and
measured negative.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskPhysioNet(nn.Module):
    """Shared encoder, one classification head plus optional regression heads.

    Only heads that are enabled are constructed, so an ablation rung carries
    exactly the parameters it needs and nothing else.
    """

    def __init__(self, encoder: nn.Module, hidden_dim: int = 128, dropout: float = 0.3,
                 n_descriptors: int = 19, use_ph: bool = False, use_bdecf: bool = False,
                 use_apgar: bool = False, use_descriptors: bool = False):
        super().__init__()
        self.encoder = encoder
        latent = getattr(encoder, "latent_dim", 128)
        self.distress = nn.Sequential(
            nn.Linear(latent, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.ph = nn.Linear(latent, 1) if use_ph else None
        self.bdecf = nn.Linear(latent, 1) if use_bdecf else None
        self.apgar = nn.Linear(latent, 1) if use_apgar else None
        self.descriptors = nn.Linear(latent, n_descriptors) if use_descriptors else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        if isinstance(z, tuple):
            z = z[0]
        out = {"distress": self.distress(z).squeeze(-1)}
        if self.ph is not None:
            out["ph"] = self.ph(z).squeeze(-1)
        if self.bdecf is not None:
            out["bdecf"] = self.bdecf(z).squeeze(-1)
        if self.apgar is not None:
            out["apgar"] = self.apgar(z).squeeze(-1)
        if self.descriptors is not None:
            out["descriptors"] = self.descriptors(z)
        return out


def focal_bce(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss, gamma=2.0, pos_weight=1.0 -- matching the protocol sweep.

    pos_weight stays 1.0 because the sqrt-inverse WeightedRandomSampler already
    rebalances each batch; stacking both is the double-rebalancing documented at
    train_ctg_crossformer.py:419.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
    return ((1 - p_t) ** gamma * bce).mean()


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """MSE that skips missing values (BDecf is absent for 10 patients).

    Returns a zero that still carries grad_fn when a batch has no valid target,
    so the backward pass does not break on an all-missing batch.
    """
    if mask is None:
        mask = torch.isfinite(target)
    mask = mask & torch.isfinite(target)
    if not mask.any():
        return (pred * 0.0).sum()
    d = (pred[mask] - target[mask]) ** 2
    return d.mean()


def multitask_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                   lam: float = 0.3, gamma: float = 2.0) -> Dict[str, torch.Tensor]:
    """Composite loss. Auxiliary terms share one weight so the ladder varies
    WHICH targets are present, not how they are weighted -- otherwise a rung
    could win on tuning rather than on information."""
    losses = {"distress": focal_bce(out["distress"], batch["y"], gamma)}
    for key in ("ph", "bdecf", "apgar"):
        if key in out:
            losses[key] = lam * masked_mse(out[key], batch[key])
    if "descriptors" in out:
        losses["descriptors"] = lam * masked_mse(out["descriptors"], batch["descriptors"])
    losses["total"] = sum(losses.values())
    return losses
