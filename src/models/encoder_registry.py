"""
Encoder registry -- lets the CRP pretraining and fine-tuning pipeline run with
any temporal encoder, not just the CrossFormer.

All encoders in this project share one signature:

    (Batch, 2, 4800)  ->  (Batch, latent_dim=128)

so they are interchangeable as long as the classification head reads the
128-d latent. That last clause is the catch, and it is worth stating plainly:

HEAD MISMATCH (read before comparing to published CrossFormer numbers).
CTGCrossformerForClassification does NOT call encoder.forward(). It re-runs the
encoder's internal stages and classifies from the 256-d mean-pooled transformer
output (ctg_crossformer.py:263-284), bypassing the encoder's final Linear(->128)
+ LayerNorm. Two consequences:

  1. Clinical-relational pretraining shapes the 128-d latent, but the CrossFormer
     classifier never reads it. CRP still transfers -- gradients flow through the
     shared trunk -- but the pretrained projection layer itself is discarded at
     fine-tuning time.
  2. CNN1D and MultiScaleLSTM have no 256-d intermediate to pool. Their natural
     output IS the 128-d latent.

So EncoderForClassification below classifies from the 128-d latent for every
architecture. For CNN1D/MSLSTM that is the only sensible choice and it makes CRP
fully effective. For the CrossFormer it is a DIFFERENT head from the delivered
model, which is why `--encoder crossformer_latent` exists: it runs the
CrossFormer through this same 128-d head so a three-way comparison is
apples-to-apples. `--encoder crossformer` keeps the original 256-d head and
reproduces the delivered result exactly.
"""
from typing import Optional

import torch
import torch.nn as nn

from src.models.bilstm_encoder import BiLSTMEncoder
from src.models.cnn1d_encoder import CNN1DEncoder
from src.models.ctg_crossformer import CTGCrossformerEncoder, CTGCrossformerForClassification
from src.models.gru_encoder import GRUEncoder
from src.models.multiscale_lstm import MultiScaleLSTMEncoder
from src.models.patchctg import PatchCTGEncoder
from src.models.patchtst import PatchTSTEncoder
from src.models.tcn_encoder import TCNEncoder

ENCODER_NAMES = ["crossformer", "crossformer_latent", "cnn1d", "mslstm",
                 "bilstm", "gru", "tcn", "patchctg", "patchtst"]

# The five encoders added 2026-09-03 for the protocol sweep all share the
# (B, C, 4800) -> (B, 128) contract, but they do NOT agree on what the latent
# argument is called: GRUEncoder and TCNEncoder take `hidden_dim`, the rest take
# `latent_dim`. They also do not all SET a .latent_dim attribute, which
# EncoderForClassification reads to size its head -- so build_encoder sets it.
_LATENT_KWARG = {"bilstm": "latent_dim", "gru": "hidden_dim", "tcn": "hidden_dim",
                 "patchctg": "latent_dim", "patchtst": "latent_dim"}
_SIMPLE_ENCODERS = {"bilstm": BiLSTMEncoder, "gru": GRUEncoder, "tcn": TCNEncoder,
                    "patchctg": PatchCTGEncoder, "patchtst": PatchTSTEncoder}


def build_encoder(name: str, m_cfg: Optional[dict] = None, latent_dim: int = 128) -> nn.Module:
    """Construct a bare encoder by name. m_cfg is the `model:` block of the YAML."""
    m_cfg = m_cfg or {}
    ch = m_cfg.get("in_channels", 2)
    if name in ("crossformer", "crossformer_latent"):
        if ch != 2:
            raise ValueError(
                "CTGCrossformerEncoder is a dual-branch architecture: it splits the "
                "input into an FHR branch and a UC branch and asserts C == 2. It "
                f"cannot take {ch} channels. Use --in_channels 2, or cnn1d/mslstm "
                "for a missingness-mask ablation.")
        return CTGCrossformerEncoder(
            in_channels=2, seq_len=4800,
            cnn_channels=m_cfg.get("cnn_channels", 128),
            n_heads_cross=m_cfg.get("n_heads_cross", 4),
            n_heads_tf=m_cfg.get("n_heads_tf", 8),
            n_tf_layers=m_cfg.get("n_tf_layers", 4),
            d_ff=m_cfg.get("d_ff", 512),
            dropout=m_cfg.get("dropout", 0.1),
            latent_dim=m_cfg.get("latent_dim", latent_dim),
        )
    if name == "cnn1d":
        return CNN1DEncoder(in_channels=ch, seq_len=4800,
                            latent_dim=m_cfg.get("latent_dim", latent_dim))
    if name == "mslstm":
        return MultiScaleLSTMEncoder(
            in_channels=ch, seq_len=4800,
            hidden_size=m_cfg.get("mslstm_hidden_size", 64),
            num_layers=m_cfg.get("mslstm_num_layers", 2),
            dropout=m_cfg.get("mslstm_dropout", 0.2),
            latent_dim=m_cfg.get("latent_dim", latent_dim),
        )
    if name in _SIMPLE_ENCODERS:
        ld = m_cfg.get("latent_dim", latent_dim)
        enc = _SIMPLE_ENCODERS[name](in_channels=ch, seq_len=4800,
                                     **{_LATENT_KWARG[name]: ld})
        if not hasattr(enc, "latent_dim"):
            enc.latent_dim = ld
        return enc
    raise ValueError(f"unknown encoder '{name}'. choose from {ENCODER_NAMES}")


class EncoderForClassification(nn.Module):
    """
    Architecture-agnostic binary head on top of the 128-d latent.

    Mirrors the CrossFormer head's shape (latent -> hidden -> ReLU -> dropout ->
    1) so the only thing that differs between architectures is the encoder.
    """

    def __init__(self, encoder: nn.Module, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        latent = getattr(encoder, "latent_dim", 128)
        self.classifier = nn.Sequential(
            nn.Linear(latent, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if isinstance(z, tuple):
            z = z[0]
        return self.classifier(z)


def build_classifier(name: str, encoder: nn.Module, m_cfg: Optional[dict] = None) -> nn.Module:
    """Wrap an encoder in the appropriate classification head for its family."""
    m_cfg = m_cfg or {}
    hidden = m_cfg.get("classifier_hidden_dim", 128)
    drop = m_cfg.get("classifier_dropout", 0.3)
    if name == "crossformer":
        # Original 256-d pooled head -- reproduces the delivered model exactly.
        return CTGCrossformerForClassification(encoder=encoder, hidden_dim=hidden, dropout=drop)
    return EncoderForClassification(encoder=encoder, hidden_dim=hidden, dropout=drop)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
