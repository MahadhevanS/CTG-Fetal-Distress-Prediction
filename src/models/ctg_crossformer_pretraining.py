"""
Self-supervised pretraining scaffold for CTGCrossformerEncoder (see
docs/model8_crossformer_run_history.md, 2026-08-17 entry, and the
associated plan). Prototype, kept in its own file -- exactly the pattern
`knowledge_infused_framework_wide_distress.py`'s `CTGCrossformerDualLatentEncoder`
already established for exposing an extra intermediate tensor: subclass
CTGCrossformerEncoder, inherit __init__ unchanged (no new params added), only
override forward(). This guarantees state_dict() key/shape parity with the
base CTGCrossformerEncoder, so a pretrained checkpoint from this file loads
directly into the plain CTGCrossformerEncoder used everywhere else in this
project (train_ctg_crossformer.py, train_knowledge_infused.py) via
load_state_dict(..., strict=True).

Only CTGCrossformerSSLEncoder's weights are ever saved/reused downstream --
CTGReconstructionDecoder and CTGCrossformerSSLPretrainer exist purely to
drive the pretext-task training loop and are discarded afterward.
"""

from typing import Tuple

import torch
import torch.nn as nn

from src.models.ctg_crossformer import CTGCrossformerEncoder


class CTGCrossformerSSLEncoder(CTGCrossformerEncoder):
    """Same weights/architecture as CTGCrossformerEncoder (subclass, not a
    fork -- state_dicts are interchangeable). forward() additionally returns
    tf_out (B,150,256), the per-token transformer output immediately before
    the mean-pooling + latent_adapter compression, needed by the
    reconstruction decoder to localize masked positions."""

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

        tf_out = self.transformer_encoder(f_fused)  # (B, 150, 256)
        pooled = tf_out.mean(dim=1)
        z = self.latent_adapter(pooled)
        return z, tf_out


class CTGReconstructionDecoder(nn.Module):
    """Throwaway pretraining-only module: maps tf_out (B,150,256) back to a
    raw-signal reconstruction (B,2,4800), mirroring the encoder's CNN
    downsampling (pools 4,4,2 -> 4800/32=150) in reverse via ConvTranspose1d.
    Exact output lengths: 150->300->1200->4800 (verified: (150-1)*2-2+4=300,
    (300-1)*4-4+8=1200, (1200-1)*4-4+8=4800)."""

    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.up1 = nn.Sequential(  # 150 -> 300
            nn.ConvTranspose1d(d_model, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(  # 300 -> 1200
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.up3 = nn.Sequential(  # 1200 -> 4800
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_conv = nn.Conv1d(32, 2, kernel_size=7, padding=3)

    def forward(self, tf_out: torch.Tensor) -> torch.Tensor:
        h = tf_out.transpose(1, 2)  # (B, 256, 150)
        h = self.up1(h)
        h = self.up2(h)
        h = self.dropout(self.up3(h))
        return self.out_conv(h)  # (B, 2, 4800)


class CTGCrossformerSSLPretrainer(nn.Module):
    """Thin wrapper tying the SSL encoder to the reconstruction decoder for
    the pretraining loop. Not saved/reused -- see module docstring."""

    def __init__(self, encoder: CTGCrossformerSSLEncoder, decoder: CTGReconstructionDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        _, tf_out = self.encoder(x_masked)
        return self.decoder(tf_out)


def masked_reconstruction_loss(
    recon: torch.Tensor, X_clean: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked-only MSE -- reconstruction is only supervised where mask=True;
    unmasked (already-visible) positions contribute nothing to the loss."""
    mask_f = mask.float()
    return ((recon - X_clean) ** 2 * mask_f).sum() / mask_f.sum().clamp(min=1.0)
