"""
CTG-CrossFormer: Hybrid CNN-Transformer with Bidirectional Cross-Attention
==========================================================================
Implementation based on Dang et al. (2026):
"A Hybrid CNN-Transformer with Cross-Attention for Automated Fetal Distress Detection"

Architectural Hierarchy (Section 3.2):
1. Dual CNN Branches with Squeeze-and-Excitation (SE) Modules:
   - Independent 1D Conv branches for FHR and UC signals.
   - 3 Conv blocks with kernel sizes (7, 5, 3) and max pooling (4, 4, 2) => 32x downsampling.
   - SE module (r=4) for channel recalibration.
   - Outputs: F_FHR, F_UC ∈ R^(B × 150 × 128)
2. Bidirectional Cross-Attention Module:
   - Cross-attention between FHR (query) and UC (key/value), and vice versa.
   - 4 heads, d_k=32. Residual connection + LayerNorm.
   - Concatenation (256 dims) + Fusion Bottleneck (Linear -> ReLU -> LayerNorm) => F_fused ∈ R^(B × 150 × 256)
3. Global Pre-LN Transformer Encoder:
   - 4 layers, 8 heads, d_model=256, d_ff=512, GELU, 0.1 dropout + learnable 1D positional embedding.
4. Universal Latent Projection / Classification Head:
   - Global Average Pooling over temporal dimension L=150.
   - Latent Adapter: Linear(256 -> 128) + LayerNorm(128) => (Batch, 128) for KI-MTF framework.
   - Classification Head: Linear(256 -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 1) for standalone benchmark.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SqueezeAndExcitation1D(nn.Module):
    """
    1D Squeeze-and-Excitation (SE) Module for channel recalibration.
    Paper Eq (1): s = σ(W2 · ReLU(W1 · GAP(x)))
    """
    def __init__(self, channels: int, reduction_ratio: int = 4):
        super().__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channels, Length)
        b, c, _ = x.shape
        gap = x.mean(dim=-1)  # (Batch, Channels)
        scale = self.fc(gap).view(b, c, 1)  # (Batch, Channels, 1)
        return x * scale


class SingleChannelCNNBranch(nn.Module):
    """
    3-Stage 1D-CNN hierarchy with SE modules for a single CTG channel (FHR or UC).
    Downsamples length 4800 by 32x -> 150 temporal positions.
    Channels: 1 -> 32 -> 64 -> 128.
    Kernels: 7, 5, 3.
    Pools: 4, 4, 2 (4 * 4 * 2 = 32).
    """
    def __init__(self, in_channels: int = 1, reduction_ratio: int = 4):
        super().__init__()
        # Block 1: Conv(1->32, k=7, p=3) -> BN -> ReLU -> MaxPool(4) -> SE(32)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            SqueezeAndExcitation1D(32, reduction_ratio=reduction_ratio)
        )
        # Block 2: Conv(32->64, k=5, p=2) -> BN -> ReLU -> MaxPool(4) -> SE(64)
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            SqueezeAndExcitation1D(64, reduction_ratio=reduction_ratio)
        )
        # Block 3: Conv(64->128, k=3, p=1) -> BN -> ReLU -> MaxPool(2) -> SE(128)
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SqueezeAndExcitation1D(128, reduction_ratio=reduction_ratio)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 1, 4800)
        h = self.block1(x)  # (Batch, 32, 1200)
        h = self.block2(h)  # (Batch, 64, 300)
        h = self.block3(h)  # (Batch, 128, 150)
        # Transpose to (Batch, Sequence_Len=150, Channels=128)
        return h.transpose(1, 2)


class BidirectionalCrossAttentionModule(nn.Module):
    """
    Bidirectional Cross-Attention between FHR and UC features.
    Eq (2): F'_FHR = LN(F_FHR + MHA(Q=F_FHR, K=F_UC, V=F_UC))
    Eq (3): F'_UC  = LN(F_UC  + MHA(Q=F_UC, K=F_FHR, V=F_FHR))
    Followed by concatenation & fusion bottleneck: Linear(256->256) -> ReLU -> LayerNorm(256)
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha_fhr_uc = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.mha_uc_fhr = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ln_fhr = nn.LayerNorm(d_model)
        self.ln_uc = nn.LayerNorm(d_model)

        # Fusion bottleneck for concatenated features (128 + 128 = 256)
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(2 * d_model)
        )

    def forward(self, f_fhr: torch.Tensor, f_uc: torch.Tensor) -> torch.Tensor:
        # f_fhr: (Batch, 150, 128), f_uc: (Batch, 150, 128)
        attn_fhr, _ = self.mha_fhr_uc(query=f_fhr, key=f_uc, value=f_uc)
        f_prime_fhr = self.ln_fhr(f_fhr + attn_fhr)

        attn_uc, _ = self.mha_uc_fhr(query=f_uc, key=f_fhr, value=f_fhr)
        f_prime_uc = self.ln_uc(f_uc + attn_uc)

        # Concatenate along channel dimension -> (Batch, 150, 256)
        f_concat = torch.cat([f_prime_fhr, f_prime_uc], dim=-1)
        f_fused = self.fusion(f_concat)  # (Batch, 150, 256)
        return f_fused


class CTGCrossformerEncoder(nn.Module):
    """
    CTG-CrossFormer Backbone Encoder for CTG signals.
    Transforms (Batch, Channels=2, Seq_Len=4800) into a 128-dimensional latent vector z
    for compatibility with the Knowledge-Infused Multi-Task Framework (Model 8).
    """
    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 4800,
        cnn_channels: int = 128,
        n_heads_cross: int = 4,
        n_heads_tf: int = 8,
        n_tf_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Downsampled sequence length: 4800 / 32 = 150
        self.num_positions = seq_len // 32

        # 1. Dual CNN Branches with Squeeze-and-Excitation
        self.fhr_branch = SingleChannelCNNBranch(in_channels=1, reduction_ratio=4)
        self.uc_branch = SingleChannelCNNBranch(in_channels=1, reduction_ratio=4)

        # 2. Bidirectional Cross-Attention Module
        self.cross_attn = BidirectionalCrossAttentionModule(
            d_model=cnn_channels, n_heads=n_heads_cross, dropout=dropout
        )

        # 3. Fused dimension = 256 (128 + 128)
        fused_dim = cnn_channels * 2

        # Learnable 1D Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_positions, fused_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_dropout = nn.Dropout(dropout)

        # 4. Global Pre-LN Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_dim,
            nhead=n_heads_tf,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_tf_layers,
            enable_nested_tensor=False
        )

        # 5. Universal Latent Adapter for KI-MTF (256 -> 128)
        self.latent_adapter = nn.Sequential(
            nn.Linear(fused_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input CTG signal of shape (Batch, 2, 4800)
               Channel 0: FHR, Channel 1: UC
        Returns:
            z: Latent representation of shape (Batch, 128)
        """
        B, C, L = x.shape
        assert C == 2, f"Expected 2 input channels (FHR and UC), got {C}"
        assert L == self.seq_len, f"Expected sequence length {self.seq_len}, got {L}"

        # Split channels: FHR and UC
        x_fhr = x[:, 0:1, :]  # (Batch, 1, 4800)
        x_uc  = x[:, 1:2, :]  # (Batch, 1, 4800)

        # Stage 1: Dual CNN + SE feature extraction -> (Batch, 150, 128)
        f_fhr = self.fhr_branch(x_fhr)
        f_uc  = self.uc_branch(x_uc)

        # Stage 2: Cross-Attention & Fusion -> (Batch, 150, 256)
        f_fused = self.cross_attn(f_fhr, f_uc)

        # Add Positional Embeddings
        f_fused = f_fused + self.pos_embed[:, :f_fused.size(1), :]
        f_fused = self.pos_dropout(f_fused)

        # Stage 3: Global Transformer Encoder -> (Batch, 150, 256)
        tf_out = self.transformer_encoder(f_fused)

        # Stage 4: Global Average Pooling across temporal dimension L=150 -> (Batch, 256)
        pooled = tf_out.mean(dim=1)

        # Stage 5: Map to Universal Latent Vector -> (Batch, 128)
        z = self.latent_adapter(pooled)
        return z


class CTGCrossformerForClassification(nn.Module):
    """
    Standalone Paper Replication Classifier.
    Attaches the exact Section 3.2.4 Classification Head to CTGCrossformerEncoder:
    Global Avg Pool (256) -> Linear(256 -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 1)
    """
    def __init__(
        self,
        encoder: CTGCrossformerEncoder,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        self.encoder = encoder
        # Paper Classification Head: 256 -> 128 -> 1 (binary logit)
        fused_dim = encoder.cross_attn.fusion[0].in_features  # 256
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CTG signal (Batch, 2, 4800)
        Returns:
            Binary distress logit (Batch, 1)
        """
        # Re-run encoder stages up to transformer global pooling
        x_fhr = x[:, 0:1, :]
        x_uc  = x[:, 1:2, :]

        f_fhr = self.encoder.fhr_branch(x_fhr)
        f_uc  = self.encoder.uc_branch(x_uc)
        f_fused = self.encoder.cross_attn(f_fhr, f_uc)
        f_fused = f_fused + self.encoder.pos_embed[:, :f_fused.size(1), :]
        f_fused = self.encoder.pos_dropout(f_fused)

        tf_out = self.encoder.transformer_encoder(f_fused)
        pooled = tf_out.mean(dim=1)  # (Batch, 256)

        logits = self.classifier(pooled)  # (Batch, 1)
        return logits
