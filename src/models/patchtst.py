"""
PatchTST (Patch Time Series Transformer) Encoder for CTG Fetal Distress Prediction
===================================================================================

Phase 3 Model 7: PatchTST Baseline Encoder conforming to:
- Input Shape: (Batch, 2, 4800)
- Output Latent Shape: (Batch, 128)
- GE Patent US12094611B2 Non-Infringement Boundary: Continuous end-to-end signal
  encoding without longitudinal graphical pattern matching or bounding-box loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTSTEncoder(nn.Module):
    """
    PatchTST Temporal Encoder for multi-channel Cardiotocography (CTG) time-series.

    Transforms input (Batch, Channels=2, Seq_Len=4800) into a 128-dimensional
    latent representation using channel-independent patchification and
    multi-head self-attention Transformer blocks.

    Args:
        in_channels (int): Number of input signal channels (default 2: FHR and UC).
        seq_len (int): Temporal sequence length (default 4800 for 20 min @ 4 Hz).
        patch_len (int): Length of each temporal patch (default 16 samples = 4 sec).
        stride (int): Stride between patches (default 16 = non-overlapping patches).
        d_model (int): Hidden dimension of the Transformer (default 128).
        n_heads (int): Number of multi-head self-attention heads (default 8).
        n_layers (int): Number of Transformer encoder layers (default 3).
        d_ff (int): Dimension of feedforward network (default 512).
        dropout (float): Dropout probability (default 0.1).
        latent_dim (int): Final output latent representation dimension (default 128).
    """
    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 4800,
        patch_len: int = 16,
        stride: int = 16,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.latent_dim = latent_dim

        # Calculate number of patches per channel
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Linear projection from patch vector to d_model
        self.patch_embed = nn.Linear(patch_len, d_model)

        # Learnable 1D Positional Embeddings for patch sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder Blocks (Pre-LN for training stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False
        )

        # Adaptive pooling across patches: (B * C, d_model, N) -> (B * C, d_model, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Bottleneck Linear Projection to fixed latent dimension R^128
        self.latent_head = nn.Sequential(
            nn.Linear(in_channels * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input CTG signal of shape (Batch, 2, 4800)
                              Channel 0: Baseline-corrected FHR, Channel 1: UC
        Returns:
            torch.Tensor: Latent representation of shape (Batch, 128)
        """
        B, C, L = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        assert L == self.seq_len, f"Expected sequence length {self.seq_len}, got {L}"

        # 1. Patchify sequence: (B, C, L) -> (B, C, N, P)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        N = patches.size(2)  # Number of patches (300 for seq_len=4800, patch_len=16, stride=16)

        # 2. Channel-Independent Reshape: (B * C, N, P)
        x_patched = patches.contiguous().view(B * C, N, self.patch_len)

        # 3. Patch Linear Embedding: (B * C, N, d_model)
        enc_out = self.patch_embed(x_patched)

        # 4. Add Positional Embeddings & Dropout
        enc_out = enc_out + self.pos_embed[:, :N, :]
        enc_out = self.dropout(enc_out)

        # 5. Transformer Encoder: (B * C, N, d_model)
        enc_out = self.transformer_encoder(enc_out)

        # 6. Global Pooling over Patch Dimension: (B * C, N, d_model) -> (B * C, d_model, 1)
        enc_out = enc_out.transpose(1, 2)
        pooled = self.pool(enc_out).squeeze(-1)  # (B * C, d_model)

        # 7. Channel Recombination: (B, C * d_model)
        pooled_channels = pooled.view(B, C * self.d_model)

        # 8. Projection to Universal Latent Vector: (B, 128)
        latent = self.latent_head(pooled_channels)
        return latent


class PatchTSTForClassification(nn.Module):
    """
    Benchmarking Wrapper attaching a standardized MLP classification head
    to the PatchTSTEncoder latent representation.
    """
    def __init__(self, encoder: PatchTSTEncoder, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Signal shape (Batch, 2, 4800)
        Returns:
            torch.Tensor: Unnormalized logits of shape (Batch, 1)
        """
        latent = self.encoder(x)          # (Batch, 128)
        logits = self.classifier(latent)  # (Batch, 1)
        return logits
