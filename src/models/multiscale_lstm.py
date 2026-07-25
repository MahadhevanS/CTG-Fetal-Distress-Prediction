"""
Multi-Scale LSTM Temporal Encoder for CTG Fetal Distress Prediction
====================================================================

Phase 3 Model 5: Multi-Scale LSTM Literature Baseline conforming to:
- Input Shape: (Batch, 2, 4800)
- Output Latent Shape: (Batch, 128)
- GE Patent US12094611B2 Non-Infringement Boundary: Continuous end-to-end signal
  encoding across multiple temporal resolutions without longitudinal graphical
  pattern matching or bounding-box loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLSTMEncoder(nn.Module):
    """
    Multi-Scale LSTM Temporal Encoder for multi-channel Cardiotocography (CTG) time-series.

    Captures temporal dependencies across multiple resolutions (fine, medium, coarse)
    using parallel 1D convolutional scale projections coupled with parallel Bidirectional LSTMs.

    Args:
        in_channels (int): Number of input signal channels (default 2: FHR and UC).
        seq_len (int): Temporal sequence length (default 4800 for 20 min @ 4 Hz).
        hidden_size (int): Hidden dimension per LSTM branch direction (default 64 -> 128 bi-dir).
        num_layers (int): Number of LSTM layers per scale branch (default 2).
        dropout (float): Dropout probability (default 0.2).
        latent_dim (int): Final output latent representation dimension (default 128).
    """
    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 4800,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        # 1. Multi-scale 1D Convolutional Front-ends for temporal resolution extraction
        # Fine scale (Scale 1): stride 2, receptive field ~7
        self.scale1_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        # Medium scale (Scale 2): stride 8, receptive field ~15
        self.scale2_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=8, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        # Coarse scale (Scale 3): stride 32, receptive field ~31
        self.scale3_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=31, stride=32, padding=15),
            nn.BatchNorm1d(32),
            nn.GELU()
        )

        # 2. Parallel Bidirectional LSTM Branches
        lstm_out_dim = hidden_size * 2  # Bidirectional

        self.lstm1 = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.lstm2 = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.lstm3 = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 3. Latent Projection Head combining all 3 temporal scales into R^128
        total_scale_dim = lstm_out_dim * 3  # 128 * 3 = 384
        self.latent_head = nn.Sequential(
            nn.Linear(total_scale_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
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

        # 1. Multi-scale Convolutional Feature Extraction
        # Scale 1: (B, 32, 2400)
        s1 = self.scale1_conv(x).transpose(1, 2)  # (B, 2400, 32)
        # Scale 2: (B, 32, 600)
        s2 = self.scale2_conv(x).transpose(1, 2)  # (B, 600, 32)
        # Scale 3: (B, 32, 150)
        s3 = self.scale3_conv(x).transpose(1, 2)  # (B, 150, 32)

        # 2. Process each scale through parallel BiLSTMs
        out1, _ = self.lstm1(s1)  # (B, 2400, 128)
        out2, _ = self.lstm2(s2)  # (B, 600, 128)
        out3, _ = self.lstm3(s3)  # (B, 150, 128)

        # 3. Global Temporal Pooling per scale (average pooling over sequence dimension)
        pool1 = out1.mean(dim=1)  # (B, 128)
        pool2 = out2.mean(dim=1)  # (B, 128)
        pool3 = out3.mean(dim=1)  # (B, 128)

        # 4. Concatenate multi-scale representations: (B, 384)
        multi_scale_repr = torch.cat([pool1, pool2, pool3], dim=1)

        # 5. Project to Universal Latent Vector: (B, 128)
        latent = self.latent_head(multi_scale_repr)
        return latent


class MultiScaleLSTMForClassification(nn.Module):
    """
    Benchmarking Wrapper attaching a standardized MLP classification head
    to the MultiScaleLSTMEncoder latent representation.
    """
    def __init__(self, encoder: MultiScaleLSTMEncoder, hidden_dim: int = 64, dropout: float = 0.2):
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
