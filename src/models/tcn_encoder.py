import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ChopCausalPadding(nn.Module):
    """
    Slices off trailing padding from sequence dimension to enforce strict temporal causality.
    """
    def __init__(self, pad_size: int):
        super(ChopCausalPadding, self).__init__()
        self.pad_size = pad_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_size <= 0:
            return x
        return x[:, :, :-self.pad_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Causal Dilated Residual Block for Temporal Convolutional Network (TCN).
    """
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chop1 = ChopCausalPadding(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chop2 = ChopCausalPadding(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chop1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chop2, self.bn2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = self.net(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network (TCN) Encoder with causal dilated convolutions.
    Conforms to universal signature:
      Input:  (Batch, 2, 4800) [Channel 0: FHR, Channel 1: UC]
      Output: (Batch, 128)     [1D latent representation]

    Patent Compliance (GE US12094611B2):
    Encodes raw 2D continuous sequence tensors directly to R^128 without explicit
    graphical pattern extraction, bounding box proposals, or post-hoc shape matching loops.
    """
    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 4800,
        hidden_dim: int = 128,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super(TCNEncoder, self).__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        if num_channels is None:
            # 6 dilated layers with channels 32, 64, 64, 128, 128, 128
            num_channels = [32, 64, 64, 128, 128, 128]

        # Temporal downsampling stem: (Batch, 2, 4800) -> (Batch, num_channels[0], 2400)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, num_channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU(inplace=True)
        )

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = num_channels[i - 1] if i > 0 else num_channels[0]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(
                    n_inputs=in_ch,
                    n_outputs=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout
                )
            )

        self.tcn = nn.Sequential(*layers)

        # Global temporal pooling & projection
        last_ch = num_channels[-1]
        self.fc_proj = nn.Sequential(
            nn.Linear(last_ch * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, 2, 4800)
        Returns:
            latent: Tensor of shape (Batch, 128)
        """
        if x.dim() != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected input shape (Batch, {self.in_channels}, {self.seq_len}), got {tuple(x.shape)}")

        # 1. Temporal stem downsampling: (Batch, 2, 4800) -> (Batch, num_channels[0], 2400)
        x_stem = self.stem(x)

        # 2. Causal Dilated Convolutions: (Batch, num_channels[0], 2400) -> (Batch, last_ch, 2400)
        features = self.tcn(x_stem)

        # 2. Global Mean + Max Pooling across time dimension
        mean_pool = torch.mean(features, dim=2)
        max_pool, _ = torch.max(features, dim=2)
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (Batch, last_ch * 2)

        # 3. Latent projection: (Batch, last_ch * 2) -> (Batch, 128)
        latent = self.fc_proj(pooled)
        return latent
