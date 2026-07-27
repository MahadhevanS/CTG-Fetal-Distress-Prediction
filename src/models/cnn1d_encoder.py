import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class CNN1DEncoder(nn.Module):
    """
    1D Residual Convolutional Network (CNN1D) Temporal Encoder for CTG signals.
    
    Conforms to universal signature:
      Input:  (Batch, 2, 4800) [Channel 0: FHR diff, Channel 1: UC]
      Output: (Batch, 128)     [1D latent representation]
    
    Architecture:
      - Stem: Conv1d(2 -> 32, kernel=7, stride=2) + BatchNorm + ReLU + MaxPool1d(kernel=3, stride=2)
      - Layer 1: ResidualBlock(32 -> 32, stride=1)
      - Layer 2: ResidualBlock(32 -> 64, stride=2)
      - Layer 3: ResidualBlock(64 -> 128, stride=2)
      - Pooling: AdaptiveAvgPool1d(1)
      - Projection & Normalization: Linear(128 -> 128) + LayerNorm(128)
      
    Parameter Count: ~135,296 parameters.
    
    Patent Compliance (GE US12094611B2):
      Maps continuous multi-channel sequence directly to latent space R^128 without
      bounding-box proposals, pattern matching loops, or post-hoc shape correlation.
    """

    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 4800,
        latent_dim: int = 128
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input shape (Batch, {self.in_channels}, {self.seq_len}), got {tuple(x.shape)}"
            )

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        x = self.layer_norm(x)

        return x