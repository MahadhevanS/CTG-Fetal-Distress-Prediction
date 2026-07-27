import torch
import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM (BiLSTM) Temporal Encoder for CTG signals.

    Conforms to universal signature:
      Input:  (Batch, 2, 4800) [Channel 0: FHR diff, Channel 1: UC]
      Output: (Batch, 128)     [1D latent representation]

    Architecture:
      - 2-layer Bidirectional LSTM over 4800 sequence steps (hidden_size=64, concat=128)
      - Concatenation of final forward & backward hidden states (shape: Batch, 128)
      - Projection & Normalization: Linear(128 -> 128) + LayerNorm(128)

    Parameter Count: ~159,232 parameters.

    Patent Compliance (GE US12094611B2):
      Encodes raw 2D continuous sequence tensors directly to R^128 without explicit
      graphical pattern extraction, bounding box proposals, or post-hoc shape matching loops.
    """

    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 4800,
        latent_dim: int = 128,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        input_size: int = None,
    ):
        super().__init__()
        self.in_channels = input_size if input_size is not None else in_channels
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.lstm = nn.LSTM(
            input_size=self.in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.fc = nn.Linear(hidden_size * 2, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input shape (Batch, {self.in_channels}, {self.seq_len}), got {tuple(x.shape)}"
            )

        # (Batch, 2, 4800) -> (Batch, 4800, 2)
        x = x.permute(0, 2, 1)

        _, (hidden, _) = self.lstm(x)

        # Forward hidden state [-2], Backward hidden state [-1]
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        # Concatenate: (Batch, 128)
        x = torch.cat((forward_hidden, backward_hidden), dim=1)

        # Project & Normalize
        x = self.fc(x)
        x = self.layer_norm(x)

        return x