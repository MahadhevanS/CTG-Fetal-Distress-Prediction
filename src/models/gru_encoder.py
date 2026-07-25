import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUEncoder(nn.Module):
    """
    Multi-layer Gated Recurrent Unit (GRU) Temporal Encoder for CTG signals.
    Conforms to universal signature:
      Input:  (Batch, 2, 4800) [Channel 0: FHR, Channel 1: UC]
      Output: (Batch, 128)     [1D latent representation]
    
    Patent Compliance (GE US12094611B2):
    Maps continuous multi-channel signals directly to latent space without
    graphical bounding box detection or cross-temporal shape-matching loops.
    """
    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 4800,
        hidden_dim: int = 128,
        gru_hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(GRUEncoder, self).__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 1D Conv Stem for continuous temporal feature extraction and downsampling
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, gru_hidden, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(gru_hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(gru_hidden, gru_hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(gru_hidden),
            nn.ReLU(inplace=True)
        )

        # Multi-layer GRU
        self.gru = nn.GRU(
            input_size=gru_hidden,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        num_directions = 2 if bidirectional else 1
        gru_out_dim = gru_hidden * num_directions

        # Latent projection head (combines global mean + max pooling)
        self.fc_proj = nn.Sequential(
            nn.Linear(gru_out_dim * 2, hidden_dim),
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

        # 1. Temporal stem downsampling: (Batch, 2, 4800) -> (Batch, gru_hidden, 1200)
        x_stem = self.stem(x)

        # 2. Reshape for GRU: (Batch, gru_hidden, 1200) -> (Batch, 1200, gru_hidden)
        x_seq = x_stem.transpose(1, 2)

        # 3. GRU forward pass: (Batch, 1200, gru_out_dim)
        gru_out, _ = self.gru(x_seq)

        # 4. Temporal pooling (Global Mean + Global Max pooling across sequence length)
        mean_pool = torch.mean(gru_out, dim=1)
        max_pool, _ = torch.max(gru_out, dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (Batch, gru_out_dim * 2)

        # 5. Latent projection to (Batch, 128)
        latent = self.fc_proj(pooled)
        return latent
