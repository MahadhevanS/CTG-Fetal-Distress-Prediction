import torch
import torch.nn as nn

class UniversalClassifier(nn.Module):
    """
    Standardized classification model wrapping a temporal encoder
    with a 2-layer MLP classification head for binary distress prediction.
    """
    def __init__(self, encoder: nn.Module, latent_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.2):
        super(UniversalClassifier, self).__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw signal tensor (Batch, 2, 4800)
        Returns:
            logits: Tensor of shape (Batch, 1) - unnormalized log odds
        """
        latent = self.encoder(x)  # (Batch, 128)
        logits = self.head(latent) # (Batch, 1)
        return logits
