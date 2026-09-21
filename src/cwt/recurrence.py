"""
PyTorch-based GPU Recurrence Plot (RP) generator for FHR time series.

Converts 1D FHR sequences into 2D phase-space distance / recurrence representations
via time-delay embedding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrencePlot(nn.Module):
    """
    Computes 2D Recurrence Plot / Phase Space Distance Matrix on GPU.

    Args:
        dimension (int): Embedding dimension m (default: 3).
        time_delay (int): Embedding delay tau in samples (default: 4 -> 1 second at 4 Hz).
        target_size (int): Output 2D matrix resolution (default: 128 -> 128x128).
    """
    def __init__(self, dimension: int = 3, time_delay: int = 4, target_size: int = 128):
        super().__init__()
        self.dimension = dimension
        self.time_delay = time_delay
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, 1, Time) e.g. (B, 1, 4800)
        Returns:
            rp: Tensor of shape (Batch, 1, target_size, target_size)
        """
        B, C, N = x.shape
        x_flat = x.squeeze(1) # (B, N)

        # Downsample the raw signal first to make phase-space embedding compact
        # E.g., downsample from 4800 to target_size * 2 or similar
        subsampled_len = self.target_size + (self.dimension - 1) * self.time_delay
        x_sub = F.interpolate(x_flat.unsqueeze(1), size=subsampled_len, mode='linear', align_corners=False).squeeze(1)

        # Build embedded phase space vectors: shape (B, target_size, dimension)
        embedded_list = []
        for d in range(self.dimension):
            start = d * self.time_delay
            end = start + self.target_size
            embedded_list.append(x_sub[:, start:end].unsqueeze(-1))
        
        # Phase space trajectory: (B, target_size, dimension)
        traj = torch.cat(embedded_list, dim=-1)

        # Pairwise Euclidean distance matrix: ||y_i - y_j||_2
        # (B, target_size, 1, dim) - (B, 1, target_size, dim) -> (B, target_size, target_size)
        diff = traj.unsqueeze(2) - traj.unsqueeze(1)
        dist = torch.norm(diff, p=2, dim=-1) # (B, target_size, target_size)

        # Normalize distance matrix per window to [0, 1]
        max_dist = dist.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        rp = 1.0 - (dist / max_dist) # Invert so higher value = closer recurrence
        rp = rp.unsqueeze(1) # (B, 1, target_size, target_size)
        return rp
