"""
Physiologically Safe CTG Signal Augmentation Module
=====================================================
Provides data augmentation for CTG waveforms that preserves all clinically
meaningful morphological features (baseline FHR level, STV/LTV, deceleration
patterns, uterine contraction shapes).

IMPORTANT safety rules enforced by this module:
  ✓ Additive Gaussian noise (sensor-level)
  ✓ Slow baseline drift (physiological drift within ±5 bpm)
  ✓ Amplitude scaling (±5% max — stay within physiological range)
  ✓ Temporal jitter (minimal global shift ±5 samples = ±1.25 s)
  ✓ Missing sensor simulation (contiguous zero blocks <3% of signal)
  ✗ Time reversal (non-physiological)
  ✗ Frequency domain warping (distorts deceleration timing)
  ✗ Channel mixing (FHR ≠ UC — must stay independent)
  ✗ Aggressive time warping (distorts pattern timing)

Usage:
    augmentor = PhysiologicalAugmentor(p=0.5)
    X_aug = augmentor(X_batch)  # (B, 2, 4800) → (B, 2, 4800)

    # Or use as a collate_fn wrapper:
    loader = DataLoader(dataset, collate_fn=augmentor.collate_fn)
"""

import random
from typing import Optional

import torch
import torch.nn as nn


class PhysiologicalAugmentor(nn.Module):
    """
    On-the-fly physiologically safe CTG signal augmentation.

    Each augmentation is applied independently with probability `p`.
    All operations preserve the (B, 2, 4800) tensor shape and operate
    on normalized (Z-scored) signals — so magnitudes are in std units.

    Args:
        p:                  Global probability of applying any augmentation (default 0.5).
        noise_std:          Gaussian noise std in normalized units (default 0.02).
        drift_max_amp:      Max baseline drift amplitude in normalized units (default 0.05).
        scale_range:        (min, max) multiplicative scaling factor (default (0.97, 1.03)).
        jitter_max:         Max temporal shift in samples (default 5 samples = 1.25 s).
        missing_max_ratio:  Maximum fraction of signal to zero out (default 0.03 = 3%).
        missing_block_max:  Maximum contiguous block size for missing sim (default 48 samples = 12 s).
        channel_independent: Apply augmentations independently to each channel (default True).
        seed:               Optional random seed for reproducibility.
    """

    def __init__(
        self,
        p: float = 0.5,
        noise_std: float = 0.02,
        drift_max_amp: float = 0.05,
        scale_range: tuple = (0.97, 1.03),
        jitter_max: int = 5,
        missing_max_ratio: float = 0.03,
        missing_block_max: int = 48,
        channel_independent: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.p = p
        self.noise_std = noise_std
        self.drift_max_amp = drift_max_amp
        self.scale_range = scale_range
        self.jitter_max = jitter_max
        self.missing_max_ratio = missing_max_ratio
        self.missing_block_max = missing_block_max
        self.channel_independent = channel_independent
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to a batch of CTG signals.

        Args:
            X: CTG signal batch of shape (B, 2, 4800).

        Returns:
            Augmented tensor of same shape (B, 2, 4800).
        """
        if not self.training or self.p == 0.0:
            return X

        X = X.clone()
        B, C, L = X.shape

        for b in range(B):
            if random.random() > self.p:
                continue
            for c in range(C):
                # Each augmentation applied independently with p=0.5
                if random.random() < 0.5:
                    X[b, c] = self._add_gaussian_noise(X[b, c])
                if random.random() < 0.5:
                    X[b, c] = self._add_baseline_drift(X[b, c])
                if random.random() < 0.3:
                    X[b, c] = self._scale_amplitude(X[b, c])
                if random.random() < 0.3:
                    X[b, c] = self._temporal_jitter(X[b, c])
                if random.random() < 0.2:
                    X[b, c] = self._simulate_missing(X[b, c])

        return X

    def _add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add zero-mean Gaussian noise (sensor noise simulation)."""
        return x + torch.randn_like(x) * self.noise_std

    def _add_baseline_drift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add a slow sinusoidal drift over the full window.
        Drift period is random (80–400 s), amplitude is bounded by drift_max_amp.
        """
        L = x.shape[-1]
        period_samples = random.randint(80 * 4, 400 * 4)  # 80–400 s at 4 Hz
        phase = random.uniform(0, 2 * 3.14159)
        amp = random.uniform(0.01, self.drift_max_amp)
        t = torch.arange(L, dtype=x.dtype, device=x.device)
        drift = amp * torch.sin(2 * 3.14159 * t / period_samples + phase)
        return x + drift

    def _scale_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply by a small random scale factor."""
        scale = random.uniform(*self.scale_range)
        return x * scale

    def _temporal_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shift the signal by a random number of samples (circular shift).
        Maximum shift: jitter_max samples (default 5 samples = 1.25 s at 4 Hz).
        """
        shift = random.randint(-self.jitter_max, self.jitter_max)
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-1)

    def _simulate_missing(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simulate missing sensor data by zeroing out a small contiguous block.
        Block length is uniformly drawn up to missing_block_max.
        Position is random, ensuring total missing < missing_max_ratio.
        """
        L = x.shape[-1]
        max_block = min(self.missing_block_max, int(L * self.missing_max_ratio))
        if max_block <= 0:
            return x
        block_len = random.randint(1, max_block)
        start = random.randint(0, L - block_len)
        x = x.clone()
        x[start: start + block_len] = 0.0
        return x

    def collate_fn(self, batch):
        """
        Drop-in replacement collate_fn for DataLoader.
        Applies augmentation to the X tensor in each batch item.

        Works with MultiTaskCTGDataset which returns (X, y_primary, y_figo, y_features).
        """
        X = torch.stack([item[0] for item in batch])
        rest = [
            torch.stack([item[i] for item in batch])
            for i in range(1, len(batch[0]))
        ]
        X_aug = self.forward(X)
        return (X_aug, *rest)
