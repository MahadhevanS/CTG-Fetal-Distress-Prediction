"""
Balanced Batch Sampling & Hard Example Mining for CTG Multi-Task Training
=========================================================================
Provides two complementary sampling strategies that work alongside
(not replacing) the dynamic pos_weight in BCEWithLogitsLoss:

  1. BalancedBatchSampler:
     Constructs each mini-batch with an equal number of positive (distress)
     and negative (normal) samples. Eliminates majority-class dominance within
     each batch gradient, producing more stable and informative updates.

  2. HardExampleMiner:
     After a configurable warm-up period, tracks per-sample loss history and
     up-weights windows that are consistently hard to classify (high loss).
     Works with PyTorch's WeightedRandomSampler to bias batch construction.

Usage:
    # Option A: BalancedBatchSampler (simple, no warm-up needed)
    from src.training.samplers import BalancedBatchSampler
    sampler = BalancedBatchSampler(dataset.y_primary, batch_size=64)
    train_loader = DataLoader(dataset, batch_sampler=sampler)

    # Option B: HardExampleMiner (activates after warm-up)
    from src.training.samplers import HardExampleMiner
    from torch.utils.data import WeightedRandomSampler
    miner = HardExampleMiner(n_samples=len(train_idx), warmup_epochs=5)
    # In training loop — after each forward pass:
    miner.update_losses(sample_indices, per_sample_losses)
    # Build sampler for next epoch:
    weights = miner.get_sample_weights()
    sampler = WeightedRandomSampler(weights, num_samples=len(train_idx), replacement=True)
"""

import math
import random
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Sampler


# =============================================================================
# 1. Balanced Batch Sampler
# =============================================================================

class BalancedBatchSampler(Sampler):
    """
    Yields balanced mini-batches with equal positive and negative samples.

    Each batch contains:
        batch_size // 2  positive (distress) samples
        batch_size // 2  negative (normal) samples

    Positive and negative indices are shuffled independently each epoch and
    cycled through until all samples have appeared (approximately once).

    Args:
        y_primary:  Binary labels tensor or array of shape (N,).
        batch_size: Total mini-batch size (must be even). Default: 64.
        drop_last:  Drop the final incomplete batch. Default: True.
    """

    def __init__(
        self,
        y_primary: torch.Tensor,
        batch_size: int = 64,
        drop_last: bool = True,
    ):
        super().__init__()
        if batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even for balanced sampling, got {batch_size}")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.half = batch_size // 2

        y_np = np.array(y_primary.cpu().numpy() if isinstance(y_primary, torch.Tensor) else y_primary)
        self.pos_indices = np.where(y_np == 1)[0].tolist()
        self.neg_indices = np.where(y_np == 0)[0].tolist()

        if len(self.pos_indices) == 0:
            raise ValueError("No positive samples found in dataset for BalancedBatchSampler.")
        if len(self.neg_indices) == 0:
            raise ValueError("No negative samples found in dataset for BalancedBatchSampler.")

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle both pools independently
        pos_pool = self.pos_indices.copy()
        neg_pool = self.neg_indices.copy()
        random.shuffle(pos_pool)
        random.shuffle(neg_pool)

        # Cycle shorter pool to match the longer one
        n_batches = max(len(pos_pool), len(neg_pool)) // self.half
        pos_cycle = (pos_pool * math.ceil(n_batches * self.half / max(len(pos_pool), 1)))[:n_batches * self.half]
        neg_cycle = (neg_pool * math.ceil(n_batches * self.half / max(len(neg_pool), 1)))[:n_batches * self.half]

        for i in range(n_batches):
            batch = (
                pos_cycle[i * self.half: (i + 1) * self.half]
                + neg_cycle[i * self.half: (i + 1) * self.half]
            )
            random.shuffle(batch)  # Mix positives and negatives within batch
            yield batch

    def __len__(self) -> int:
        return max(len(self.pos_indices), len(self.neg_indices)) // self.half


# =============================================================================
# 2. Hard Example Miner
# =============================================================================

class HardExampleMiner:
    """
    Tracks per-sample loss history and generates sample weights for up-sampling
    consistently hard-to-classify windows.

    After `warmup_epochs` the miner begins returning skewed weights:
        weight_i ∝ (1 - hard_ratio) * uniform + hard_ratio * loss_normalized_i

    where `hard_ratio` controls how aggressively hard examples are prioritized.

    Args:
        n_samples:     Total number of training windows per fold.
        warmup_epochs: Epochs before mining activates. Default: 5.
        ema_decay:     Decay for per-sample loss EMA (default 0.9).
        hard_ratio:    Fraction of weight allocated to hard examples (default 0.6).
                       At 0.6: 60% of batch weight from hard samples, 40% uniform.
    """

    def __init__(
        self,
        n_samples: int,
        warmup_epochs: int = 5,
        ema_decay: float = 0.9,
        hard_ratio: float = 0.6,
    ):
        self.n_samples = n_samples
        self.warmup_epochs = warmup_epochs
        self.ema_decay = ema_decay
        self.hard_ratio = hard_ratio
        self._epoch = 0
        # EMA loss per sample — initialized to 1.0 (uniform)
        self._loss_ema = np.ones(n_samples, dtype=np.float32)
        self._update_count = np.zeros(n_samples, dtype=np.int32)

    def update_losses(
        self,
        indices: List[int],
        losses: List[float],
    ) -> None:
        """
        Update EMA loss for each sample seen in this batch.

        Args:
            indices: Sample indices in the training subset (0-based).
            losses:  Per-sample scalar losses (same order as indices).
        """
        for idx, loss in zip(indices, losses):
            if 0 <= idx < self.n_samples:
                self._loss_ema[idx] = (
                    self.ema_decay * self._loss_ema[idx]
                    + (1.0 - self.ema_decay) * loss
                )
                self._update_count[idx] += 1

    def step_epoch(self) -> None:
        """Call once per epoch to advance the miner's epoch counter."""
        self._epoch += 1

    def get_sample_weights(self) -> np.ndarray:
        """
        Returns sampling weight per sample.

        Before warmup: uniform weights (all 1.0).
        After warmup:  blended uniform + loss-proportional weights.

        Returns:
            np.ndarray of shape (n_samples,), positive floats.
            Pass directly to torch.utils.data.WeightedRandomSampler.
        """
        if self._epoch < self.warmup_epochs:
            return np.ones(self.n_samples, dtype=np.float32)

        # Normalize loss EMA to [0, 1]
        loss_range = self._loss_ema.max() - self._loss_ema.min()
        if loss_range < 1e-8:
            normalized = np.ones(self.n_samples, dtype=np.float32)
        else:
            normalized = (self._loss_ema - self._loss_ema.min()) / loss_range

        # Blend: hard_ratio * loss-based + (1 - hard_ratio) * uniform
        weights = (
            self.hard_ratio * normalized
            + (1.0 - self.hard_ratio) * np.ones(self.n_samples, dtype=np.float32)
        )
        # Ensure all weights are positive
        weights = np.clip(weights, 1e-6, None)
        return weights

    def reset(self) -> None:
        """Reset miner state for a new fold."""
        self._epoch = 0
        self._loss_ema = np.ones(self.n_samples, dtype=np.float32)
        self._update_count = np.zeros(self.n_samples, dtype=np.int32)

    @property
    def is_active(self) -> bool:
        """True if mining has started (past warmup period)."""
        return self._epoch >= self.warmup_epochs
