"""
Masked-reconstruction corruption for self-supervised CTG-CrossFormer
pretraining (see docs/model8_crossformer_run_history.md, 2026-08-17 entry).

Unlike PhysiologicalAugmentor (light regularization noise, no mask returned,
~1% missing-block cap -- too weak for a reconstruction pretext task), this
masks ~50% of each channel in multi-second blocks and returns the mask so a
decoder's reconstruction loss can be computed only at masked positions.
"""

import random
from typing import Optional, Tuple

import torch

from src.training.augmentation import PhysiologicalAugmentor


def _build_block_mask(
    n_samples: int,
    mask_ratio: float,
    block_size_range: Tuple[int, int],
) -> torch.Tensor:
    """Boolean (n_samples,) mask, True=masked, built from randomly placed,
    randomly sized, non-overlapping blocks until ~mask_ratio is covered."""
    mask = torch.zeros(n_samples, dtype=torch.bool)
    target_count = int(n_samples * mask_ratio)
    masked_count = 0
    # Cap attempts so a bad draw sequence can't spin forever near the target.
    for _ in range(n_samples):
        if masked_count >= target_count:
            break
        block_len = random.randint(*block_size_range)
        block_len = min(block_len, n_samples)
        start = random.randint(0, n_samples - block_len)
        block = slice(start, start + block_len)
        newly_masked = (~mask[block]).sum().item()
        mask[block] = True
        masked_count += newly_masked
    return mask


def mask_ctg_signal(
    X_clean: torch.Tensor,
    mask_ratio: float = 0.5,
    block_size_range: Tuple[int, int] = (16, 160),
    channel_independent: bool = True,
    mask_value: float = 0.0,
    light_augment: bool = True,
    augmentor: Optional[PhysiologicalAugmentor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        X_clean: (B, 2, 4800) clean, Z-normalized signal.
        mask_ratio: target fraction of each channel to mask (default 0.5).
        block_size_range: (min, max) block length in samples (default
            16-160 = 4-40s at 4Hz, ~0.5-5 encoder tokens per block).
        channel_independent: mask FHR and UC at different positions --
            forces reconstruction to lean on cross-channel context
            (BidirectionalCrossAttentionModule) rather than pure local
            interpolation within one channel.
        mask_value: value written at masked positions (default 0.0, matching
            the missing-data convention used throughout the preprocessing
            pipeline).
        light_augment: additionally corrupt the whole signal (masked input
            only, not the reconstruction target) with PhysiologicalAugmentor
            at reduced strength -- prevents the trivial "position is masked
            iff value==mask_value" shortcut and matches the noise/scale
            perturbation the encoder already sees during supervised
            fine-tuning.
        augmentor: reuse an existing PhysiologicalAugmentor instance
            (e.g. to avoid reseeding python's `random` module repeatedly);
            constructs a default p=0.3 instance if not provided.

    Returns:
        (X_masked, mask): X_masked (B,2,4800) is the corrupted input to feed
        the encoder; mask (B,2,4800) bool, True=masked. Reconstruction loss
        should always be computed against the caller's original X_clean at
        mask==True positions -- light_augment corrupts what the model SEES,
        never the ground-truth target.
    """
    B, C, L = X_clean.shape
    assert C == 2, f"Expected 2 channels (FHR, UC), got {C}"

    X_masked = X_clean.clone()
    if light_augment:
        if augmentor is None:
            augmentor = PhysiologicalAugmentor(p=0.3)
        augmentor.train()
        X_masked = augmentor(X_masked)

    mask = torch.zeros(B, C, L, dtype=torch.bool)
    for b in range(B):
        if channel_independent:
            for c in range(C):
                mask[b, c] = _build_block_mask(L, mask_ratio, block_size_range)
        else:
            shared = _build_block_mask(L, mask_ratio, block_size_range)
            mask[b, 0] = shared
            mask[b, 1] = shared

    mask = mask.to(X_clean.device)
    X_masked = X_masked.masked_fill(mask, mask_value)
    return X_masked, mask
