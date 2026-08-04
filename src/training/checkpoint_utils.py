"""
EMA & SWA Checkpoint Utilities for Model 8 Training
=====================================================
Provides:
  - ExponentialMovingAverage (EMA): shadows model weights during training.
    Validation always uses the EMA snapshot, giving more stable metrics.
  - StochasticWeightAveraging (SWA): averages the last N best checkpoints
    at the end of training to improve generalization without retraining.

Usage:
    # EMA — attach at fold start, update after each optimizer step
    ema = ExponentialMovingAverage(model, decay=0.999)
    # ... training loop ...
    optimizer.step()
    ema.update()            # update shadow weights
    # ... validation ...
    with ema.average_parameters():
        metrics = evaluate(model, val_loader)

    # SWA — collect checkpoint paths during training, average at fold end
    swa = StochasticWeightAveraging()
    swa.add_checkpoint(model.state_dict())
    averaged_state = swa.average()
    model.load_state_dict(averaged_state)

References:
  - EMA: Standard practice in vision transformers (e.g., DINO, EfficientNet)
  - SWA: Izmailov et al. (2018) "Averaging Weights Leads to Wider Optima"
"""

import copy
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# =============================================================================
# Exponential Moving Average (EMA)
# =============================================================================

class ExponentialMovingAverage:
    """
    Maintains an exponential moving average of model parameters.

    After each optimizer step, call ema.update() to update the shadow copy:
        shadow_p = decay * shadow_p + (1 - decay) * p

    During validation, use the context manager `with ema.average_parameters():`
    to temporarily swap in the EMA weights, then restore the training weights.

    Args:
        model:  The model whose parameters to track.
        decay:  EMA decay rate (0.999 is standard for epoch-level EMA;
                0.9 is faster for step-level). Default: 0.9995.
        device: Device for shadow parameters (matches model device by default).
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9995,
        device: Optional[torch.device] = None,
    ):
        self.decay = decay
        self.model = model
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}

        # Initialize shadow as a copy of current parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(
                    device if device else param.device
                )

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow parameters after each optimizer step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def _apply_shadow(self) -> None:
        """Swap training weights with EMA shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def _restore(self) -> None:
        """Restore training weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    @contextmanager
    def average_parameters(self):
        """
        Context manager: temporarily apply EMA weights for validation.

        Usage:
            with ema.average_parameters():
                val_loss = evaluate(model, val_loader)
        """
        self._apply_shadow()
        try:
            yield
        finally:
            self._restore()

    def state_dict(self) -> Dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]


# =============================================================================
# Stochastic Weight Averaging (SWA)
# =============================================================================

class StochasticWeightAveraging:
    """
    Stochastic Weight Averaging (SWA) — Izmailov et al. (2018).

    Collects model state dicts during training (e.g., last N best-AUROC epochs)
    and computes their arithmetic mean to obtain a flatter, better-generalizing
    model at zero extra training cost.

    Usage:
        swa = StochasticWeightAveraging(max_checkpoints=10)
        # Inside training loop at each epoch (or when AUROC improves):
        swa.add_checkpoint(model.state_dict())
        # At end of fold:
        averaged_state = swa.average()
        model.load_state_dict(averaged_state)
        # Evaluate averaged model on validation set

    Args:
        max_checkpoints: Maximum number of checkpoints to retain (FIFO).
                         Set to 0 for unlimited. Default: 10.
    """

    def __init__(self, max_checkpoints: int = 10):
        self.max_checkpoints = max_checkpoints
        self._checkpoints: List[Dict[str, torch.Tensor]] = []

    def add_checkpoint(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Add a model state_dict snapshot."""
        # Deep-copy to avoid aliasing with the live model
        self._checkpoints.append(
            {k: v.clone().cpu() for k, v in state_dict.items()}
        )
        if self.max_checkpoints > 0 and len(self._checkpoints) > self.max_checkpoints:
            self._checkpoints.pop(0)  # FIFO eviction

    def average(self) -> Dict[str, torch.Tensor]:
        """
        Compute element-wise arithmetic mean of all collected checkpoints.

        Returns:
            Averaged state dict. Load into model with model.load_state_dict(avg_state).

        Raises:
            ValueError if no checkpoints have been added.
        """
        if not self._checkpoints:
            raise ValueError("No checkpoints collected. Call add_checkpoint() first.")

        avg_state: Dict[str, torch.Tensor] = {}
        for key in self._checkpoints[0]:
            tensors = [ckpt[key].float() for ckpt in self._checkpoints]
            avg_state[key] = torch.stack(tensors).mean(dim=0)

        return avg_state

    def reset(self) -> None:
        """Clear all stored checkpoints (call at fold start)."""
        self._checkpoints.clear()

    @property
    def n_checkpoints(self) -> int:
        return len(self._checkpoints)
