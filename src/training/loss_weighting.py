"""
Dynamic Multi-Task Loss Weighting Module
=========================================
Provides four strategies for balancing the composite multi-task loss in Model 8:

    L_total = L_distress + λ₁·L_FIGO + λ₂·L_features + λ₃·L_knowledge

Strategies:
  1. FixedWeighting           — current default (static λ values from config)
  2. UncertaintyWeighting     — Kendall et al. (2018): learns log-variance per task
  3. DynamicWeightAveraging   — Liu et al. (2019): weights ∝ relative loss-rate change
  4. GradNorm                 — Chen et al. (2018): gradient-norm balancing

Usage:
    from src.training.loss_weighting import build_loss_weighter

    weighter = build_loss_weighter(
        method="uncertainty",   # or "dwa", "gradnorm", "fixed"
        n_tasks=4,
        device=device,
    )

    # Per-step usage:
    loss_components = {"distress": l_d, "figo": l_f, "features": l_ft, "knowledge": l_k}
    total_loss, weights = weighter.weight(loss_components, model=model)  # model only for gradnorm

    # For optimizers that need weighter parameters (UncertaintyWeighting):
    optimizer.add_param_group({"params": weighter.parameters()})

References:
  - Kendall et al. (2018) "Multi-Task Learning Using Uncertainty" CVPR
  - Liu et al. (2019) "End-to-End Multi-Task Learning with Attention" CVPR
  - Chen et al. (2018) "GradNorm: Gradient Normalization for Adaptive Loss Balancing" ICML
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# Task ordering: matches ABLATION_VARIANTS task inclusion order
TASK_NAMES = ["distress", "figo", "features", "knowledge"]


# =============================================================================
# Base Interface
# =============================================================================

class BaseLossWeighter(nn.Module):
    """Abstract base for all loss weighting strategies."""

    def weight(
        self,
        loss_components: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Combines loss components into a single scalar total loss.

        Args:
            loss_components: Dict mapping task name → scalar loss tensor.
                             Only present tasks are included (ablation controls this).
            model:           Model reference (only used by GradNorm).
            epoch:           Current epoch number (used by DWA).

        Returns:
            (total_loss_tensor, weights_dict)
        """
        raise NotImplementedError


# =============================================================================
# 1. Fixed Weighting (Baseline)
# =============================================================================

class FixedWeighting(BaseLossWeighter):
    """
    Static λ-based weighting. Replicates the original fixed-weight behavior.
    Weights are applied as L = L_distress + λ_figo·L_figo + λ_feat·L_feat + λ_know·L_know.
    """

    def __init__(
        self,
        lambda_figo: float = 0.3,
        lambda_features: float = 0.2,
        lambda_knowledge: float = 0.1,
    ):
        super().__init__()
        self._weights = {
            "distress": 1.0,
            "figo": lambda_figo,
            "features": lambda_features,
            "knowledge": lambda_knowledge,
        }

    def weight(
        self,
        loss_components: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = sum(
            self._weights.get(name, 1.0) * loss
            for name, loss in loss_components.items()
        )
        weights_out = {name: self._weights.get(name, 1.0) for name in loss_components}
        return total, weights_out


# =============================================================================
# 2. Uncertainty Weighting (Kendall et al., 2018)
# =============================================================================

class UncertaintyWeighting(BaseLossWeighter):
    """
    Homoscedastic uncertainty-based weighting (Kendall et al., 2018).

    Each task t gets a learnable log-variance parameter σ_t. The total loss is:
        L = Σ_t  [1/(2·exp(log_σ²_t)) · L_t  +  0.5·log_σ²_t]

    The σ² parameters are optimized jointly with the model weights via the optimizer.
    To use, add weighter.parameters() to the optimizer:
        optimizer = AdamW([
            {"params": model.parameters()},
            {"params": weighter.parameters(), "lr": 1e-3}
        ])

    Args:
        task_names: List of task names that will be present (drives param init).
        init_log_sigma2: Initial value for log(σ²) per task (default 0.0 → σ²=1).
    """

    def __init__(
        self,
        task_names: List[str] = None,
        init_log_sigma2: float = 0.0,
    ):
        super().__init__()
        if task_names is None:
            task_names = TASK_NAMES
        self.task_names = task_names
        # Learnable log(σ²) per task; initialized to 0 → σ²=1 → unit weight
        self.log_sigma2 = nn.ParameterDict({
            name.replace(".", "_"): nn.Parameter(torch.tensor(init_log_sigma2))
            for name in task_names
        })

    def weight(
        self,
        loss_components: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = torch.tensor(0.0, device=next(iter(loss_components.values())).device)
        weights_out: Dict[str, float] = {}

        for name, loss in loss_components.items():
            key = name.replace(".", "_")
            if key in self.log_sigma2:
                log_s2 = self.log_sigma2[key]
                # L_weighted = (1/2σ²)·L + (1/2)·log(σ²)
                inv_sigma2 = torch.exp(-log_s2)
                total = total + 0.5 * inv_sigma2 * loss + 0.5 * log_s2
                weights_out[name] = float(0.5 * inv_sigma2.detach().cpu())
            else:
                # Unknown task → unit weight
                total = total + loss
                weights_out[name] = 1.0

        return total, weights_out


# =============================================================================
# 3. Dynamic Weight Averaging (Liu et al., 2019)
# =============================================================================

class DynamicWeightAveraging(BaseLossWeighter):
    """
    Dynamic Weight Averaging (DWA) from Liu et al. (2019).

    At each epoch, weights are proportional to the relative rate of loss change:
        r_t(epoch) = L_t(epoch-1) / L_t(epoch-2)
        w_t(epoch) = exp(r_t / T)   (T is softmax temperature, default 2.0)
        λ_t = K · softmax(w_t)      (K = number of tasks)

    Rationale: Tasks whose loss decreased slowly (high r_t) receive more weight.

    Args:
        n_tasks: Number of tasks (for scaling final weights).
        temperature: Softmax temperature for weight sharpness (default 2.0).
        history_len: Number of past epochs to keep for loss history (default 2).
    """

    def __init__(
        self,
        task_names: List[str] = None,
        temperature: float = 2.0,
    ):
        super().__init__()
        if task_names is None:
            task_names = TASK_NAMES
        self.task_names = task_names
        self.T = temperature
        # Stores the running mean loss per task for each epoch (list of epoch-dicts)
        self._loss_history: List[Dict[str, float]] = []
        self._current_weights: Dict[str, float] = {t: 1.0 for t in task_names}

    def update_history(self, epoch_losses: Dict[str, float]) -> None:
        """Call once per epoch with mean training losses to update DWA weights."""
        self._loss_history.append(epoch_losses)
        if len(self._loss_history) < 2:
            # Not enough history — keep uniform weights
            return

        prev = self._loss_history[-2]
        curr = self._loss_history[-1]
        n_tasks = len(epoch_losses)

        rates = {}
        for t in epoch_losses:
            if t in prev and prev[t] > 1e-8:
                rates[t] = curr[t] / prev[t]
            else:
                rates[t] = 1.0

        # Softmax over rates
        rate_tensor = torch.tensor([rates.get(t, 1.0) for t in self.task_names], dtype=torch.float32)
        weights = torch.softmax(rate_tensor / self.T, dim=0) * n_tasks
        self._current_weights = {
            t: float(weights[i]) for i, t in enumerate(self.task_names)
        }

    def weight(
        self,
        loss_components: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = sum(
            self._current_weights.get(name, 1.0) * loss
            for name, loss in loss_components.items()
        )
        weights_out = {name: self._current_weights.get(name, 1.0) for name in loss_components}
        return total, weights_out


# =============================================================================
# 4. GradNorm (Chen et al., 2018)
# =============================================================================

class GradNorm(BaseLossWeighter):
    """
    GradNorm (Chen et al., 2018): gradient-norm-based adaptive loss balancing.

    Maintains per-task loss weights as learnable parameters. At each step,
    computes gradient norms for each task and adjusts weights so that all tasks
    have similar gradient magnitudes relative to their initial loss scale.

    IMPORTANT: GradNorm requires access to the shared encoder's last layer
    parameters to compute per-task gradient norms. Pass `model` and
    `shared_layer_name` at init.

    Args:
        task_names:         Task name list.
        alpha:              Restoring force strength (0 = no restoring, default 1.5).
        shared_layer_name:  Attribute name of the shared encoder layer for grad-norm
                            computation (default: "encoder" — the PatchTST backbone).
    """

    def __init__(
        self,
        task_names: List[str] = None,
        alpha: float = 1.5,
        shared_layer_name: str = "encoder",
    ):
        super().__init__()
        if task_names is None:
            task_names = TASK_NAMES
        self.task_names = task_names
        self.alpha = alpha
        self.shared_layer_name = shared_layer_name

        # Learnable log-scale per-task weights (log to keep positive)
        self.log_weights = nn.ParameterDict({
            name.replace(".", "_"): nn.Parameter(torch.zeros(1))
            for name in task_names
        })
        self._initial_losses: Dict[str, float] = {}
        self._initialized = False

    @property
    def weights(self) -> Dict[str, torch.Tensor]:
        return {name.replace("_", "."): torch.exp(self.log_weights[name.replace(".", "_")])
                for name in self.task_names}

    def weight(
        self,
        loss_components: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Initialize reference losses on first call
        if not self._initialized:
            for name, loss in loss_components.items():
                self._initial_losses[name] = float(loss.detach())
            self._initialized = True

        weights_out: Dict[str, float] = {}
        total = torch.tensor(0.0, device=next(iter(loss_components.values())).device)

        for name, loss in loss_components.items():
            key = name.replace(".", "_")
            w = torch.exp(self.log_weights.get(key, torch.zeros(1))).squeeze()
            total = total + w * loss
            weights_out[name] = float(w.detach())

        # GradNorm auxiliary loss is computed externally in the training loop
        # (requires per-task backward passes). Here we just return the weighted sum.
        # The training script should call compute_gradnorm_loss() separately.
        return total, weights_out

    def compute_gradnorm_loss(
        self,
        loss_components: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        """
        Computes the GradNorm regularization loss on per-task weight parameters.
        Call this separately and backprop only through log_weights.

        Returns the GradNorm auxiliary scalar loss.
        """
        # Get shared layer parameters
        shared_module = getattr(model, self.shared_layer_name, None)
        if shared_module is None:
            return torch.tensor(0.0)

        # Compute gradient norms per task w.r.t. shared encoder's last param group
        try:
            shared_params = list(shared_module.parameters())[-1]
        except (StopIteration, IndexError):
            return torch.tensor(0.0)

        grad_norms: Dict[str, torch.Tensor] = {}
        for name, loss in loss_components.items():
            key = name.replace(".", "_")
            w = torch.exp(self.log_weights.get(key, torch.zeros(1))).squeeze()
            weighted_loss = w * loss
            grad = torch.autograd.grad(
                weighted_loss, shared_params,
                retain_graph=True, create_graph=True, allow_unused=True
            )
            if grad[0] is not None:
                grad_norms[name] = torch.norm(grad[0])
            else:
                grad_norms[name] = torch.tensor(0.0, requires_grad=True)

        if not grad_norms:
            return torch.tensor(0.0)

        mean_norm = torch.stack(list(grad_norms.values())).mean().detach()

        # GradNorm loss: push all task norms toward mean_norm
        gradnorm_loss = torch.tensor(0.0, device=shared_params.device)
        for name, g_norm in grad_norms.items():
            L0 = self._initial_losses.get(name, 1.0)
            L_cur = float(loss_components[name].detach())
            relative_progress = L_cur / (L0 + 1e-8)
            target_norm = mean_norm * (relative_progress ** self.alpha)
            gradnorm_loss = gradnorm_loss + torch.abs(g_norm - target_norm)

        return gradnorm_loss


# =============================================================================
# Factory
# =============================================================================

def build_loss_weighter(
    method: str = "fixed",
    task_names: Optional[List[str]] = None,
    lambda_figo: float = 0.3,
    lambda_features: float = 0.2,
    lambda_knowledge: float = 0.1,
    **kwargs,
) -> BaseLossWeighter:
    """
    Factory function to build a loss weighter by method name.

    Args:
        method:           One of 'fixed', 'uncertainty', 'dwa', 'gradnorm'.
        task_names:       Task names (default: all 4 tasks).
        lambda_figo:      Used by FixedWeighting only.
        lambda_features:  Used by FixedWeighting only.
        lambda_knowledge: Used by FixedWeighting only.
        **kwargs:         Passed to the selected weighter constructor.

    Returns:
        BaseLossWeighter instance.
    """
    method = method.lower().strip()
    if task_names is None:
        task_names = TASK_NAMES

    if method in ("fixed", "static"):
        return FixedWeighting(
            lambda_figo=lambda_figo,
            lambda_features=lambda_features,
            lambda_knowledge=lambda_knowledge,
        )
    elif method in ("uncertainty", "uncertainty_weighting", "kendall"):
        return UncertaintyWeighting(task_names=task_names, **kwargs)
    elif method in ("dwa", "dynamic_weight_averaging"):
        return DynamicWeightAveraging(task_names=task_names, **kwargs)
    elif method in ("gradnorm", "grad_norm"):
        return GradNorm(task_names=task_names, **kwargs)
    else:
        raise ValueError(
            f"Unknown loss weighting method: '{method}'. "
            f"Choose from: 'fixed', 'uncertainty', 'dwa', 'gradnorm'."
        )
