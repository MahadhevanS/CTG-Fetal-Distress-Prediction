"""
Hyperparameter Tuning Runner for Model 8 (Knowledge-Infused Multi-Task Framework)
==================================================================================
tune_hyperparameters.py only tunes the 7 standalone single-task encoders -- it has
no awareness of Model 8's multi-task loop at all. Model 8's current hyperparameters
(LR, OneCycle schedule) were borrowed wholesale from the standalone CrossFormer
classifier's own tuning, never tuned for Model 8's own multi-task loss landscape
(distress + FIGO + features + knowledge, or + criteria). This script closes that
gap: randomized search directly over train_and_evaluate_model8(), reusing a base
model8_*.yaml config for everything NOT being searched (backbone architecture,
paths, sampling method, augmentation).

Two things intentionally NOT searched, both settled empirically this session
rather than arbitrarily excluded:
  - EMA is forced OFF for every trial. A full 50-epoch/5-fold run (2026-08-15)
    confirmed EMA hurts CrossFormer's `full` AUROC by ~0.043 (0.7893 -> 0.7461),
    consistently across folds -- not a short-diagnostic artifact. Re-discovering
    this per trial would waste search budget on a settled question.
  - head_hidden_dim/head_dropout are left at whatever the base config specifies.
    For CrossFormer these were deliberately matched to CTGCrossformerForClassification's
    own head capacity (128/0.3) for a specific, already-documented architectural
    reason (see configs/model8_crossformer_config.yaml) -- not a knob to
    blindly re-search alongside everything else.

PROXY METRIC CAVEAT (learned the hard way from the EMA diagnostic this session):
the default --epochs/--k-folds here are a cheap proxy (fewer epochs, fewer folds
than the real 50/5), not the final answer. The winning config from this search
should be re-validated with a real full-length run (--ablation <variant>, full
epochs/folds, no proxy) before being trusted -- exactly the mistake the EMA
diagnostic almost made (a 20-epoch/1-fold read that looked directionally right
but wasn't confirmed until the full-length re-run).

Usage:
    python src/training/tune_model8_hyperparameters.py \\
        --config configs/model8_crossformer_config.yaml \\
        --ablation full \\
        --n-trials 12 --epochs 15 --k-folds 3 \\
        --output-config configs/model8_tuned_hyperparameters.yaml
"""

import argparse
import os
import random
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.training.train_knowledge_infused import train_and_evaluate_model8, ABLATION_VARIANTS

# Search space for Model 8's multi-task training loop. Values centered on the
# current CrossFormer config's settings (lr=0.0003, pct_start=0.1, wd=1e-5,
# batch=32, lambda_figo=0.3/features=0.2/knowledge=0.1, loss_weighting=fixed)
# so the current config is itself one reachable sample, not excluded from the
# search.
SEARCH_SPACE: Dict[str, List[Any]] = {
    "lr": [0.0001, 0.0002, 0.0003, 0.0005, 0.0008],
    "onecycle_pct_start": [0.1, 0.2, 0.3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "batch_size": [16, 32, 64],
    "lambda_figo": [0.1, 0.2, 0.3, 0.5],
    "lambda_features": [0.1, 0.2, 0.3],
    "lambda_knowledge": [0.05, 0.1, 0.2],
    "loss_weighting_method": ["fixed", "uncertainty", "dwa", "gradnorm"],
}


def sample_hyperparameters(search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Randomly samples a parameter configuration from the search space dictionary."""
    return {name: random.choice(choices) for name, choices in search_space.items()}


def tune_model8(
    base_config_path: str,
    ablation: str,
    n_trials: int,
    epochs: int,
    k_folds: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    with open(base_config_path, "r") as f:
        base_cfg = yaml.safe_load(f) or {}

    backbone_cfg = base_cfg.get("backbone", {})
    heads_cfg = base_cfg.get("heads", {})
    path_cfg = base_cfg.get("paths", {})
    sampling_cfg = base_cfg.get("sampling", {})
    augmentation_cfg = base_cfg.get("augmentation", {})
    calibration_cfg = base_cfg.get("calibration", {})
    curriculum_cfg = base_cfg.get("curriculum", {})
    error_analysis_cfg = base_cfg.get("error_analysis", {})
    train_cfg = base_cfg.get("training", {})
    loss_cfg = base_cfg.get("loss", {})

    data_dir = path_cfg.get("data_dir", "data/processed/")
    scaler_path = path_cfg.get("feature_scaler_path", "data/processed/feature_scaler.npz")
    gradient_clip = train_cfg.get("gradient_clip", 1.0)
    lambda_consistency = loss_cfg.get("lambda_consistency", 0.5)
    lambda_criteria = loss_cfg.get("lambda_criteria", 0.3)

    print("\n" + "=" * 70)
    print(f" Model 8 Hyperparameter Tuning | Backbone config: {base_config_path}")
    print(f" Ablation: {ablation} | Trials: {n_trials} | Epochs/Trial: {epochs} | Folds/Trial: {k_folds}")
    print(" (Proxy metric -- re-validate the winner with a full-length run before trusting it)")
    print(" Search Space:")
    for k, v in SEARCH_SPACE.items():
        print(f"   - {k}: {v}")
    print(" EMA: forced OFF for every trial (confirmed harmful at full scale, 2026-08-15)")
    print("=" * 70 + "\n")

    best_score = -1.0
    best_params: Dict[str, Any] = {}
    best_metrics: Dict[str, float] = {}
    trial_log = []

    for trial_idx in range(1, n_trials + 1):
        params = sample_hyperparameters(SEARCH_SPACE)
        print(f"\n--- [Trial {trial_idx}/{n_trials}] Sampled Params: {params} ---")

        if dry_run:
            sim_auroc = random.uniform(0.65, 0.85)
            sim_sens90 = random.uniform(30.0, 55.0)
            score = 0.6 * sim_auroc + 0.4 * (sim_sens90 / 100.0)
            print(f" [DRY RUN] Simulated -> AUROC: {sim_auroc:.4f} | Sens@90%Spec: {sim_sens90:.2f}% | Score: {score:.4f}")
            trial_log.append({"trial": trial_idx, "params": params, "auroc": sim_auroc, "score": score})
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = {"auroc": sim_auroc, "sens_at_90spec": sim_sens90}
            continue

        results = train_and_evaluate_model8(
            data_dir=data_dir,
            scaler_path=scaler_path,
            save_dir="checkpoints/model8_tuning_tmp/",
            results_dir="checkpoints/model8_tuning_tmp/results/",
            ablation=ablation,
            epochs=epochs,
            batch_size=params["batch_size"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            k_folds=k_folds,
            gradient_clip=gradient_clip,
            lambda_figo=params["lambda_figo"],
            lambda_features=params["lambda_features"],
            lambda_knowledge=params["lambda_knowledge"],
            lambda_consistency=lambda_consistency,
            dry_run=False,
            lambda_criteria=lambda_criteria,
            backbone_cfg=backbone_cfg,
            heads_cfg=heads_cfg,
            loss_weighting_cfg={"method": params["loss_weighting_method"]},
            scheduler_cfg={
                "type": "onecycle",
                "onecycle_max_lr": params["lr"],
                "onecycle_pct_start": params["onecycle_pct_start"],
            },
            ema_cfg={"enabled": False},
            swa_cfg={"enabled": False},
            sampling_cfg=sampling_cfg,
            augmentation_cfg=augmentation_cfg,
            calibration_cfg=calibration_cfg,
            curriculum_cfg=curriculum_cfg,
            error_analysis_cfg=error_analysis_cfg,
        )

        if not results:
            print(f" [WARNING] Trial {trial_idx} returned empty metrics -- skipping.")
            continue

        auroc_mean = results.get("auroc", (0.0, 0.0))[0]
        sens90_mean = results.get("sens_at_90spec", (0.0, 0.0))[0]
        # Same composite objective as tune_hyperparameters.py, for consistency
        # across the project's two tuning scripts.
        score = 0.6 * auroc_mean + 0.4 * (sens90_mean / 100.0)

        print(f" Trial {trial_idx} Result -> Score: {score:.4f} | AUROC: {auroc_mean:.4f} | Sens@90%Spec: {sens90_mean:.2f}%")
        trial_log.append({"trial": trial_idx, "params": params, "auroc": auroc_mean, "score": score})

        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = {m: results[m][0] for m in results}

    print("\n" + "=" * 70)
    print(" Model 8 Hyperparameter Tuning Complete")
    print(f" Best Score: {best_score:.4f}")
    print(f" Best Params: {best_params}")
    print(f" Best Metrics: {best_metrics}")
    print("\n Full trial log (sorted by score):")
    for t in sorted(trial_log, key=lambda r: r["score"], reverse=True):
        print(f"   Trial {t['trial']:>2} | Score: {t['score']:.4f} | AUROC: {t['auroc']:.4f} | Params: {t['params']}")
    print("=" * 70 + "\n")

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "trial_log": trial_log,
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Model 8 (Knowledge-Infused Multi-Task Framework)")
    parser.add_argument("--config", type=str, default="configs/model8_crossformer_config.yaml",
                         help="Base Model 8 YAML config -- supplies backbone/paths/sampling/augmentation; "
                              "hyperparameters in SEARCH_SPACE override its training/loss/scheduler/loss_weighting sections per trial.")
    parser.add_argument("--ablation", type=str, default="full",
                         help=f"Ablation variant to tune against: {ABLATION_VARIANTS}")
    parser.add_argument("--output-config", type=str, default="configs/model8_tuned_hyperparameters.yaml",
                         help="Target YAML path to save the winning hyperparameters")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of random search trials")
    parser.add_argument("--epochs", type=int, default=15,
                         help="Epochs per trial (proxy -- less than the real 50; re-validate the winner at full length)")
    parser.add_argument("--k-folds", type=int, default=3,
                         help="Folds per trial (proxy -- less than the real 5; re-validate the winner at full length)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate trials without real training (tests the search loop itself)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trial sampling and training")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.ablation not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation '{args.ablation}'. Options: {ABLATION_VARIANTS}")

    result = tune_model8(
        base_config_path=args.config,
        ablation=args.ablation,
        n_trials=args.n_trials,
        epochs=args.epochs,
        k_folds=args.k_folds,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        all_tuned = {}
        if os.path.exists(args.output_config):
            try:
                with open(args.output_config, "r") as f:
                    all_tuned = yaml.safe_load(f) or {}
            except Exception:
                all_tuned = {}

        backbone_name = (yaml.safe_load(open(args.config)) or {}).get("backbone", {}).get("model", "unknown")
        all_tuned[f"{backbone_name}__{args.ablation}"] = {
            "params": result["best_params"],
            "score": result["best_score"],
            "metrics": result["best_metrics"],
            "proxy_epochs": args.epochs,
            "proxy_k_folds": args.k_folds,
            "base_config": args.config,
        }

        os.makedirs(os.path.dirname(args.output_config), exist_ok=True)
        with open(args.output_config, "w") as f:
            yaml.dump(all_tuned, f, default_flow_style=False, sort_keys=False)
        print(f"[SUCCESS] Best hyperparameters saved to {args.output_config}")
        print("[REMINDER] This was a proxy search (reduced epochs/folds) -- re-validate the winner "
              "with a full-length run (--ablation, full epochs/folds) before trusting it.")


if __name__ == "__main__":
    main()
