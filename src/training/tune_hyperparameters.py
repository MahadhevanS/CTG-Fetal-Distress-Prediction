"""
Hyperparameter Tuning Runner for CTG Fetal Distress Prediction Temporal Encoders
================================================================================
Automated hyperparameter optimization framework across all 7 temporal encoders
(1D CNN, BiLSTM, GRU, TCN, MultiScale-LSTM, PatchCTG, PatchTST).

Evaluates candidate hyperparameter combinations using Stratified Patient-Level Cross-Validation
and saves the optimal parameter configurations to configs/tuned_hyperparameters.yaml.
"""

import argparse
import os
import random
import sys
from typing import Dict, Any, List

import yaml
import torch

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.training.train import train_and_evaluate, MODEL_REGISTRY

# Search spaces for each encoder architecture
SEARCH_SPACES: Dict[str, Dict[str, List[Any]]] = {
    "cnn1d": {
        "lr": [0.0001, 0.0003, 0.0005, 0.001],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "batch_size": [16, 32, 64],
        "latent_dim": [128],
    },
    "bilstm": {
        "lr": [0.0001, 0.0003, 0.0005, 0.001],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "hidden_size": [32, 64, 128],
        "num_layers": [1, 2, 3],
        "dropout": [0.1, 0.2, 0.3],
        "batch_size": [16, 32, 64],
    },
    "gru": {
        "lr": [0.0001, 0.0003, 0.0005, 0.001],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "gru_hidden": [32, 64, 128],
        "num_layers": [1, 2, 3],
        "dropout": [0.1, 0.2, 0.3],
        "batch_size": [16, 32, 64],
    },
    "tcn": {
        "lr": [0.0001, 0.0003, 0.0005, 0.001],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "kernel_size": [3, 5, 7],
        "dropout": [0.1, 0.2, 0.3],
        "batch_size": [16, 32, 64],
    },
    "multiscale_lstm": {
        "lr": [0.0001, 0.0003, 0.0005, 0.001],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "hidden_size": [32, 64, 128],
        "num_layers": [1, 2, 3],
        "dropout": [0.1, 0.2, 0.3],
        "batch_size": [16, 32, 64],
    },
    "patchctg": {
        "lr": [0.0001, 0.0003, 0.0005],
        "weight_decay": [1e-5, 1e-4],
        "patch_len": [8, 16, 32],
        "stride": [8, 16],
        "n_heads": [4, 8],
        "n_layers": [2, 3, 4],
        "dropout": [0.1, 0.2],
        "batch_size": [16, 32, 64],
    },
    "patchtst": {
        "lr": [0.0001, 0.0003, 0.0005],
        "weight_decay": [1e-5, 1e-4],
        "patch_len": [8, 16, 32],
        "stride": [8, 16],
        "n_heads": [4, 8],
        "n_layers": [2, 3, 4],
        "dropout": [0.1, 0.2],
        "batch_size": [16, 32, 64],
    },
}


def sample_hyperparameters(search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Randomly samples a parameter configuration from a search space dictionary."""
    sampled = {}
    for param_name, choices in search_space.items():
        sampled[param_name] = random.choice(choices)
    return sampled


def tune_model(
    model_name: str,
    data_dir: str,
    n_trials: int = 5,
    epochs: int = 10,
    k_folds: int = 5,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Executes randomized hyperparameter search trials for a specific encoder architecture.
    Returns dictionary containing best hyperparameter trial parameters and evaluation metrics.
    """
    model_key = model_name.lower().replace("-", "").replace("_", "").strip()
    
    # Map common aliases to search space keys
    key_mapping = {
        "1dcnn": "cnn1d", "cnn": "cnn1d", "cnn1dencoder": "cnn1d",
        "bilstmencoder": "bilstm", "lstm": "bilstm",
        "multiscalelstm": "multiscale_lstm",
    }
    search_key = key_mapping.get(model_key, model_key)
    
    if search_key not in SEARCH_SPACES:
        raise ValueError(f"No search space defined for model '{model_name}'. Options: {list(SEARCH_SPACES.keys())}")

    search_space = SEARCH_SPACES[search_key]
    print("\n" + "=" * 70)
    print(f" Starting Hyperparameter Tuning for Model: {model_name.upper()}")
    print(f" Trials: {n_trials} | Epochs/Trial: {epochs} | Folds: {k_folds}")
    print(" Search Space:")
    for k, v in search_space.items():
        print(f"   - {k}: {v}")
    print("=" * 70 + "\n")

    best_score = -1.0
    best_config = {}
    best_metrics = {}

    for trial_idx in range(1, n_trials + 1):
        print(f"\n--- [Trial {trial_idx}/{n_trials}] Sampling Configuration ---")
        params = sample_hyperparameters(search_space)
        print(f" Sampled Params: {params}")

        if dry_run:
            print(f" [DRY RUN] Simulating training for trial {trial_idx}...")
            sim_score = random.uniform(0.65, 0.85)
            if sim_score > best_score:
                best_score = sim_score
                best_config = params
                best_metrics = {"auroc": sim_score, "sens_at_90spec": sim_score * 40.0}
            continue

        lr = params.pop("lr", 0.0005)
        weight_decay = params.pop("weight_decay", 1e-4)
        batch_size = params.pop("batch_size", 32)
        encoder_kwargs = params  # remaining items are encoder-specific args

        results = train_and_evaluate(
            model_name=search_key,
            data_dir=data_dir,
            k_folds=k_folds,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            encoder_kwargs=encoder_kwargs,
            save_dir="checkpoints/tuning_tmp",
            dry_run=dry_run,
        )

        if not results:
            print(f"[WARNING] Trial {trial_idx} returned empty metrics.")
            continue

        auroc_mean = results.get("auroc", (0.0, 0.0))[0]
        sens90_mean = results.get("sens_at_90spec", (0.0, 0.0))[0]

        # Objective function: 60% AUROC + 40% Sens@90%Spec normalized
        score = 0.6 * auroc_mean + 0.4 * (sens90_mean / 100.0)

        print(
            f" Trial {trial_idx} Result -> Score: {score:.4f} | "
            f"Val AUROC: {auroc_mean:.4f} | Sens@90%Spec: {sens90_mean:.2f}%"
        )

        if score > best_score:
            best_score = score
            best_config = {
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                **encoder_kwargs,
            }
            best_metrics = {m: results[m][0] for m in results}

    print("\n" + "=" * 70)
    print(f" Hyperparameter Tuning Complete for {model_name.upper()}")
    print(f" Best Score: {best_score:.4f}")
    print(f" Optimal Configuration: {best_config}")
    print("=" * 70 + "\n")

    return {
        "best_config": best_config,
        "best_score": best_score,
        "best_metrics": best_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for CTG Temporal Encoders")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Target model architecture name (e.g. cnn1d, bilstm, gru, tcn, multiscale_lstm, patchctg, patchtst, all)",
    )
    parser.add_argument("--data-dir", type=str, default="data/processed/", help="Path to preprocessed dataset")
    parser.add_argument(
        "--output-config",
        type=str,
        default="configs/tuned_hyperparameters.yaml",
        help="Target YAML path to save tuned hyperparameters",
    )
    parser.add_argument("--n-trials", type=int, default=3, help="Number of random hyperparameter search trials per model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per trial")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of cross-validation folds per trial")
    parser.add_argument("--dry-run", action="store_true", help="Run quick dry-run test without training")

    args = parser.parse_args()

    models_to_tune = (
        ["cnn1d", "bilstm", "gru", "tcn", "multiscale_lstm", "patchctg", "patchtst"]
        if args.model.lower() == "all"
        else [args.model.lower()]
    )

    # Load existing config if available
    all_tuned_configs = {}
    if os.path.exists(args.output_config):
        try:
            with open(args.output_config, "r") as f:
                all_tuned_configs = yaml.safe_load(f) or {}
        except Exception:
            all_tuned_configs = {}

    for model_name in models_to_tune:
        res = tune_model(
            model_name=model_name,
            data_dir=args.data_dir,
            n_trials=args.n_trials,
            epochs=args.epochs,
            k_folds=args.k_folds,
            dry_run=args.dry_run,
        )
        all_tuned_configs[model_name] = res["best_config"]

    if not args.dry_run:
        os.makedirs(os.path.dirname(args.output_config), exist_ok=True)
        with open(args.output_config, "w") as f:
            yaml.dump(all_tuned_configs, f, default_flow_style=False)
        print(f"\n[SUCCESS] Optimal hyperparameter configurations saved to {args.output_config}")


if __name__ == "__main__":
    main()
