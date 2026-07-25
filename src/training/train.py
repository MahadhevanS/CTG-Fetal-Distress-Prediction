"""
Universal Model Training Module for CTG Fetal Distress Prediction
===================================================================

Supports dynamic model selection, dataset loading, patient-level stratified 5-fold CV,
and evaluation across all implemented temporal encoders.
"""

import os
import sys
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import yaml
except ImportError:
    yaml = None

import torch
import torch.nn as nn
import numpy as np

from src.models import (
    MultiScaleLSTMEncoder,
    MultiScaleLSTMForClassification,
    PatchCTGEncoder,
    PatchCTGForClassification
)


MODEL_REGISTRY = {
    "multiscale_lstm": (MultiScaleLSTMEncoder, MultiScaleLSTMForClassification),
    "patchctg": (PatchCTGEncoder, PatchCTGForClassification),
}

# Add PatchTST if available in workspace
try:
    from src.models.patchtst import PatchTSTEncoder, PatchTSTForClassification
    MODEL_REGISTRY["patchtst"] = (PatchTSTEncoder, PatchTSTForClassification)
except ImportError:
    pass


def load_config(config_path):
    if yaml is None or not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model(model_name: str, config: dict):
    model_name = model_name.lower().strip()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not recognized. Available choices: {list(MODEL_REGISTRY.keys())}")

    encoder_cls, classifier_cls = MODEL_REGISTRY[model_name]

    if model_name == "multiscale_lstm":
        encoder = encoder_cls(
            in_channels=2,
            seq_len=4800,
            hidden_size=config.get("model", {}).get("hidden_dim", 64),
            latent_dim=128
        )
    elif model_name == "patchctg":
        encoder = encoder_cls(
            in_channels=2,
            seq_len=4800,
            patch_len=16,
            stride=16,
            d_model=config.get("model", {}).get("hidden_dim", 128),
            latent_dim=128
        )
    elif model_name == "patchtst":
        encoder = encoder_cls(
            in_channels=2,
            seq_len=4800,
            patch_len=16,
            stride=16,
            d_model=config.get("model", {}).get("hidden_dim", 128),
            latent_dim=128
        )
    else:
        encoder = encoder_cls(latent_dim=128)

    model = classifier_cls(encoder)
    return encoder, model


def main():
    parser = argparse.ArgumentParser(description="Train CTG Fetal Distress Temporal Encoder Models")
    parser.add_argument("--config", type=str, default="configs/local.yaml", help="Path to configuration file")
    parser.add_argument("--model", type=str, default="multiscale_lstm", choices=list(MODEL_REGISTRY.keys()),
                        help="Model architecture to train and evaluate")
    parser.add_argument("--dry_run", action="store_true", help="Run quick dry-run verification")
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config) if os.path.exists(args.config) else {}
    
    data_path = config.get('data', {}).get('processed_path', 'data/processed/')
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', 'checkpoints/')
    
    print(f"Selected Model: {args.model}")
    print(f"Using data path: {data_path}")
    print(f"Saving checkpoints to: {checkpoint_dir}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    encoder, full_model = build_model(args.model, config)
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Successfully instantiated '{args.model}' encoder ({total_params:,} trainable parameters).")
    
    if args.model == "multiscale_lstm":
        from src.models.train_multiscale_lstm import main as train_multiscale_lstm_main
        import sys
        sys.argv = ["train_multiscale_lstm.py", "--data_dir", data_path, "--checkpoint_dir", checkpoint_dir]
        if args.dry_run:
            sys.argv.append("--dry_run")
        train_multiscale_lstm_main()
    elif args.model == "patchctg":
        from src.models.train_patchctg import main as train_patchctg_main
        import sys
        sys.argv = ["train_patchctg.py", "--data_dir", data_path, "--checkpoint_dir", checkpoint_dir]
        if args.dry_run:
            sys.argv.append("--dry_run")
        train_patchctg_main()
    else:
        print(f"Training routine for {args.model} ready.")


if __name__ == "__main__":
    main()
