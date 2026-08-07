"""
Generate Feature Scaler Artifact (data/processed/feature_scaler.npz)
=====================================================================
Computes per-feature mean and standard deviation across the 8 clinical target
features in train_dataset.pt:
    [0]: Baseline FHR (bpm)
    [1]: STV (bpm)
    [2]: LTV (bpm)
    [3]: Acceleration Count
    [4]: Early Deceleration Count
    [5]: Late Deceleration Count
    [6]: Variable Deceleration Count
    [7]: Prolonged Deceleration Count

Saves these stats into data/processed/feature_scaler.npz under keys:
    - 'feature_means'
    - 'feature_stds'
"""

import os
import numpy as np
import torch

def main():
    train_pt = "data/processed/train_dataset.pt"
    save_path = "data/processed/feature_scaler.npz"

    if not os.path.exists(train_pt):
        print(f"Error: {train_pt} not found!")
        return

    print(f"Loading training dataset from {train_pt}...")
    data = torch.load(train_pt, weights_only=False)

    if 'y_features' not in data:
        print("Error: 'y_features' key not found in train_dataset.pt!")
        return

    y_features = data['y_features'].numpy()  # (6266, 8)
    means = y_features.mean(axis=0).astype(np.float32)
    stds = y_features.std(axis=0).astype(np.float32)
    stds = np.clip(stds, 1e-6, None)  # Prevent division by zero

    print("\nCalculated Clinical Feature Scaler Stats (8 features):")
    feature_names = [
        "Baseline FHR (bpm)", "STV (bpm)", "LTV (bpm)", "Accel Count",
        "Early Decels", "Late Decels", "Var Decels", "Prolonged Decels"
    ]
    for idx, (name, m, s) in enumerate(zip(feature_names, means, stds)):
        print(f"  [{idx}] {name:<25}: Mean = {m:10.4f}, Std = {s:10.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, feature_means=means, feature_stds=stds)
    print(f"\n[Saved] Feature scaler artifact successfully generated at: {save_path}")

if __name__ == "__main__":
    main()
