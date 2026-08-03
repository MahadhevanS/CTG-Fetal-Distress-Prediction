"""
Multi-Task CTG Dataset Module
===============================
Provides MultiTaskCTGDataset — a PyTorch Dataset that loads all four
training targets required for Model 8 knowledge-infused multi-task training:

    X           (N, 2, 4800)  — Z-normalized CTG signal windows
    y_primary   (N,)          — Binary distress label (pH ≤ 7.15, float32)
    y_figo      (N,)          — FIGO 3-class label (0=Normal, 1=Suspicious, 2=Pathological, long)
    y_features  (N, 8)        — Physiological feature regression targets (float32):
                                [0] Baseline FHR (bpm)
                                [1] STV (bpm)
                                [2] LTV (bpm)
                                [3] Acceleration Count
                                [4] Early Deceleration Count
                                [5] Late Deceleration Count
                                [6] Variable Deceleration Count
                                [7] Prolonged Deceleration Count

Also provides:
    load_all_multitask_splits() — aggregates train/val/test .pt files into a single dataset.
    load_feature_scaler_stats() — loads mean/std per feature from stored scaler artifacts.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiTaskCTGDataset(Dataset):
    """
    PyTorch Dataset for Knowledge-Infused Multi-Task CTG training (Model 8).

    Loads all three supervised targets alongside the signal tensor so that
    the training loop can compute the composite multi-task loss in a single pass.

    Args:
        X:           Tensor of shape (N, 2, 4800) — Z-normalized CTG signal windows.
        y_primary:   Tensor of shape (N,) float32 — Binary distress label.
        y_figo:      Tensor of shape (N,) long    — FIGO 3-class label.
        y_features:  Tensor of shape (N, 8) float32 — Physiological features.
        patient_ids: Optional list of patient ID strings for fold splitting.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y_primary: torch.Tensor,
        y_figo: torch.Tensor,
        y_features: torch.Tensor,
        patient_ids: Optional[List[str]] = None,
    ):
        assert len(X) == len(y_primary) == len(y_figo) == len(y_features), (
            f"Tensor length mismatch: X={len(X)}, y_primary={len(y_primary)}, "
            f"y_figo={len(y_figo)}, y_features={len(y_features)}"
        )
        self.X = X
        self.y_primary = y_primary.float()
        self.y_figo = y_figo.long()
        self.y_features = y_features.float()
        self.patient_ids = (
            patient_ids if patient_ids is not None
            else [f"p_{i}" for i in range(len(X))]
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (X[idx], y_primary[idx], y_figo[idx], y_features[idx])
        """
        return (
            self.X[idx],
            self.y_primary[idx],
            self.y_figo[idx],
            self.y_features[idx],
        )


def _load_pt_split(pt_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Loads a single .pt split file and extracts all multi-task targets.

    Expected .pt structure (from preprocessing pipeline v1.0):
        ds["X"]          (N, 2, 4800)
        ds["y_primary"]  (N,)
        ds["y_figo"]     (N,)
        ds["y_features"] (N, 8)
        ds["metadata"]   list of (patient_id, ...) tuples (optional)

    Returns:
        Tuple of (X, y_primary, y_figo, y_features, patient_ids)
    """
    ds = torch.load(pt_path, map_location="cpu", weights_only=False)

    X = ds["X"]
    y_primary = ds["y_primary"]

    # y_figo: may be absent in older .pt files — fall back to zeros
    if "y_figo" in ds and ds["y_figo"] is not None:
        y_figo = ds["y_figo"]
    else:
        print(f"[WARNING] y_figo not found in {pt_path}. Filling with zeros.")
        y_figo = torch.zeros(len(X), dtype=torch.long)

    # y_features: may be absent in older .pt files — fall back to zeros
    if "y_features" in ds and ds["y_features"] is not None:
        y_features = ds["y_features"]
    else:
        print(f"[WARNING] y_features not found in {pt_path}. Filling with zeros.")
        y_features = torch.zeros(len(X), 8, dtype=torch.float32)

    # Patient IDs from metadata
    if "metadata" in ds and ds["metadata"]:
        patient_ids = [str(item[0]) for item in ds["metadata"]]
    else:
        patient_ids = [f"rec_{i}" for i in range(len(X))]

    return X, y_primary, y_figo, y_features, patient_ids


def load_all_multitask_splits(
    data_dir: str,
    splits: Optional[List[str]] = None,
) -> Tuple[MultiTaskCTGDataset, List[str]]:
    """
    Aggregates train / val / test .pt split files into a single MultiTaskCTGDataset.
    Used by the 5-fold CV training loop which re-creates folds from the full dataset.

    Args:
        data_dir: Path to the directory containing train_dataset.pt, val_dataset.pt,
                  test_dataset.pt (and optionally others).
        splits:   Which splits to load. Defaults to ["train", "val", "test"].

    Returns:
        Tuple of:
          - MultiTaskCTGDataset containing all aggregated windows.
          - List[str] of patient IDs (same order as dataset indices).

    Raises:
        FileNotFoundError: If no .pt files are found in data_dir.
    """
    if splits is None:
        splits = ["train", "val", "test"]

    X_list, yp_list, yf_list, yfeat_list, pids_list = [], [], [], [], []

    for split_name in splits:
        pt_path = os.path.join(data_dir, f"{split_name}_dataset.pt")
        if not os.path.exists(pt_path):
            print(f"[INFO] {pt_path} not found — skipping split '{split_name}'.")
            continue
        X, y_primary, y_figo, y_features, patient_ids = _load_pt_split(pt_path)
        X_list.append(X)
        yp_list.append(y_primary)
        yf_list.append(y_figo)
        yfeat_list.append(y_features)
        pids_list.extend(patient_ids)

    if not X_list:
        raise FileNotFoundError(
            f"No .pt dataset files found in '{data_dir}' for splits {splits}."
        )

    X_all = torch.cat(X_list, dim=0)
    yp_all = torch.cat(yp_list, dim=0)
    yf_all = torch.cat(yf_list, dim=0)
    yfeat_all = torch.cat(yfeat_list, dim=0)

    dataset = MultiTaskCTGDataset(X_all, yp_all, yf_all, yfeat_all, patient_ids=pids_list)
    print(
        f"[MultiTaskCTGDataset] Loaded {len(dataset)} windows | "
        f"{len(set(pids_list))} unique patients | "
        f"Distress prevalence: {yp_all.mean().item():.3f}"
    )
    return dataset, pids_list


def load_feature_scaler_stats(
    scaler_path: str,
    n_features: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads per-feature mean and std from the stored CTU-CHB signal scaler.

    IMPORTANT: The CTU signal scaler (ctu_signal_scaler.npz) stores stats for
    the 2 signal channels (FHR diff, UC). The 8 clinical feature targets
    (Baseline, STV, LTV, Accels, Decels) have their own scale. If a separate
    feature scaler exists at feature_scaler_path, load that instead.

    This function is a convenience loader. The returned arrays should be passed
    to figo_rule_loss_normalized() after converting to torch.Tensor.

    Args:
        scaler_path: Path to .npz scaler file.
        n_features:  Number of features expected (default 8).

    Returns:
        Tuple of (feature_means, feature_stds), both shape (n_features,), dtype float32.
        Returns zeros/ones (identity transform) if scaler file not found.
    """
    if not os.path.exists(scaler_path):
        print(
            f"[WARNING] Scaler not found at {scaler_path}. "
            f"Using identity transform (means=0, stds=1). "
            f"Knowledge loss will operate on normalized values."
        )
        return np.zeros(n_features, dtype=np.float32), np.ones(n_features, dtype=np.float32)

    data = np.load(scaler_path)
    if "feature_means" in data and "feature_stds" in data:
        means = data["feature_means"].astype(np.float32)
        stds = data["feature_stds"].astype(np.float32)
        stds = np.clip(stds, 1e-6, None)  # prevent division by zero
        return means, stds

    # Fallback: signal scaler has 'mean' and 'std' for 2 channels — not usable for 8 features
    print(
        "[WARNING] Scaler file does not contain 'feature_means'/'feature_stds' keys. "
        "Using identity transform. Ensure pipeline generates a feature-specific scaler."
    )
    return np.zeros(n_features, dtype=np.float32), np.ones(n_features, dtype=np.float32)
