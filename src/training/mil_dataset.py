"""
Multiple-Instance Learning (MIL) bag dataset for patient-level CTG modelling.

A "bag" is one patient: all of that patient's 20-minute windows, carrying a
single patient-level label (umbilical pH <= 7.15). This is the objective the
clinical task actually poses -- prior work in this project trained and scored
per-window, which is a proxy for it.

REQUIRES uniform-stride data (data/processed_mil/, built by
src/preprocessing/pipeline_mil.py). On label-conditional-stride data such as
data/processed/, bag size alone predicts the label at AUROC 0.9947 and every
patient-level number is an artifact -- see scripts/audit_bag_size_leak.py.

Bags are variable-length (patients differ in recording length and how many
windows survive quality filtering), so collate_mil_bags() pads to the batch
maximum and returns a boolean mask. Padded slots must be excluded from
attention (set to -inf pre-softmax) and from all per-window losses.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MILBagDataset(Dataset):
    """
    Groups windows into per-patient bags.

    Args:
        X:          (N, 2, 4800) signal windows.
        y_patient:  (N,) patient-level label broadcast to each window.
        y_primary:  (N,) window-level (horizon-relabelled) auxiliary label.
        y_figo:     (N,) FIGO 3-class per window.
        y_features: (N, 8) clinical features per window.
        patient_ids: length-N list of patient id strings.
        starts:     (N,) window start sample index (temporal ordering; used by
                    trajectory features and by the explainability module).
        indices:    Optional subset of window indices to restrict to (e.g. one
                    CV fold). Bags are formed only from these.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y_patient: torch.Tensor,
        y_primary: torch.Tensor,
        y_figo: torch.Tensor,
        y_features: torch.Tensor,
        patient_ids: List[str],
        starts: np.ndarray,
        indices: Optional[np.ndarray] = None,
    ):
        self.X = X
        self.y_patient = y_patient
        self.y_primary = y_primary
        self.y_figo = y_figo
        self.y_features = y_features
        self.starts = np.asarray(starts)

        pid_arr = np.asarray(patient_ids)
        if indices is None:
            indices = np.arange(len(X))
        indices = np.asarray(indices)

        # Group the selected window indices by patient, each bag ordered in time
        # so trajectory features and attention traces are chronological.
        bags: Dict[str, List[int]] = {}
        for i in indices:
            bags.setdefault(pid_arr[i], []).append(int(i))

        self.patient_order: List[str] = sorted(bags)
        self.bags: List[np.ndarray] = []
        for p in self.patient_order:
            idx = np.array(bags[p], dtype=np.int64)
            self.bags.append(idx[np.argsort(self.starts[idx])])

        # One label per bag -- max is a safety net; every window of a patient
        # already carries the same y_patient value by construction.
        self.bag_labels = np.array(
            [int(self.y_patient[b].max()) for b in self.bags], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, i: int) -> Dict:
        idx = self.bags[i]
        return {
            "X": self.X[idx],                       # (n_i, 2, 4800)
            "y_patient": float(self.bag_labels[i]),  # scalar
            "y_primary": self.y_primary[idx].float(),
            "y_figo": self.y_figo[idx].long(),
            "y_features": self.y_features[idx].float(),
            "starts": torch.as_tensor(self.starts[idx], dtype=torch.float32),
            "patient_id": self.patient_order[i],
            "n_windows": len(idx),
        }

    @property
    def bag_size_leak_auroc(self) -> float:
        """
        Self-audit: AUROC of bag size predicting the bag label. Must be ~0.5.
        Cheap enough to assert at the start of every training run.
        """
        from sklearn.metrics import roc_auc_score

        sizes = np.array([len(b) for b in self.bags], dtype=float)
        if len(set(self.bag_labels.tolist())) < 2:
            return float("nan")
        return float(roc_auc_score(self.bag_labels, sizes))


def collate_mil_bags(batch: List[Dict]) -> Dict:
    """
    Pads variable-length bags to the batch maximum.

    Returns dict with:
        X          (B, n_max, 2, 4800)
        mask       (B, n_max) bool -- True = real window, False = padding
        y_patient  (B,)
        y_primary  (B, n_max)
        y_figo     (B, n_max)
        y_features (B, n_max, 8)
        starts     (B, n_max)
        patient_ids list[str]
    """
    B = len(batch)
    n_max = max(item["n_windows"] for item in batch)
    C, L = batch[0]["X"].shape[1], batch[0]["X"].shape[2]
    n_feat = batch[0]["y_features"].shape[1]

    X = torch.zeros(B, n_max, C, L, dtype=torch.float32)
    mask = torch.zeros(B, n_max, dtype=torch.bool)
    y_primary = torch.zeros(B, n_max, dtype=torch.float32)
    y_figo = torch.zeros(B, n_max, dtype=torch.long)
    y_features = torch.zeros(B, n_max, n_feat, dtype=torch.float32)
    starts = torch.zeros(B, n_max, dtype=torch.float32)
    y_patient = torch.zeros(B, dtype=torch.float32)
    patient_ids = []

    for b, item in enumerate(batch):
        n = item["n_windows"]
        X[b, :n] = item["X"]
        mask[b, :n] = True
        y_primary[b, :n] = item["y_primary"]
        y_figo[b, :n] = item["y_figo"]
        y_features[b, :n] = item["y_features"]
        starts[b, :n] = item["starts"]
        y_patient[b] = item["y_patient"]
        patient_ids.append(item["patient_id"])

    return {
        "X": X, "mask": mask, "y_patient": y_patient,
        "y_primary": y_primary, "y_figo": y_figo, "y_features": y_features,
        "starts": starts, "patient_ids": patient_ids,
    }


def load_mil_splits(
    data_dir: str, splits: Optional[List[str]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], np.ndarray]:
    """
    Aggregates uniform-stride split files into flat tensors for MILBagDataset.

    Returns (X, y_patient, y_primary, y_figo, y_features, patient_ids, starts).

    Raises if y_patient is absent -- that key is what distinguishes a
    pipeline_mil.py dataset from the older label-conditional-stride ones, so its
    absence almost certainly means the wrong data_dir was passed.
    """
    if splits is None:
        splits = ["train", "val", "test"]

    Xs, yps, y1s, yfs, yfeats, pids, starts = [], [], [], [], [], [], []
    for split in splits:
        path = os.path.join(data_dir, f"{split}_dataset.pt")
        if not os.path.exists(path):
            print(f"[INFO] {path} not found -- skipping split '{split}'.")
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        if "y_patient" not in d:
            raise KeyError(
                f"{path} has no 'y_patient' key. MIL requires a uniform-stride "
                f"dataset built by src/preprocessing/pipeline_mil.py -- older "
                f"datasets (data/processed/) use a label-conditional stride and "
                f"are invalid for patient-level modelling."
            )
        Xs.append(d["X"])
        yps.append(d["y_patient"])
        y1s.append(d["y_primary"])
        yfs.append(d["y_figo"])
        yfeats.append(d["y_features"])
        pids.extend([str(m[0]) for m in d["metadata"]])
        starts.extend([int(m[1]) for m in d["metadata"]])

    if not Xs:
        raise FileNotFoundError(f"No MIL dataset splits found in '{data_dir}'.")

    return (
        torch.cat(Xs), torch.cat(yps), torch.cat(y1s),
        torch.cat(yfs), torch.cat(yfeats), pids, np.array(starts),
    )
