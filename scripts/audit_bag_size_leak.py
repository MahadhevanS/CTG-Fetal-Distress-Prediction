"""
Bag-size leak audit -- STANDING GUARD for all patient-level (MIL) work.

Measures how well a patient's WINDOW COUNT alone predicts their label. If a
dataset's windowing depends on the label (as data/processed/ does -- pipeline.py
picks a 30s stride for distress patients and 10min for normal ones), bag size
becomes a label proxy and every max/top-k patient-level aggregation reports that
artifact rather than fetal distress.

Measured 2026-08-18:
    data/processed/            -> 0.9947  (SEVERE leak, unusable for MIL)
    data/processed_paper_match/-> 0.5467  (clean)

Pass criterion: AUROC ~0.5 (report flags anything >= 0.60).

Usage:
    python scripts/audit_bag_size_leak.py --data_dir data/processed_mil/
    python scripts/audit_bag_size_leak.py --data_dir data/processed/ --data_dir data/processed_mil/
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

FLAG_THRESHOLD = 0.60


def audit_split(pt_path: str):
    """Returns (n_patients, n_windows, prevalence, auroc_from_count, stats) or None."""
    if not os.path.exists(pt_path):
        return None
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    pids = np.array([m[0] for m in d["metadata"]])

    # Prefer an explicit patient-level label when the dataset carries one
    # (pipeline_mil.py adds y_patient); otherwise fall back to max-over-windows
    # of y_primary, the same rule create_patient_level_folds() uses.
    if "y_patient" in d:
        per_window_label = d["y_patient"].numpy()
    else:
        per_window_label = d["y_primary"].numpy()

    counts = Counter(pids)
    uniq = np.array(sorted(counts))
    labels = np.array([int(per_window_label[pids == p].max()) for p in uniq])
    sizes = np.array([counts[p] for p in uniq], dtype=float)

    if len(set(labels)) < 2:
        return None

    auroc = roc_auc_score(labels, sizes)
    stats = {
        "pos_median": float(np.median(sizes[labels == 1])),
        "neg_median": float(np.median(sizes[labels == 0])),
        "pos_range": (int(sizes[labels == 1].min()), int(sizes[labels == 1].max())),
        "neg_range": (int(sizes[labels == 0].min()), int(sizes[labels == 0].max())),
    }
    return len(uniq), len(pids), float(labels.mean()), auroc, stats


def audit_data_dir(data_dir: str) -> bool:
    """Audits every split present. Returns True if all splits pass."""
    print(f"\n{'='*66}")
    print(f" {data_dir}")
    print(f"{'='*66}")

    all_pass = True
    found_any = False
    for split in ["train", "val", "test"]:
        res = audit_split(os.path.join(data_dir, f"{split}_dataset.pt"))
        if res is None:
            continue
        found_any = True
        n_pat, n_win, prev, auroc, st = res
        verdict = "PASS" if auroc < FLAG_THRESHOLD else "*** LEAK ***"
        print(f"\n  [{split}] {n_pat} patients | {n_win} windows | patient prevalence {prev:.3f}")
        print(f"    windows/patient : distress median {st['pos_median']:.0f} "
              f"(range {st['pos_range'][0]}-{st['pos_range'][1]})")
        print(f"                      normal   median {st['neg_median']:.0f} "
              f"(range {st['neg_range'][0]}-{st['neg_range'][1]})")
        print(f"    AUROC(window count -> patient label): {auroc:.4f}   {verdict}")
        if auroc >= FLAG_THRESHOLD:
            all_pass = False

    if not found_any:
        print("  [WARNING] no *_dataset.pt splits found here.")
        return False
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Bag-size leak audit for patient-level modelling")
    parser.add_argument("--data_dir", action="append", default=None,
                        help="Dataset directory to audit (repeatable).")
    args = parser.parse_args()

    data_dirs = args.data_dir or ["data/processed_mil/"]

    results = {}
    for d in data_dirs:
        results[d] = audit_data_dir(d)

    print(f"\n{'='*66}")
    print(" SUMMARY")
    print(f"{'='*66}")
    for d, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {d}")
    print(f"\n  Pass criterion: AUROC(window count -> label) < {FLAG_THRESHOLD} on every split.")

    if not all(results.values()):
        print("\n  A FAILING dataset must NOT be used for patient-level/MIL work --")
        print("  bag size is acting as a label proxy. Window-level results on it are")
        print("  still valid (a window is scored on its own signal alone).")
        sys.exit(1)


if __name__ == "__main__":
    main()
