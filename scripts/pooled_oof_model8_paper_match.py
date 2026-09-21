"""
Thin driver: pooled OOF AUROC for Model 8 (wide-distress `full`) trained on
the paper-matched dataset. Reuses scripts/ensemble_oof_eval.py's
compute_oof_array()/build_wide_distress_model() unchanged -- Model 8's own
fold algorithm (create_patient_level_folds(), joint-stratified on y_figo) is
exactly what produced these checkpoints in the first place, so reconstruction
here is already consistent by construction. No new fold logic needed.
"""

import os
import sys

import torch
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.ensemble_oof_eval import compute_oof_array, build_wide_distress_model
from src.training.multi_task_dataset import load_all_multitask_splits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = "data/processed_paper_match/"
    checkpoint_dir = "checkpoints/model8_crossformer_widedistress_paper_match/"

    dataset, patient_ids = load_all_multitask_splits(data_dir)

    oof = compute_oof_array(
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix="model8_full",
        build_fn=build_wide_distress_model,
        X=dataset.X, y_all=dataset.y_primary, patient_ids=patient_ids,
        device=device, secondary_labels=dataset.y_figo,
    )

    pooled = roc_auc_score(dataset.y_primary.numpy(), oof)
    print(f"\nPooled OOF AUROC (Model 8 wide-distress full, paper-matched data): {pooled:.4f}")


if __name__ == "__main__":
    main()
