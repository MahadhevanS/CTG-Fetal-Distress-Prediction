"""
External validation handoff -- Step 0: produce deployable model artifacts.

Everything in Phases 13-16 evaluated candidates via cross-validation and
saved PREDICTIONS on CTU-UHB. No script anywhere in this project ever
persisted an actual fitted classifier object for P6, Model 3, or the
parity fusion -- there is nothing to literally hand to an external-cohort
researcher and say "apply this to your patients." This script closes that
gap: it fits ONE final version of each candidate on the FULL 547-patient
CTU-UHB cohort (no held-out fold) and saves it as a portable artifact.

This is standard "lock the final model before external validation"
practice -- these final fits are expected to look mildly optimistic on
CTU-UHB itself (they were fit on all of it), which is precisely why the
external cohort, not another CTU-UHB re-evaluation, is the real test.
Nothing here changes Phase 12.1, which remains the authoritative,
separately-frozen production artifact.

Produces:
  models/external_validation_handoff/p6_final_classifier.joblib
  models/external_validation_handoff/p6_final_scaler.joblib
  models/external_validation_handoff/model3_final_scorer.pt
  models/external_validation_handoff/parity_final_model.joblib
  models/external_validation_handoff/parity_final_scaler.joblib
  models/external_validation_handoff/manifest.json  -- exact provenance,
    fitting data, feature definitions, and how to apply each artifact.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import get_patient_scores_at_horizon_corrected, to_logit, from_logit, select_lambda_trainfold
from src.models.phase16_causal_attention import train_scorer, score_full_sequence, pooled_prediction_from_logits
from scripts.phase16_temporal_attention_model import build_patient_data, carve_inner_validation

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
TRAJ_PATH = "results/phase9c_state_trajectory/state_trajectory_features.npz"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
OUT_DIR = "models/external_validation_handoff"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    df = pd.read_csv(ROLLING_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    patient_ids = df["patient_id"].values
    t_del = df["time_before_delivery_min"].values
    y_715 = df["primary_label_715"].values
    y_pat = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])

    data_traj = np.load(TRAJ_PATH)
    X_p6 = data_traj["X_state_trajectory"]  # (8517, 40)

    print("================================================================================")
    print("  PREPARING EXTERNAL VALIDATION ARTIFACTS  (fit once, on all 547 CTU-UHB patients)")
    print("================================================================================")

    # ---------------- 1. P6-final ----------------
    print("\n[1/3] P6-final classifier (LogisticRegression on 40-D state-trajectory features)")
    scaler_p6 = StandardScaler()
    X_p6_s = scaler_p6.fit_transform(X_p6)
    clf_p6 = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
    clf_p6.fit(X_p6_s, y_715)
    pred_p6_window_final = clf_p6.predict_proba(X_p6_s)[:, 1]

    sw_final = get_patient_scores_at_horizon_corrected(pred_p6_window_final, patient_ids, clean_pids, t_del, 0)
    auc_insample = roc_auc_score(y_pat, sw_final)
    print(f"  In-sample delivery AUROC (fit-on-everything, NOT a validation metric): {auc_insample:.4f}")
    print("  (Compare to the locked out-of-fold Phase 12.1 number, 0.6872, for context only --")
    print("   this number is expected to look better since it is not held out. It is not a claim of improvement.)")

    joblib.dump(clf_p6, os.path.join(OUT_DIR, "p6_final_classifier.joblib"))
    joblib.dump(scaler_p6, os.path.join(OUT_DIR, "p6_final_scaler.joblib"))

    # ---------------- 2. Model 3-final ----------------
    print("\n[2/3] Model 3-final attention scorer (trained on P6-final's own window-level scores)")
    y_lookup = {p: int(v) for p, v in zip(clean_pids, y_pat)}
    patient_data_final, t_del_final = build_patient_data(pred_p6_window_final, patient_ids, df, clean_pids, y_lookup)

    inner_tr, inner_val = carve_inner_validation(clean_pids, y_lookup, frac=0.15, seed=42)
    scorer3_final, val_loss, n_epochs = train_scorer(inner_tr, inner_val, patient_data_final, use_elapsed=True, seed=42)
    print(f"  Trained {n_epochs} epochs, inner-val loss={val_loss:.4f}")
    torch.save(scorer3_final.state_dict(), os.path.join(OUT_DIR, "model3_final_scorer.pt"))

    # ---------------- 3. Parity-final ----------------
    print("\n[3/3] Parity-final fusion (univariate parity model + logit-modulation lambda)")
    meta = pd.read_csv(METADATA_PATH)
    if "record_id" in meta.columns:
        meta = meta.set_index("record_id")
    parity_pat = np.array([float(meta.loc[int(p), "parity"]) for p in clean_pids])

    scaler_parity = StandardScaler()
    X_parity_s = scaler_parity.fit_transform(parity_pat.reshape(-1, 1))
    clf_parity = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_parity.fit(X_parity_s, y_pat)
    p_parity_final = clf_parity.predict_proba(X_parity_s)[:, 1]

    lam_final, _ = select_lambda_trainfold(to_logit(sw_final), to_logit(p_parity_final), y_pat)
    print(f"  Selected fusion lambda (fit on all 547, for reference -- fold lambdas were 1.3-1.5): {lam_final:.2f}")

    joblib.dump(clf_parity, os.path.join(OUT_DIR, "parity_final_model.joblib"))
    joblib.dump(scaler_parity, os.path.join(OUT_DIR, "parity_final_scaler.joblib"))

    # ---------------- manifest ----------------
    manifest = {
        "purpose": "Frozen artifacts for external-cohort validation. Fit once on all 547 CTU-UHB patients (no held-out fold) -- NOT a re-validation of internal performance.",
        "governance": "Phase 12.1 is unaffected and remains the sole locked production artifact. These are new, separate, exploratory candidates.",
        "p6_final": {
            "file": "p6_final_classifier.joblib", "scaler": "p6_final_scaler.joblib",
            "input": "40-D X_state_trajectory feature vector per window (see docs/phase16_protocol.md Section 0(a) and reports/phase12_final_scientific_synthesis.md for the exact feature definitions)",
            "how_to_apply": "scaler.transform(X_new_40d) -> clf.predict_proba(...)[:,1] gives a window-level P6 risk score, exactly as done throughout Phases 8-16.",
            "in_sample_delivery_auroc_reference_only": round(float(auc_insample), 4),
        },
        "model3_final": {
            "file": "model3_final_scorer.pt",
            "architecture": "AttentionScorer(in_dim=2, hidden=8) -- src/models/phase16_causal_attention.py",
            "input": "causal sequence of (r_t, elapsed_t) pairs, r_t = P6-final's window-level score, elapsed_t = minutes since that patient's own first retained window",
            "how_to_apply": "src.models.phase16_causal_attention.score_full_sequence + pooled_prediction_from_logits, using the causally-eligible prefix at whatever horizon is being evaluated (src.evaluation.phase13_common.eligible_prefix_length)",
            "trained_epochs": n_epochs, "inner_val_loss": round(float(val_loss), 4),
            "note": "Trained on P6-final's own (in-sample) window scores for internal consistency of the deployable chain -- external cohort scores must come from applying p6_final_classifier fresh, not from any CTU-UHB-derived score.",
        },
        "parity_final": {
            "file": "parity_final_model.joblib", "scaler": "parity_final_scaler.joblib",
            "fusion_lambda": round(float(lam_final), 4),
            "fusion_equation": "logit(p_fused) = logit(p_P6_final) + lambda * logit(p_parity_final)",
            "input_covariate": "parity (integer count, raw from clinical_metadata.csv, admission-time, causally safe)",
            "how_to_apply": "scaler.transform(parity_value) -> clf.predict_proba(...)[:,1] gives p_parity; fuse via the equation above using P6-final's own delivery-horizon score as p_P6_final.",
        },
        "frozen_huber_reference_not_refit": {
            "path": "models/continuous_clinical_huber/huber_fold_{0-4}.joblib + scaler_fold_{0-4}.joblib",
            "note": "Not refit here -- for a new external cohort, use the mean of the 5 existing fold models' predictions as the Huber-derived risk input feature (same ensembling convention already used for this project's other delivered models), rather than attempting a from-scratch Huber refit.",
        },
        "cohort_used_for_fitting": {"n_patients": len(clean_pids), "n_windows": len(df), "source": "CTU-UHB, same 547-patient cohort as every prior phase"},
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nSaved all artifacts + manifest -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
