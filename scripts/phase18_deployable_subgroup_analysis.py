"""
Phase 18 Stage 1h -- subgroup, residual, and M3-risk-stratified analysis for
the frozen deployable A3 fusion mechanism (A3_selected: p_fused = alpha *
p_M3(t) + (1-alpha) * p_parity, alpha ~= 0.33-0.35).

Items 5-6 of the user-directed next-stage plan:
  5. Subgroup and residual analyses -- error disagreement (which patients
     does fusion correct vs. harm relative to M3-only) and parity-group-
     stratified performance.
  6. Test whether parity helps primarily in the INTERMEDIATE M3-risk
     region -- the central hypothesis:
         "Parity acts primarily as a maternal-context modifier that
          improves the reliability of earlier CTG-based risk estimation,
          rather than providing a uniformly useful additive contribution
          at every prediction horizon."
     The horizon-dependent half of this hypothesis (parity matters far
     more at ≥20m/30m -- earlier, less-mature CTG evidence -- than at
     delivery) is already established by the 5-way comparison
     (deployable_fusion_5way_comparison.csv): CV delta vs. M3-only grows
     from +0.012 (delivery) to +0.065 (>=30m), monotonically. This script
     tests the OTHER half directly and for the first time: whether, WITHIN
     a single horizon, the fusion gain concentrates in patients whose
     M3-only risk estimate is intermediate/ambiguous rather than already
     extreme in either direction.

Reuses the frozen, sanity-checked per-fold models from
scripts/phase18_deployable_freeze_and_operational.py (re-fit fresh here,
deterministically, and cross-checked against that script's own saved
operational-metric predictions before anything is computed from them).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import to_logit
from src.models.phase16_causal_attention import predict_all_prefixes, eligible_prefix_length
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data
from scripts.phase18_deployable_freeze_and_operational import fit_parity_lr, fit_deployable_bundle

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
DATA_DIR = "data/processed_clinical"
CHECKPOINT_DIR = "results/phase16/checkpoints"
OUT_DIR = "results/phase18_fusion_ablation"

HORIZONS_OF_INTEREST = [0, 20]  # delivery (weak fusion gain) vs >=20m (strong fusion gain) -- bookends of the hypothesis


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    n_patients = len(clean_pids)

    df_rolling = pd.read_csv(ROLLING_PATH)
    df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    y_lookup = {p: int(df_rolling[df_rolling["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}
    y_pat = np.array([y_lookup[p] for p in clean_pids])

    meta = pd.read_csv(METADATA_PATH).set_index("record_id")
    parity_by_pid = {str(p): float(meta.loc[int(p), "parity"]) for p in clean_pids}
    parity_raw_pat = np.array([parity_by_pid[p] for p in clean_pids])

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_ids_arr = p6["patient_ids"]
    pred_cv_window = p6["pred_unweighted_cv"]
    patient_data_cv, t_del_cv = build_patient_data(pred_cv_window, patient_ids_arr, df_rolling, clean_pids, y_lookup)

    completed = load_completed_units(CHECKPOINT_DIR)
    fold_of = {p: folds_blob["assignment"][p][0] for p in clean_pids}
    fold_scorers = {f: load_scorer_checkpoint(completed[f"Model_3_magnitude_position_fold{f}"], in_dim=2, hidden=8) for f in range(5)}

    print("================================================================================")
    print("  PHASE 18 -- SUBGROUP, RESIDUAL, AND M3-RISK-STRATIFIED ANALYSIS                  ")
    print("================================================================================")

    m3_seq_cv = {}
    for pid in clean_pids:
        r_seq, elapsed_seq, y, T_i = patient_data_cv[pid]
        m3_seq_cv[pid] = predict_all_prefixes(fold_scorers[fold_of[pid]], r_seq, elapsed_seq, True, T_i)

    p_parity_oof = np.zeros(n_patients)
    for f_idx in range(5):
        tr_pids = [p for p in clean_pids if fold_of[p] != f_idx]
        p_par = fit_parity_lr(np.array([parity_by_pid[p] for p in tr_pids]), np.array([y_lookup[p] for p in tr_pids]),
                               np.array([parity_by_pid[p] for p in clean_pids]))
        te_mask = np.array([fold_of[p] == f_idx for p in clean_pids])
        p_parity_oof[te_mask] = p_par[te_mask]
    parity_prob_by_pid = {p: p_parity_oof[i] for i, p in enumerate(clean_pids)}

    # ---------------- refit canonical per-fold models (deterministic, verified pattern) ----------------
    canon_score_fns = {}
    for f_idx in range(5):
        tr_pids = [p for p in clean_pids if fold_of[p] != f_idx]
        score_fn, _ = fit_deployable_bundle(tr_pids, m3_seq_cv, parity_prob_by_pid, y_lookup)
        canon_score_fns[f_idx] = score_fn

    m3_only_by_h = {h: np.zeros(n_patients) for h in HORIZONS_OF_INTEREST}
    a3_by_h = {h: np.zeros(n_patients) for h in HORIZONS_OF_INTEREST}
    for i, pid in enumerate(clean_pids):
        _, _, _, T_i = patient_data_cv[pid]
        f_idx = fold_of[pid]
        par_p = parity_prob_by_pid[pid]
        for h in HORIZONS_OF_INTEREST:
            k = eligible_prefix_length(t_del_cv[pid], h, T_i)
            z_k = m3_seq_cv[pid][k - 1]
            m3_only_by_h[h][i] = z_k
            a3_by_h[h][i] = canon_score_fns[f_idx]("A3_selected", z_k, par_p)

    # sanity check vs already-saved CV AUROC
    ref = pd.read_csv(os.path.join(OUT_DIR, "deployable_fusion_results.csv"))
    h_name_map = {0: "Delivery", 20: ">=20m"}
    for h in HORIZONS_OF_INTEREST:
        c = roc_auc_score(y_pat, a3_by_h[h])
        r = ref[(ref["horizon"] == h_name_map[h]) & (ref["model"] == "A3_selected_once")]["cv_auroc"].iloc[0]
        status = "OK" if abs(c - r) < 1e-3 else "MISMATCH"
        print(f"  sanity h={h}: refit={c:.4f} vs saved={r:.4f} [{status}]")

    # ================================================================
    # ITEM 5a -- error disagreement (delivery, fixed operating point)
    # ================================================================
    print("\n--- Item 5a: error disagreement (M3-only vs A3_selected), delivery, 80%-sens threshold ---")
    op = pd.read_csv(os.path.join(OUT_DIR, "deployable_fusion_operational_metrics.csv")).set_index("system")
    th_m3 = op.loc["M3_only", "mean_threshold"]
    th_a3 = op.loc["A3_selected", "mean_threshold"]

    pred_m3_bin = (m3_only_by_h[0] >= th_m3).astype(int)
    pred_a3_bin = (a3_by_h[0] >= th_a3).astype(int)
    correct_m3 = (pred_m3_bin == y_pat); correct_a3 = (pred_a3_bin == y_pat)
    corrected = np.where((~correct_m3) & correct_a3)[0]
    harmed = np.where(correct_m3 & (~correct_a3))[0]
    error_disagreement = {
        "corrected_by_fusion": int(len(corrected)), "harmed_by_fusion": int(len(harmed)),
        "net": int(len(corrected) - len(harmed)),
        "positives_corrected": int(np.sum(y_pat[corrected] == 1)), "positives_harmed": int(np.sum(y_pat[harmed] == 1)),
        "negatives_corrected": int(np.sum(y_pat[corrected] == 0)), "negatives_harmed": int(np.sum(y_pat[harmed] == 0)),
    }
    print(f"  {json.dumps(error_disagreement, indent=2)}")
    with open(os.path.join(OUT_DIR, "deployable_error_disagreement.json"), "w") as fh:
        json.dump(error_disagreement, fh, indent=2)

    # ================================================================
    # ITEM 5b -- parity-group-stratified performance, delivery
    # ================================================================
    print("\n--- Item 5b: parity-group-stratified performance, delivery ---")
    parity_cat = np.clip(parity_raw_pat.astype(int), 0, 3)  # 0,1,2,3+ (fixed buckets, defined before analysis)
    subgroup_rows = []
    for cat, name in [(0, "parity=0"), (1, "parity=1"), (2, "parity=2"), (3, "parity>=3")]:
        mask = parity_cat == cat
        n = int(np.sum(mask)); pos = int(np.sum(y_pat[mask]))
        if pos > 0 and (n - pos) > 0:
            sens_m3 = np.sum((pred_m3_bin[mask] == 1) & (y_pat[mask] == 1)) / pos
            sens_a3 = np.sum((pred_a3_bin[mask] == 1) & (y_pat[mask] == 1)) / pos
            far_m3 = np.sum((pred_m3_bin[mask] == 1) & (y_pat[mask] == 0)) / (n - pos)
            far_a3 = np.sum((pred_a3_bin[mask] == 1) & (y_pat[mask] == 0)) / (n - pos)
            auc_m3 = roc_auc_score(y_pat[mask], m3_only_by_h[0][mask])
            auc_a3 = roc_auc_score(y_pat[mask], a3_by_h[0][mask])
        else:
            sens_m3 = sens_a3 = far_m3 = far_a3 = auc_m3 = auc_a3 = float("nan")
        row = {"parity_group": name, "n": n, "n_positive": pos,
               "m3_sens": round(sens_m3, 3) if not np.isnan(sens_m3) else None,
               "a3_sens": round(sens_a3, 3) if not np.isnan(sens_a3) else None,
               "m3_far": round(far_m3, 3) if not np.isnan(far_m3) else None,
               "a3_far": round(far_a3, 3) if not np.isnan(far_a3) else None,
               "m3_auroc": round(auc_m3, 4) if not np.isnan(auc_m3) else None,
               "a3_auroc": round(auc_a3, 4) if not np.isnan(auc_a3) else None}
        subgroup_rows.append(row)
        print(f"  {name}: n={n} pos={pos}  M3 sens/far/auc={row['m3_sens']}/{row['m3_far']}/{row['m3_auroc']}  "
              f"A3 sens/far/auc={row['a3_sens']}/{row['a3_far']}/{row['a3_auroc']}")
    pd.DataFrame(subgroup_rows).to_csv(os.path.join(OUT_DIR, "deployable_parity_subgroup_analysis.csv"), index=False)

    # ================================================================
    # ITEM 6 -- M3-risk-tertile-stratified fusion gain (the central hypothesis)
    # ================================================================
    print("\n--- Item 6: M3-risk-tertile-stratified fusion gain (delivery and >=20m) ---")
    print("  Hypothesis: does the A3-vs-M3only gain concentrate in the INTERMEDIATE M3-risk tertile?")
    tertile_rows = []
    for h in HORIZONS_OF_INTEREST:
        m3_h = m3_only_by_h[h]; a3_h = a3_by_h[h]
        # per-fold TRAINING-derived tertile cutoffs (no leakage), applied to that fold's test patients
        tertile_label = np.full(n_patients, -1, dtype=int)
        for f_idx in range(5):
            tr_mask = np.array([fold_of[p] != f_idx for p in clean_pids])
            te_mask = np.array([fold_of[p] == f_idx for p in clean_pids])
            cuts = np.percentile(m3_h[tr_mask], [33.33, 66.67])
            lab = np.zeros(np.sum(te_mask), dtype=int)
            m3_te = m3_h[te_mask]
            lab[m3_te > cuts[0]] = 1
            lab[m3_te > cuts[1]] = 2
            tertile_label[te_mask] = lab

        for t_idx, t_name in [(0, "Low M3-risk"), (1, "Mid M3-risk"), (2, "High M3-risk")]:
            mask = tertile_label == t_idx
            n = int(np.sum(mask)); pos = int(np.sum(y_pat[mask]))
            if pos > 2 and (n - pos) > 2:
                auc_m3_t = roc_auc_score(y_pat[mask], m3_h[mask])
                auc_a3_t = roc_auc_score(y_pat[mask], a3_h[mask])
                delta = auc_a3_t - auc_m3_t
            else:
                auc_m3_t = auc_a3_t = delta = float("nan")
            # accuracy-shift within stratum, using the SAME operational thresholds as item 5a (delivery only; for h=20 report score-delta instead)
            mean_score_delta = float(np.mean(a3_h[mask] - m3_h[mask]))
            row = {"horizon_min": h, "stratum": t_name, "n": n, "n_positive": pos,
                   "m3_auroc": round(auc_m3_t, 4) if not np.isnan(auc_m3_t) else None,
                   "a3_auroc": round(auc_a3_t, 4) if not np.isnan(auc_a3_t) else None,
                   "auroc_delta": round(delta, 4) if not np.isnan(delta) else None,
                   "mean_score_delta_a3_minus_m3": round(mean_score_delta, 4)}
            tertile_rows.append(row)
            print(f"  h={h:>3}m {t_name:<13}: n={n} pos={pos}  M3-AUROC={row['m3_auroc']}  A3-AUROC={row['a3_auroc']}  "
                  f"delta={row['auroc_delta']}  mean_score_shift={row['mean_score_delta_a3_minus_m3']:+.4f}")

    df_tertile = pd.DataFrame(tertile_rows)
    df_tertile.to_csv(os.path.join(OUT_DIR, "deployable_m3_risk_tertile_analysis.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}/deployable_m3_risk_tertile_analysis.csv")

    # ---------------- continuous version: correlate score-delta with M3-only score (rank), delivery + >=20m ----------------
    print("\n--- Continuous check: correlation of (A3 - M3only) score-shift with M3-only's own score ---")
    corr_rows = []
    for h in HORIZONS_OF_INTEREST:
        delta = a3_by_h[h] - m3_only_by_h[h]
        # "distance from 0.5" as an ambiguity proxy: is the fusion shift larger when M3 is closer to 0.5 (ambiguous)?
        ambiguity = -np.abs(m3_only_by_h[h] - 0.5)  # higher = more ambiguous (closer to 0.5)
        r, p = stats.spearmanr(ambiguity, np.abs(delta))
        corr_rows.append({"horizon_min": h, "spearman_r_abs_shift_vs_ambiguity": round(r, 4), "p": round(p, 4)})
        print(f"  h={h:>3}m: Spearman r(|A3-M3| shift, ambiguity=-|M3-0.5|) = {r:+.4f} (p={p:.4f})")
    pd.DataFrame(corr_rows).to_csv(os.path.join(OUT_DIR, "deployable_ambiguity_correlation.csv"), index=False)

    print("\n--- Execution complete. ---")


if __name__ == "__main__":
    main()
