"""
Phase 26, Problems 3-5 -- calibration, PPV/NPV/false-alert burden, and a threshold-performance curve for the
LOCKED URM (docs/phase26_full_clinical_evaluation_protocol.md).

IMPORTANT CORRECTION (found while building this): results/phase18_fusion_ablation/stage1_patient_level_predictions.csv
is an EARLIER, exploratory Stage-1 ablation snapshot (dated 2026-09-15, docs/phase18_m3_parity_fusion_ablation_protocol.md)
-- its own AUROC (0.7382/0.6913/0.6659/0.6470) does NOT match the frozen headline URM numbers (0.7335/0.6921/0.6638/0.6631)
that every other phase in this study reproduces and gates against. It is NOT the deployed URM's predictions and is not
used here. Instead this script regenerates the exact A3_selected_once sequence from the frozen artifacts (per-fold TAM
checkpoints + fit_parity_lr + fit_deployable_bundle, imported directly from phase18_deployable_freeze_and_operational.py
and phase18_deployable_fusion.py so the fitting code is bit-identical, not reimplemented) for both CV and the held-out
test partition, and gates both against the frozen headline AUROCs before computing anything else.
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import torch
from src.models.phase16_causal_attention import predict_all_prefixes, eligible_prefix_length
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data
from scripts.phase18_deployable_freeze_and_operational import fit_parity_lr, fit_deployable_bundle, build_pooled_training_set

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
DATA_DIR = "data/processed_clinical"
CHECKPOINT_DIR = "results/phase16/checkpoints"
OUT = "results/phase26_clinical_evaluation"; os.makedirs(OUT, exist_ok=True)
FS_HZ = 4.0
H = [0, 10, 20, 30]
FROZEN_URM_CV = {0: 0.7335, 10: 0.6921, 20: 0.6638, 30: 0.6631}
FROZEN_URM_TEST = {0: 0.7433, 10: 0.6988, 20: 0.7558, 30: 0.7754}


def confusion_metrics(y, yhat):
    tp = int(((yhat == 1) & (y == 1)).sum()); fp_ = int(((yhat == 1) & (y == 0)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp_, 1); ppv = tp / max(tp + fp_, 1); npv = tn / max(tn + fn, 1)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-9)
    return dict(tp=tp, fp=fp_, tn=tn, fn=fn, sensitivity=round(sens, 4), specificity=round(spec, 4), ppv=round(ppv, 4),
               npv=round(npv, 4), f1=round(f1, 4), false_alert_rate=round(fp_ / max(fp_ + tn, 1), 4))


def calib_row(y, s):
    s = np.clip(np.asarray(s, float), 1e-6, 1 - 1e-6); yv = np.asarray(y, float)
    lg = np.log(s / (1 - s))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lg[:, None], yv)
    edges = np.linspace(0, 1, 11); b = np.clip(np.digitize(s, edges) - 1, 0, 9); ece = 0.0; rel = []
    for k in range(10):
        mk = b == k
        if mk.sum():
            ece += mk.mean() * abs(yv[mk].mean() - s[mk].mean())
            rel.append({"bin": f"{edges[k]:.1f}-{edges[k + 1]:.1f}", "n": int(mk.sum()), "mean_pred": round(float(s[mk].mean()), 4),
                       "observed_rate": round(float(yv[mk].mean()), 4)})
    return {"brier": round(float(brier_score_loss(yv, s)), 4), "cal_intercept": round(float(m.intercept_[0]), 3),
            "cal_slope": round(float(m.coef_[0][0]), 3), "ece10": round(float(ece), 4), "n": len(s), "n_pos": int(yv.sum())}, rel


# ---------------------------------------------------------------- regenerate the exact URM sequences (CV + test)
def build_urm_sequences():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    fold_of = {p: folds_blob["assignment"][p][0] for p in clean_pids}
    df_rolling = pd.read_csv(ROLLING_PATH); df_rolling["patient_id"] = df_rolling["patient_id"].astype(str)
    y_lookup = {p: int(df_rolling[df_rolling["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}
    meta = pd.read_csv(METADATA_PATH).set_index("record_id")
    parity_by_pid = {str(p): float(meta.loc[int(p), "parity"]) for p in clean_pids}
    parity_pat = np.array([parity_by_pid[p] for p in clean_pids]); y_pat = np.array([y_lookup[p] for p in clean_pids])

    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    train_val_pids = sorted(p for p in clean_pids if p not in test_pids)
    trval_mask = np.array([p in set(train_val_pids) for p in clean_pids]); test_mask_pat = ~trval_mask

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_data_cv, t_del_cv = build_patient_data(p6["pred_unweighted_cv"], p6["patient_ids"], df_rolling, clean_pids, y_lookup)
    patient_data_test, t_del_test = build_patient_data(p6["pred_unweighted_test"], p6["patient_ids"], df_rolling, test_pids, y_lookup)

    completed = load_completed_units(CHECKPOINT_DIR)
    fold_scorers = {f: load_scorer_checkpoint(completed[f"Model_3_magnitude_position_fold{f}"], in_dim=2, hidden=8) for f in range(5)}
    test_scorer = load_scorer_checkpoint(completed["Model_3_magnitude_position_testmodel"], in_dim=2, hidden=8)

    m3_seq_cv = {}
    for pid in clean_pids:
        r_seq, elapsed_seq, y, T_i = patient_data_cv[pid]
        m3_seq_cv[pid] = predict_all_prefixes(fold_scorers[fold_of[pid]], r_seq, elapsed_seq, True, T_i)
    m3_seq_test = {}
    for pid in test_pids:
        r_seq, elapsed_seq, y, T_i = patient_data_test[pid]
        m3_seq_test[pid] = predict_all_prefixes(test_scorer, r_seq, elapsed_seq, True, T_i)

    # parity, OOF for CV, train_val-fit for test -- exactly phase18_deployable_fusion.py's own recipe
    p_parity_oof = np.zeros(len(clean_pids))
    for f in range(5):
        te_mask = np.array([fold_of[p] == f for p in clean_pids]); tr_mask = ~te_mask
        p_parity_oof[te_mask] = fit_parity_lr(parity_pat[tr_mask], y_pat[tr_mask], parity_pat)[te_mask]
    parity_prob_by_pid = {p: p_parity_oof[i] for i, p in enumerate(clean_pids)}
    p_parity_test = fit_parity_lr(parity_pat[trval_mask], y_pat[trval_mask], parity_pat)[test_mask_pat]
    parity_prob_test_by_pid = {p: p_parity_test[i] for i, p in enumerate(test_pids)}

    # CV: per-fold fresh fit (fit_deployable_bundle), applied to that fold's held-out patients
    fused_seq_cv = {}
    for f in range(5):
        te_pids = [p for p in clean_pids if fold_of[p] == f]; tr_pids = [p for p in clean_pids if fold_of[p] != f]
        score_fn, _ = fit_deployable_bundle(tr_pids, m3_seq_cv, parity_prob_by_pid, y_lookup)
        for pid in te_pids:
            par_p = parity_prob_by_pid[pid]
            fused_seq_cv[pid] = np.array([score_fn("A3_selected", z_k, par_p) for z_k in m3_seq_cv[pid]])

    # test: train_val -> test, once
    score_fn_test, _ = fit_deployable_bundle(train_val_pids, m3_seq_cv, parity_prob_by_pid, y_lookup)
    fused_seq_test = {}
    for pid in test_pids:
        par_p = parity_prob_test_by_pid[pid]
        fused_seq_test[pid] = np.array([score_fn_test("A3_selected", z_k, par_p) for z_k in m3_seq_test[pid]])

    return dict(clean_pids=clean_pids, fold_of=fold_of, fused_seq_cv=fused_seq_cv, t_del_cv=t_del_cv, y_lookup=y_lookup,
               test_pids=test_pids, fused_seq_test=fused_seq_test, t_del_test=t_del_test)


def horizon_scores(pids, seq, tdel, y_lookup):
    out = {h: np.zeros(len(pids)) for h in H}
    y = np.array([y_lookup[p] for p in pids])
    for i, p in enumerate(pids):
        s = seq[p]
        for h in H:
            k = eligible_prefix_length(tdel[p], h, len(s)) - 1
            out[h][i] = s[k]
    return out, y


def gate(name, pids, seq, tdel, y_lookup, ref):
    out, y = horizon_scores(pids, seq, tdel, y_lookup)
    ok = True
    for h in H:
        auc = roc_auc_score(y, out[h]); good = abs(auc - ref[h]) <= 0.001; ok &= good
        print(f"  [{name}] gate h={h:>2}m: {auc:.4f} vs frozen {ref[h]:.4f} [{'PASS' if good else 'FAIL'}]")
    return ok, out, y


def sweep_thresholds(pids, fold_of, seq, tdel, y_lookup, hours):
    y = np.array([y_lookup[p] for p in pids]); n = len(pids)
    delivery_scores = np.array([seq[p][-1] for p in pids])
    total_hours = float(sum(hours[p] for p in pids))
    rows = []
    for target in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        alerted = np.zeros(n, bool); lead = np.full(n, np.nan); fold_thr = {}
        for f in set(fold_of[p] for p in pids):
            tr = [p for p in pids if fold_of[p] != f]
            trs = np.array([seq[p][-1] for p in tr]); ytr = np.array([y_lookup[p] for p in tr])
            fold_thr[f] = float(np.percentile(trs[ytr == 1], (1 - target) * 100))
        for i, p in enumerate(pids):
            thr = fold_thr[fold_of[p]]; s = seq[p]; idx = np.where(s >= thr)[0]
            if len(idx):
                alerted[i] = True
                if y[i] == 1: lead[i] = tdel[p][idx[0]]
        yhat = alerted.astype(int)
        cm = confusion_metrics(y, yhat)
        l = lead[(y == 1) & ~np.isnan(lead)]
        rows.append({"target_sensitivity": target, "mean_threshold": round(float(np.mean(list(fold_thr.values()))), 4), **cm,
                     "false_alerts_per_patient": round(cm["fp"] / n, 4), "false_alerts_per_hour": round(cm["fp"] / total_hours, 5),
                     "median_lead_min": round(float(np.median(l)), 2) if len(l) else None,
                     "pct_ge10min": round(float(np.mean(l >= 10)), 4) if len(l) else None,
                     "pct_ge20min": round(float(np.mean(l >= 20)), 4) if len(l) else None,
                     "pct_ge30min": round(float(np.mean(l >= 30)), 4) if len(l) else None})
    return pd.DataFrame(rows), total_hours


def main():
    print("=== regenerating the exact continuous URM scores (CV + test), gated against the frozen headline numbers ===")
    d = build_urm_sequences()
    ok_cv, horiz_cv, y_cv = gate("CV", d["clean_pids"], d["fused_seq_cv"], d["t_del_cv"], d["y_lookup"], FROZEN_URM_CV)
    ok_test, horiz_test, y_test = gate("TEST", d["test_pids"], d["fused_seq_test"], d["t_del_test"], d["y_lookup"], FROZEN_URM_TEST)
    if not (ok_cv and ok_test):
        print("GATE FAILED -- stopping."); return
    print("  both gates PASSED\n")

    # ---------------- Problem 3: calibration ----------------
    print("=== Problem 3: calibration (URM, per horizon, on the gate-verified exact sequence) ===")
    calib_rows, reliab = [], {}
    for split_name, horiz, y in [("cv", horiz_cv, y_cv), ("test", horiz_test, y_test)]:
        for h in H:
            r, rel = calib_row(y, horiz[h]); r.update({"split": split_name, "horizon_min": h}); calib_rows.append(r)
            reliab[f"{split_name}_{h}m"] = rel
            print(f"  {split_name:<4} {h:>2}m: brier={r['brier']} slope={r['cal_slope']} intercept={r['cal_intercept']} ece10={r['ece10']}")
    pd.DataFrame(calib_rows).to_csv(os.path.join(OUT, "calibration.csv"), index=False)
    json.dump(reliab, open(os.path.join(OUT, "reliability_tables.json"), "w"), indent=2)

    # ---------------- Problems 4 & 5: classification metrics + threshold-performance curve (canonical CV) ----------------
    print("\n=== Problems 4-5: classification metrics + false-alert burden across target sensitivities (canonical CV) ===")
    rolling = pd.read_csv(ROLLING_PATH); rolling["patient_id"] = rolling["patient_id"].astype(str)
    hrs = (rolling.groupby("patient_id")["end_sample"].max() / (FS_HZ * 3600.0)).to_dict()
    df_sweep, total_hours = sweep_thresholds(d["clean_pids"], d["fold_of"], d["fused_seq_cv"], d["t_del_cv"], d["y_lookup"], hrs)

    ref_sens, ref_far = 0.8636, 0.6247   # deployable_fusion_operational_metrics.csv, A3_selected (published operating point)
    row80 = df_sweep[df_sweep.target_sensitivity == 0.80].iloc[0].to_dict()
    diff_sens, diff_far = abs(row80["sensitivity"] - ref_sens), abs(row80["false_alert_rate"] - ref_far)
    print(f"  cross-check vs deployable_fusion_operational_metrics.csv: sens {row80['sensitivity']} vs {ref_sens} (diff {diff_sens:.4f}), "
          f"FAR {row80['false_alert_rate']} vs {ref_far} (diff {diff_far:.4f})")
    row80["published_operational_metrics_diff"] = {"sensitivity": round(diff_sens, 4), "false_alert_rate": round(diff_far, 4)}
    print(" ", row80)
    json.dump(row80, open(os.path.join(OUT, "operating_point_80pct.json"), "w"), indent=2, default=float)

    df_sweep.to_csv(os.path.join(OUT, "threshold_performance_curve.csv"), index=False)
    print(df_sweep[["target_sensitivity", "sensitivity", "specificity", "ppv", "npv", "f1", "false_alert_rate",
                    "false_alerts_per_patient", "false_alerts_per_hour", "median_lead_min", "pct_ge20min", "pct_ge30min"]].to_string(index=False))
    print(f"\n  total monitored hours across all {len(d['clean_pids'])} patients: {total_hours:.1f}")
    print(f"\nSaved -> {OUT}/")


if __name__ == "__main__":
    main()
