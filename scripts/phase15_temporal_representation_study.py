"""
Phase 15 -- Temporal Risk Representation: Exploratory Methodology Study.
Implements docs/phase15_temporal_risk_representation_protocol.md exactly.

Sections implemented:
  4  Core representations: single-window, P90, Max, Mean, Median
  6  Temporal persistence/structure descriptors (fixed, non-outcome threshold)
  8  Operational metrics
  9  Recording-duration analysis
  10 P90 vs Max descriptive comparison

No model is retrained. All inputs are the frozen Phase 13.0A window-level
P6 predictions, verified aligned and leakage-free before this protocol was
written (Section 0 of the protocol).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import get_patient_scores_at_horizon_corrected, get_eligible_window_mask
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
DATA_DIR = "data/processed_clinical"
OUT_DIR = "results/phase15"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [0, 10, 20, 30]
REPS = ["p90", "max", "mean", "median"]


# --------------------------------------------------------------------------
# Section 4: core representations
# --------------------------------------------------------------------------
def pool(pred_arr, patient_ids, clean_pids, t_del, h_val, op):
    scores = []
    for pid in clean_pids:
        idx = np.where(patient_ids == str(pid))[0]
        t_pts = t_del[idx]
        elig = get_eligible_window_mask(t_pts, h_val)
        if not np.any(elig):
            elig = np.ones_like(t_pts, dtype=bool)
        p_elig = pred_arr[idx[elig]]
        if op == "p90":
            scores.append(float(np.percentile(p_elig, 90)))
        elif op == "max":
            scores.append(float(np.max(p_elig)))
        elif op == "mean":
            scores.append(float(np.mean(p_elig)))
        elif op == "median":
            scores.append(float(np.percentile(p_elig, 50)))
        else:
            raise ValueError(op)
    return np.array(scores)


def eligible_window_count(patient_ids, clean_pids, t_del, h_val):
    counts = []
    for pid in clean_pids:
        idx = np.where(patient_ids == str(pid))[0]
        t_pts = t_del[idx]
        elig = get_eligible_window_mask(t_pts, h_val)
        counts.append(int(np.sum(elig)) if np.any(elig) else len(t_pts))
    return np.array(counts)


# --------------------------------------------------------------------------
# Section 6: temporal persistence/structure descriptors
# --------------------------------------------------------------------------
def compute_persistence_descriptors(pred_arr, patient_ids, clean_pids, t_del, h_val, tau):
    """Returns dict of {descriptor_name: np.array} for the given fixed tau."""
    persistence, longest_run, slope, recent_minus_earlier = [], [], [], []
    for pid in clean_pids:
        idx = np.where(patient_ids == str(pid))[0]
        t_pts = t_del[idx]
        elig_mask = get_eligible_window_mask(t_pts, h_val)
        if not np.any(elig_mask):
            elig_mask = np.ones_like(t_pts, dtype=bool)
        elig_idx = idx[elig_mask]
        # chronological order = descending time_before_delivery (earliest window first)
        order = np.argsort(-t_del[elig_idx])
        seq = pred_arr[elig_idx][order]
        n = len(seq)

        above = seq > tau
        persistence.append(float(np.mean(above)))

        best = cur = 0
        for v in above:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        longest_run.append(int(best))

        if n >= 2:
            x = np.arange(n, dtype=np.float64)
            sl, _, _, _, _ = stats.linregress(x, seq)
            slope.append(float(sl))
        else:
            slope.append(0.0)

        if n >= 2:
            half = n // 2
            earlier = seq[:max(half, 1)]
            later = seq[max(half, 1):] if n - max(half, 1) > 0 else seq[-1:]
            recent_minus_earlier.append(float(np.mean(later) - np.mean(earlier)))
        else:
            recent_minus_earlier.append(0.0)

    return {
        "persistence": np.array(persistence), "longest_run": np.array(longest_run, dtype=np.float64),
        "slope": np.array(slope), "recent_minus_earlier": np.array(recent_minus_earlier),
    }


def fit_tau_train_only(pred_arr, patient_ids, tr_pids):
    tr_mask = np.isin(patient_ids, list(tr_pids))
    return float(np.percentile(pred_arr[tr_mask], 75))


# --------------------------------------------------------------------------
# Section 8: operational metrics
# --------------------------------------------------------------------------
def compute_operational_metrics(patient_scores, df, clean_pids, y_pat, target_sens=0.80):
    pos = patient_scores[y_pat == 1]
    threshold = float(np.percentile(pos, (1.0 - target_sens) * 100)) if len(pos) > 0 else 0.5
    alerted = patient_scores >= threshold
    tp = int(np.sum(alerted & (y_pat == 1)))
    fp = int(np.sum(alerted & (y_pat == 0)))
    tn = int(np.sum(~alerted & (y_pat == 0)))
    fn = int(np.sum(~alerted & (y_pat == 1)))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    far = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"threshold": round(threshold, 4), "sensitivity": round(sens, 4),
            "specificity": round(spec, 4), "false_alert_rate": round(far, 4)}


def compute_lead_time_metrics(pred_window, patient_ids, clean_pids, t_del, df, y_pat, target_sens=0.80):
    delivery_scores = pool(pred_window, patient_ids, clean_pids, t_del, 0, "max")  # last-window-only proxy not needed; use SW at h=0
    sw0 = get_patient_scores_at_horizon_corrected(pred_window, patient_ids, clean_pids, t_del, 0)
    pos = sw0[y_pat == 1]
    threshold = float(np.percentile(pos, (1.0 - target_sens) * 100)) if len(pos) > 0 else 0.5

    lead_times = []
    for i, pid in enumerate(clean_pids):
        if y_pat[i] != 1:
            continue
        idx = np.where(patient_ids == str(pid))[0]
        order = np.argsort(-t_del[idx])  # earliest first
        seq_scores = pred_window[idx][order]
        seq_t = t_del[idx][order]
        alert_pos = np.where(seq_scores >= threshold)[0]
        lead_times.append(float(seq_t[alert_pos[0]]) if len(alert_pos) > 0 else 0.0)
    lead_times = np.array(lead_times)
    return {
        "median_lead_min": round(float(np.median(lead_times)), 2) if len(lead_times) else 0.0,
        "pct_detected_ge20min": round(float(np.mean(lead_times >= 20)), 4) if len(lead_times) else 0.0,
        "pct_detected_ge30min": round(float(np.mean(lead_times >= 30)), 4) if len(lead_times) else 0.0,
    }


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_ids = p6["patient_ids"]
    t_del = p6["t_del"]
    pred_cv = p6["pred_unweighted_cv"]
    pred_test = p6["pred_unweighted_test"]

    df = pd.read_csv(ROLLING_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    y_pat = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in clean_pids])

    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    train_val_pids = [p for p in clean_pids if p not in test_pids]
    y_test_pat = np.array([df[df["patient_id"] == p]["primary_label_715"].iloc[0] for p in test_pids])

    print("================================================================================")
    print("  PHASE 15 -- TEMPORAL RISK REPRESENTATION STUDY                                  ")
    print("================================================================================")

    # ============================ Section 4: core table ============================
    core_rows = []
    op_rows = []
    sw_cv_by_h, rep_cv_by_h = {}, {h: {} for h in HORIZONS}
    for h in HORIZONS:
        sw_cv = get_patient_scores_at_horizon_corrected(pred_cv, patient_ids, clean_pids, t_del, h)
        sw_test = get_patient_scores_at_horizon_corrected(pred_test, patient_ids, test_pids, t_del, h)
        sw_cv_by_h[h] = sw_cv
        auc_sw_cv = roc_auc_score(y_pat, sw_cv)
        auc_sw_test = roc_auc_score(y_test_pat, sw_test)

        core_rows.append({"horizon_min": h, "representation": "single_window", "cv_auroc": round(auc_sw_cv, 4),
                           "cv_delta": 0.0, "cv_p": None, "cv_ci_low": None, "cv_ci_high": None,
                           "test_auroc": round(auc_sw_test, 4), "test_delta": 0.0, "test_p": None})
        op_sw = compute_operational_metrics(sw_cv, df, clean_pids, y_pat)
        lead_sw = compute_lead_time_metrics(pred_cv, patient_ids, clean_pids, t_del, df, y_pat)
        op_rows.append({"horizon_min": h, "representation": "single_window", **op_sw, **lead_sw})

        for op in REPS:
            cand_cv = pool(pred_cv, patient_ids, clean_pids, t_del, h, op)
            cand_test = pool(pred_test, patient_ids, test_pids, t_del, h, op)
            rep_cv_by_h[h][op] = cand_cv
            auc_cv = roc_auc_score(y_pat, cand_cv)
            auc_test = roc_auc_score(y_test_pat, cand_test)
            auprc_cv = average_precision_score(y_pat, cand_cv)

            boot_cv = paired_patient_bootstrap(y_pat, sw_cv, cand_cv, n_boot=2000, seed=42)
            pdel_cv, _, _ = delong_roc_test(y_pat, cand_cv, sw_cv)
            boot_test = paired_patient_bootstrap(y_test_pat, sw_test, cand_test, n_boot=2000, seed=42)
            pdel_test, _, _ = delong_roc_test(y_test_pat, cand_test, sw_test)

            core_rows.append({
                "horizon_min": h, "representation": op, "cv_auroc": round(auc_cv, 4),
                "cv_delta": round(auc_cv - auc_sw_cv, 4), "cv_p": round(boot_cv["p_value"], 4),
                "cv_ci_low": round(boot_cv["ci_95_low"], 4), "cv_ci_high": round(boot_cv["ci_95_high"], 4),
                "cv_delong_p": round(pdel_cv, 4), "cv_auprc": round(auprc_cv, 4),
                "test_auroc": round(auc_test, 4), "test_delta": round(auc_test - auc_sw_test, 4),
                "test_p": round(boot_test["p_value"], 4), "test_delong_p": round(pdel_test, 4),
            })
            op_metrics = compute_operational_metrics(cand_cv, df, clean_pids, y_pat)
            # representation-specific alert score for lead-time: use the pooled score itself as the alerting signal
            lead_metrics_rep = compute_lead_time_for_representation(pred_cv, patient_ids, clean_pids, t_del, df, y_pat, op)
            op_rows.append({"horizon_min": h, "representation": op, **op_metrics, **lead_metrics_rep})

            print(f"h={h:>3}m {op:>7s}  CV: sw={auc_sw_cv:.4f} rep={auc_cv:.4f} (d{auc_cv-auc_sw_cv:+.4f} p={boot_cv['p_value']:.3f})  "
                  f"TEST: sw={auc_sw_test:.4f} rep={auc_test:.4f} (d{auc_test-auc_sw_test:+.4f} p={boot_test['p_value']:.3f})")

    pd.DataFrame(core_rows).to_csv(os.path.join(OUT_DIR, "phase15_core_representations.csv"), index=False)
    pd.DataFrame(op_rows).to_csv(os.path.join(OUT_DIR, "phase15_operational_metrics.csv"), index=False)

    # ============================ Section 6: persistence descriptors ============================
    print("\n--- Section 6: temporal persistence/structure descriptors (delivery, >=30m) ---")
    persist_rows = []
    for h in [0, 30]:
        desc_cv = {name: np.zeros(len(clean_pids)) for name in ["persistence", "longest_run", "slope", "recent_minus_earlier"]}
        taus_used = []
        for f_idx in range(5):
            te_pids = [p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx]
            tr_pids = [p for p in clean_pids if p not in te_pids]
            tau = fit_tau_train_only(pred_cv, patient_ids, tr_pids)
            taus_used.append(tau)
            te_mask = np.array([p in te_pids for p in clean_pids])
            descs = compute_persistence_descriptors(pred_cv, patient_ids, te_pids, t_del, h, tau)
            for name in desc_cv:
                desc_cv[name][te_mask] = descs[name]

        tau_test = fit_tau_train_only(pred_cv, patient_ids, train_val_pids)
        descs_test = compute_persistence_descriptors(pred_test, patient_ids, test_pids, t_del, h, tau_test)

        sw_cv = sw_cv_by_h[h]
        auc_sw_cv = roc_auc_score(y_pat, sw_cv)
        for name in desc_cv:
            auc_cv = roc_auc_score(y_pat, desc_cv[name])
            auc_test = roc_auc_score(y_test_pat, descs_test[name])
            boot_cv = paired_patient_bootstrap(y_pat, sw_cv, desc_cv[name], n_boot=2000, seed=42)
            row = {"horizon_min": h, "descriptor": name, "tau_folds": [round(t, 4) for t in taus_used], "tau_test": round(tau_test, 4),
                   "cv_auroc": round(auc_cv, 4), "cv_delta_vs_sw": round(auc_cv - auc_sw_cv, 4),
                   "cv_p": round(boot_cv["p_value"], 4), "test_auroc": round(auc_test, 4)}
            persist_rows.append(row)
            print(f"h={h:>3}m {name:>20s}  CV AUROC={auc_cv:.4f} (d{auc_cv-auc_sw_cv:+.4f} p={boot_cv['p_value']:.3f})  TEST AUROC={auc_test:.4f}")

    pd.DataFrame(persist_rows).to_csv(os.path.join(OUT_DIR, "phase15_persistence_descriptors.csv"), index=False)

    # ============================ Section 9: recording-duration analysis ============================
    print("\n--- Section 9: recording-duration analysis (delivery horizon) ---")
    n_windows_h0 = eligible_window_count(patient_ids, clean_pids, t_del, 0)
    duration_rows = []
    sw0 = sw_cv_by_h[0]
    for op in REPS:
        rep_vals = rep_cv_by_h[0][op]
        r_val, p_val = stats.pearsonr(n_windows_h0, rep_vals)
        r_delta, p_delta = stats.pearsonr(n_windows_h0, rep_vals - sw0)
        duration_rows.append({"representation": op, "pearson_r_value_vs_nwindows": round(r_val, 4), "p_value": round(p_val, 4),
                               "pearson_r_delta_vs_nwindows": round(r_delta, 4), "p_delta": round(p_delta, 4)})
        print(f"  {op:>7s}: r(value, n_windows)={r_val:+.4f} (p={p_val:.3f})   r(delta_vs_sw, n_windows)={r_delta:+.4f} (p={p_delta:.3f})")
    pd.DataFrame(duration_rows).to_csv(os.path.join(OUT_DIR, "phase15_duration_analysis.csv"), index=False)

    # ============================ Section 10: P90 vs Max descriptive ============================
    print("\n--- Section 10: P90 vs Max descriptive comparison (delivery horizon) ---")
    p90_h0 = rep_cv_by_h[0]["p90"]
    max_h0 = rep_cv_by_h[0]["max"]
    gap = p90_h0 - max_h0
    gap_pos = gap[y_pat == 1]
    gap_neg = gap[y_pat == 0]
    p90_max_summary = {
        "mean_gap_all": round(float(np.mean(gap)), 4), "median_gap_all": round(float(np.median(gap)), 4),
        "mean_gap_positive_patients": round(float(np.mean(gap_pos)), 4), "mean_gap_negative_patients": round(float(np.mean(gap_neg)), 4),
        "frac_patients_p90_equals_max_within_0.01": round(float(np.mean(np.abs(gap) < 0.01)), 4),
    }
    print(f"  mean(P90-Max)={p90_max_summary['mean_gap_all']:.4f}  positive-patients={p90_max_summary['mean_gap_positive_patients']:.4f}  "
          f"negative-patients={p90_max_summary['mean_gap_negative_patients']:.4f}  frac(P90~=Max)={p90_max_summary['frac_patients_p90_equals_max_within_0.01']:.3f}")
    with open(os.path.join(OUT_DIR, "phase15_p90_vs_max_descriptive.json"), "w") as fh:
        json.dump(p90_max_summary, fh, indent=2)

    print(f"\nSaved all Phase 15 outputs -> {OUT_DIR}/")


def compute_lead_time_for_representation(pred_window, patient_ids, clean_pids, t_del, df, y_pat, op, target_sens=0.80):
    """Lead-time metrics using the representation's own rolling pooled score as the alerting signal at each point in time."""
    # Rolling alert signal: at each causal instant, use pool(op) over windows observed so far (t_i >= running "now").
    # Approximate "now" sweep using each patient's own observed t_del values as the candidate alert-check instants.
    delivery_scores = []
    for pid in clean_pids:
        idx = np.where(patient_ids == str(pid))[0]
        t_pts = t_del[idx]
        elig = get_eligible_window_mask(t_pts, 0)
        p_elig = pred_window[idx[elig]]
        if op == "p90":
            delivery_scores.append(float(np.percentile(p_elig, 90)))
        elif op == "max":
            delivery_scores.append(float(np.max(p_elig)))
        elif op == "mean":
            delivery_scores.append(float(np.mean(p_elig)))
        else:
            delivery_scores.append(float(np.percentile(p_elig, 50)))
    delivery_scores = np.array(delivery_scores)
    pos = delivery_scores[y_pat == 1]
    threshold = float(np.percentile(pos, (1.0 - target_sens) * 100)) if len(pos) > 0 else 0.5

    lead_times = []
    for i, pid in enumerate(clean_pids):
        if y_pat[i] != 1:
            continue
        idx = np.where(patient_ids == str(pid))[0]
        order = np.argsort(-t_del[idx])
        seq_t = t_del[idx][order]
        seq_scores = pred_window[idx][order]
        alerted_at = None
        for k in range(len(seq_t)):
            causal_scores = seq_scores[:k + 1]
            if op == "p90":
                s = np.percentile(causal_scores, 90)
            elif op == "max":
                s = np.max(causal_scores)
            elif op == "mean":
                s = np.mean(causal_scores)
            else:
                s = np.percentile(causal_scores, 50)
            if s >= threshold:
                alerted_at = seq_t[k]
                break
        lead_times.append(float(alerted_at) if alerted_at is not None else 0.0)
    lead_times = np.array(lead_times)
    return {
        "median_lead_min": round(float(np.median(lead_times)), 2) if len(lead_times) else 0.0,
        "pct_detected_ge20min": round(float(np.mean(lead_times >= 20)), 4) if len(lead_times) else 0.0,
        "pct_detected_ge30min": round(float(np.mean(lead_times >= 30)), 4) if len(lead_times) else 0.0,
    }


if __name__ == "__main__":
    main()
