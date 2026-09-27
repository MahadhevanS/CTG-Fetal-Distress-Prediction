"""
Phase 19 Step 2 -- is TAM (Model 3) equivalent to a 2-parameter pooling?

Step 1 (phase19_attention_diagnostics.py) found TAM's learned attention is
dominated by RECENCY (logit rises ~0.15/min of elapsed time; weights are
monotone in time for essentially every patient) with only a mild positive
tilt toward higher-risk windows. That suggests:

    w_i(at prefix k)  proportional to  exp( lam * (tau_i - tau_k) + kap * r_i ),  i <= k
    z_k = sum_i w_i r_i

with two parameters (lam: recency decay per minute, kap: risk sharpening).
Fit under EXACTLY TAM's training objective (BCE of the pooled score on
every causal truncation, per-patient normalized 1/T_i), on the canonical
training folds only, by grid search. Compared, on the same folds and the
same eligible-prefix horizon logic, against:
  - PRS single window, Max, P90 (the fixed baselines)
  - TAM (frozen Phase 16 checkpoints)
  - ablations: recency-only (kap=0), risk-only (lam=0)
Exploratory diagnostic: no claim is made from this script alone; any
promising variant gets a fold-resplit check before being reported.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import get_patient_scores_at_horizon_corrected, get_eligible_window_mask
from src.models.phase16_causal_attention import predict_at_horizon_for_patients, eligible_prefix_length
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data
from scripts.phase11_bootstrap import paired_patient_bootstrap

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
CHECKPOINT_DIR = "results/phase16/checkpoints"
OUT_DIR = "results/phase19_attention"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [0, 10, 20, 30]
LAM_GRID = np.round(np.arange(0.0, 0.41, 0.025), 4)
KAP_GRID = np.arange(0.0, 16.1, 1.0)


def pad_patients(pids, patient_data):
    Tmax = max(patient_data[p][3] for p in pids)
    N = len(pids)
    R = np.zeros((N, Tmax)); TAU = np.zeros((N, Tmax)); L = np.zeros(N, dtype=int); Y = np.zeros(N)
    for j, p in enumerate(pids):
        r, el, y, T = patient_data[p]
        R[j, :T] = r.numpy(); TAU[j, :T] = el.numpy(); L[j] = T; Y[j] = float(y.item())
    return R, TAU, L, Y


def all_prefix_scores(R, TAU, L, lam, kap):
    """z[n, k] = pooled score of patient n at causal prefix k (0-indexed), shape (N, Tmax)."""
    N, T = R.shape
    tau_k = TAU[:, :, None]            # (N,K,1)
    tau_i = TAU[:, None, :]            # (N,1,I)
    logw = lam * (tau_i - tau_k) + kap * R[:, None, :]      # (N,K,I)
    idx = np.arange(T)
    causal = idx[None, :, None] >= idx[None, None, :]        # i <= k
    valid_i = idx[None, None, :] < L[:, None, None]
    logw = np.where(causal & valid_i, logw, -np.inf)
    logw = logw - logw.max(axis=2, keepdims=True)
    w = np.exp(logw); w = w / w.sum(axis=2, keepdims=True)
    return (w * R[:, None, :]).sum(axis=2)


def pooled_bce(R, TAU, L, Y, lam, kap):
    z = np.clip(all_prefix_scores(R, TAU, L, lam, kap), 1e-6, 1 - 1e-6)
    T = R.shape[1]
    valid_k = np.arange(T)[None, :] < L[:, None]
    ll = -(Y[:, None] * np.log(z) + (1 - Y[:, None]) * np.log(1 - z))
    per_pat = (ll * valid_k).sum(axis=1) / L
    return per_pat.mean()


def fit_grid(R, TAU, L, Y, lam_grid, kap_grid):
    best, best_loss = (0.0, 0.0), np.inf
    for lam in lam_grid:
        for kap in kap_grid:
            loss = pooled_bce(R, TAU, L, Y, lam, kap)
            if loss < best_loss:
                best_loss, best = loss, (float(lam), float(kap))
    return best, best_loss


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    pids_arr = np.array(clean_pids); n = len(clean_pids)
    fold_of = {p: folds_blob["assignment"][p][0] for p in clean_pids}

    df = pd.read_csv(ROLLING_PATH); df["patient_id"] = df["patient_id"].astype(str)
    y_lookup = {p: int(df[df["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}
    y_pat = np.array([y_lookup[p] for p in clean_pids])
    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    pred_cv = p6["pred_unweighted_cv"]; pid_arr_w = p6["patient_ids"]; t_del_w = p6["t_del"]
    patient_data, t_del_by_pid = build_patient_data(pred_cv, pid_arr_w, df, clean_pids, y_lookup)

    completed = load_completed_units(CHECKPOINT_DIR)
    scorers = {f: load_scorer_checkpoint(completed[f"Model_3_magnitude_position_fold{f}"], in_dim=2, hidden=8) for f in range(5)}

    print("=" * 90); print("  PHASE 19 -- is TAM a 2-parameter recency x risk pooling?"); print("=" * 90)

    variants = {"full_2param": (LAM_GRID, KAP_GRID), "recency_only": (LAM_GRID, np.array([0.0])), "risk_only": (np.array([0.0]), KAP_GRID)}
    preds = {"TAM": {h: np.zeros(n) for h in HORIZONS}}
    for v in variants:
        preds[v] = {h: np.zeros(n) for h in HORIZONS}
    fitted = []

    for f in range(5):
        te = [p for p in clean_pids if fold_of[p] == f]; tr = [p for p in clean_pids if fold_of[p] != f]
        te_mask = np.isin(pids_arr, te)
        Rtr, TAUtr, Ltr, Ytr = pad_patients(tr, patient_data)
        # TAM
        for h in HORIZONS:
            out = predict_at_horizon_for_patients(scorers[f], te, patient_data, t_del_by_pid, True, h)
            preds["TAM"][h][te_mask] = [out[p][0] for p in pids_arr[te_mask]]
        Rte, TAUte, Lte, Yte = pad_patients(te, patient_data)
        for v, (lg, kg) in variants.items():
            (lam, kap), loss = fit_grid(Rtr, TAUtr, Ltr, Ytr, lg, kg)
            fitted.append({"variant": v, "fold": f, "lam_per_min": lam, "kap": kap, "train_pooled_bce": round(loss, 4)})
            z_te = all_prefix_scores(Rte, TAUte, Lte, lam, kap)
            for h in HORIZONS:
                vals = []
                for j, p in enumerate(te):
                    k = eligible_prefix_length(t_del_by_pid[p], h, patient_data[p][3])
                    vals.append(z_te[j, k - 1])
                preds[v][h][te_mask] = vals
        print(f"  fold {f} fitted: " + ", ".join(f"{d['variant']}=(lam={d['lam_per_min']},kap={d['kap']})" for d in fitted[-3:]))

    pd.DataFrame(fitted).to_csv(os.path.join(OUT_DIR, "parametric_pooling_fitted_params.csv"), index=False)

    # baselines
    sw = {h: get_patient_scores_at_horizon_corrected(pred_cv, pid_arr_w, clean_pids, t_del_w, h) for h in HORIZONS}
    def pool(h, op):
        out = []
        for p in clean_pids:
            idx = np.where(pid_arr_w == str(p))[0]
            el = get_eligible_window_mask(t_del_w[idx], h)
            if not np.any(el): el = np.ones_like(el, dtype=bool)
            v = pred_cv[idx[el]]
            out.append(np.max(v) if op == "max" else (np.percentile(v, 90) if op == "p90" else np.mean(v)))
        return np.array(out)
    base = {"PRS_single_window": sw, "Max": {h: pool(h, "max") for h in HORIZONS}, "P90": {h: pool(h, "p90") for h in HORIZONS},
            "Mean": {h: pool(h, "mean") for h in HORIZONS}}
    allm = {**base, **preds}

    rows = []
    for h in HORIZONS:
        for name, d in allm.items():
            auc = roc_auc_score(y_pat, d[h])
            b_tam = paired_patient_bootstrap(y_pat, preds["TAM"][h], d[h], n_boot=2000, seed=42) if name != "TAM" else None
            b_prs = paired_patient_bootstrap(y_pat, sw[h], d[h], n_boot=2000, seed=42) if name != "PRS_single_window" else None
            rows.append({"horizon_min": h, "model": name, "cv_auroc": round(auc, 4),
                         "delta_vs_PRS": round(auc - roc_auc_score(y_pat, sw[h]), 4), "p_vs_PRS": round(b_prs["p_value"], 4) if b_prs else None,
                         "delta_vs_TAM": round(auc - roc_auc_score(y_pat, preds["TAM"][h]), 4), "p_vs_TAM": round(b_tam["p_value"], 4) if b_tam else None})
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT_DIR, "parametric_pooling_results.csv"), index=False)
    for h in HORIZONS:
        print(f"\n--- horizon {h}m ---")
        print(df_out[df_out.horizon_min == h].drop(columns="horizon_min").to_string(index=False))
    print(f"\nSaved -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
