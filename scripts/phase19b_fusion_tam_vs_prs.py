"""
Phase 19b -- does the deployable fusion (URM) need TAM? (docs/phase19b_tam_vs_prs_fusion_protocol.md)

U_TAM : p = alpha * p_TAM(t) + (1-alpha) * p_parity      (URM control)
U_PRS : p = alpha * p_PRS_latest_window(t) + (1-alpha) * p_parity
Same per-split function, same alpha-selection procedure for both; only the CTG score sequence differs.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.phase16_causal_attention import predict_all_prefixes, eligible_prefix_length
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data
from scripts.phase19_pooler_experiment import fast_auc, boot_all, summarize_delta

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
DATA_DIR = "data/processed_clinical"
CHECKPOINT_DIR = "results/phase16/checkpoints"
STAB_PATH = "results/phase18_fusion_ablation/deployable_fusion_resplit_sensitivity.csv"
OUT_DIR = "results/phase19b_fusion"
os.makedirs(OUT_DIR, exist_ok=True)

H = [0, 10, 20, 30]
ALPHA_GRID = np.linspace(0.0, 1.0, 21)
RESPLIT_SEEDS = [11, 22, 33, 44, 55]
PRS_LOCKED = {0: 0.6872, 10: 0.6859, 20: 0.6238, 30: 0.5828}
ARMS = ["U_TAM", "U_PRS"]


def fit_parity(par_tr, y_tr, par_all):
    sc = StandardScaler(); Xtr = sc.fit_transform(par_tr.reshape(-1, 1))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(Xtr, y_tr)
    return clf.predict_proba(sc.transform(par_all.reshape(-1, 1)))[:, 1]


def pooled_set(pids, seq, par, y):
    z, p, yy, w = [], [], [], []
    for pid in pids:
        s = seq[pid]; T = len(s)
        z.extend(s.tolist()); p.extend([par[pid]] * T); yy.extend([y[pid]] * T); w.extend([1.0 / T] * T)
    return np.array(z), np.array(p), np.array(yy), np.array(w)


def select_alpha(pids, seq, par, y):
    z, p, yy, w = pooled_set(pids, seq, par, y)
    best, best_auc = 0.5, -1.0
    for a in ALPHA_GRID:
        auc = roc_auc_score(yy, a * z + (1 - a) * p, sample_weight=w)
        if auc > best_auc:
            best_auc, best = auc, float(a)
    return best


def fit_split_models(tr_pids, seqs, par_raw, y):
    """Parity fit fresh on tr_pids (in-sample for training rows) + alpha per arm. Returns (par_prob_fn, alphas)."""
    p_all_pids = list(par_raw.keys())
    fitted = fit_parity(np.array([par_raw[p] for p in tr_pids]), np.array([y[p] for p in tr_pids]),
                        np.array([par_raw[p] for p in p_all_pids]))
    par = dict(zip(p_all_pids, fitted))
    return par, {a: select_alpha(tr_pids, seqs[a], par, y) for a in ARMS}


def horizon_scores(pids, seq, tdel, alpha, par):
    out = np.zeros((len(pids), 4))
    for j, pid in enumerate(pids):
        s = seq[pid]; T = len(s)
        for a, h in enumerate(H):
            k = eligible_prefix_length(tdel[pid], h, T) - 1
            out[j, a] = alpha * s[k] + (1 - alpha) * par[pid] if alpha is not None else s[k]
    return out


def run_cv(assign, clean_pids, seqs, tdel, par_raw, y):
    n = len(clean_pids); row = {p: i for i, p in enumerate(clean_pids)}
    res = {a: np.zeros((n, 4)) for a in ARMS}
    refs = {"TAM": np.zeros((n, 4)), "PRS": np.zeros((n, 4)), "PARITY": np.zeros((n, 4))}
    alphas = {a: [] for a in ARMS}; fold_par = {}
    for f in range(5):
        te = [p for p in clean_pids if assign[p] == f]; tr = [p for p in clean_pids if assign[p] != f]
        par, al = fit_split_models(tr, seqs, par_raw, y)
        rows = [row[p] for p in te]
        for a in ARMS:
            res[a][rows] = horizon_scores(te, seqs[a], tdel, al[a], par); alphas[a].append(al[a])
        refs["TAM"][rows] = horizon_scores(te, seqs["U_TAM"], tdel, None, par)
        refs["PRS"][rows] = horizon_scores(te, seqs["U_PRS"], tdel, None, par)
        refs["PARITY"][rows] = np.tile(np.array([par[p] for p in te])[:, None], (1, 4))
        for p in te: fold_par[p] = par[p]
        fold_par.update({p: par[p] for p in te})
    return res, refs, alphas, (assign, fold_par)


def main():
    with open(FOLDS_PATH) as fh:
        fb = json.load(fh)
    clean_pids = sorted(fb["assignment"].keys()); pids_arr = np.array(clean_pids)
    canon = {p: fb["assignment"][p][0] for p in clean_pids}
    df = pd.read_csv(ROLLING_PATH); df["patient_id"] = df["patient_id"].astype(str)
    y = {p: int(df[df["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}
    y_pat = np.array([y[p] for p in clean_pids])
    meta = pd.read_csv(METADATA_PATH).set_index("record_id")
    par_raw = {p: float(meta.loc[int(p), "parity"]) for p in clean_pids}
    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    pdata, tdel = build_patient_data(p6["pred_unweighted_cv"], p6["patient_ids"], df, clean_pids, y)
    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"])); trval = [p for p in clean_pids if p not in test_pids]
    pdata_t, tdel_t = build_patient_data(p6["pred_unweighted_test"], p6["patient_ids"], df, test_pids, y)
    y_test = np.array([y[p] for p in test_pids])

    completed = load_completed_units(CHECKPOINT_DIR)
    fscorers = {f: load_scorer_checkpoint(completed[f"Model_3_magnitude_position_fold{f}"], 2, 8) for f in range(5)}
    tscorer = load_scorer_checkpoint(completed["Model_3_magnitude_position_testmodel"], 2, 8)
    seqs = {"U_TAM": {}, "U_PRS": {}}
    for p in clean_pids:
        r, el, _, T = pdata[p]
        seqs["U_TAM"][p] = predict_all_prefixes(fscorers[canon[p]], r, el, True, T)   # frozen canonical-fold scorer
        seqs["U_PRS"][p] = r.numpy().astype(float)
    seqs_t = {"U_TAM": {}, "U_PRS": {}}
    for p in test_pids:
        r, el, _, T = pdata_t[p]
        seqs_t["U_TAM"][p] = predict_all_prefixes(tscorer, r, el, True, T)
        seqs_t["U_PRS"][p] = r.numpy().astype(float)

    print("=" * 88 + "\n  PHASE 19b -- does the deployable fusion need TAM?\n" + "=" * 88)
    res, refs, alphas, _ = run_cv(canon, clean_pids, seqs, tdel, par_raw, y)

    # ---------------- gate ----------------
    ref = pd.read_csv(STAB_PATH); ref = ref[(ref.split == "CANONICAL") & (ref.model == "A3_selected_once")].set_index("horizon_min")["auroc"]
    ok = True
    for a, h in enumerate(H):
        c = fast_auc(y_pat, res["U_TAM"][:, a]); g1 = abs(c - ref[h]) <= 0.001
        pr = fast_auc(y_pat, refs["PRS"][:, a]); g2 = abs(pr - PRS_LOCKED[h]) <= 0.001; ok &= g1 and g2
        print(f"  gate h={h:>2}m: U_TAM={c:.4f} vs Phase18-stability {ref[h]:.4f} [{'PASS' if g1 else 'FAIL'}] | PRS={pr:.4f} vs {PRS_LOCKED[h]:.4f} [{'PASS' if g2 else 'FAIL'}]")
    if not ok:
        print("  GATE FAILED -- stopping per protocol."); return
    print("  gate PASSED")

    # ---------------- canonical comparison + test ----------------
    scores = {**res, **{"TAM": refs["TAM"], "PRS_ref": refs["PRS"], "PARITY": refs["PARITY"]}}
    boot = boot_all(y_pat, scores)
    can_rows = []
    for nm in scores:
        r = {"model": nm, **{f"auroc_{h}m": round(fast_auc(y_pat, scores[nm][:, a]), 4) for a, h in enumerate(H)}}
        r["M"] = round(np.mean([r[f"auroc_{h}m"] for h in H]), 4); can_rows.append(r)
    pd.DataFrame(can_rows).to_csv(os.path.join(OUT_DIR, "canonical_auroc.csv"), index=False)
    d_can = summarize_delta(boot, "U_PRS", "U_TAM", y_pat, scores)
    pd.DataFrame([d_can]).to_csv(os.path.join(OUT_DIR, "canonical_delta_UPRS_minus_UTAM.csv"), index=False)
    print(pd.DataFrame(can_rows).to_string(index=False))
    print("  U_PRS - U_TAM:", {k: round(v, 4) for k, v in d_can.items()})
    print(f"  selected alpha per fold  U_TAM={alphas['U_TAM']}  U_PRS={alphas['U_PRS']}")

    par_t, al_t = fit_split_models(trval, {**{a: {**seqs[a]} for a in ARMS}}, par_raw, y)
    tres = {a: horizon_scores(test_pids, seqs_t[a], tdel_t, al_t[a], par_t) for a in ARMS}
    t_M = {a: np.mean([fast_auc(y_test, tres[a][:, i]) for i in range(4)]) for a in ARMS}
    print(f"  test partition: M(U_TAM)={t_M['U_TAM']:.4f}  M(U_PRS)={t_M['U_PRS']:.4f}  dM={t_M['U_PRS'] - t_M['U_TAM']:+.4f}  alpha={al_t}")
    pd.DataFrame([{"arm": a, "test_M": round(t_M[a], 4), "alpha": al_t[a],
                   **{f"test_auroc_{h}m": round(fast_auc(y_test, tres[a][:, i]), 4) for i, h in enumerate(H)}} for a in ARMS]
                 ).to_csv(os.path.join(OUT_DIR, "test_partition.csv"), index=False)

    # ---------------- resplits ----------------
    rrows = []
    for rs in RESPLIT_SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs); assign = {}
        for f, (_, te_idx) in enumerate(skf.split(pids_arr, y_pat)):
            for p in pids_arr[te_idx]: assign[p] = f
        rres, rrefs, ral, _ = run_cv(assign, clean_pids, seqs, tdel, par_raw, y)
        rb = boot_all(y_pat, rres)
        rec = {"resplit_seed": rs, "alpha_TAM_mean": round(float(np.mean(ral["U_TAM"])), 3), "alpha_PRS_mean": round(float(np.mean(ral["U_PRS"])), 3)}
        rec.update(summarize_delta(rb, "U_PRS", "U_TAM", y_pat, rres))
        rec.update({f"M_TAM": round(float(np.mean([fast_auc(y_pat, rres["U_TAM"][:, i]) for i in range(4)])), 4),
                    f"M_PRS": round(float(np.mean([fast_auc(y_pat, rres["U_PRS"][:, i]) for i in range(4)])), 4)})
        rrows.append(rec)
        print(f"  resplit {rs}: dM={rec['dM']:+.4f} (p={rec['dM_p']:.3f})  per-horizon " + " ".join(f"{h}m={rec[f'd{h}']:+.4f}" for h in H))
    dfr = pd.DataFrame(rrows); dfr.to_csv(os.path.join(OUT_DIR, "resplit_delta_UPRS_minus_UTAM.csv"), index=False)

    # ---------------- verdict (mechanical) ----------------
    dM = d_can["dM"]; p = d_can["dM_p"]
    can_min = min(d_can[f"d{h}"] for h in H); res_min = min(dfr[f"d{h}"].mean() for h in H)
    n_pos = int((dfr.dM > 0).sum()); n_ge = int((dfr.dM >= -0.005).sum()); n_lt = int((dfr.dM < -0.005).sum())
    if dM > 0 and p < 0.05 and n_pos == 5: tier = "PRS-BETTER"
    elif dM >= -0.005 and n_ge >= 4 and can_min >= -0.020 and res_min >= -0.020: tier = "NON-INFERIOR (TAM removable)"
    elif dM < -0.005 and p < 0.05 and n_lt >= 4: tier = "TAM-NEEDED"
    else: tier = "INCONCLUSIVE"
    verdict = {"tier": tier, "canonical_dM": round(dM, 4), "canonical_p": round(p, 4), "resplits_dM_gt0": n_pos,
               "resplits_dM_ge_-0.005": n_ge, "resplits_dM_lt_-0.005": n_lt,
               "canonical_worst_horizon_delta": round(can_min, 4), "resplit_mean_worst_horizon_delta": round(res_min, 4),
               "test_dM": round(t_M["U_PRS"] - t_M["U_TAM"], 4)}
    json.dump(verdict, open(os.path.join(OUT_DIR, "verdict.json"), "w"), indent=2)
    print("\n  VERDICT:", verdict)

    # ---------------- secondary: operational metrics (canonical folds, 80% target sensitivity) ----------------
    print("\n  operational (80% target sens, per-fold training-only threshold, patient-level ever-alerted):")
    op = []
    variants = ["U_TAM", "U_PRS", "TAM", "PRS_ref"]
    for v in variants:
        alerted = np.zeros(len(clean_pids), dtype=bool); leads = []; ths = []
        for f in range(5):
            te = [q for q in clean_pids if canon[q] == f]; tr = [q for q in clean_pids if canon[q] != f]
            par, al = fit_split_models(tr, seqs, par_raw, y)
            seq = seqs["U_TAM"] if v in ("U_TAM", "TAM") else seqs["U_PRS"]
            a_ = al[v] if v in ARMS else None
            def fused(q, k):
                return a_ * seq[q][k] + (1 - a_) * par[q] if a_ is not None else seq[q][k]
            trs = np.array([fused(q, len(seq[q]) - 1) for q in tr]); try_ = np.array([y[q] for q in tr])
            th = float(np.percentile(trs[try_ == 1], 20.0)); ths.append(th)
            for q in te:
                T = len(seq[q]); sc = np.array([fused(q, k) for k in range(T)]); ai = np.where(sc >= th)[0]
                if len(ai):
                    alerted[clean_pids.index(q)] = True
                    if y[q] == 1: leads.append(float(tdel[q][ai[0]]))
        lead = np.array(leads)
        row = {"system": v, "achieved_sens": round(float(alerted[y_pat == 1].mean()), 4), "FAR": round(float(alerted[y_pat == 0].mean()), 4),
               "median_lead_min": round(float(np.median(lead)), 1), "pct_ge20": round(float(np.mean(lead >= 20)), 4), "pct_ge30": round(float(np.mean(lead >= 30)), 4)}
        op.append(row); print("   ", row)
    pd.DataFrame(op).to_csv(os.path.join(OUT_DIR, "operational_metrics.csv"), index=False)
    print("\nSaved -> results/phase19b_fusion/")


if __name__ == "__main__":
    main()
