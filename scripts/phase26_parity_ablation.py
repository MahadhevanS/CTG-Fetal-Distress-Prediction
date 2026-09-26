"""
Phase 26, Problem 6 -- does parity carry independent information, or is it a proxy / does the fusion mechanism
itself manufacture an apparent gain from any low-information covariate? (docs/phase26_full_clinical_evaluation_protocol.md
section 5). Uses the frozen per-fold TAM checkpoints (the literal deployed CTG component); only the covariate
model changes across arms; fusion mechanism (alpha grid, sample-weighted pooled training AUROC) unchanged from URM.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from src.models.phase16_causal_attention import predict_all_prefixes, eligible_prefix_length
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data
from scripts.phase11_bootstrap import paired_patient_bootstrap, fast_auc

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
METADATA_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
CHECKPOINT_DIR = "results/phase16/checkpoints"
OUT = "results/phase26_clinical_evaluation"; os.makedirs(OUT, exist_ok=True)
H = [0, 10, 20, 30]
ALPHA_GRID = np.linspace(0.0, 1.0, 21)
RNG_SEED = 42


def fit_lr_1d(Xtr_raw, ytr, Xall_raw):
    sc = StandardScaler(); a = sc.fit_transform(Xtr_raw.reshape(-1, 1))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(a, ytr)
    return clf.predict_proba(sc.transform(Xall_raw.reshape(-1, 1)))[:, 1]


def fit_lr_nd(Xtr_raw, ytr, Xall_raw):
    sc = StandardScaler(); a = sc.fit_transform(Xtr_raw)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(a, ytr)
    return clf.predict_proba(sc.transform(Xall_raw))[:, 1]


def select_alpha(tr_pids, m3_seq, cov_prob, y_lookup):
    z, c, yy, w = [], [], [], []
    for pid in tr_pids:
        s = m3_seq[pid]; T = len(s)
        z.extend(s.tolist()); c.extend([cov_prob[pid]] * T); yy.extend([y_lookup[pid]] * T); w.extend([1.0 / T] * T)
    z, c, yy, w = map(np.array, (z, c, yy, w))
    best, best_auc = 0.5, -1.0
    for a in ALPHA_GRID:
        auc = roc_auc_score(yy, a * z + (1 - a) * c, sample_weight=w)
        if auc > best_auc: best_auc, best = auc, float(a)
    return best


def horizon_scores(pids, m3_seq, cov_prob, alpha, tdel):
    out = {h: np.zeros(len(pids)) for h in H}
    for i, p in enumerate(pids):
        s = m3_seq[p]
        for h in H:
            k = eligible_prefix_length(tdel[p], h, len(s)) - 1
            out[h][i] = alpha * s[k] + (1 - alpha) * cov_prob[p] if alpha is not None else s[k]
    return out


def run_arm(name, clean_pids, fold_of, m3_seq, tdel, y_lookup, cov_builder, ctg_only=False, parity_only=False):
    n = len(clean_pids); horiz = {h: np.zeros(n) for h in H}; alphas = []
    for f in range(5):
        te = [p for p in clean_pids if fold_of[p] == f]; tr = [p for p in clean_pids if fold_of[p] != f]
        cov_prob = cov_builder(tr, clean_pids, f) if not ctg_only else {p: 0.0 for p in clean_pids}
        if parity_only:
            alpha = 0.0
        elif ctg_only:
            alpha = 1.0
        else:
            alpha = select_alpha(tr, m3_seq, cov_prob, y_lookup)
        alphas.append(alpha)
        te_rows = [clean_pids.index(p) for p in te]
        hs = horizon_scores(te, m3_seq, cov_prob, alpha, tdel)
        for h in H:
            horiz[h][te_rows] = hs[h][[te.index(p) for p in te]]
    return horiz, alphas


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    fold_of = {p: folds_blob["assignment"][p][0] for p in clean_pids}
    df = pd.read_csv(ROLLING_PATH); df["patient_id"] = df["patient_id"].astype(str)
    y_lookup = {p: int(df[df["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}
    y_pat = np.array([y_lookup[p] for p in clean_pids])

    meta = pd.read_csv(METADATA_PATH).set_index("record_id")
    parity_raw = {p: float(meta.loc[int(p), "parity"]) for p in clean_pids}
    age_raw = {p: float(meta.loc[int(p), "age"]) for p in clean_pids}
    gest_raw = {p: float(meta.loc[int(p), "gest. weeks"]) for p in clean_pids}
    grav_raw = {p: meta.loc[int(p), "gravidity"] for p in clean_pids}   # may be NaN (4 missing)
    rng = np.random.default_rng(RNG_SEED)
    random_demo = {p: v for p, v in zip(clean_pids, rng.normal(size=len(clean_pids)))}   # ONE fixed draw per patient

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_data_cv, t_del_cv = build_patient_data(p6["pred_unweighted_cv"], p6["patient_ids"], df, clean_pids, y_lookup)
    completed = load_completed_units(CHECKPOINT_DIR)
    fold_scorers = {f: load_scorer_checkpoint(completed[f"Model_3_magnitude_position_fold{f}"], in_dim=2, hidden=8) for f in range(5)}
    m3_seq = {}
    for pid in clean_pids:
        r_seq, elapsed_seq, y, T_i = patient_data_cv[pid]
        m3_seq[pid] = predict_all_prefixes(fold_scorers[fold_of[pid]], r_seq, elapsed_seq, True, T_i)

    def cov_parity(tr, allp, f):
        pred = fit_lr_1d(np.array([parity_raw[p] for p in tr]), np.array([y_lookup[p] for p in tr]), np.array([parity_raw[p] for p in allp]))
        return dict(zip(allp, pred))

    def cov_shuffled_parity(tr, allp, f):
        rng_f = np.random.default_rng(1000 + f)
        shuffled_vals = np.array([parity_raw[p] for p in tr]); rng_f.shuffle(shuffled_vals)   # permute training patients' parity among themselves
        pred = fit_lr_1d(shuffled_vals, np.array([y_lookup[p] for p in tr]), np.array([parity_raw[p] for p in allp]))
        return dict(zip(allp, pred))

    def cov_random_demo(tr, allp, f):
        pred = fit_lr_1d(np.array([random_demo[p] for p in tr]), np.array([y_lookup[p] for p in tr]), np.array([random_demo[p] for p in allp]))
        return dict(zip(allp, pred))

    def cov_maternal(tr, allp, f):
        cols = lambda pids: np.column_stack([[parity_raw[p] for p in pids], [age_raw[p] for p in pids], [gest_raw[p] for p in pids],
                                             [grav_raw[p] for p in pids]])
        Xtr = cols(tr); med = np.nanmedian(Xtr, axis=0); Xtr = np.where(np.isnan(Xtr), med, Xtr)
        Xall = cols(allp); Xall = np.where(np.isnan(Xall), med, Xall)
        pred = fit_lr_nd(Xtr, np.array([y_lookup[p] for p in tr]), Xall)
        return dict(zip(allp, pred))

    arms = {
        "CTG only": (None, dict(ctg_only=True)),
        "Parity only": (cov_parity, dict(parity_only=True)),
        "CTG + parity (= URM)": (cov_parity, {}),
        "CTG + shuffled parity": (cov_shuffled_parity, {}),
        "CTG + random demographic var.": (cov_random_demo, {}),
        "CTG + additional maternal vars.": (cov_maternal, {}),
    }
    scores = {}; alphas_by_arm = {}
    for name, (builder, kw) in arms.items():
        horiz, alphas = run_arm(name, clean_pids, fold_of, m3_seq, t_del_cv, y_lookup, builder, **kw)
        scores[name] = np.column_stack([horiz[h] for h in H]); alphas_by_arm[name] = alphas
        m = np.mean([fast_auc(y_pat, horiz[h]) for h in H])
        print(f"  {name:<32} M={m:.4f}  per-horizon " + " ".join(f"{h}m={fast_auc(y_pat, horiz[h]):.4f}" for h in H) +
              f"  alpha={[round(a,2) for a in alphas]}", flush=True)

    rows = []
    for name in arms:
        m_arm = np.mean([fast_auc(y_pat, scores[name][:, i]) for i in range(4)])
        rec = {"arm": name, "M": round(m_arm, 4), **{f"auc_{h}m": round(fast_auc(y_pat, scores[name][:, i]), 4) for i, h in enumerate(H)},
               "alpha_per_fold": [round(a, 2) for a in alphas_by_arm[name]]}
        for ref in ("CTG only", "CTG + parity (= URM)"):
            if name == ref: continue
            b = paired_patient_bootstrap(y_pat, scores[ref][:, 0], scores[name][:, 0], n_boot=2000, seed=42)
            dM = m_arm - np.mean([fast_auc(y_pat, scores[ref][:, i]) for i in range(4)])
            rec[f"dM_vs_{ref}"] = round(dM, 4)
            rec[f"p_delivery_vs_{ref}"] = round(b["p_value"], 4)
            rec[f"ci_delivery_vs_{ref}"] = [round(b["ci_95_low"], 4), round(b["ci_95_high"], 4)]
        rows.append(rec)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT, "parity_ablation.csv"), index=False)
    json.dump(rows, open(os.path.join(OUT, "parity_ablation.json"), "w"), indent=2, default=float)
    print("\nSaved ->", OUT)


if __name__ == "__main__":
    main()
