"""
Phase 19 -- adaptive temporal pooling, Routes 1 and 2 (docs/phase19_adaptive_pooling_protocol.md, frozen
before this file was written).

Arms: A0 (control = TAM inputs + pooled BCE), A1 (+4 reliability cues), A2 (horizon-grid ranking loss),
A3 (both). One vectorized full-batch trainer replicates train_scorer's optimisation exactly (Adam lr .01,
wd 1e-4, max 200 epochs, patience 10 on the arm's own objective on an inner-validation split); nothing is tuned.

Stages:  canonical  -> sanity gates, canonical CV + held-out test, saves scores
         resplit    -> 5 independent fold resplits (PRS scores frozen; only the pooler is retrained)
         verdict    -> applies the pre-registered decision rule mechanically
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import get_patient_scores_at_horizon_corrected
from src.models.phase16_causal_attention import (AttentionScorer, eligible_prefix_length,
                                                  predict_at_horizon_for_patients)
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data, carve_inner_validation

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
X40_PATH = "results/phase9c_state_trajectory/state_trajectory_features.npz"
X40_NAMES_PATH = "results/phase9c_state_trajectory/state_trajectory_feature_names.json"
DATA_DIR = "data/processed_clinical"
CHECKPOINT_DIR = "results/phase16/checkpoints"
OUT_DIR = "results/phase19_pooler"
os.makedirs(OUT_DIR, exist_ok=True)

H_EVAL = [0, 10, 20, 30]
H_TRAIN = list(range(0, 36, 5))
SEEDS = [42, 43, 44]
RESPLIT_SEEDS = [11, 22, 33, 44, 55]
ARMS = {"A0": dict(ext=False, obj="bce"), "A1": dict(ext=True, obj="bce"),
        "A2": dict(ext=False, obj="rank"), "A3": dict(ext=True, obj="rank")}
FROZEN_TAM_CV = {0: 0.7216, 10: 0.6683, 20: 0.6015, 30: 0.5976}
GATE_TOL = 0.006
CUE_COLS = ["dom_sev_uterine", "dom_sev_coupling"]
FS_HZ = 4.0
MAX_EPOCHS, PATIENCE, LR, WD = 200, 10, 0.01, 1e-4


# ---------------------------------------------------------------- data
class Seqs:
    pass


def build_seqs(pred_arr, pid_arr, df, X40, col_idx, pids, y_lookup):
    tdel_all = df["time_before_delivery_min"].values
    ss_all = df["start_sample"].values
    recs = []
    for pid in pids:
        idx = np.where(pid_arr == str(pid))[0]
        tv = tdel_all[idx]
        o = np.argsort(-tv); io = idx[o]
        r = pred_arr[io].astype(np.float32)
        el = ss_all[io] / (FS_HZ * 60.0); el = (el - el[0]).astype(np.float32)
        d = np.r_[0.0, np.abs(np.diff(r))].astype(np.float32)
        s3 = np.array([np.std(r[max(0, t - 2):t + 1]) for t in range(len(r))], dtype=np.float32)
        cues = np.column_stack([d, s3, X40[io, col_idx[0]], X40[io, col_idx[1]]]).astype(np.float32)
        recs.append((r, el, cues, tv[o], y_lookup[pid]))
    N = len(pids); Tmax = max(len(x[0]) for x in recs)
    S = Seqs(); S.pids = list(pids)
    S.R = torch.zeros(N, Tmax); S.EL = torch.zeros(N, Tmax); S.CUE = torch.zeros(N, Tmax, 4)
    S.L = torch.zeros(N, dtype=torch.long); S.Y = torch.zeros(N)
    S.KTR = torch.zeros(N, len(H_TRAIN), dtype=torch.long); S.KEV = torch.zeros(N, len(H_EVAL), dtype=torch.long)
    for j, (r, el, cues, tdel, y) in enumerate(recs):
        T = len(r)
        S.R[j, :T] = torch.tensor(r); S.EL[j, :T] = torch.tensor(el); S.CUE[j, :T] = torch.tensor(cues)
        S.L[j] = T; S.Y[j] = float(y)
        for a, h in enumerate(H_TRAIN):
            S.KTR[j, a] = eligible_prefix_length(tdel, h, T) - 1
        for a, h in enumerate(H_EVAL):
            S.KEV[j, a] = eligible_prefix_length(tdel, h, T) - 1
    S.row = {p: j for j, p in enumerate(S.pids)}
    return S


def cue_stats(S, rows):
    """mean/std of the 4 cues over the valid windows of the given patient rows (training patients only)."""
    vals = [S.CUE[j, :int(S.L[j])] for j in rows]
    v = torch.cat(vals, 0)
    return v.mean(0), v.std(0).clamp_min(1e-6)


def make_inputs(S, ext, mu=None, sd=None):
    base = torch.stack([S.R, S.EL / 60.0], -1)
    if not ext:
        return base
    return torch.cat([base, (S.CUE - mu) / sd], -1)


# ---------------------------------------------------------------- model / objectives
def pooled_all_prefix(scorer, X, R, L):
    N, T, _ = X.shape
    e = scorer(X)                                            # (N,T)
    idx = torch.arange(T)
    mask = (idx[None, :, None] >= idx[None, None, :]) & (idx[None, None, :] < L[:, None, None])
    logits = e[:, None, :].expand(N, T, T).masked_fill(~mask, float("-inf"))
    w = torch.softmax(logits, dim=2)
    return (w * R[:, None, :]).sum(-1)                        # (N,K) pooled score at every causal prefix


def obj_bce(z, S, sub):
    zc = z.clamp(1e-6, 1 - 1e-6)
    Y = S.Y[:, None]
    ll = -(Y * torch.log(zc) + (1 - Y) * torch.log(1 - zc))
    valid = torch.arange(z.shape[1])[None, :] < S.L[:, None]
    per = (ll * valid).sum(1) / S.L
    return per[sub].mean()


def obj_rank(z, S, sub):
    zs = z.gather(1, S.KTR).clamp(1e-6, 1 - 1e-6)
    lz = torch.log(zs / (1 - zs))                             # (N,H)
    ysub = S.Y[sub]
    pos = sub[ysub == 1]; neg = sub[ysub == 0]
    tot = 0.0
    for a in range(lz.shape[1]):
        diff = lz[pos, a][:, None] - lz[neg, a][None, :]
        tot = tot + F.softplus(-diff).mean()
    return tot / lz.shape[1]


def fit_arm(S, X, tr_idx, val_idx, obj, seed):
    torch.manual_seed(seed)
    sc = AttentionScorer(X.shape[-1], hidden=8)
    opt = torch.optim.Adam(sc.parameters(), lr=LR, weight_decay=WD)
    fn = obj_bce if obj == "bce" else obj_rank
    best, best_state, bad, n_ep = float("inf"), None, 0, 0
    for ep in range(MAX_EPOCHS):
        n_ep = ep + 1
        sc.train(); opt.zero_grad()
        loss = fn(pooled_all_prefix(sc, X, S.R, S.L), S, tr_idx)
        loss.backward(); opt.step()
        sc.eval()
        with torch.no_grad():
            v = fn(pooled_all_prefix(sc, X, S.R, S.L), S, val_idx).item()
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in sc.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    sc.load_state_dict(best_state); sc.eval()
    return sc, n_ep, best


def predict_eval(sc, S, X):
    with torch.no_grad():
        z = pooled_all_prefix(sc, X, S.R, S.L)
        return z.gather(1, S.KEV).numpy()                     # (N,4)


# ---------------------------------------------------------------- statistics
def fast_auc(y, s):
    npos = int(y.sum()); nneg = len(y) - npos
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def boot_all(y, scores, n_boot=2000, seed=42):
    """scores: name -> (N,4). One shared set of patient resamples for every model (paired)."""
    rng = np.random.default_rng(seed); n = len(y); names = list(scores)
    out = {nm: [] for nm in names}
    for _ in range(n_boot):
        bi = rng.choice(n, n, replace=True); yb = y[bi]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        for nm in names:
            out[nm].append([fast_auc(yb, scores[nm][bi, h]) for h in range(len(H_EVAL))])
    return {nm: np.array(v) for nm, v in out.items()}


def summarize_delta(boot, arm, ref, y, scores):
    d = boot[arm] - boot[ref]                                  # (B,4)
    dM = d.mean(1)
    def pv(x): return float(min(1.0, 2 * min(np.mean(x <= 0), np.mean(x >= 0))))
    rec = {"dM": float(np.mean([fast_auc(y, scores[arm][:, h]) - fast_auc(y, scores[ref][:, h]) for h in range(4)])),
           "dM_p": pv(dM), "dM_ci_lo": float(np.percentile(dM, 2.5)), "dM_ci_hi": float(np.percentile(dM, 97.5))}
    for a, h in enumerate(H_EVAL):
        rec[f"d{h}"] = float(fast_auc(y, scores[arm][:, a]) - fast_auc(y, scores[ref][:, a]))
        rec[f"p{h}"] = pv(d[:, a])
    return rec


# ---------------------------------------------------------------- one split (canonical or resplit)
def run_split(S, y_lookup, clean_pids, assign, offset, arms, seeds, log, tag):
    n = len(clean_pids)
    oof = {a: {s: np.zeros((n, 4)) for s in seeds} for a in arms}
    for f in range(5):
        te = [p for p in clean_pids if assign[p] == f]
        tr_all = [p for p in clean_pids if assign[p] != f]
        inner_tr, inner_val = carve_inner_validation(tr_all, y_lookup, seed=42 + f + offset)
        tr_idx = torch.tensor([S.row[p] for p in inner_tr]); val_idx = torch.tensor([S.row[p] for p in inner_val])
        te_rows = np.array([S.row[p] for p in te])
        mu, sd = cue_stats(S, [S.row[p] for p in tr_all])
        for a in arms:
            X = make_inputs(S, ARMS[a]["ext"], mu, sd)
            for s in seeds:
                sc, n_ep, best = fit_arm(S, X, tr_idx, val_idx, ARMS[a]["obj"], s)
                oof[a][s][te_rows] = predict_eval(sc, S, X)[te_rows]
                log.append({"split": tag, "arm": a, "seed": s, "fold": f, "epochs": n_ep,
                            "cap_hit": n_ep >= MAX_EPOCHS, "best_val_obj": round(best, 5)})
    return oof


def ensemble(oof_a, seeds):
    return np.mean([oof_a[s] for s in seeds], axis=0)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all", choices=["canonical", "resplit", "verdict", "all"])
    args = ap.parse_args()

    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys()); pids_arr = np.array(clean_pids)
    canon_assign = {p: folds_blob["assignment"][p][0] for p in clean_pids}
    df = pd.read_csv(ROLLING_PATH); df["patient_id"] = df["patient_id"].astype(str)
    y_lookup = {p: int(df[df["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}
    y_pat = np.array([y_lookup[p] for p in clean_pids])
    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    pid_arr, t_del_arr = p6["patient_ids"], p6["t_del"]
    pred_cv, pred_test = p6["pred_unweighted_cv"], p6["pred_unweighted_test"]
    X40 = np.load(X40_PATH)["X_state_trajectory"]
    names = json.load(open(X40_NAMES_PATH)); col_idx = [names.index(c) for c in CUE_COLS]
    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    train_val_pids = [p for p in clean_pids if p not in test_pids]
    y_test = np.array([y_lookup[p] for p in test_pids])

    S = build_seqs(pred_cv, pid_arr, df, X40, col_idx, clean_pids, y_lookup)
    S_test = build_seqs(pred_test, pid_arr, df, X40, col_idx, test_pids, y_lookup)
    prs_cv = np.column_stack([get_patient_scores_at_horizon_corrected(pred_cv, pid_arr, clean_pids, t_del_arr, h) for h in H_EVAL])
    prs_test = np.column_stack([get_patient_scores_at_horizon_corrected(pred_test, pid_arr, test_pids, t_del_arr, h) for h in H_EVAL])

    if args.stage in ("canonical", "all"):
        print("=" * 88 + "\n  PHASE 19 -- canonical stage\n" + "=" * 88)
        # ---- equivalence check: vectorized pooling == frozen TAM inference ----
        completed = load_completed_units(CHECKPOINT_DIR)
        pdata, tdel_by = build_patient_data(pred_cv, pid_arr, df, clean_pids, y_lookup)
        te0 = [p for p in clean_pids if canon_assign[p] == 0]
        sc0 = load_scorer_checkpoint(completed["Model_3_magnitude_position_fold0"], in_dim=2, hidden=8)
        X0 = make_inputs(S, False)
        maxdiff = 0.0
        with torch.no_grad():
            zv = pooled_all_prefix(sc0, X0, S.R, S.L)
        for h_i, h in enumerate(H_EVAL):
            ref = predict_at_horizon_for_patients(sc0, te0, pdata, tdel_by, True, h)
            for p in te0:
                maxdiff = max(maxdiff, abs(ref[p][0] - float(zv[S.row[p], S.KEV[S.row[p], h_i]])))
        print(f"  equivalence check (vectorized pooling vs frozen TAM inference, fold 0): max|diff| = {maxdiff:.2e}")
        assert maxdiff < 1e-4, "vectorized pooling does not match the frozen implementation -- stop"

        # ---- sanity gate: A0 seed 42 reproduces frozen TAM ----
        glog = []
        g = run_split(S, y_lookup, clean_pids, canon_assign, 0, ["A0"], [42], glog, "gate")
        ok = True
        for a, h in enumerate(H_EVAL):
            auc = fast_auc(y_pat, g["A0"][42][:, a])
            good = abs(auc - FROZEN_TAM_CV[h]) <= GATE_TOL; ok &= good
            print(f"  gate h={h:>2}m: A0(seed42)={auc:.4f} vs frozen TAM {FROZEN_TAM_CV[h]:.4f}  [{'PASS' if good else 'FAIL'}]")
        if not ok:
            print("  SANITY GATE FAILED -- stopping per protocol. Do not adjust arms."); return
        print("  sanity gate PASSED")

        log = []
        oof = run_split(S, y_lookup, clean_pids, canon_assign, 0, list(ARMS), SEEDS, log, "canonical")
        ens = {a: ensemble(oof[a], SEEDS) for a in ARMS}
        # frozen TAM reference (context only)
        tam = np.zeros((len(clean_pids), 4))
        for f in range(5):
            te = [p for p in clean_pids if canon_assign[p] == f]
            scf = load_scorer_checkpoint(completed[f"Model_3_magnitude_position_fold{f}"], in_dim=2, hidden=8)
            for a, h in enumerate(H_EVAL):
                out = predict_at_horizon_for_patients(scf, te, pdata, tdel_by, True, h)
                for p in te: tam[S.row[p], a] = out[p][0]
        scores = {**ens, "PRS": prs_cv, "TAM_frozen": tam}

        # ---- held-out test partition: train+val -> test, once per arm/seed ----
        inner_tr, inner_val = carve_inner_validation(train_val_pids, y_lookup, seed=123)
        tr_idx = torch.tensor([S.row[p] for p in inner_tr]); val_idx = torch.tensor([S.row[p] for p in inner_val])
        mu, sd = cue_stats(S, [S.row[p] for p in train_val_pids])
        test_ens = {}
        for a in ARMS:
            X = make_inputs(S, ARMS[a]["ext"], mu, sd); Xt = make_inputs(S_test, ARMS[a]["ext"], mu, sd)
            outs = []
            for s in SEEDS:
                sc, n_ep, best = fit_arm(S, X, tr_idx, val_idx, ARMS[a]["obj"], s)
                outs.append(predict_eval(sc, S_test, Xt))
                log.append({"split": "test_model", "arm": a, "seed": s, "fold": -1, "epochs": n_ep,
                            "cap_hit": n_ep >= MAX_EPOCHS, "best_val_obj": round(best, 5)})
            test_ens[a] = np.mean(outs, axis=0)
        test_scores = {**test_ens, "PRS": prs_test}

        print("\n  bootstrapping (B=2000, paired) ...")
        boot = boot_all(y_pat, scores)
        rows = []
        for nm in scores:
            r = {"model": nm}
            for a, h in enumerate(H_EVAL):
                r[f"auroc_{h}m"] = round(fast_auc(y_pat, scores[nm][:, a]), 4)
            r["mean_auroc_M"] = round(np.mean([r[f"auroc_{h}m"] for h in H_EVAL]), 4)
            if nm in test_scores:
                r["test_M"] = round(np.mean([fast_auc(y_test, test_scores[nm][:, a]) for a in range(4)]), 4)
            rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "canonical_auroc.csv"), index=False)
        drows = []
        for arm in ["A1", "A2", "A3"]:
            for ref in ["A0", "PRS"]:
                rec = {"arm": arm, "vs": ref}; rec.update(summarize_delta(boot, arm, ref, y_pat, scores)); drows.append(rec)
        for ref in ["PRS"]:
            rec = {"arm": "A0", "vs": ref}; rec.update(summarize_delta(boot, "A0", ref, y_pat, scores)); drows.append(rec)
        pd.DataFrame(drows).to_csv(os.path.join(OUT_DIR, "canonical_deltas.csv"), index=False)
        pd.DataFrame(log).to_csv(os.path.join(OUT_DIR, "training_log_canonical.csv"), index=False)
        # per-seed spread + test deltas
        spread = [{"arm": a, "seed": s, **{f"auroc_{h}m": round(fast_auc(y_pat, oof[a][s][:, i]), 4) for i, h in enumerate(H_EVAL)}}
                  for a in ARMS for s in SEEDS]
        pd.DataFrame(spread).to_csv(os.path.join(OUT_DIR, "canonical_seed_spread.csv"), index=False)
        trows = [{"arm": a, "test_M": round(np.mean([fast_auc(y_test, test_ens[a][:, i]) for i in range(4)]), 4),
                  "test_dM_vs_A0": round(np.mean([fast_auc(y_test, test_ens[a][:, i]) - fast_auc(y_test, test_ens["A0"][:, i]) for i in range(4)]), 4)}
                 for a in ARMS]
        pd.DataFrame(trows).to_csv(os.path.join(OUT_DIR, "test_partition.csv"), index=False)
        pd.DataFrame({"patient_id": clean_pids, **{f"{a}_{h}m": ens[a][:, i] for a in ARMS for i, h in enumerate(H_EVAL)}}
                     ).to_csv(os.path.join(OUT_DIR, "canonical_scores.csv"), index=False)
        print(pd.DataFrame(rows).to_string(index=False))
        print(pd.DataFrame(drows)[["arm", "vs", "dM", "dM_p", "dM_ci_lo", "dM_ci_hi", "d0", "d10", "d20", "d30"]].round(4).to_string(index=False))
        print(pd.DataFrame(trows).to_string(index=False))
        print(f"  epoch-cap hits: {int(pd.DataFrame(log).cap_hit.sum())} of {len(log)} trainings")

    if args.stage in ("resplit", "all"):
        print("=" * 88 + "\n  PHASE 19 -- resplit stage\n" + "=" * 88)
        path = os.path.join(OUT_DIR, "resplit_deltas.csv")
        done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        rrows = done.to_dict("records") if len(done) else []
        rlog = []
        for rs in RESPLIT_SEEDS:
            if len(done) and (done["resplit_seed"] == rs).any():
                print(f"  resplit {rs}: already done, skipping"); continue
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
            assign = {}
            for f, (_, te_idx) in enumerate(skf.split(pids_arr, y_pat)):
                for p in pids_arr[te_idx]: assign[p] = f
            oof = run_split(S, y_lookup, clean_pids, assign, 100 + rs, list(ARMS), SEEDS, rlog, f"resplit_{rs}")
            scores = {**{a: ensemble(oof[a], SEEDS) for a in ARMS}, "PRS": prs_cv}
            boot = boot_all(y_pat, scores)
            for arm in ["A1", "A2", "A3"]:
                rec = {"resplit_seed": rs, "arm": arm, "vs": "A0"}; rec.update(summarize_delta(boot, arm, "A0", y_pat, scores)); rrows.append(rec)
            pd.DataFrame(rrows).to_csv(path, index=False)
            print(f"  resplit {rs} done: " + ", ".join(f"{r['arm']} dM={r['dM']:+.4f} (p={r['dM_p']:.3f})" for r in rrows[-3:]))
        if rlog:
            old = os.path.join(OUT_DIR, "training_log_resplit.csv")
            prev = pd.read_csv(old) if os.path.exists(old) else pd.DataFrame()
            pd.concat([prev, pd.DataFrame(rlog)]).to_csv(old, index=False)

    if args.stage in ("verdict", "all"):
        print("=" * 88 + "\n  PHASE 19 -- verdict (pre-registered rule, mechanical)\n" + "=" * 88)
        can = pd.read_csv(os.path.join(OUT_DIR, "canonical_deltas.csv")); res = pd.read_csv(os.path.join(OUT_DIR, "resplit_deltas.csv"))
        verdict = {}
        for arm in ["A1", "A2", "A3"]:
            c = can[(can.arm == arm) & (can.vs == "A0")].iloc[0]
            r = res[res.arm == arm]
            c1 = bool(c.dM >= 0.005 and c.dM_p < 0.05)
            n_pos = int((r.dM > 0).sum()); n_sig = int(((r.dM > 0) & (r.dM_p < 0.05)).sum())
            c2 = bool(n_pos == 5 and n_sig >= 3)
            can_min = min(c[f"d{h}"] for h in H_EVAL)
            res_min = min(r[f"d{h}"].mean() for h in H_EVAL)
            c3 = bool(can_min >= -0.020 and res_min >= -0.020)
            if c1 and c2 and c3: tier = "CONFIRMED"
            elif c.dM > 0 and n_pos >= 4: tier = "SUGGESTIVE"
            else: tier = "NOT SUPPORTED"
            verdict[arm] = {"tier": tier, "C1": c1, "C2": c2, "C3": c3, "canonical_dM": round(float(c.dM), 4), "canonical_p": round(float(c.dM_p), 4),
                            "resplits_positive": n_pos, "resplits_p_lt_05": n_sig,
                            "canonical_worst_horizon_delta": round(float(can_min), 4), "resplit_mean_worst_horizon_delta": round(float(res_min), 4)}
            print(f"  {arm}: {tier}  {verdict[arm]}")
        json.dump(verdict, open(os.path.join(OUT_DIR, "verdict.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
