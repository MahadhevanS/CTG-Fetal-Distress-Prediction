"""
Statistical validation of the selective knowledge layer -- the project's novelty claim.

The claim: clinical knowledge raises precision in the high-confidence region
(low recall) and lowers it in the screening region (high recall), so it should be
applied selectively rather than fused permanently.

The evidence so far is a single pooled measurement on 51 positive test windows,
with no confidence interval and no check that the crossover is reproducible. That
is the weakest link in the project and the first thing a reviewer would attack.

Three checks, none requiring retraining:

  1. BOOTSTRAP CONFIDENCE INTERVALS on the precision delta at each recall level.
     Resampling is at PATIENT level, not window level -- windows from one patient
     are strongly correlated (87.5% overlap at a 2.5-min stride), so window-level
     resampling would understate the interval badly.

  2. PER-FOLD REPLICATION. Does the sign flip appear independently in each of the
     5 CV folds, or only after pooling? A pooled effect driven by one fold is not
     a finding.

  3. THIRD PARTITION. The validation split (1,162 windows) has been used for
     neither training nor any test-set claim, so it is an untouched independent
     check.

A claim that survives all three is defensible. One that does not should not be
built into a thesis chapter.
"""
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.knowledge.figo import derive_figo_criteria_flags_torch
from src.models.ctg_crossformer import CTGCrossformerEncoder, CTGCrossformerForClassification
from src.training.train import create_patient_level_folds

CK = "checkpoints/ctg_crossformer_invfreq"
RECALLS = [0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
N_BOOT = 2000
RNG = np.random.default_rng(42)


def build(dev):
    e = CTGCrossformerEncoder(in_channels=2, seq_len=4800, cnn_channels=128, n_heads_cross=4,
                              n_heads_tf=8, n_tf_layers=4, d_ff=512, dropout=0.1, latent_dim=128)
    return CTGCrossformerForClassification(encoder=e, hidden_dim=128, dropout=0.3).to(dev)


@torch.no_grad()
def score(m, X, idx, dev, bs=64):
    o = np.zeros(len(idx), np.float32)
    for s in range(0, len(idx), bs):
        c = idx[s:s + bs]
        o[s:s + len(c)] = torch.sigmoid(m(X[c].to(dev)).squeeze(-1)).cpu().numpy()
    return o


def ensemble_score(X, dev):
    acc = np.zeros(len(X))
    for k in range(1, 6):
        m = build(dev)
        m.load_state_dict(torch.load(f"{CK}/ctg_crossformer_fold_{k}_best.pth",
                                     map_location=dev, weights_only=True))
        m.eval()
        acc += score(m, X, np.arange(len(X)), dev)
        del m
        torch.cuda.empty_cache()
    return acc / 5.0


def prec_at_recall(y, s, r):
    if y.sum() < 2 or len(set(y.tolist())) < 2:
        return np.nan
    p, rc, _ = precision_recall_curve(y, s)
    i = np.where(rc >= r)[0]
    return float(p[i].max()) if len(i) else np.nan


def knowledge_features(yf, ext_path):
    E = np.delete(np.load(ext_path), 7, axis=1)   # drop constant uc_tachysystole
    return np.column_stack([derive_figo_criteria_flags_torch(yf.float()).numpy(), E])


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- CV pool: OOF model scores + fitted knowledge combiner ---------- #
    d = torch.load("data/processed_mil/train_dataset.pt", map_location="cpu", weights_only=False)
    X, y, yf = d["X"], d["y_primary"].numpy(), d["y_features"]
    pids = np.array([m[0] for m in d["metadata"]])
    folds = create_patient_level_folds(list(pids), torch.as_tensor(y), k_folds=5,
                                       secondary_labels=d["y_figo"])
    K = knowledge_features(yf, "data/processed_mil/train_extended_features.npy")

    oof = np.full(len(y), np.nan, np.float32)
    for k, (_, vi) in enumerate(folds, 1):
        m = build(dev)
        m.load_state_dict(torch.load(f"{CK}/ctg_crossformer_fold_{k}_best.pth",
                                     map_location=dev, weights_only=True))
        m.eval()
        oof[vi] = score(m, X, vi, dev)
        del m
        torch.cuda.empty_cache()

    combiner = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    combiner.fit(np.column_stack([oof, K]), y)

    # =================== CHECK 2: per-fold replication =================== #
    print("=" * 78)
    print(" CHECK 2 -- does the crossover replicate INDEPENDENTLY in each CV fold?")
    print("=" * 78)
    print(f"  {'fold':<7}" + "".join(f"{int(r*100):>9}%" for r in RECALLS))
    print("  " + "-" * 74)
    signs = {r: [] for r in RECALLS}
    for fi, (tr, va) in enumerate(folds, 1):
        c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        c.fit(np.column_stack([oof[tr], K[tr]]), y[tr])
        fused = c.predict_proba(np.column_stack([oof[va], K[va]]))[:, 1]
        row = []
        for r in RECALLS:
            a = prec_at_recall(y[va], oof[va], r)
            b = prec_at_recall(y[va], fused, r)
            dlt = b - a
            row.append(dlt)
            if not np.isnan(dlt):
                signs[r].append(np.sign(dlt))
        print(f"  {fi:<7}" + "".join(f"{v:>+10.3f}" if not np.isnan(v) else f"{'n/a':>10}" for v in row))
    print("  " + "-" * 74)
    print(f"  {'agree':<7}" + "".join(
        f"{int(sum(1 for x in signs[r] if x > 0))}/{len(signs[r]):>8}" for r in RECALLS))
    print("\n  (folds where knowledge IMPROVED precision, out of folds with a defined value)")

    # ============ CHECKS 1 & 3: bootstrap CI on val and test ============= #
    for split in ["val", "test"]:
        dt = torch.load(f"data/processed_mil/{split}_dataset.pt", map_location="cpu", weights_only=False)
        Xt, yt, yft = dt["X"], dt["y_primary"].numpy(), dt["y_features"]
        pt_ids = np.array([m[0] for m in dt["metadata"]])
        Kt = knowledge_features(yft, f"data/processed_mil/{split}_extended_features.npy")
        base = ensemble_score(Xt, dev)
        fused = combiner.predict_proba(np.column_stack([base, Kt]))[:, 1]

        label = ("CHECK 3 -- VALIDATION split (independent, never used for any claim)"
                 if split == "val" else "CHECK 1 -- TEST split, bootstrap CIs")
        print("\n" + "=" * 78)
        print(f" {label}")
        print(f" {len(yt)} windows, {int(yt.sum())} positive, {len(set(pt_ids))} patients")
        print("=" * 78)

        uniq = np.array(sorted(set(pt_ids)))
        idx_by_p = {p: np.where(pt_ids == p)[0] for p in uniq}

        print(f"  {'recall':<10}{'model':<10}{'+know':<10}{'delta':<10}{'95% CI':<20}{'p(delta>0)'}")
        print("  " + "-" * 72)
        for r in RECALLS:
            a = prec_at_recall(yt, base, r)
            b = prec_at_recall(yt, fused, r)
            deltas = []
            for _ in range(N_BOOT):
                samp = RNG.choice(uniq, size=len(uniq), replace=True)
                ii = np.concatenate([idx_by_p[p] for p in samp])
                if yt[ii].sum() < 3:
                    continue
                da = prec_at_recall(yt[ii], base[ii], r)
                db = prec_at_recall(yt[ii], fused[ii], r)
                if not (np.isnan(da) or np.isnan(db)):
                    deltas.append(db - da)
            if len(deltas) < 100:
                print(f"  {int(r*100):>3}%      insufficient bootstrap samples")
                continue
            deltas = np.array(deltas)
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            pgt = float((deltas > 0).mean())
            sig = "  *" if (lo > 0 or hi < 0) else ""
            print(f"  {int(r*100):>3}%      {a:<10.4f}{b:<10.4f}{b-a:<+10.4f}"
                  f"[{lo:+.3f}, {hi:+.3f}]{'':<4}{pgt:.3f}{sig}")
        print("\n  * = 95% CI excludes zero.  Bootstrap resamples PATIENTS, not windows.")


if __name__ == "__main__":
    main()
