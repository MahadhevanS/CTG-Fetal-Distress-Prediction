"""
Early warning, STAGES 0-3: the boring baselines, before any deep model.

    python scripts/figo_ew_baselines.py

Deliberately unglamorous. The plan's instruction is that these results decide
what the next experiment is, so nothing here is tuned beyond a fixed,
pre-declared grid, and every arm is scored the same way.

STAGES
------
  1  current state / clock / anchor descriptors (LR and small tree)
  2  persistence and transition features from the epochs BEFORE the anchor
  3  temporal descriptor model: level, trend, instability, delta over a
     history window, LR and small gradient boosting
  4  context-length sweep on a FIXED cohort (0 / 10 / 20 minutes of history)

Patient-grouped folds are reused from the detection track
(`data/processed_figo/folds.json`) so no patient can appear in two roles and
no new partition is introduced. The early-warning cohort is a subset of those
patients; the fold assignment is unchanged.

WHAT WOULD MAKE ANY OF THIS MEANINGLESS
----------------------------------------
Three things, all guarded in src/figo_state/earlywarning.py rather than here:
observability applied to both classes, no feature drawn from after the
anchor, and no use of quantities unavailable at prediction time
(`n_epochs_in_record`, `minutes_before_end`). The clock arm exists precisely
so that a model that has quietly learned "how far into the recording am I"
shows up as one.
"""

import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))

from figo_state.earlywarning import (HORIZON_EPOCHS, build_anchors,  # noqa: E402
                                     fixed_cohort, history_features)
from figo_state.protocol import FigoProtocol                        # noqa: E402

DATA = os.path.join(BASE, "data", "processed_figo")
OUT = os.path.join(BASE, "results", "figo_earlywarning")
N_BOOT = 4000
SEED = 0


def load():
    z = np.load(os.path.join(DATA, "epochs.npz"), allow_pickle=True)
    names = [str(s) for s in z["descriptor_names"]]
    meta = pd.DataFrame({c: z[c] for c in z.files
                         if c not in ("X", "F", "descriptor_names",
                                      "fhr_valid", "uc_valid")})
    order = meta.sort_values(["record_id", "epoch_index"]).index.values
    return (z["F"][order].astype(np.float32),
            meta.iloc[order].reset_index(drop=True), names)


def patient_bootstrap(y, s, pid, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    pats = np.array(sorted(set(pid)))
    idx = {q: np.where(pid == q)[0] for q in pats}
    v = []
    for _ in range(n):
        b = rng.choice(pats, len(pats), replace=True)
        rows = np.concatenate([idx[q] for q in b])
        if len(np.unique(y[rows])) < 2:
            continue
        v.append(roc_auc_score(y[rows], s[rows]))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v \
        else (float("nan"), float("nan"))


def paired_delta(y, s1, s2, pid, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    pats = np.array(sorted(set(pid)))
    idx = {q: np.where(pid == q)[0] for q in pats}
    v = []
    for _ in range(n):
        b = rng.choice(pats, len(pats), replace=True)
        rows = np.concatenate([idx[q] for q in b])
        if len(np.unique(y[rows])) < 2:
            continue
        v.append(roc_auc_score(y[rows], s1[rows]) - roc_auc_score(y[rows], s2[rows]))
    v = np.array(v)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def oof_predict(X, y, pid, P, kind, repeat=0):
    """Out-of-fold probabilities through the frozen patient-grouped folds."""
    oof = np.zeros(len(y))
    seen = np.zeros(len(y), bool)
    for tr, te in P.folds(repeat=repeat):
        if len(np.unique(y[tr])) < 2:
            continue
        if kind == "lr":
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=3000, class_weight="balanced")
            m.fit(sc.transform(X[tr]), y[tr])
            p = m.predict_proba(sc.transform(X[te]))[:, 1]
        elif kind == "tree":
            m = DecisionTreeClassifier(max_depth=4, min_samples_leaf=25,
                                       class_weight="balanced",
                                       random_state=SEED).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
        elif kind == "gbm":
            m = HistGradientBoostingClassifier(
                max_iter=200, max_depth=3, learning_rate=0.05,
                min_samples_leaf=25, l2_regularization=1.0,
                random_state=SEED).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
        else:
            raise ValueError(kind)
        oof[te] = p
        seen[te] = True
    assert seen.all(), "some anchors never appeared in a test fold"
    return oof


def report(name, y, s, pid, results, verbose=True):
    lo, hi = patient_bootstrap(y, s, pid)
    fpr, tpr, _ = roc_curve(y, s)
    spec = 1 - fpr
    j = int(np.argmax(np.minimum(tpr, spec)))
    r = dict(name=name, n=int(len(y)), prevalence=float(y.mean()),
             auroc=float(roc_auc_score(y, s)),
             auprc=float(average_precision_score(y, s)),
             ci_lo=lo, ci_hi=hi,
             bal_sens=float(tpr[j]), bal_spec=float(spec[j]),
             spec_at_90sens=float(spec[tpr >= 0.90].max()) if (tpr >= 0.90).any() else 0.0)
    results[name] = r
    if verbose:
        print(f"  {name:44s} AUROC {r['auroc']:.4f} [{lo:.3f}-{hi:.3f}]  "
              f"AUPRC {r['auprc']:.4f}  bal {r['bal_sens']:.3f}/{r['bal_spec']:.3f}")
    return r


def gate(a):
    if a >= 0.85:
        return "TARGET REACHED -- freeze and replicate."
    if a >= 0.80:
        return "VERY PROMISING -- replication becomes mandatory."
    if a >= 0.75:
        return "PROMISING -- one controlled refinement."
    if a >= 0.70:
        return "SIGNAL EXISTS but probably not enough -- check the representation adds information."
    return "WEAK -- do not tune. AUROC < 0.70."


def main():
    os.makedirs(OUT, exist_ok=True)
    F, meta, names = load()
    anchors = build_anchors(meta)
    states = meta.y_state.values.astype(int)

    print("=" * 78)
    print("STAGE 0 -- FROZEN TASK")
    print("=" * 78)
    print(f"  Normal anchor -> Abnormal within {HORIZON_EPOCHS * 10} min")
    print(f"  anchors {len(anchors)} | positives {int(anchors.y.sum())} "
          f"({100 * anchors.y.mean():.1f}%) | patients {anchors.record_id.nunique()}")
    print(f"  negatives require all {HORIZON_EPOCHS} horizon epochs present AND readable")
    print(f"  the same test is applied to positives (inclusion is outcome-independent)")
    for k in range(4):
        sub = anchors[anchors.n_hist >= k]
        print(f"    >={k} prior epochs: {len(sub):5d} anchors, "
              f"{100 * sub.y.mean():.1f}% positive, {sub.record_id.nunique()} patients")

    y_all = anchors.y.values.astype(int)
    pid_all = anchors.record_id.values.astype(str)
    worst = meta.groupby("record_id").y_state.max().to_dict()
    P_all = FigoProtocol.load_or_create(
        pid_all, {str(k): int(v) for k, v in worst.items()},
        path=os.path.join(DATA, "folds.json"))

    results: Dict[str, Dict] = {}
    X_now, cols_now = history_features(F, states, anchors, 0, names)

    print("\n" + "=" * 78)
    print("STAGE 1 -- THE BORING BASELINES  (full cohort, n=%d)" % len(anchors))
    print("=" * 78)
    report("1a  current state (all anchors are Normal)", y_all,
           np.zeros(len(y_all)) + 0.5, pid_all, results) \
        if False else print("  1a  current state: constant by construction "
                            "(every anchor is Normal) -- AUROC undefined, as expected")
    report("1b  clock: epoch index alone", y_all,
           anchors.epoch_index.values.astype(float), pid_all, results)
    report("1c  anchor descriptors -> LR", y_all,
           oof_predict(X_now, y_all, pid_all, P_all, "lr"), pid_all, results)
    report("1d  anchor descriptors -> tree(d=4)", y_all,
           oof_predict(X_now, y_all, pid_all, P_all, "tree"), pid_all, results)
    report("1e  anchor descriptors -> GBM", y_all,
           oof_predict(X_now, y_all, pid_all, P_all, "gbm"), pid_all, results)

    # ------------------------------------------------------------------ 2/3
    MIN_HIST = 2          # fixes the cohort for every context comparison
    coh = fixed_cohort(anchors, MIN_HIST)
    y = coh.y.values.astype(int)
    pid = coh.record_id.values.astype(str)
    P = FigoProtocol.load_or_create(
        pid, {str(k): int(v) for k, v in worst.items()},
        path=os.path.join(DATA, "folds.json"))

    print("\n" + "=" * 78)
    print(f"STAGES 2-4 -- FIXED COHORT: anchors with >={MIN_HIST} prior epochs")
    print("=" * 78)
    print(f"  n={len(coh)}  positives {int(y.sum())} ({100 * y.mean():.1f}%)  "
          f"patients {coh.record_id.nunique()}")
    print("  cohort is held CONSTANT across context lengths so that context is")
    print("  not confounded with anchor position -- deeper-history anchors are")
    print("  later in labour and deteriorate more often.")

    print("\n  re-baselined on this cohort:")
    Xc0, _ = history_features(F, states, coh, 0, names)
    report("   clock: epoch index alone", y,
           coh.epoch_index.values.astype(float), pid, results)
    base_lr = oof_predict(Xc0, y, pid, P, "lr")
    report("   anchor descriptors -> LR   [context 0]", y, base_lr, pid, results)
    base_gbm = oof_predict(Xc0, y, pid, P, "gbm")
    report("   anchor descriptors -> GBM  [context 0]", y, base_gbm, pid, results)

    print("\n  STAGE 2/3 -- with history (level, trend, instability, delta,")
    print("               plus FIGO-state persistence over the window):")
    ctx_scores = {}
    for ctx in (1, 2):
        Xc, cols = history_features(F, states, coh, ctx, names)
        for kind, tag in (("lr", "LR"), ("gbm", "GBM")):
            s = oof_predict(Xc, y, pid, P, kind)
            ctx_scores[(ctx, kind)] = s
            report(f"   history {ctx * 10:2d} min -> {tag:3s} [{Xc.shape[1]} features]",
                   y, s, pid, results)

    print("\n" + "=" * 78)
    print("STAGE 4 -- DOES HISTORY ADD ANYTHING?  (paired, same cohort)")
    print("=" * 78)
    for kind, tag, base in (("lr", "LR", base_lr), ("gbm", "GBM", base_gbm)):
        for ctx in (1, 2):
            d, lo, hi = paired_delta(y, ctx_scores[(ctx, kind)], base, pid)
            mark = "EXCLUDES 0" if (lo > 0 or hi < 0) else "crosses 0"
            print(f"  {tag:3s}  context {ctx * 10:2d} min - context 0 : "
                  f"{d:+.4f}  [{lo:+.4f},{hi:+.4f}]  {mark}")

    best = max((r for r in results.values() if "clock" not in r["name"]),
               key=lambda r: r["auroc"])
    print("\n" + "=" * 78)
    print(f"STAGE 9 GATE on the best non-clock arm: {best['name']} "
          f"(AUROC {best['auroc']:.4f})")
    print(f"  {gate(best['auroc'])}")
    print("=" * 78)

    json.dump(results, open(os.path.join(OUT, "stage0_3_baselines.json"), "w"),
              indent=2, default=float)
    print(f"\nwrote {OUT}/stage0_3_baselines.json")


if __name__ == "__main__":
    main()
