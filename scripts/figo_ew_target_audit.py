"""
TARGET-INFORMATION AUDIT for early warning. No model is being proposed here.

    python scripts/figo_ew_target_audit.py

The Stage 0-4 baselines returned AUROC 0.5842 against a clock at 0.5681, and
every trajectory-vs-instantaneous comparison crossed zero
(docs/figo_early_warning.md). Before spending compute on raw FHR or a
temporal network, the question is whether ANY clinically sensible
formulation of "deterioration" on this corpus carries predictable signal.

FIVE AUDITS
-----------
  A  WHERE the event happens. Is the 30-minute target really a 10-minute
     target with noise appended?
  B  HORIZON. 10 / 20 / 30 / 40 minutes scored separately, each with its own
     observability-matched anchor set and its own clock control.
  C  CONCEPT PERSISTENCE -- the information ceiling. The label is exactly
     NOT(baseline_normal AND variability_normal AND no_repetitive_decels).
     If those three concepts are near-independent across the horizon, then
     the future label is near-independent of anything measurable now, and no
     architecture recovers it. This is the audit that can end the question
     rather than merely score it.
  D  SEVERITY. Does the signal improve when the event is required to be
     unambiguous (two criteria failing, or a large excursion) rather than one
     marginal threshold crossing?
  E  ORACLE CEILING. Given the true future concepts, the label is exact by
     construction; given the true CURRENT concepts, how well can the future
     label be predicted at all? That separates "our features are bad" from
     "the future is not determined by the present".

A NOTE ON MARGIN-BASED EXCLUSION, LEARNED THE HARD WAY
-------------------------------------------------------
Audit D is tempting to run as "drop the ambiguous cases and re-score". This
project has already produced a spectacular artefact that way: excluding
horizon-boundary windows once gave AUROC 0.9832, which turned out to be two
positives against 984 negatives after the exclusion removed 96% of the
positive class (docs/auroc_ceiling_analysis.md). So every exclusion below
reports the surviving class balance, and any arm whose positive count falls
below 50 is marked untrustworthy rather than quoted.
"""

import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))

from figo_state.earlywarning import history_features                # noqa: E402
from figo_state.protocol import FigoProtocol                        # noqa: E402

DATA = os.path.join(BASE, "data", "processed_figo")
OUT = os.path.join(BASE, "results", "figo_earlywarning")
MIN_POS = 50          # below this an arm is reported but not trusted


def load():
    z = np.load(os.path.join(DATA, "epochs.npz"), allow_pickle=True)
    names = [str(s) for s in z["descriptor_names"]]
    meta = pd.DataFrame({c: z[c] for c in z.files
                         if c not in ("X", "F", "descriptor_names",
                                      "fhr_valid", "uc_valid")})
    order = meta.sort_values(["record_id", "epoch_index"]).index.values
    return (z["F"][order].astype(np.float32),
            meta.iloc[order].reset_index(drop=True), names)


def score(F, states, anchors, names, worst, tag, results, note=""):
    """Clock and anchor-descriptor LR through the frozen patient folds."""
    if len(anchors) == 0 or anchors.y.nunique() < 2:
        print(f"  {tag:46s} degenerate")
        return None
    y = anchors.y.values.astype(int)
    pid = anchors.record_id.values.astype(str)
    P = FigoProtocol.load_or_create(pid, {str(k): int(v) for k, v in worst.items()},
                                    path=os.path.join(DATA, "folds.json"))
    X, _ = history_features(F, states, anchors, 0, names)
    oof = np.zeros(len(y))
    for tr, te in P.folds(repeat=0):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=3000, class_weight="balanced")
        m.fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    clock = roc_auc_score(y, anchors.epoch_index.values)
    lr = roc_auc_score(y, oof)
    npos = int(y.sum())
    flag = "  [n_pos<50, NOT TRUSTED]" if npos < MIN_POS else ""
    print(f"  {tag:46s} n={len(y):5d} pos={npos:4d} ({100*y.mean():4.1f}%) "
          f"clock={clock:.4f} LR={lr:.4f}  d={lr-clock:+.4f}{flag}")
    results[tag] = dict(n=len(y), n_pos=npos, prevalence=float(y.mean()),
                        clock=float(clock), lr=float(lr),
                        delta=float(lr - clock), trusted=npos >= MIN_POS,
                        note=note)
    return results[tag]


def anchors_for(meta, horizon, positive_fn, require_normal=True):
    """Generic anchor builder. positive_fn(future_states, future_rows) -> int."""
    out: List[Dict] = []
    for rid, g in meta.groupby("record_id", sort=False):
        s = g.y_state.values
        rows = g.index.values
        n = len(s)
        for t in range(n):
            if require_normal and s[t] != 0:
                continue
            fs = s[t + 1: t + 1 + horizon]
            fr = rows[t + 1: t + 1 + horizon]
            if len(fs) < horizon or np.any(fs == -1):
                continue
            y = positive_fn(fs, fr)
            if y is None:
                continue
            out.append(dict(anchor_row=int(rows[t]), record_id=str(rid),
                            epoch_index=int(t), n_hist=int(t),
                            hist_rows=rows[:t].tolist(), y=int(y)))
    return pd.DataFrame(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    F, meta, names = load()
    states = meta.y_state.values.astype(int)
    worst = meta.groupby("record_id").y_state.max().to_dict()
    results: Dict[str, Dict] = {}

    def rule(t):
        print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

    # ------------------------------------------------------------------ A
    rule("A. WHERE DOES THE EVENT HAPPEN?")
    a30 = anchors_for(meta, 3, lambda fs, fr: int(np.any(fs >= 1)))
    pos = a30[a30.y == 1]
    firsts = []
    for _, r in pos.iterrows():
        ar = int(r.anchor_row)
        for k in (1, 2, 3):
            if states[ar + k] >= 1:
                firsts.append(k)
                break
    firsts = np.array(firsts)
    print(f"  of {len(pos)} positives, the FIRST abnormal epoch is at:")
    for k in (1, 2, 3):
        c = int((firsts == k).sum())
        print(f"    t+{k} ({k*10:2d} min): {c:4d}  ({100*c/len(firsts):5.1f}%)")
    print(f"\n  -> {100*(firsts==1).mean():.1f}% of the 30-minute target is decided by")
    print("     the very next epoch. The horizon adds little beyond t+1.")
    results["A_first_event_epoch"] = {f"t+{k}": int((firsts == k).sum())
                                      for k in (1, 2, 3)}

    print("\n  predictability of each sub-target separately:")
    score(F, states, anchors_for(meta, 1, lambda fs, fr: int(fs[0] >= 1)),
          names, worst, "abnormal at t+1 exactly (10 min)", results)
    score(F, states,
          anchors_for(meta, 2, lambda fs, fr: (int(fs[1] >= 1) if fs[0] == 0 else None)),
          names, worst, "abnormal at t+2, t+1 still Normal", results)
    score(F, states,
          anchors_for(meta, 3, lambda fs, fr: (int(fs[2] >= 1)
                                               if (fs[0] == 0 and fs[1] == 0) else None)),
          names, worst, "abnormal at t+3, t+1/t+2 still Normal", results)

    # ------------------------------------------------------------------ B
    rule("B. HORIZON SWEEP  (each with its own observability-matched anchors)")
    for h in (1, 2, 3, 4):
        score(F, states, anchors_for(meta, h, lambda fs, fr: int(np.any(fs >= 1))),
              names, worst, f"any abnormal within {h*10:2d} min", results)

    # ------------------------------------------------------------------ C
    rule("C. CONCEPT PERSISTENCE -- the information ceiling")
    print("  The label is exactly NOT(baseline_normal AND variability_normal")
    print("  AND no_repetitive_decels). If those concepts do not persist across")
    print("  the horizon, the future label is not determined by the present.\n")
    idx = {n: names.index(n) for n in
           ("baseline_bpm", "variability_bpm", "decel_repetitive")}
    print(f"  {'concept':22s}{'lag 1':>10s}{'lag 2':>10s}{'lag 3':>10s}   (Pearson r)")
    persist = {}
    for nm, j in idx.items():
        rs = []
        for lag in (1, 2, 3):
            xs, ys = [], []
            for rid, g in meta.groupby("record_id", sort=False):
                rows = g.index.values
                s = g.y_state.values
                for t in range(len(rows) - lag):
                    if s[t] == -1 or s[t + lag] == -1:
                        continue
                    v0, v1 = F[rows[t], j], F[rows[t + lag], j]
                    if np.isfinite(v0) and np.isfinite(v1):
                        xs.append(v0)
                        ys.append(v1)
            rs.append(float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else np.nan)
        persist[nm] = rs
        print(f"  {nm:22s}" + "".join(f"{r:10.3f}" for r in rs))
    results["C_concept_persistence"] = persist

    print("\n  binary flags -- does the flag at t predict the flag at t+lag?  (AUROC)")
    print(f"  {'flag':22s}{'lag 1':>10s}{'lag 2':>10s}{'lag 3':>10s}")
    flag_auc = {}
    for nm in ("baseline_normal", "variability_normal", "no_repetitive_decels"):
        aucs = []
        for lag in (1, 2, 3):
            xs, ys = [], []
            for rid, g in meta.groupby("record_id", sort=False):
                v = g[nm].values
                s = g.y_state.values
                for t in range(len(v) - lag):
                    if s[t] == -1 or s[t + lag] == -1:
                        continue
                    xs.append(1 - v[t])
                    ys.append(1 - v[t + lag])
            aucs.append(float(roc_auc_score(ys, xs))
                        if len(set(ys)) > 1 else np.nan)
        flag_auc[nm] = aucs
        print(f"  {nm:22s}" + "".join(f"{a:10.3f}" for a in aucs))
    results["C_flag_persistence_auroc"] = flag_auc

    # ------------------------------------------------------------------ D
    rule("D. SEVERITY -- is an UNAMBIGUOUS event more predictable?")

    def n_failed(row_idx):
        return int(meta.baseline_normal.iloc[row_idx] == 0) \
             + int(meta.variability_normal.iloc[row_idx] == 0) \
             + int(meta.no_repetitive_decels.iloc[row_idx] == 0)

    def ge2_criteria(fs, fr):
        for s_, r_ in zip(fs, fr):
            if s_ >= 1 and n_failed(r_) >= 2:
                return 1
        return 0

    score(F, states, anchors_for(meta, 3, lambda fs, fr: int(np.any(fs >= 1))),
          names, worst, "any abnormal (baseline target)", results)
    score(F, states, anchors_for(meta, 3, ge2_criteria), names, worst,
          ">=2 normality criteria fail", results,
          note="unambiguous abnormality, not a single marginal crossing")
    score(F, states, anchors_for(meta, 3, lambda fs, fr: int(np.any(fs == 2))),
          names, worst, "Pathological within 30 min", results)

    iv = names.index("variability_bpm")
    ib = names.index("baseline_bpm")

    def big_excursion(fs, fr):
        for s_, r_ in zip(fs, fr):
            if s_ < 1:
                continue
            v, b = F[r_, iv], F[r_, ib]
            if (np.isfinite(v) and (v < 4.0 or v > 26.0)) or \
               (np.isfinite(b) and (b < 105 or b > 165)):
                return 1
        return 0

    score(F, states, anchors_for(meta, 3, big_excursion), names, worst,
          "clear excursion (>1 bpm past a threshold)", results,
          note="margin-based; check surviving class balance")

    # ------------------------------------------------------------------ E
    rule("E. ORACLE -- what if the CURRENT concepts were measured perfectly?")
    print("  Feed the LR the TRUE current rule-concepts instead of all descriptors.")
    print("  This is the ceiling for any model that perfectly perceives the")
    print("  present but must still infer the future.\n")
    a = a30
    y = a.y.values.astype(int)
    pid = a.record_id.values.astype(str)
    P = FigoProtocol.load_or_create(pid, {str(k): int(v) for k, v in worst.items()},
                                    path=os.path.join(DATA, "folds.json"))
    cols = [names.index(c) for c in ("baseline_bpm", "variability_bpm",
                                     "decel_repetitive")]
    Xo = F[a.anchor_row.values][:, cols]
    oof = np.zeros(len(y))
    for tr, te in P.folds(repeat=0):
        sc = StandardScaler().fit(Xo[tr])
        m = LogisticRegression(max_iter=3000, class_weight="balanced")
        m.fit(sc.transform(Xo[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(Xo[te]))[:, 1]
    print(f"  true current rule-concepts -> LR : AUROC {roc_auc_score(y, oof):.4f}")
    results["E_oracle_current_concepts"] = float(roc_auc_score(y, oof))

    # ------------------------------------------------------------------
    rule("VERDICT")
    trusted = {k: v for k, v in results.items()
               if isinstance(v, dict) and v.get("trusted")}
    if trusted:
        best = max(trusted.items(), key=lambda kv: kv[1]["lr"])
        print(f"  strongest trustworthy target: {best[0]}")
        print(f"    AUROC {best[1]['lr']:.4f}  (clock {best[1]['clock']:.4f}, "
              f"delta {best[1]['delta']:+.4f}, n_pos {best[1]['n_pos']})")
        print(f"\n  Stage 9 gate: {'PROCEED to raw FHR' if best[1]['lr'] >= 0.70 else 'STOP -- every trustworthy target is below 0.70'}")

    json.dump(results, open(os.path.join(OUT, "target_audit.json"), "w"),
              indent=2, default=float)
    print(f"\nwrote {OUT}/target_audit.json")


if __name__ == "__main__":
    main()
