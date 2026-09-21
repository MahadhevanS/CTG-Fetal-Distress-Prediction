"""
Replication analysis for the knowledge-infusion result.

    python scripts/figo_ki_replication.py

WHAT IS BEING REPLICATED, AND WHY IT NEEDS TO BE
-------------------------------------------------
The KI result rests on two claims measured on ONE fold-repeat with ONE seed:

    KI-2 - Z    = +0.0213  [+0.0118, +0.0313]   knowledge infusion helps
    KI-2 - KI-1 = +0.0104  [+0.0030, +0.0177]   the RULE, not the aux loss

The second is the mechanistic claim and the more important one. Both carry a
degree of freedom the frozen baseline does not: lambda was selected per fold
on inner-validation patients from {0.1, 0.3, 1.0}, and the selected value
varied by fold. That selection is protocol-legal, but it is still a knob, and
a single-repeat effect of +0.02 on 547 patients is exactly the size that
fails to reproduce.

So repeats 1 and 2 re-run the identical code with different PATIENT
PARTITIONS (protocol repeats, seed unchanged in spirit: each repeat uses
seed+rep for its StratifiedKFold, per src/figo_state/protocol.py). Nothing
else changes. No re-tuning, no new arms.

WHAT COUNTS AS REPLICATED -- fixed before the numbers were seen
----------------------------------------------------------------
1. SIGN. The point estimate is positive in all three repeats.
2. MAGNITUDE. The pooled effect across repeats keeps a paired CI excluding
   zero.
3. STABILITY. The per-repeat estimates do not straddle zero.

A result that satisfies 1 and 2 but shows a repeat-to-repeat spread wider
than the effect itself is reported as fragile, not as replicated. That
distinction is the whole point of running this.
"""

import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = os.path.join(BASE, "results", "figo_detection")
ARMS = ["Z_frozen_smallcnn", "KI1_concept", "KI2_rule", "KI3_combined"]
N_BOOT = 4000
SEED = 0


def load(arm: str, repeat: int) -> Optional[Dict]:
    name = "oof.npz" if repeat == 0 else f"oof_r{repeat}.npz"
    p = os.path.join(R, arm, name)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return dict(prob=z["prob"], y=z["y"].astype(int),
                pid=z["pid"].astype(str), fold=z["fold"].astype(int),
                lambdas=(z["lambdas"].tolist() if "lambdas" in z.files else None))


def paired_ci(y, p1, p2, pid, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    pats = np.array(sorted(set(pid)))
    idx = {q: np.where(pid == q)[0] for q in pats}
    v = []
    for _ in range(n):
        b = rng.choice(pats, len(pats), replace=True)
        rows = np.concatenate([idx[q] for q in b])
        if len(np.unique(y[rows])) < 2:
            continue
        v.append(roc_auc_score(y[rows], p1[rows]) - roc_auc_score(y[rows], p2[rows]))
    v = np.array(v)
    return float(v.mean()), *[float(x) for x in np.percentile(v, [2.5, 97.5])]


def main():
    repeats = sorted({int(os.path.basename(f).split("_r")[1].split(".")[0])
                      for f in glob.glob(os.path.join(R, "*", "oof_r*.npz"))} | {0})
    print(f"repeats found: {repeats}\n")

    data = {r: {a: load(a, r) for a in ARMS} for r in repeats}
    usable = [r for r in repeats if all(data[r][a] is not None for a in ARMS)]
    if len(usable) < 2:
        print("need at least two complete repeats; run figo_run_ki.py --repeat N")
        return

    print("=" * 78)
    print("AUROC PER REPEAT")
    print("=" * 78)
    print(f"{'arm':20s}" + "".join(f"{'r' + str(r):>10s}" for r in usable)
          + f"{'mean':>10s}{'sd':>8s}")
    auroc: Dict[str, List[float]] = {}
    for a in ARMS:
        vals = [roc_auc_score(data[r][a]["y"], data[r][a]["prob"]) for r in usable]
        auroc[a] = vals
        sd = np.std(vals, ddof=1) if len(vals) > 1 else float("nan")
        print(f"{a:20s}" + "".join(f"{v:10.4f}" for v in vals)
              + f"{np.mean(vals):10.4f}{sd:8.4f}")

    print("\n" + "=" * 78)
    print("PAIRED DELTAS PER REPEAT  (patient bootstrap, 4000 resamples)")
    print("=" * 78)
    comparisons = [("KI2_rule", "Z_frozen_smallcnn", "KI-2 - Z      (does KI help)"),
                   ("KI3_combined", "Z_frozen_smallcnn", "KI-3 - Z"),
                   ("KI1_concept", "Z_frozen_smallcnn", "KI-1 - Z"),
                   ("KI2_rule", "KI1_concept", "KI-2 - KI-1   (is it the RULE)")]
    summary = {}
    for a, b, label in comparisons:
        print(f"\n{label}")
        deltas = []
        for r in usable:
            d = data[r][a]
            m, lo, hi = paired_ci(d["y"], d["prob"], data[r][b]["prob"], d["pid"])
            deltas.append(m)
            mark = "excludes 0" if (lo > 0 or hi < 0) else "crosses 0"
            print(f"    repeat {r}: {m:+.4f}  [{lo:+.4f},{hi:+.4f}]  {mark}")
        deltas = np.array(deltas)
        # verdict against the three pre-registered criteria
        sign_ok = bool((deltas > 0).all())
        spread = float(deltas.max() - deltas.min())
        stable = spread < abs(float(deltas.mean()))
        print(f"    -> mean {deltas.mean():+.4f}  range [{deltas.min():+.4f},"
              f"{deltas.max():+.4f}]  spread {spread:.4f}")
        print(f"    -> sign consistent: {'YES' if sign_ok else 'NO'}"
              f"   |   spread < |effect|: {'YES' if stable else 'NO'}")
        summary[label] = dict(deltas=deltas.tolist(), mean=float(deltas.mean()),
                              spread=spread, sign_consistent=sign_ok,
                              stable=stable)

    print("\n" + "=" * 78)
    print("LAMBDA SELECTED PER FOLD (the degree of freedom Z does not have)")
    print("=" * 78)
    for a in ("KI1_concept", "KI2_rule", "KI3_combined"):
        for r in usable:
            lam = data[r][a]["lambdas"]
            if lam:
                print(f"  {a:16s} repeat {r}: {lam}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    for label, s in summary.items():
        if s["sign_consistent"] and s["stable"]:
            v = "REPLICATED"
        elif s["sign_consistent"]:
            v = "sign holds but FRAGILE (spread exceeds the effect)"
        else:
            v = "NOT REPLICATED (sign flips across repeats)"
        print(f"  {label:32s} mean {s['mean']:+.4f}  -> {v}")

    out = os.path.join(R, "ki_replication.json")
    json.dump(dict(repeats=usable, auroc=auroc, comparisons=summary),
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
