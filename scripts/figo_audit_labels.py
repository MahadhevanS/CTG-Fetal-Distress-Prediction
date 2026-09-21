"""
GATE 1 -- is the FIGO-state label valid, sufficiently represented, and are
its transitions predictable at all?

    python scripts/figo_audit_labels.py

Runs before any model is trained, and is the gate the user pre-registered:
if the pathological state is extremely rare or the labels are obviously
unstable, stop and redesign the label rather than reaching for a bigger
network.

Reports, in order:
  1. state prevalence, epoch-level and patient-level
  2. which pathological criterion fires, and how often
  3. descriptor distributions against the FIGO bands they are thresholded on
  4. the state transition matrix -- is deterioration a thing that happens?
  5. early-warning label availability at each horizon, including how much is
     censored and how much survives the eligibility restriction
  6. the persistence/base-rate baseline that any model must beat
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))
from figo_state import rules as RL  # noqa: E402

DATA = os.path.join(BASE, "data", "processed_figo")
NAME = {-1: "UNREADABLE", 0: "Normal", 1: "Suspicious", 2: "Pathological"}


def load():
    z = np.load(os.path.join(DATA, "epochs.npz"), allow_pickle=True)
    cols = [k for k in z.files if k not in ("X", "F", "descriptor_names")]
    meta = pd.DataFrame({c: z[c] for c in cols})
    F = pd.DataFrame(z["F"], columns=[str(s) for s in z["descriptor_names"]])
    manifest = json.load(open(os.path.join(DATA, "manifest.json")))
    return meta, F, manifest


def rule(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def main():
    meta, F, man = load()
    em = man["epoch_minutes"]
    n = len(meta)
    print(f"ruleset {man['ruleset_version']} | epoch {em:.0f} min | "
          f"{meta.record_id.nunique()} patients | {n} epochs")

    # ------------------------------------------------------------------ 1
    rule("1. STATE PREVALENCE")
    vc = meta.y_state.value_counts().sort_index()
    print(f"{'state':14s}{'epochs':>8s}{'%':>8s}{'patients':>10s}{'% pat':>8s}")
    npat = meta.record_id.nunique()
    for s in (-1, 0, 1, 2):
        e = int(vc.get(s, 0))
        p = meta.loc[meta.y_state == s, "record_id"].nunique()
        print(f"{NAME[s]:14s}{e:8d}{100 * e / n:8.1f}{p:10d}{100 * p / npat:8.1f}")

    readable = meta[meta.y_state >= 0]
    print(f"\nof READABLE epochs only (n={len(readable)}):")
    for s in (0, 1, 2):
        e = int((readable.y_state == s).sum())
        print(f"  {NAME[s]:14s}{e:8d}{100 * e / len(readable):8.1f}%")
    print("\npublished intrapartum reference: Normal ~60-70%, Suspicious ~20-30%,"
          "\nPathological ~5-10%. Deviation on Pathological is the number to watch.")

    # ------------------------------------------------------------------ 2
    rule("2. WHICH PATHOLOGICAL CRITERION FIRES")
    crit = ["path_bradycardia", "path_reduced_var", "path_increased_var",
            "path_rep_decels", "path_prolonged"]
    path = meta[meta.y_state == 2]
    print(f"{'criterion':26s}{'epochs':>8s}{'% of path':>11s}{'patients':>10s}")
    for c in crit:
        k = int(meta[c].sum())
        p = meta.loc[meta[c] == 1, "record_id"].nunique()
        share = 100 * k / max(1, len(path))
        print(f"{c:26s}{k:8d}{share:11.1f}{p:10d}")
    if len(path):
        multi = path[crit].sum(axis=1)
        print(f"\npathological epochs firing >1 criterion: "
              f"{int((multi > 1).sum())}/{len(path)}")
    print("\nNOT IMPLEMENTED: sinusoidal pattern (>30 min). The Pathological class"
          "\nis under-inclusive by exactly that criterion -- stated, not hidden.")

    rule("2b. WHY EPOCHS ARE SUSPICIOUS")
    susp = meta[meta.y_state == 1]
    for c, lbl in (("baseline_normal", "baseline outside 110-160"),
                   ("variability_normal", "variability outside 5-25 / unreadable"),
                   ("no_repetitive_decels", "repetitive decelerations")):
        k = int((susp[c] == 0).sum())
        print(f"  {lbl:42s}{k:6d}  ({100 * k / max(1, len(susp)):5.1f}% of Suspicious)")

    # ------------------------------------------------------------------ 3
    rule("3. DESCRIPTORS vs THE FIGO BANDS THEY ARE THRESHOLDED ON")
    ok = meta.y_state >= 0
    b, v = F.baseline_bpm[ok.values], F.variability_bpm[ok.values]
    meas = F.variability_measurable[ok.values] > 0
    print(f"baseline   mean {b.mean():6.1f}  median {b.median():6.1f}")
    print(f"  <100 (path) {100 * (b < 100).mean():5.1f}%   "
          f"110-160 (normal) {100 * ((b >= 110) & (b <= 160)).mean():5.1f}%   "
          f">160 {100 * (b > 160).mean():5.1f}%")
    print(f"variability  mean {v[meas].mean():6.1f}  median {v[meas].median():6.1f} "
          f"(measurable on {100 * meas.mean():.1f}% of readable epochs)")
    vm = v[meas]
    print(f"  <5 (reduced) {100 * (vm < 5).mean():5.1f}%   "
          f"5-25 (normal) {100 * ((vm >= 5) & (vm <= 25)).mean():5.1f}%   "
          f">25 (increased) {100 * (vm > 25).mean():5.1f}%")
    print("  [features.py on the 20-min substrate read 44.2% >25 and 0.4% <5]")
    print(f"\ncontractions/epoch  mean {F.n_contractions[ok.values].mean():.2f}"
          f"   (normal labour: <=5 per 10 min)")
    print(f"decelerations/epoch mean {F.n_decels[ok.values].mean():.2f}"
          f"   [features.py: 3.44 variable alone per 20-min window]")
    print(f"repetitive-decel epochs {100 * F.decel_repetitive[ok.values].mean():.1f}%")
    print(f"baseline carried from previous epoch: "
          f"{100 * F.baseline_from_carry[ok.values].mean():.1f}%")

    # ------------------------------------------------------------------ 4
    rule("4. STATE TRANSITIONS (consecutive non-overlapping epochs)")
    T = np.zeros((4, 4), dtype=int)
    idx = {-1: 3, 0: 0, 1: 1, 2: 2}
    for _, g in meta.sort_values(["record_id", "epoch_index"]).groupby("record_id"):
        s = g.y_state.values
        for a, bnext in zip(s[:-1], s[1:]):
            T[idx[a], idx[bnext]] += 1
    lbl = ["Normal", "Suspicious", "Pathological", "UNREADABLE"]
    print("from / to".ljust(14) + "".join(f"{x:>14s}" for x in lbl) + f"{'n':>8s}")
    for i, r in enumerate(lbl):
        tot = T[i].sum()
        cells = "".join(f"{T[i, j]:6d} ({100 * T[i, j] / max(1, tot):4.1f}%)"
                        for j in range(4))
        print(f"{r:14s}{cells}{tot:8d}")
    print("\nNo overlap between consecutive epochs, so these are genuine")
    print("transitions -- not the 87.5%-shared-signal autocorrelation that the")
    print("2.5-min-stride substrate produces.")

    print(f"\nNormal -> Pathological (next epoch)      : {T[0, 2]:4d}")
    print(f"Suspicious -> Pathological (next epoch)  : {T[1, 2]:4d}")
    print("If Normal->Pathological is near-empty, the clinically meaningful")
    print("early-warning task is Suspicious->Pathological, and that decision")
    print("comes from this table rather than from a preference.")

    # ------------------------------------------------------------------ 5
    rule("5. EARLY-WARNING LABEL AVAILABILITY")
    print(f"{'horizon':>8s}{'n=1':>8s}{'n=0':>8s}{'censored':>10s}{'prev%':>8s}"
          f"{'elig n=1':>10s}{'elig n=0':>10s}{'elig prev%':>12s}{'pat n=1':>9s}")
    for h, c in ((20, "y_det20"), (30, "y_det30"), (40, "y_det40")):
        y = meta[c].values
        p1, p0, cen = int((y == 1).sum()), int((y == 0).sum()), int((y == -1).sum())
        prev = 100 * p1 / max(1, p1 + p0)
        el = meta[(meta.eligible_ew == 1) & (meta[c] >= 0)]
        e1, e0 = int((el[c] == 1).sum()), int((el[c] == 0).sum())
        eprev = 100 * e1 / max(1, e1 + e0)
        npos = el.loc[el[c] == 1, "record_id"].nunique()
        print(f"{h:>8d}{p1:8d}{p0:8d}{cen:10d}{prev:8.1f}{e1:10d}{e0:10d}"
              f"{eprev:12.1f}{npos:9d}")
    print("\n'elig' = current epoch is Normal or Suspicious, i.e. genuine")
    print("prediction rather than recognition of an already-pathological state.")

    # ------------------------------------------------------------------ 6
    rule("6. BASELINES ANY MODEL MUST BEAT (30-min horizon, eligible anchors)")
    el = meta[(meta.eligible_ew == 1) & (meta.y_det30 >= 0)].copy()
    y = el.y_det30.values
    if len(np.unique(y)) < 2:
        print("  degenerate -- only one class present")
    else:
        from sklearn.metrics import average_precision_score, roc_auc_score
        print(f"  n = {len(y)} anchors, {int(y.sum())} positive "
              f"({100 * y.mean():.1f}%)")
        # persistence: current state is the whole prediction
        for nm, sc in (("current state (0/1/2 as a score)", el.y_state.values),
                       ("is currently Suspicious", (el.y_state == 1).astype(float)),
                       ("epoch index (a clock, no signal)", el.epoch_index.values),
                       ("minutes from start", el.minutes_from_start.values)):
            try:
                print(f"  {nm:38s} AUROC {roc_auc_score(y, sc):.4f}   "
                      f"AUPRC {average_precision_score(y, sc):.4f}")
            except ValueError as e:
                print(f"  {nm:38s} n/a ({e})")
        print("\n  'epoch index' is the time-confound control. It must sit near")
        print("  0.50. On the previous substrate the equivalent probe reached")
        print("  0.84 and that invalidated the whole label.")

    rule("VERDICT INPUTS")
    npath = int((meta.y_state == 2).sum())
    npath_pat = meta.loc[meta.y_state == 2, "record_id"].nunique()
    print(f"  pathological epochs   {npath}  ({100 * npath / n:.1f}%)")
    print(f"  patients with any     {npath_pat}/{npat}")
    el30 = meta[(meta.eligible_ew == 1) & (meta.y_det30 >= 0)]
    print(f"  30-min eligible anchors {len(el30)}, positives "
          f"{int((el30.y_det30 == 1).sum())}")


if __name__ == "__main__":
    main()
