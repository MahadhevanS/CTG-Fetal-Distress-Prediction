"""
THE IMMUTABLE EVALUATION HARNESS for FIGO-state detection.

    python scripts/figo_eval_detection.py results/figo_detection/*/oof.npz
    python scripts/figo_eval_detection.py --compare A_fhr C_cross

Every detection model must be scored through this file and nothing else.

WHY IT IS A SEPARATE FILE FROM EVERY TRAINING SCRIPT
-----------------------------------------------------
This project has already measured what happens when each experiment brings
its own scoring: src/training/protocol.py's docstring records per-fold AUROC
swinging 0.72-0.91 between scripts that each built their own folds, "far
larger than the effect sizes being compared". Two numbers produced by two
scripts were then not comparable however carefully each was measured.

A training script may not compute its own headline metric. It writes an
`oof.npz` and stops. This file turns that into numbers.

WHAT AN oof.npz MUST CONTAIN
-----------------------------
    prob        (N,) float, out-of-fold P(abnormal) for every readable epoch
    y           (N,) int,   0 = Normal, 1 = Suspicious or Pathological
    pid         (N,) str,   record_id, for patient-grouped bootstrap
    fold        (N,) int,   which test fold each row was predicted in
    name        str,        model name for tables
    n_params    int,        parameter count
Anything else is carried through to the report untouched.

OPERATING POINTS -- BOTH DIRECTIONS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS
----------------------------------------------------------------------------
Fixed-specificity views (sens @ 90/95% spec) keep comparability with this
repo's earlier work and with the CTG literature. Fixed-sensitivity views
(spec @ 85/90/95% sens) are the clinically binding direction for a screening
alarm and are where the stated 89/89 target lives.

The headline threshold is NOT chosen by maximising anything on the data it is
then scored on. It is selected inside each training fold on that fold's inner
validation patients, by a rule fixed in advance -- lowest threshold reaching
TARGET_SENSITIVITY -- and applied blind to the held-out patients. A threshold
picked on the test predictions is a fitted parameter and its sensitivity and
specificity are training numbers.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))

RESULTS = os.path.join(BASE, "results", "figo_detection")

TARGET_SENSITIVITY = 0.90     # the pre-specified selection rule
GOAL_SENS = 0.89              # the stated success criterion
GOAL_SPEC = 0.89
N_BOOT = 2000
SEED = 0


# --------------------------------------------------------------------------
def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray,
                              target: float = TARGET_SENSITIVITY) -> float:
    """
    Lowest score threshold that still reaches `target` sensitivity.

    Fixed in advance and applied identically everywhere. Called on VALIDATION
    predictions inside a training fold; never on the predictions it will be
    scored against.
    """
    order = np.argsort(-p)
    ys = y[order]
    ps = p[order]
    npos = max(int(y.sum()), 1)
    tp = np.cumsum(ys)
    sens = tp / npos
    hit = np.where(sens >= target)[0]
    if len(hit) == 0:
        return float(ps.min())
    return float(ps[hit[0]])


def point_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2 * ppv * sens / (ppv + sens) if ppv + sens else 0.0
    return dict(threshold=float(thr), tp=tp, fp=fp, tn=tn, fn=fn,
                sensitivity=sens, specificity=spec, ppv=ppv, npv=npv,
                balanced_accuracy=(sens + spec) / 2, f1=f1)


def roc_views(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Both directions of the ROC, plus the max-min balanced point."""
    fpr, tpr, thr = roc_curve(y, p)
    spec = 1 - fpr
    out: Dict[str, float] = {}
    for t in (0.90, 0.95):
        m = spec >= t
        out[f"sens_at_{int(t * 100)}spec"] = float(tpr[m].max()) if m.any() else 0.0
    for t in (0.85, 0.90, 0.95):
        m = tpr >= t
        out[f"spec_at_{int(t * 100)}sens"] = float(spec[m].max()) if m.any() else 0.0
    j = int(np.argmax(np.minimum(tpr, spec)))
    out["balanced_sens"] = float(tpr[j])
    out["balanced_spec"] = float(spec[j])
    out["balanced_threshold"] = float(thr[j])
    out["min_sens_spec"] = float(min(tpr[j], spec[j]))
    return out


def patient_bootstrap(y: np.ndarray, p: np.ndarray, pid: np.ndarray,
                      thr: Optional[float] = None,
                      n_boot: int = N_BOOT) -> Dict[str, List[float]]:
    """
    Resample PATIENTS, never epochs.

    Epochs within a patient share a fetus, a sensor placement and a labour, so
    they are not independent draws. The previous substrate measured epoch-level
    resampling as understating the interval by 2.7x
    (scripts/audit_evaluation_protocol.py, `clustering` probe).
    """
    rng = np.random.default_rng(SEED)
    pats = np.array(sorted(set(pid)))
    idx = {q: np.where(pid == q)[0] for q in pats}
    auroc, auprc, sens, spec = [], [], [], []
    for _ in range(n_boot):
        b = rng.choice(pats, len(pats), replace=True)
        rows = np.concatenate([idx[q] for q in b])
        yy, pp = y[rows], p[rows]
        if len(np.unique(yy)) < 2:
            continue
        auroc.append(roc_auc_score(yy, pp))
        auprc.append(average_precision_score(yy, pp))
        if thr is not None:
            m = point_metrics(yy, pp, thr)
            sens.append(m["sensitivity"])
            spec.append(m["specificity"])
    return dict(auroc=auroc, auprc=auprc, sensitivity=sens, specificity=spec)


def ci(v: List[float]) -> str:
    if not v:
        return "     n/a     "
    lo, hi = np.percentile(v, [2.5, 97.5])
    return f"[{lo:.3f}-{hi:.3f}]"


# --------------------------------------------------------------------------
def evaluate(path: str, verbose: bool = True) -> Dict:
    z = np.load(path, allow_pickle=True)
    y = z["y"].astype(int)
    p = z["prob"].astype(float)
    pid = z["pid"].astype(str)
    fold = z["fold"].astype(int) if "fold" in z.files else np.zeros(len(y), int)
    name = str(z["name"]) if "name" in z.files else os.path.basename(path)
    n_params = int(z["n_params"]) if "n_params" in z.files else -1
    # Threshold chosen INSIDE the training folds, carried in the file.
    thr = float(z["threshold"]) if "threshold" in z.files else None

    res: Dict = dict(name=name, path=path, n=int(len(y)),
                     n_positive=int(y.sum()), prevalence=float(y.mean()),
                     n_patients=int(len(set(pid))), n_params=n_params)
    res["auroc"] = float(roc_auc_score(y, p))
    res["auprc"] = float(average_precision_score(y, p))
    res["brier"] = float(brier_score_loss(y, p))
    res.update(roc_views(y, p))

    per_fold = []
    for k in sorted(set(fold)):
        m = fold == k
        if len(np.unique(y[m])) < 2:
            continue
        per_fold.append(float(roc_auc_score(y[m], p[m])))
    res["fold_auroc"] = per_fold
    res["fold_auroc_mean"] = float(np.mean(per_fold)) if per_fold else float("nan")
    res["fold_auroc_sd"] = float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else float("nan")

    if thr is not None:
        res["operating_point"] = point_metrics(y, p, thr)
        res["operating_point"]["selected_by"] = (
            f"lowest threshold reaching {TARGET_SENSITIVITY:.0%} sensitivity "
            f"on inner-validation patients of each training fold")

    boot = patient_bootstrap(y, p, pid, thr)
    res["ci"] = {k: ([float(x) for x in np.percentile(v, [2.5, 97.5])] if v else None)
                 for k, v in boot.items()}

    if verbose:
        print(f"\n{'=' * 74}\n{name}   ({n_params:,} params)" if n_params > 0
              else f"\n{'=' * 74}\n{name}")
        print("=" * 74)
        print(f"  n {res['n']} epochs / {res['n_patients']} patients | "
              f"{res['n_positive']} abnormal ({100 * res['prevalence']:.1f}%)")
        print(f"\n  DISCRIMINATION")
        print(f"    AUROC              {res['auroc']:.4f}  {ci(boot['auroc'])}")
        print(f"    AUPRC              {res['auprc']:.4f}  {ci(boot['auprc'])}")
        if per_fold:
            print(f"    per fold           " +
                  " ".join(f"{v:.3f}" for v in per_fold) +
                  f"   mean {res['fold_auroc_mean']:.4f} "
                  f"sd {res['fold_auroc_sd']:.4f}")
        print(f"\n  FIXED SPECIFICITY (comparability with prior work)")
        print(f"    sens @ 90% spec    {res['sens_at_90spec']:.4f}")
        print(f"    sens @ 95% spec    {res['sens_at_95spec']:.4f}")
        print(f"\n  FIXED SENSITIVITY (the clinically binding direction)")
        print(f"    spec @ 85% sens    {res['spec_at_85sens']:.4f}")
        print(f"    spec @ 90% sens    {res['spec_at_90sens']:.4f}   <-- headline")
        print(f"    spec @ 95% sens    {res['spec_at_95sens']:.4f}")
        print(f"\n  BALANCED POINT (max of min(sens, spec))")
        print(f"    sens/spec          {res['balanced_sens']:.4f} / "
              f"{res['balanced_spec']:.4f}")
        if thr is not None:
            o = res["operating_point"]
            print(f"\n  PRE-SPECIFIED OPERATING POINT (threshold {thr:.4f}, "
                  f"chosen in training folds)")
            print(f"    sensitivity        {o['sensitivity']:.4f}  "
                  f"{ci(boot['sensitivity'])}")
            print(f"    specificity        {o['specificity']:.4f}  "
                  f"{ci(boot['specificity'])}")
            print(f"    PPV / NPV          {o['ppv']:.4f} / {o['npv']:.4f}")
            print(f"    balanced acc / F1  {o['balanced_accuracy']:.4f} / "
                  f"{o['f1']:.4f}")
        print(f"\n  CALIBRATION")
        print(f"    Brier              {res['brier']:.4f}")
        goal = (res["balanced_sens"] >= GOAL_SENS and
                res["balanced_spec"] >= GOAL_SPEC)
        print(f"\n  GOAL {GOAL_SENS:.0%}/{GOAL_SPEC:.0%}: "
              f"{'MET' if goal else 'not met'}"
              f"   (best balanced point {res['balanced_sens']:.3f}/"
              f"{res['balanced_spec']:.3f})")
    return res


def gate(auroc: float) -> str:
    """PHASE 7, pre-registered. Reported, never negotiated after the fact."""
    if auroc >= 0.85:
        return "LEVEL 1 -- original target reached. STOP architecture work, validate."
    if auroc >= 0.80:
        return "LEVEL 2 -- freeze architecture, move to operating-point work."
    if auroc >= 0.79:
        return "LEVEL 3 -- interesting. Operating point + robustness, do NOT enlarge."
    if auroc >= 0.75:
        return "LEVEL 4 / 0.75-0.79 -- ONE controlled refinement permitted, then reassess."
    return "KILL CRITERION -- AUROC < 0.75 after the interaction experiment. STOP."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="oof.npz files (glob ok)")
    ap.add_argument("--baseline", default="A_fhr",
                    help="model name to report deltas against")
    ap.add_argument("--gate_on", default="C_cross",
                    help="arm the Phase 7 gate is evaluated on (the hypothesis, "
                         "not the best-scoring control)")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    paths: List[str] = []
    for p in (args.paths or [os.path.join(RESULTS, "*", "oof.npz")]):
        paths.extend(sorted(glob.glob(p)) or ([p] if os.path.exists(p) else []))
    if not paths:
        print("no oof.npz found. Run scripts/figo_run_detection.py first.")
        return

    out = [evaluate(p) for p in paths]

    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print("=" * 100)
    print(f"{'model':26s}{'params':>9s}{'AUROC':>8s}{'95% CI':>16s}"
          f"{'AUPRC':>8s}{'sp@90se':>9s}{'se@90sp':>9s}{'bal se/sp':>13s}")
    base = next((r for r in out if r["name"] == args.baseline), None)
    for r in sorted(out, key=lambda x: -x["auroc"]):
        lo, hi = r["ci"]["auroc"] or (float("nan"), float("nan"))
        pstr = f"{r['n_params']:,}" if r["n_params"] > 0 else "-"
        print(f"{r['name']:26s}{pstr:>9s}{r['auroc']:8.4f}"
              f"  [{lo:.3f}-{hi:.3f}]{r['auprc']:8.4f}"
              f"{r['spec_at_90sens']:9.4f}{r['sens_at_90spec']:9.4f}"
              f"   {r['balanced_sens']:.3f}/{r['balanced_spec']:.3f}")

    if base is not None:
        print(f"\ndelta vs {args.baseline} (AUROC {base['auroc']:.4f})")
        for r in sorted(out, key=lambda x: -x["auroc"]):
            if r["name"] == args.baseline:
                continue
            d = r["auroc"] - base["auroc"]
            print(f"  {r['name']:26s}{d:+.4f}")

    # The Phase 7 gate is defined on the INTERACTION experiment, not on
    # whichever arm happened to score highest. Reading it off the best arm
    # would let a control pass a gate that exists to judge the hypothesis.
    interaction = next((r for r in out if r["name"] == args.gate_on), None)
    if interaction is None:
        print(f"\nPHASE 7 GATE: arm {args.gate_on!r} not present, gate not evaluated.")
    else:
        print(f"\nPHASE 7 GATE, evaluated on the interaction arm "
              f"{interaction['name']} (AUROC {interaction['auroc']:.4f}):")
        print(f"  {gate(interaction['auroc'])}")
        best = max(out, key=lambda x: x["auroc"])
        if best["name"] != interaction["name"]:
            print(f"  (best arm overall is {best['name']} at "
                  f"{best['auroc']:.4f}; it is a control, not the hypothesis)")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        json.dump(out, open(args.json_out, "w"), indent=2, default=float)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
