"""
End-to-end clinical review: one raw CTG recording in, the full clinician-facing
output out -- risk timeline, patient report, and per-window FIGO explanations.

This is the complete delivered path in one command. It chains:

    raw .dat/.hea  ->  preprocessing (identical to training)
                   ->  5-fold CRP ensemble risk per 20-minute window
                   ->  FIGO 2015 criteria per window (deterministic)
                   ->  patient-level timeline report
                   ->  per-window explanation narrative

SCORES vs EXPLANATIONS. The risk score is the 5-fold ensemble mean, which is the
delivered model. The FIGO criteria are computed from the trace and are model-
independent. The saliency/attention line ("model attended most to minute N") is
necessarily from a SINGLE fold (fold 1), since the five folds have different
attention maps -- it is an illustration of where one model looked, not an
ensemble property, and it is labelled as such in the output.

OPERATING POINT. Defaults to the delivered raw ensemble at threshold 0.30.
Pass --calibrated to use per-fold Platt scaling plus the validation-derived
threshold from calibration.json (see docs/calibration_test_plan.md); at 0.30 the
raw ensemble sits at ~49% sensitivity on test, whereas the calibrated operating
point targets 80%.

Usage:
    python scripts/run_clinical_review.py --record 2045 --minutes 50
    python scripts/run_clinical_review.py --record 2045 --minutes 50 --calibrated
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
_PP = os.path.join(BASE, "src", "preprocessing")
if _PP not in sys.path:
    sys.path.insert(0, _PP)

from ingestion import load_ctu_chb_record, load_clinical_metadata
from signal_quality import assess_signal_quality
from src.explainability.ctg_explainer import CTGExplainer
from src.explainability.patient_report import build_patient_report
from src.inference.ctg_inference import CTGInference, WINDOW_SAMPLES, STRIDE_SAMPLES, MAX_MISSING_RATIO

FS = 4.0
RAW_DIR = "data/raw/ctu-chb-intrapartum"
W = 78


def rule(title=""):
    print("=" * W)
    if title:
        print(f" {title}")
        print("=" * W)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", default="2045", help="CTU-CHB record id, e.g. 2045")
    ap.add_argument("--minutes", type=float, default=50.0, help="minutes of trace to review (from the end)")
    ap.add_argument("--ckpt_dir", default="checkpoints/ctg_crossformer_crp")
    ap.add_argument("--scaler", default="data/processed_mil/ctu_signal_scaler.npz")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--calibrated", action="store_true",
                    help="apply per-fold Platt calibration + validation-derived threshold")
    ap.add_argument("--explain", choices=["flagged", "all", "peak"], default="flagged",
                    help="which windows to narrate (default: flagged, plus the peak window)")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------- input
    rule("1. INPUT")
    fhr, uc, fs = load_ctu_chb_record(os.path.join(RAW_DIR, a.record))
    keep = int(a.minutes * 60 * fs)
    fhr, uc = fhr[-keep:], uc[-keep:]
    print(f"  record            : {a.record}")
    print(f"  reviewed segment  : last {a.minutes:g} min ({len(fhr)} samples @ {fs:g} Hz)")
    print(f"  FHR range         : {fhr[fhr > 0].min():.0f} - {fhr.max():.0f} bpm")
    print(f"  missing samples   : {np.mean(fhr == 0) * 100:.1f}%")

    ph = None
    try:
        md = load_clinical_metadata(os.path.join(RAW_DIR, "clinical_metadata.csv"))
        md.columns = [c.strip().lower() for c in md.columns]
        ph = float(md.set_index("record_id").loc[int(a.record), "ph"])
    except Exception:
        pass

    # ------------------------------------------------------------ inference
    eng = CTGInference(a.ckpt_dir, a.scaler, device=dev, threshold=a.threshold)

    platt = None
    thr = a.threshold
    if a.calibrated:
        cal_path = os.path.join(a.ckpt_dir, "calibration.json")
        if not os.path.exists(cal_path):
            sys.exit(f"[ABORT] --calibrated needs {cal_path}. Run: python scripts/calibrate_crp.py")
        cal = json.load(open(cal_path))
        platt = [(c["slope"], c["intercept"]) for c in cal["platt_coefficients"]]
        thr = cal["threshold"]

    mode = ("calibrated (per-fold Platt, validation-derived threshold)"
            if a.calibrated else "raw ensemble (delivered default)")
    print(f"  model             : {a.ckpt_dir} (5-fold ensemble)")
    print(f"  scoring mode      : {mode}")
    print(f"  threshold         : {thr:.4f}")

    # window the trace exactly as the inference engine does
    windows, starts = [], []
    skipped = []
    for s in range(0, max(len(fhr) - WINDOW_SAMPLES + 1, 0), STRIDE_SAMPLES):
        fw, uw = fhr[s:s + WINDOW_SAMPLES], uc[s:s + WINDOW_SAMPLES]
        if not assess_signal_quality(fw, max_missing_ratio=MAX_MISSING_RATIO):
            skipped.append(round(s / (FS * 60), 1))
            continue
        windows.append(eng.preprocess_window(fw, uw))
        starts.append(s)

    if not windows:
        sys.exit("  No analysable windows (all failed the >30% missing-sample quality gate).")
    print(f"  windows analysed  : {len(windows)}"
          + (f"  ({len(skipped)} skipped for signal quality at min {skipped})" if skipped else ""))

    # ensemble risk per window, with optional per-fold calibration
    xs = torch.tensor(np.stack([x for x, _ in windows]), dtype=torch.float32).to(dev)
    with torch.no_grad():
        per_fold = np.stack([torch.sigmoid(m(xs).squeeze(-1)).cpu().numpy() for m in eng.models])
    if platt is not None:
        c = np.clip(per_fold, 1e-7, 1 - 1e-7)
        z = np.log(c / (1 - c))
        for i, (sl, ic) in enumerate(platt):
            z[i] = sl * z[i] + ic
        per_fold = 1.0 / (1.0 + np.exp(-z))
    risk = per_fold.mean(0)

    # ---------------------------------------------------- explanations
    # FIGO criteria + saliency come from the explainer; risk is overridden with
    # the ensemble value so the narrative reports the delivered model's number.
    exps = []
    with CTGExplainer(eng.models[0], dev, threshold=thr) as ex:
        for i, (x, feats) in enumerate(windows):
            e = ex.explain(torch.tensor(x, dtype=torch.float32),
                           torch.tensor(feats, dtype=torch.float32))
            e["risk_score"] = float(risk[i])
            e["flagged"] = bool(risk[i] >= thr)
            e["threshold"] = thr
            exps.append(e)
        narrations = [ex.narrate(e) for e in exps]

    # ------------------------------------------------------ patient report
    print()
    rule("2. PATIENT REPORT -- what the clinician sees first")
    rep = build_patient_report(a.record, exps, np.array(starts), threshold=thr, fs=FS)
    print(rep.render(width=W))

    # -------------------------------------------------- window explanations
    print()
    rule("3. WINDOW-LEVEL EXPLANATION -- why each flag fired")
    peak_i = int(np.argmax(risk))
    if a.explain == "all":
        idx = list(range(len(exps)))
    elif a.explain == "peak":
        idx = [peak_i]
    else:
        idx = sorted(set([i for i, e in enumerate(exps) if e["flagged"]] + [peak_i]))

    for i in idx:
        m = starts[i] / (FS * 60)
        tag = "FLAGGED" if exps[i]["flagged"] else "not flagged"
        extra = "  <- peak risk" if i == peak_i else ""
        print(f"\n--- window @ minute {m:.0f}-{m + 20:.0f}  [{tag}]{extra} ---")
        print(narrations[i])
        print(f"  per-fold risk: {np.round(per_fold[:, i], 3).tolist()}")

    if a.explain == "flagged" and len(idx) == 1 and not exps[peak_i]["flagged"]:
        print("\n  (no windows flagged; the peak-risk window is shown above)")

    # ------------------------------------------------------------ closing
    print()
    rule("4. SUMMARY")
    print(f"  peak risk {risk.max():.3f} at minute {starts[peak_i] / (FS * 60):.0f} | "
          f"mean {risk.mean():.3f} | {int((risk >= thr).sum())}/{len(risk)} windows flagged")
    print(f"  risk trend {rep.risk_trend:+.3f}/hour | longest sustained run "
          f"{rep.sustained_run} windows")
    print()
    print("  The saliency line in section 3 is from fold 1 only and illustrates where")
    print("  one model looked; it is not an ensemble property. FIGO criteria are")
    print("  computed from the trace and are model-independent.")
    print("  Decision support only -- verify against the strip.")
    if ph is not None:
        print()
        print(f"  [ground truth, never shown to the model] umbilical pH {ph:.3f} -> "
              f"{'ACIDOTIC' if ph <= 7.15 else 'normal'}")
    rule()


if __name__ == "__main__":
    main()
