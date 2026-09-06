"""
Phase 9A -- validate this project's FHR event detectors against expert consensus.

WHY
---
Every one of the 19 clinical descriptors underpinning the 0.7271 baseline, and
therefore every null result in Phases 5-8, is produced by THIS REPO'S detectors:

    calculate_iterative_baseline   (src/preprocessing/baseline.py)
    detect_accelerations           (src/preprocessing/features.py)
    detect_decelerations           (src/preprocessing/features.py)

They have never been checked against expert ground truth, because none was
available. FHRMA (Boudet et al., distributed inside CTGDL v5) provides
expert-consensus baseline / acceleration / deceleration annotations on 156
recordings. This script is that check.

FHRMA has NO pH outcome. This validates the measurement instrument only; it
cannot and does not make any claim about acidaemia prediction.

WHAT IS MEASURED
  A. baseline agreement, per sample (mean abs diff, RMSE, bias, % within
     5/10 bpm, correlation)
  B. event detection, per event (sensitivity, precision, F1) and per sample
     (Dice), for accelerations and decelerations
  C. the ablation that localises any error:
       our baseline    + our event rule   vs expert events
       EXPERT baseline + our event rule   vs expert events
     If the second is much better, the fault is in the baseline estimator
     rather than in the event rule.

TWO PREPROCESSING VARIANTS
  A "as-shipped": exactly what pipeline_clinical.py does -- cubic interpolation
     of gaps <= 15 s, longer gaps left as 0.0, then low-pass, then baseline.
  B "gaps-filled": same but every gap interpolated first. The difference
     isolates damage caused by residual zeros reaching the baseline estimator.

Note on interpretation: disagreement is NOT automatically our error. FHRMA
experts annotate by their own consensus criteria, which need not match the FIGO
15 bpm / 15 s rule these detectors implement. Direction and magnitude are
reported so the two causes can be told apart.

Run from the repo root:
    python scripts/validate_detectors_fhrma.py
"""
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.preprocessing.baseline import calculate_iterative_baseline
from src.preprocessing.filtering import (apply_lowpass_filter,
                                         interpolate_missing, remove_spikes)

FHRMA = "data/raw/fhrma/CTGDL_FHRMA_ano_csv"
OUT = "results/phase9"
FS = 4.0
WINDOW = int(20 * 60 * FS)            # 4800, the unit the 19 descriptors use
MAX_MISSING = 0.50                    # pipeline_clinical's window gate
THRESH_BPM = 15.0                     # FIGO, as implemented in features.py
DUR_SAMPLES = int(15 * FS)


def runs(mask):
    """Contiguous True spans as (start, end) pairs."""
    m = np.asarray(mask).astype(int)
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def our_events(fhr, baseline, kind):
    """The span logic of detect_accelerations / detect_decelerations, verbatim.

    Those functions return only counts, so the identical rule is reproduced
    here to obtain spans. Thresholds and duration are unchanged.
    """
    delta = (fhr - baseline) if kind == "acc" else (baseline - fhr)
    return [(s, e) for s, e in runs(delta >= THRESH_BPM)
            if (e - s) >= DUR_SAMPLES]


def match(pred, true):
    """Event-level matching by any temporal overlap. Returns TP, FP, FN."""
    used = set()
    tp = 0
    for ps, pe in pred:
        for i, (ts, te) in enumerate(true):
            if i in used:
                continue
            if ps < te and ts < pe:          # overlap
                used.add(i)
                tp += 1
                break
    return tp, len(pred) - tp, len(true) - tp


def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else np.nan


def prepare(fhr_raw, fill_all):
    """The pipeline_clinical signal chain. Returns (fhr, bad_mask)."""
    f = remove_spikes(fhr_raw.astype(np.float64).copy(), fs=FS)
    f = interpolate_missing(f, clip_min=50.0, clip_max=240.0)
    bad = f <= 0.0
    if fill_all and bad.any() and (~bad).sum() > 10:
        idx = np.arange(len(f))
        f[bad] = np.interp(idx[bad], idx[~bad], f[~bad])
        bad = np.zeros(len(f), bool)
    elif bad.any() and (~bad).sum() > 10:
        # keep the zeros (as-shipped) but remember where they are for metrics
        pass
    f = apply_lowpass_filter(f, fs=FS)
    return f, bad


def main():
    files = [f for f in sorted(glob.glob(os.path.join(FHRMA, "*.csv")))
             if not f.endswith(".csv.csv")]
    print(f"FHRMA recordings: {len(files)}")

    rows = {v: [] for v in ("as_shipped", "gaps_filled")}
    for path in files:
        d = pd.read_csv(path)
        fhr_raw = d["fhr"].to_numpy(float)
        exp_base = d["baseline"].to_numpy(float)
        exp_acc = d["acc"].to_numpy(float) > 0
        exp_dec = d["dec"].to_numpy(float) > 0

        for variant, fill in (("as_shipped", False), ("gaps_filled", True)):
            f, bad = prepare(fhr_raw, fill)
            for st in range(0, len(f) - WINDOW + 1, WINDOW):
                sl = slice(st, st + WINDOW)
                raw_w = fhr_raw[sl]
                if (raw_w <= 0).mean() > MAX_MISSING:
                    continue
                fw, bw = f[sl], bad[sl]
                eb, ea, ed = exp_base[sl], exp_acc[sl], exp_dec[sl]
                if np.isnan(eb).all():
                    continue

                ours_base = calculate_iterative_baseline(fw)
                ok = (~bw) & np.isfinite(eb)
                if ok.sum() < FS * 120:
                    continue

                diff = ours_base[ok] - eb[ok]
                rec = dict(
                    file=os.path.basename(path), start=st,
                    base_mad=float(np.mean(np.abs(diff))),
                    base_rmse=float(np.sqrt(np.mean(diff ** 2))),
                    base_bias=float(np.mean(diff)),
                    base_w5=float(np.mean(np.abs(diff) <= 5)),
                    base_w10=float(np.mean(np.abs(diff) <= 10)),
                    base_r=float(np.corrcoef(ours_base[ok], eb[ok])[0, 1])
                    if np.std(eb[ok]) > 1e-6 else np.nan,
                )

                for kind, exp_mask in (("acc", ea), ("dec", ed)):
                    true_ev = runs(exp_mask)
                    # (i) our baseline + our rule
                    pred_o = our_events(fw, ours_base, kind)
                    tp, fp, fn = match(pred_o, true_ev)
                    rec[f"{kind}_tp"], rec[f"{kind}_fp"], rec[f"{kind}_fn"] = tp, fp, fn
                    rec[f"{kind}_n_true"] = len(true_ev)
                    rec[f"{kind}_n_pred"] = len(pred_o)
                    pm = np.zeros(len(fw), bool)
                    for s, e in pred_o:
                        pm[s:e] = True
                    rec[f"{kind}_dice"] = dice(pm, exp_mask)
                    # (ii) EXPERT baseline + our rule -- localises the error
                    eb_f = np.where(np.isfinite(eb), eb, np.nanmedian(eb))
                    pred_e = our_events(fw, eb_f, kind)
                    tpe, fpe, fne = match(pred_e, true_ev)
                    rec[f"{kind}_tp_eb"], rec[f"{kind}_fp_eb"], rec[f"{kind}_fn_eb"] = tpe, fpe, fne
                rows[variant].append(rec)

    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for variant, rs in rows.items():
        df = pd.DataFrame(rs)
        df.to_csv(os.path.join(OUT, f"fhrma_detector_{variant}.csv"), index=False)
        print(f"\n{'='*78}\n{variant.upper()}  --  {len(df)} windows from "
              f"{df.file.nunique()} recordings\n{'='*78}")
        print("A. BASELINE vs expert consensus (bpm)")
        for k, lab in (("base_mad", "mean abs difference"),
                       ("base_rmse", "RMSE"),
                       ("base_bias", "bias (ours - expert)")):
            print(f"   {lab:24s} median {df[k].median():7.3f}   "
                  f"IQR {df[k].quantile(.25):6.3f} - {df[k].quantile(.75):6.3f}")
        print(f"   {'within 5 bpm':24s} {100*df.base_w5.mean():6.1f} %")
        print(f"   {'within 10 bpm':24s} {100*df.base_w10.mean():6.1f} %")
        print(f"   {'correlation':24s} median {df.base_r.median():7.3f}")

        print("\nB. EVENT DETECTION (event level, any-overlap matching)")
        print(f"   {'':12s} {'expert':>8s} {'ours':>8s} {'sens':>7s} {'prec':>7s} "
              f"{'F1':>7s} {'Dice':>7s}")
        st = {}
        for kind, lab in (("acc", "accelerations"), ("dec", "decelerations")):
            tp, fp, fn = (df[f"{kind}_tp"].sum(), df[f"{kind}_fp"].sum(),
                          df[f"{kind}_fn"].sum())
            sens = tp / max(tp + fn, 1)
            prec = tp / max(tp + fp, 1)
            f1 = 2 * sens * prec / max(sens + prec, 1e-9)
            print(f"   {lab:12s} {int(tp+fn):8d} {int(tp+fp):8d} {sens:7.3f} "
                  f"{prec:7.3f} {f1:7.3f} {df[f'{kind}_dice'].median():7.3f}")
            st[kind] = dict(sens=sens, prec=prec, f1=f1)

        print("\nC. ABLATION -- same rule on the EXPERT baseline")
        print(f"   {'':12s} {'sens':>7s} {'prec':>7s} {'F1':>7s}   (delta vs ours)")
        for kind, lab in (("acc", "accelerations"), ("dec", "decelerations")):
            tp, fp, fn = (df[f"{kind}_tp_eb"].sum(), df[f"{kind}_fp_eb"].sum(),
                          df[f"{kind}_fn_eb"].sum())
            sens = tp / max(tp + fn, 1)
            prec = tp / max(tp + fp, 1)
            f1 = 2 * sens * prec / max(sens + prec, 1e-9)
            print(f"   {lab:12s} {sens:7.3f} {prec:7.3f} {f1:7.3f}   "
                  f"({sens-st[kind]['sens']:+.3f} {prec-st[kind]['prec']:+.3f} "
                  f"{f1-st[kind]['f1']:+.3f})")
        summary[variant] = dict(
            n_windows=int(len(df)), n_recordings=int(df.file.nunique()),
            base_mad_median=float(df.base_mad.median()),
            base_within5=float(df.base_w5.mean()),
            events=st)
    json.dump(summary, open(os.path.join(OUT, "fhrma_detector_summary.json"), "w"),
              indent=1)
    print(f"\nwrote {OUT}/fhrma_detector_*.csv and fhrma_detector_summary.json")


if __name__ == "__main__":
    main()
