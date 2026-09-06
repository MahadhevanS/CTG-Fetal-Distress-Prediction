"""
Phase 9A part 2 -- can the baseline defect be fixed with code already present?

validate_detectors_fhrma.py established that calculate_iterative_baseline emits
ONE constant per 20-minute window (np.full_like, rounded to 5 bpm) while the
expert baseline moves by a median of 14.2 bpm within such a window.

src/preprocessing/baseline.py already contains asymmetric_least_squares_baseline,
a drift-tracking estimator that is imported by nothing. This compares them
against FHRMA expert consensus.

TUNING DISCLOSURE: the ALS smoothness parameter is selected against FHRMA
baseline agreement only. No CTU-UHB outcome label is involved at any point, so
this cannot leak into the frozen protocol.

Run from the repo root:
    python scripts/compare_baseline_estimators_fhrma.py
"""
import glob
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.preprocessing.baseline import (asymmetric_least_squares_baseline,
                                        calculate_iterative_baseline)
from src.preprocessing.filtering import (apply_lowpass_filter,
                                         interpolate_missing, remove_spikes)

FHRMA = "data/raw/fhrma/CTGDL_FHRMA_ano_csv"
OUT = "results/phase9"
FS = 4.0
WINDOW = int(20 * 60 * FS)
THRESH_BPM, DUR = 15.0, int(15 * FS)


def runs(mask):
    m = np.asarray(mask).astype(int)
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def our_events(fhr, base, kind):
    delta = (fhr - base) if kind == "acc" else (base - fhr)
    return [(s, e) for s, e in runs(delta >= THRESH_BPM) if (e - s) >= DUR]


def match(pred, true):
    used, tp = set(), 0
    for ps, pe in pred:
        for i, (ts, te) in enumerate(true):
            if i not in used and ps < te and ts < pe:
                used.add(i)
                tp += 1
                break
    return tp, len(pred) - tp, len(true) - tp


def prf(tp, fp, fn):
    se = tp / max(tp + fn, 1)
    pr = tp / max(tp + fp, 1)
    return se, pr, 2 * se * pr / max(se + pr, 1e-9)


def main():
    files = [f for f in sorted(glob.glob(os.path.join(FHRMA, "*.csv")))
             if not f.endswith(".csv.csv")]

    lams = [1e4, 1e5, 1e6, 1e7]
    estimators = {"current (constant)": None, "expert (upper bound)": "expert"}
    for l in lams:
        estimators[f"ALS lam={l:.0e}"] = l

    agg = {k: dict(mad=[], w5=[], acc=[0, 0, 0], dec=[0, 0, 0]) for k in estimators}
    t0 = time.time()
    nwin = 0
    for path in files:
        d = pd.read_csv(path)
        fhr_raw = d["fhr"].to_numpy(float)
        eb_all = d["baseline"].to_numpy(float)
        ea_all = d["acc"].to_numpy(float) > 0
        ed_all = d["dec"].to_numpy(float) > 0

        f = remove_spikes(fhr_raw.copy(), fs=FS)
        f = interpolate_missing(f, clip_min=50.0, clip_max=240.0)
        f = apply_lowpass_filter(f, fs=FS)

        for st in range(0, len(f) - WINDOW + 1, WINDOW):
            sl = slice(st, st + WINDOW)
            if (fhr_raw[sl] <= 0).mean() > 0.5:
                continue
            fw, eb = f[sl], eb_all[sl]
            ok = np.isfinite(eb)
            if ok.sum() < FS * 120:
                continue
            ea, ed = ea_all[sl], ed_all[sl]
            true_acc, true_dec = runs(ea), runs(ed)
            nwin += 1

            for name, param in estimators.items():
                if param is None:
                    b = calculate_iterative_baseline(fw)
                elif param == "expert":
                    b = np.where(ok, eb, np.nanmedian(eb))
                else:
                    b = asymmetric_least_squares_baseline(fw, lam=param, p=0.5,
                                                          n_iter=10)
                diff = b[ok] - eb[ok]
                agg[name]["mad"].append(float(np.mean(np.abs(diff))))
                agg[name]["w5"].append(float(np.mean(np.abs(diff) <= 5)))
                for kind, true_ev in (("acc", true_acc), ("dec", true_dec)):
                    tp, fp, fn = match(our_events(fw, b, kind), true_ev)
                    agg[name][kind][0] += tp
                    agg[name][kind][1] += fp
                    agg[name][kind][2] += fn
    print(f"{nwin} windows from {len(files)} recordings, {time.time()-t0:.0f}s\n")

    print(f"{'estimator':22s} {'MAD':>7s} {'<=5bpm':>8s} | "
          f"{'acc sens':>9s} {'acc prec':>9s} {'acc F1':>7s} | "
          f"{'dec sens':>9s} {'dec prec':>9s} {'dec F1':>7s}")
    print("-" * 100)
    summary = {}
    for name, a in agg.items():
        ase, apr, af1 = prf(*a["acc"])
        dse, dpr, df1 = prf(*a["dec"])
        print(f"{name:22s} {np.median(a['mad']):7.3f} {100*np.mean(a['w5']):7.1f}% | "
              f"{ase:9.3f} {apr:9.3f} {af1:7.3f} | {dse:9.3f} {dpr:9.3f} {df1:7.3f}")
        summary[name] = dict(mad_median=float(np.median(a["mad"])),
                             within5=float(np.mean(a["w5"])),
                             acc=dict(sens=ase, prec=apr, f1=af1),
                             dec=dict(sens=dse, prec=dpr, f1=df1))
    os.makedirs(OUT, exist_ok=True)
    json.dump(summary, open(os.path.join(OUT, "baseline_estimator_comparison.json"),
                            "w"), indent=1)
    print(f"\nwrote {OUT}/baseline_estimator_comparison.json")


if __name__ == "__main__":
    main()
