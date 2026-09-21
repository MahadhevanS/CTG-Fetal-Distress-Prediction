"""
Phase 9A part 3 -- does the baseline defect actually change any conclusion?

Phase 9A established, against FHRMA expert consensus, that
calculate_iterative_baseline emits one constant per 20-minute window while the
expert baseline moves a median of 14.2 bpm within such a window, and that a
rolling variant improves event-detection F1 (acc 0.297 -> 0.464,
dec 0.433 -> 0.497).

The question that matters for Phases 5-8 is different and is answered here:

    if the 19 descriptors are recomputed with the repaired baseline, does the
    frozen-protocol patient-level AUROC move?

If it does not, the instrument defect is real but does not affect the ranking
the project's conclusions rest on. If it does, Phases 5-8 need revisiting.

Nothing about the protocol is changed: same 547 patients, same folds.json, same
patient-level max aggregation, same logistic regression.

Run from the repo root:
    python scripts/probe_baseline_repair_impact.py
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.knowledge.extended_features import extract_extended_features
from src.preprocessing.baseline import (calculate_iterative_baseline,
                                        rolling_iterative_baseline)
from src.preprocessing.features import (calculate_variability,
                                        detect_accelerations,
                                        detect_decelerations)
from src.preprocessing.filtering import (apply_lowpass_filter,
                                         interpolate_missing, remove_spikes)
from src.preprocessing.ingestion import load_ctu_chb_record
from src.preprocessing.signal_quality import get_valid_windows
from src.training.protocol import Protocol

RAW = "data/raw/ctu-chb-intrapartum"
PROC = "data/processed_clinical"
OUT = "results/phase9"
CACHE = os.path.join(OUT, "baseline_repair_features.npz")
WIN, STRIDE, LAST_HOUR = 4800, 600, 14400
FS = 4.0


def descriptors(raw_fhr, raw_uc, baseline_fn):
    f = remove_spikes(raw_fhr.copy(), fs=FS)
    f = interpolate_missing(f, clip_min=50.0, clip_max=240.0)
    f = apply_lowpass_filter(f, fs=FS)
    u = apply_lowpass_filter(raw_uc.copy(), fs=FS)
    baseline = baseline_fn(f)
    stv, ltv = calculate_variability(f, fs=FS, baseline=baseline)
    accels = detect_accelerations(f, baseline, fs=FS)
    decels = detect_decelerations(f, baseline, u, fs=FS)
    base_val = float(np.mean(baseline))
    base8 = [base_val, stv, ltv, float(accels), float(decels["early"]),
             float(decels["late"]), float(decels["variable"]),
             float(decels["prolonged"])]
    ext = extract_extended_features(f - baseline, u, base_val, fs=FS)
    return np.concatenate([np.array(base8, np.float32), ext])


def build(pids):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if list(z["order"]) == list(pids):
            print(f"[cache] {CACHE}")
            return z["F_const"], z["F_roll"], z["pid"]
    Fc, Fr, PID = [], [], []
    t0 = time.time()
    for n, rec in enumerate(pids):
        fhr, uc, fs = load_ctu_chb_record(os.path.join(RAW, str(rec)))
        if len(fhr) > LAST_HOUR:
            fhr, uc = fhr[-LAST_HOUR:], uc[-LAST_HOUR:]
        for st in get_valid_windows(fhr, WIN, STRIDE, max_missing_ratio=0.5):
            a, b = fhr[st:st + WIN], uc[st:st + WIN]
            Fc.append(descriptors(a, b, calculate_iterative_baseline))
            Fr.append(descriptors(a, b, lambda x: rolling_iterative_baseline(
                x, fs=FS, win_sec=300.0)))
            PID.append(str(rec))
        if n % 100 == 0:
            print(f"  {n}/{len(pids)}  {time.time()-t0:.0f}s", flush=True)
    Fc, Fr, PID = np.vstack(Fc), np.vstack(Fr), np.array(PID)
    os.makedirs(OUT, exist_ok=True)
    np.savez(CACHE, F_const=Fc, F_roll=Fr, pid=PID, order=np.array(pids))
    return Fc, Fr, PID


def main():
    lab = {}
    for sp in ("train", "val", "test"):
        d = torch.load(os.path.join(PROC, f"{sp}_dataset.pt"), weights_only=False)
        ids = np.array([m[0] for m in d["metadata"]])
        y = d["y_primary"].numpy()
        for p in np.unique(ids):
            lab[p] = int(y[ids == p].max())
        del d
    pids = sorted(lab)
    print(f"cohort {len(pids)} patients, {sum(lab.values())} positive")

    Fc, Fr, pid = build(pids)
    y = np.array([lab[p] for p in pid])
    print(f"windows {len(y)}  positive {y.sum()}")

    blob = json.load(open(os.path.join(PROC, "folds.json")))
    asg = {k: v for k, v in blob["assignment"].items() if k in set(pid)}
    P = Protocol(pid, y, asg)

    def run(name, M):
        oof = np.zeros(len(M))
        cnt = np.zeros(len(M))
        for tr, te_p in P.folds(repeat=0):
            te = np.isin(pid, te_p)
            clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(max_iter=5000, C=0.1,
                                                   class_weight="balanced"))
            clf.fit(M[tr], y[tr])
            oof[te] += clf.predict_proba(M[te])[:, 1]
            cnt[te] += 1
        oof /= np.maximum(cnt, 1)
        r = P.report(name, oof, how="max")
        r["window_auroc"] = float(roc_auc_score(y, oof))
        return r

    print("\n--- frozen protocol, patient-level max aggregation ---")
    a = run("19 descriptors, CONSTANT baseline (as shipped)", Fc)
    b = run("19 descriptors, ROLLING 5-min baseline (repaired)", Fr)
    c = run("both baselines' descriptors concatenated (38)",
            np.hstack([Fc, Fr]))
    print(f"\n  window-level: constant {a['window_auroc']:.4f}  "
          f"rolling {b['window_auroc']:.4f}")
    print(f"\n  patient-level delta (rolling - constant): "
          f"{b['auroc'] - a['auroc']:+.4f}   [bar +0.0642]")
    print(f"  concatenated vs constant:                 "
          f"{c['auroc'] - a['auroc']:+.4f}")

    # how different are the two feature sets at all?
    names = ["baseline", "stv", "ltv", "accels", "dec_early", "dec_late",
             "dec_var", "dec_prolonged"]
    print("\n  per-feature change (first 8 descriptors):")
    print(f"    {'feature':14s} {'const mean':>11s} {'roll mean':>11s} {'corr':>7s}")
    for j, nm in enumerate(names):
        cc = np.corrcoef(Fc[:, j], Fr[:, j])[0, 1] if np.std(Fc[:, j]) > 0 and np.std(Fr[:, j]) > 0 else np.nan
        print(f"    {nm:14s} {Fc[:, j].mean():11.3f} {Fr[:, j].mean():11.3f} {cc:7.3f}")

    json.dump({k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                   for kk, vv in v.items()}
               for k, v in dict(constant=a, rolling=b, concat=c).items()},
              open(os.path.join(OUT, "baseline_repair_impact.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
