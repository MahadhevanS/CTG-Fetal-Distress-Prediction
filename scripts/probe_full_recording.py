"""
Phase 8, Route 3 -- does the signal discarded by the 60-minute crop help?

pipeline_clinical.py truncates every record to its last 60 minutes. Phase 7
called the discarded remainder "19 % of signal, AVAILABLE, DISCARDED" and
recommended recovering it. Phase 8 §4.2 showed the justification offered for
that recommendation was invalid (window count is a quality measure, not a length
measure, and it correlates NEGATIVELY with record length). The experiment itself
was never run. This runs it.

Structural fact worth stating up front: the discarded portion is the EARLIEST
part of the first stage, and its median duration is 11.4 minutes -- shorter than
the 20-minute window. Only 160 of 552 records discard >= 20 minutes, i.e. enough
to form even one complete window lying wholly before the final hour.

Design:
  A  last-60 crop           -- must reproduce the established 0.7271, or the
                               re-implementation is wrong and nothing else here
                               can be believed
  B  full recording, max    -- the actual question
  C  full recording, mean   -- max over a bigger bag has more chances to be
                               high, so report an aggregator that does not
  D  pre-hour windows only  -- does the discarded signal carry anything at all?
  E  bag-size control       -- longer record => longer second stage => higher
                               risk. Check that window count does not predict
                               the label, and repeat B with the bag size fixed
                               at 17 windows.

Run from the repo root:
    python scripts/probe_full_recording.py
"""
import json
import os
import sys
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
from src.preprocessing.baseline import calculate_iterative_baseline
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
OUT = "results/phase8"
CACHE = os.path.join(OUT, "full_recording_windows.npz")

# identical to pipeline_clinical.py
WINDOW_SAMPLES, STRIDE_SAMPLES, LAST_HOUR = 4800, 600, 14400
WINDOW_MAX_MISSING = 0.5
FHR_MIN_BPM, FHR_MAX_BPM = 50.0, 240.0


def window_descriptors(raw_fhr, raw_uc, fs):
    """The 19 descriptors, computed exactly as pipeline_clinical.py does."""
    f = remove_spikes(raw_fhr.copy(), fs=fs)
    f = interpolate_missing(f, clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)
    f = apply_lowpass_filter(f, fs=fs)
    u = apply_lowpass_filter(raw_uc.copy(), fs=fs)
    baseline = calculate_iterative_baseline(f)
    stv, ltv = calculate_variability(f, fs=fs, baseline=baseline)
    accels = detect_accelerations(f, baseline, fs=fs)
    decels = detect_decelerations(f, baseline, u, fs=fs)
    base_val = float(np.mean(baseline))
    base8 = [base_val, stv, ltv, float(accels), float(decels["early"]),
             float(decels["late"]), float(decels["variable"]),
             float(decels["prolonged"])]
    ext = extract_extended_features(f - baseline, u, base_val, fs=fs)
    return np.concatenate([np.array(base8, np.float32), ext])


def build(pids):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if list(z["order"]) == list(pids):
            print(f"[cache] {CACHE}")
            return z["F"], z["pid"], z["prehour"], z["cropped"]
    F, PID, PRE, CROP = [], [], [], []
    for n, rec in enumerate(pids):
        fhr, uc, fs = load_ctu_chb_record(os.path.join(RAW, str(rec)))
        full_len = len(fhr)
        hour_start = max(0, full_len - LAST_HOUR)
        for start in get_valid_windows(fhr, WINDOW_SAMPLES, STRIDE_SAMPLES,
                                       max_missing_ratio=WINDOW_MAX_MISSING):
            end = start + WINDOW_SAMPLES
            F.append(window_descriptors(fhr[start:end], uc[start:end], fs))
            PID.append(str(rec))
            PRE.append(1.0 if end <= hour_start else 0.0)
            CROP.append(1.0 if start >= hour_start else 0.0)
        if n % 100 == 0:
            print(f"  {n}/{len(pids)}", flush=True)
    F = np.vstack(F)
    PID = np.array(PID)
    PRE = np.array(PRE, np.float32)
    CROP = np.array(CROP, np.float32)
    os.makedirs(OUT, exist_ok=True)
    np.savez(CACHE, F=F, pid=PID, prehour=PRE, cropped=CROP, order=np.array(pids))
    return F, PID, PRE, CROP


def main():
    # cohort and labels from the frozen protocol
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

    F, pid, prehour, cropped = build(pids)
    y = np.array([lab[p] for p in pid])
    print(f"windows: {len(F)} total, {int(cropped.sum())} inside the last hour, "
          f"{int(prehour.sum())} wholly before it")

    blob = json.load(open(os.path.join(PROC, "folds.json")))

    def evaluate(name, mask, how="max", fixed_bag=0):
        m = mask.astype(bool)
        Fm, pm, ym = F[m], pid[m], y[m]
        if fixed_bag:
            keep = np.zeros(len(pm), bool)
            for p in np.unique(pm):
                idx = np.where(pm == p)[0]
                sel = np.linspace(0, len(idx) - 1, min(fixed_bag, len(idx)))
                keep[idx[np.unique(sel.astype(int))]] = True
            Fm, pm, ym = Fm[keep], pm[keep], ym[keep]
        asg = {k: v for k, v in blob["assignment"].items() if k in set(pm)}
        P = Protocol(pm, ym, asg)
        oof = np.zeros(len(Fm))
        cnt = np.zeros(len(Fm))
        for tr, te_p in P.folds(repeat=0):
            te = np.isin(pm, te_p)
            clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(max_iter=5000, C=0.1,
                                                   class_weight="balanced"))
            clf.fit(Fm[tr], ym[tr])
            oof[te] += clf.predict_proba(Fm[te])[:, 1]
            cnt[te] += 1
        return P.report(name, oof / np.maximum(cnt, 1), how=how)

    print("\n--- A: reproduction check (must land on 0.7271) ---")
    a = evaluate("last-60 crop, max        [established = 0.7271]", cropped)

    print("\n--- B/C: the full recording ---")
    allw = np.ones(len(F), np.float32)
    b = evaluate("full recording, max", allw)
    c = evaluate("full recording, mean", allw, how="mean")
    a_mean = evaluate("last-60 crop, mean", cropped, how="mean")

    print("\n--- D: does the discarded signal carry anything alone? ---")
    npre = np.array([prehour[pid == p].sum() for p in pids])
    print(f"    {(npre > 0).sum()}/{len(pids)} patients have any pre-hour window; "
          f"median {np.median(npre[npre > 0]):.0f} such windows")
    d_ = evaluate("pre-hour windows ONLY, max", prehour)

    print("\n--- E: bag-size controls ---")
    nw = np.array([(pid == p).sum() for p in pids])
    yl = np.array([lab[p] for p in pids])
    print(f"    windows per patient: median {np.median(nw):.0f}, "
          f"range {nw.min()}-{nw.max()}")
    print(f"    AUROC(window count -> label) = {roc_auc_score(yl, nw):.4f}")
    e = evaluate("full recording, max, bag fixed at 17", allw, fixed_bag=17)

    print(f"\nestablished {a['auroc']:.4f} | pre-registered bar +0.0642")
    for nm, r in [("full max", b), ("full mean", c), ("crop mean", a_mean),
                  ("pre-hour only", d_), ("full max, bag=17", e)]:
        print(f"  {nm:18s} {r['auroc'] - a['auroc']:+.4f}")


if __name__ == "__main__":
    main()
