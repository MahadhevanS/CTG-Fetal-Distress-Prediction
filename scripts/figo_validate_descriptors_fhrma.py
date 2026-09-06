"""
External validation of src/figo_state/descriptors.py against FHRMA expert
consensus -- the measurement instrument, not the label.

    python scripts/figo_validate_descriptors_fhrma.py

WHY THIS RUNS BEFORE ANY MODEL
------------------------------
Every FIGO state label in this project is a function of these descriptors,
so a systematically wrong descriptor is a systematically wrong label, and no
amount of model work recovers from that. FHRMA (Boudet et al., distributed in
CTGDL v5) carries per-sample expert baseline, acceleration and deceleration
annotation on 152 analysable recordings. It carries NO pH, so nothing here
can leak into the CTU-UHB outcome and nothing here licenses any claim about
acidaemia.

WHAT PHASE 9A ESTABLISHED, AND WHAT IS DIFFERENT NOW
----------------------------------------------------
docs/phase9a_detector_validation.md measured the OLD detectors on the SAME
data: acceleration F1 0.297 (sensitivity 0.204), deceleration F1 0.433
(precision 0.338, i.e. 1.8x over-detection), baseline median error 3.93 bpm.
It also found that repairing the baseline toward expert behaviour cost 0.0972
patient AUROC on the pH task -- so expert agreement was NOT a proxy objective
there.

That finding does not transfer to this project, and the reason is worth
stating precisely: on the pH task the descriptors were INPUTS to a model
predicting an external outcome, so a biased descriptor could still be
predictive. Here the descriptors DEFINE the label. There is no external
outcome to be accidentally right about. Expert agreement is the objective,
not a proxy for it.

WHAT IS COMPARED
----------------
  old   src/preprocessing/features.py + calculate_iterative_baseline, on
        20-minute windows, exactly as Phase 9A ran it
  new   src/figo_state/descriptors.py, on 10-minute epochs, with the
        validity mask that stops unrepaired dropout being read as a
        deceleration

Both are scored against the same expert annotation with the same event
matching (any temporal overlap), so the two rows are comparable.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "src", "preprocessing"))

from figo_state import descriptors as DS  # noqa: E402
from filtering import (apply_lowpass_filter, interpolate_missing,  # noqa: E402
                       remove_spikes)
from baseline import calculate_iterative_baseline  # noqa: E402

FHRMA = os.path.join(BASE, "data", "raw", "fhrma", "CTGDL_FHRMA_ano_csv")
OUT = os.path.join(BASE, "results", "figo_state")
FS = 4.0
EPOCH = int(10 * 60 * FS)          # 2400 -- the unit the new engine uses
OLD_WINDOW = int(20 * 60 * FS)     # 4800 -- the unit Phase 9A used
LOWPASS_SETTLE = int(2.0 * FS)


def runs(mask):
    m = np.asarray(mask).astype(int)
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def match(pred, true):
    """Event-level matching by any temporal overlap. Returns TP, FP, FN."""
    used, tp = set(), 0
    for ps, pe in pred:
        for i, (ts, te) in enumerate(true):
            if i in used:
                continue
            if ps < te and ts < pe:
                used.add(i)
                tp += 1
                break
    return tp, len(pred) - tp, len(true) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def dilate(mask, k):
    if k <= 0 or not mask.any():
        return mask
    return np.convolve(mask.astype(np.int16), np.ones(2 * k + 1, np.int16),
                       mode="same") > 0


def clean(fhr_raw):
    """The shared signal chain. Returns (filtered_fhr, valid_mask)."""
    f = remove_spikes(fhr_raw.astype(np.float64).copy(), fs=FS)
    f = interpolate_missing(f, clip_min=50.0, clip_max=240.0)
    invalid = (f <= 0.0)
    f = apply_lowpass_filter(f, fs=FS)
    return f, ~dilate(invalid, LOWPASS_SETTLE)


def main():
    os.makedirs(OUT, exist_ok=True)
    files = [f for f in sorted(glob.glob(os.path.join(FHRMA, "*.csv")))
             if not f.endswith(".csv.csv")]
    print(f"FHRMA recordings: {len(files)}\n")

    acc = {k: np.zeros(3, int) for k in ("old", "new")}   # tp, fp, fn
    dec = {k: np.zeros(3, int) for k in ("old", "new")}
    base_err = {"old": [], "new": []}
    n_seg = {"old": 0, "new": 0}
    var_new = []
    dice_num, dice_den = [0], [0]
    frac_ours, frac_exp = [], []

    for path in files:
        d = pd.read_csv(path)
        if not {"fhr", "baseline", "acc", "dec"} <= set(d.columns):
            continue
        fhr_raw = d["fhr"].to_numpy(float)
        exp_base = d["baseline"].to_numpy(float)
        exp_acc = d["acc"].to_numpy(float) > 0
        exp_dec = d["dec"].to_numpy(float) > 0
        toco = (d["toco"].to_numpy(float) if "toco" in d.columns
                else np.zeros_like(fhr_raw))

        f, valid = clean(fhr_raw)

        # ---------------------------------------------------- OLD, 20-min
        for s in range(0, len(f) - OLD_WINDOW + 1, OLD_WINDOW):
            e = s + OLD_WINDOW
            if valid[s:e].mean() < 0.5:
                continue
            n_seg["old"] += 1
            b = calculate_iterative_baseline(f[s:e])
            base_err["old"].append(np.abs(b - exp_base[s:e]))
            pa = [(x, y) for x, y in runs((f[s:e] - b) >= 15.0)
                  if y - x >= 60]
            pd_ = [(x, y) for x, y in runs((b - f[s:e]) >= 15.0)
                   if y - x >= 60]
            acc["old"] += np.array(match(pa, runs(exp_acc[s:e])))
            dec["old"] += np.array(match(pd_, runs(exp_dec[s:e])))

        # ---------------------------------------------------- NEW, 10-min
        prev_b = None
        for s in range(0, len(f) - EPOCH + 1, EPOCH):
            e = s + EPOCH
            v = valid[s:e]
            if v.mean() < 0.5:
                continue
            n_seg["new"] += 1
            bval, _ = DS.estimate_baseline(f[s:e], v, prev_b)
            prev_b = bval
            base_err["new"].append(np.abs(bval - exp_base[s:e]))

            # Exercise the SHIPPING delineation, not a re-implementation of
            # it: an inline copy here silently stopped tracking descriptors.py
            # once delineate_events() was introduced, and reported stale
            # numbers for a whole build.
            need = int(DS.EVENT_MIN_DURATION_S * FS)
            pa = [(x, y) for x, y in
                  DS.delineate_events(f[s:e], bval, FS, v, +1) if y - x >= need]
            pd_ = [(x, y) for x, y in
                   DS.delineate_events(f[s:e], bval, FS, v, -1) if y - x >= need]
            acc["new"] += np.array(match(pa, runs(exp_acc[s:e])))
            dec["new"] += np.array(match(pd_, runs(exp_dec[s:e])))

            # Sample-level overlap. Event matching is by ANY overlap, so it
            # cannot see a span that is far too wide; Dice can.
            mk = np.zeros(e - s, bool)
            for x, y in pd_:
                mk[x:y] = True
            dice_num[0] += int((mk & exp_dec[s:e]).sum())
            dice_den[0] += int(mk.sum() + exp_dec[s:e].sum())
            frac_ours.append(float(mk.mean()))
            frac_exp.append(float(exp_dec[s:e].mean()))

            ev = DS.event_mask(f[s:e], bval, FS, v)
            vb, ok = DS.measure_variability(f[s:e], bval, FS, ev, v)
            if ok:
                var_new.append(vb)

    print(f"{'':22s}{'sens':>8s}{'prec':>8s}{'F1':>8s}")
    res = {}
    for kind, store in (("acceleration", acc), ("deceleration", dec)):
        for v in ("old", "new"):
            tp, fp, fn = store[v]
            p, r, f1 = prf(tp, fp, fn)
            res[f"{kind}_{v}"] = dict(tp=int(tp), fp=int(fp), fn=int(fn),
                                      precision=p, sensitivity=r, f1=f1)
            print(f"{kind + ' ' + v:22s}{r:8.3f}{p:8.3f}{f1:8.3f}")
        print()

    print(f"{'baseline':22s}{'MAD':>8s}{'<=5bpm':>8s}{'segments':>10s}")
    for v in ("old", "new"):
        a = np.concatenate(base_err[v]) if base_err[v] else np.array([np.nan])
        res[f"baseline_{v}"] = dict(mad=float(np.nanmedian(a)),
                                    within5=float(np.nanmean(a <= 5)),
                                    n_segments=n_seg[v])
        print(f"{'  ' + v:22s}{np.nanmedian(a):8.2f}"
              f"{100 * np.nanmean(a <= 5):8.1f}{n_seg[v]:10d}")

    if var_new:
        vn = np.array(var_new)
        res["variability_new"] = dict(
            mean=float(vn.mean()), median=float(np.median(vn)),
            frac_below5=float((vn < 5).mean()),
            frac_normal=float(((vn >= 5) & (vn <= 25)).mean()),
            frac_above25=float((vn > 25).mean()))
        print(f"\nnew variability on FHRMA: mean {vn.mean():.1f} "
              f"median {np.median(vn):.1f} | <5 {100 * (vn < 5).mean():.1f}%  "
              f"5-25 {100 * ((vn >= 5) & (vn <= 25)).mean():.1f}%  "
              f">25 {100 * (vn > 25).mean():.1f}%")

    if dice_den[0]:
        dice = 2 * dice_num[0] / dice_den[0]
        res["deceleration_new_dice"] = float(dice)
        res["frac_time_in_decel"] = dict(ours=float(np.mean(frac_ours)),
                                         expert=float(np.mean(frac_exp)))
        print(f"\ndeceleration sample-level Dice: {dice:.3f}")
        print(f"  fraction of time inside a deceleration: "
              f"ours {np.mean(frac_ours):.3f} vs expert {np.mean(frac_exp):.3f}")
        print("  (event matching is by ANY overlap and cannot see a span that")
        print("   is far too wide; Dice and this fraction can.)")

    print("\nPhase 9A reference (old detectors, same corpus):")
    print("  acceleration F1 0.297 (sens 0.204) | deceleration F1 0.433 "
          "(prec 0.338) | baseline MAD 3.93, within 5 bpm 70.9%")
    print("  expert-baseline upper bound: acc F1 0.598, dec F1 0.616")

    with open(os.path.join(OUT, "fhrma_descriptor_validation.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}/fhrma_descriptor_validation.json")


if __name__ == "__main__":
    main()
