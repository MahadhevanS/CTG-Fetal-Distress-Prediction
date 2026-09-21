"""
Why does deceleration detection disagree with expert consensus?

    python scripts/figo_diagnose_decel_detection.py

Phase 9A measured the disagreement (sensitivity 0.604, precision 0.338) and
attributed roughly half of it to the constant baseline. It did not decompose
the remainder. This script does, because the deceleration pathway to
Pathological currently fires ZERO times
(docs/figo_state_gate1_gate2.md) and that is the binding constraint on the
whole early-warning task.

Four candidate causes, each measured separately:

  A FRAGMENTATION. One expert deceleration crossing back above the 15 bpm
    threshold briefly is detected by us as two or more events. Every extra
    fragment is a false positive, so precision collapses without us
    detecting anything spurious at all. Test: merge our runs separated by
    less than a tolerance and re-score.

  B BASELINE OFFSET. A constant baseline sits above or below the expert's
    drifting one, so a real deceleration either fails to reach 15 bpm
    (missed) or a normal segment appears to (spurious). Test: score against
    the EXPERT baseline with our rule unchanged -- Phase 9A's ablation, but
    with the fragmentation fix applied so the two effects separate.

  C DURATION FLOOR. We require 15 s. Experts annotate onset-to-recovery,
    which includes the shoulders, so an event we see for 12 s may be
    annotated as 40 s. Test: sweep the floor.

  D AMPLITUDE FLOOR. Same argument at 15 bpm.

Nothing here uses any CTU-UHB outcome. FHRMA carries no pH, so anything
tuned against it cannot leak into an acidaemia claim -- and in this project
the descriptors DEFINE the label rather than predict an outcome, so expert
agreement is the objective, not a proxy for it.
"""

import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "src", "preprocessing"))

from figo_state import descriptors as DS  # noqa: E402
from filtering import (apply_lowpass_filter, interpolate_missing,  # noqa: E402
                       remove_spikes)

FHRMA = os.path.join(BASE, "data", "raw", "fhrma", "CTGDL_FHRMA_ano_csv")
OUT = os.path.join(BASE, "results", "figo_state")
FS = 4.0
EPOCH = int(10 * 60 * FS)
SETTLE = int(2.0 * FS)


def runs(mask):
    m = np.asarray(mask).astype(int)
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def merge(spans, gap):
    """Merge spans separated by fewer than `gap` samples."""
    if not spans:
        return []
    out = [list(spans[0])]
    for s, e in spans[1:]:
        if s - out[-1][1] < gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def match(pred, true):
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
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def dilate(mask, k):
    if k <= 0 or not mask.any():
        return mask
    return np.convolve(mask.astype(np.int16), np.ones(2 * k + 1, np.int16),
                       mode="same") > 0


def load_segments():
    """Yield (fhr, valid, expert_baseline, expert_dec_mask) per 10-min epoch."""
    files = [f for f in sorted(glob.glob(os.path.join(FHRMA, "*.csv")))
             if not f.endswith(".csv.csv")]
    segs = []
    for path in files:
        d = pd.read_csv(path)
        if not {"fhr", "baseline", "dec"} <= set(d.columns):
            continue
        raw = d["fhr"].to_numpy(float)
        f = remove_spikes(raw.copy(), fs=FS)
        f = interpolate_missing(f, clip_min=50.0, clip_max=240.0)
        invalid = f <= 0.0
        f = apply_lowpass_filter(f, fs=FS)
        valid = ~dilate(invalid, SETTLE)
        eb = d["baseline"].to_numpy(float)
        ed = d["dec"].to_numpy(float) > 0
        prev = None
        for s in range(0, len(f) - EPOCH + 1, EPOCH):
            e = s + EPOCH
            if valid[s:e].mean() < 0.5:
                continue
            ours, _ = DS.estimate_baseline(f[s:e], valid[s:e], prev)
            prev = ours
            segs.append((f[s:e], valid[s:e], eb[s:e], ed[s:e], ours))
    return segs


def score(segs, baseline_kind, gap_s, amp, dur_s):
    gap = int(gap_s * FS)
    dur = int(dur_s * FS)
    tot = np.zeros(3, int)
    for f, v, eb, ed, ours in segs:
        b = eb if baseline_kind == "expert" else ours
        sp = [(s, e) for s, e in merge(runs(((b - f) >= amp) & v), gap)
              if e - s >= dur]
        tot += np.array(match(sp, runs(ed)))
    return tot


def main():
    os.makedirs(OUT, exist_ok=True)
    segs = load_segments()
    n_exp = sum(len(runs(s[3])) for s in segs)
    print(f"{len(segs)} analysable 10-min epochs | {n_exp} expert decelerations\n")

    res = {}
    print("A. FRAGMENTATION -- merge our runs separated by < gap")
    print(f"  {'gap (s)':>9s}{'ours':>8s}{'sens':>8s}{'prec':>8s}{'F1':>8s}")
    best_gap, best_f1 = 0.0, -1.0
    for g in (0, 5, 10, 15, 20, 30, 45, 60):
        tp, fp, fn = score(segs, "ours", g, 15.0, 15.0)
        p, r, f1 = prf(tp, fp, fn)
        print(f"  {g:9.0f}{tp + fp:8d}{r:8.3f}{p:8.3f}{f1:8.3f}")
        res[f"merge_gap_{g}"] = dict(sens=r, prec=p, f1=f1, n_pred=int(tp + fp))
        if f1 > best_f1:
            best_gap, best_f1 = float(g), f1
    print(f"  -> best gap {best_gap:.0f}s, F1 {best_f1:.3f}\n")

    print("B. BASELINE -- ours vs expert, at the best merge gap")
    print(f"  {'baseline':>10s}{'sens':>8s}{'prec':>8s}{'F1':>8s}")
    for kind in ("ours", "expert"):
        tp, fp, fn = score(segs, kind, best_gap, 15.0, 15.0)
        p, r, f1 = prf(tp, fp, fn)
        print(f"  {kind:>10s}{r:8.3f}{p:8.3f}{f1:8.3f}")
        res[f"baseline_{kind}"] = dict(sens=r, prec=p, f1=f1)
    print()

    print("C+D. AMPLITUDE x DURATION sweep (our baseline, best merge gap)")
    print(f"  {'amp':>5s}{'dur':>6s}{'sens':>8s}{'prec':>8s}{'F1':>8s}")
    grid = []
    for amp, dur in itertools.product((10.0, 12.0, 15.0, 18.0),
                                      (10.0, 15.0, 20.0, 30.0)):
        tp, fp, fn = score(segs, "ours", best_gap, amp, dur)
        p, r, f1 = prf(tp, fp, fn)
        grid.append((amp, dur, r, p, f1))
        print(f"  {amp:5.0f}{dur:6.0f}{r:8.3f}{p:8.3f}{f1:8.3f}")
    res["grid"] = [dict(amp=a, dur=d, sens=r, prec=p, f1=f) for a, d, r, p, f in grid]
    b = max(grid, key=lambda x: x[4])
    print(f"\n  best: amplitude {b[0]:.0f} bpm, duration {b[1]:.0f} s "
          f"-> sens {b[2]:.3f} prec {b[3]:.3f} F1 {b[4]:.3f}")

    print("\nreference points")
    print("  as shipped (gap 0, 15 bpm, 15 s)     F1 0.454 (sens 0.588 prec 0.370)")
    print("  Phase 9A old detector                F1 0.433 (sens 0.604 prec 0.338)")
    print("  Phase 9A expert-baseline upper bound F1 0.616")

    res["chosen"] = dict(merge_gap_s=best_gap, amplitude_bpm=b[0],
                         duration_s=b[1], sens=b[2], prec=b[3], f1=b[4])
    with open(os.path.join(OUT, "decel_diagnosis.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}/decel_diagnosis.json")


if __name__ == "__main__":
    main()
