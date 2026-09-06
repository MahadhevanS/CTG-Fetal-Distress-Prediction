"""
Deceleration-burden study, Phases 3 and 4.

    python scripts/figo_decel_burden_study.py

PHASE 3 -- FHRMA VALIDATION (mandatory gate)
  Expert per-sample deceleration annotation on 152 recordings, no pH.
  Compares, on identical epochs and identical contraction detection:
      1. our binary decel_repetitive
      2. our deceleration representation
      3. each continuous burden candidate
  against the same quantity computed from EXPERT decelerations. FHRMA does
  not annotate "repetitive" directly, but it is derivable: pair the EXPERT
  deceleration spans to the same detected contractions and apply the same
  rule. That gives a reference repetitiveness that isolates the deceleration
  detector from everything else.

PHASE 4 -- THRESHOLD INSTABILITY
  Quantifies, on CTU-UHB, how often one deceleration decides the FIGO state,
  and whether the continuous measures are less brittle.

NOTHING HERE IS FITTED TO ANY PREDICTION TARGET. No CTU-UHB outcome, no
early-warning label, and no pH value is read by this script.
"""

import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "src", "preprocessing"))

from figo_state import descriptors as DS                          # noqa: E402
from figo_state.burden import (DecelBurden, compute_burden,       # noqa: E402
                               deceleration_events, flip_sensitivity,
                               pair_to_contractions)
from filtering import (apply_lowpass_filter, interpolate_missing,  # noqa: E402
                       remove_spikes)

FHRMA = os.path.join(BASE, "data", "raw", "fhrma", "CTGDL_FHRMA_ano_csv")
DATA = os.path.join(BASE, "data", "processed_figo")
OUT = os.path.join(BASE, "results", "figo_burden")
FS = 4.0
EPOCH = int(10 * 60 * FS)
SETTLE = int(2.0 * FS)


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def dilate(m, k):
    if k <= 0 or not m.any():
        return m
    return np.convolve(m.astype(np.int16), np.ones(2 * k + 1, np.int16),
                       mode="same") > 0


def runs(mask):
    m = np.asarray(mask).astype(int)
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


# ==========================================================================
# PHASE 3
# ==========================================================================
def phase3() -> Dict:
    rule("PHASE 3 -- FHRMA VALIDATION")
    files = [f for f in sorted(glob.glob(os.path.join(FHRMA, "*.csv")))
             if not f.endswith(".csv.csv")]
    rows: List[Dict] = []

    for path in files:
        d = pd.read_csv(path)
        if not {"fhr", "baseline", "dec", "toco"} <= set(d.columns):
            continue
        raw = d["fhr"].to_numpy(float)
        toco = d["toco"].to_numpy(float)
        exp_dec = d["dec"].to_numpy(float) > 0

        f = remove_spikes(raw.copy(), fs=FS)
        f = interpolate_missing(f, clip_min=50.0, clip_max=240.0)
        invalid = f <= 0.0
        f = apply_lowpass_filter(f, fs=FS)
        u = apply_lowpass_filter(toco.copy(), fs=FS)
        valid = ~dilate(invalid, SETTLE)
        uc_valid = ~dilate(toco == 0.0, SETTLE)

        prev = None
        for s in range(0, len(f) - EPOCH + 1, EPOCH):
            e = s + EPOCH
            vv = valid[s:e]
            if vv.mean() < 0.5:
                continue
            b, _ = DS.estimate_baseline(f[s:e], vv, prev)
            prev = b
            uc_pk = DS.detect_contractions(u[s:e], FS, uc_valid[s:e])
            n_uc = int(uc_pk.size)
            if n_uc == 0:
                continue

            # ---- ours
            ours = deceleration_events(f[s:e], b, FS, vv)
            p_ours = pair_to_contractions(ours, uc_pk, FS)
            bur = compute_burden(f[s:e], u[s:e], FS, b, vv, uc_valid[s:e])

            # ---- expert: same rule, same contractions, expert decel spans
            need = int(DS.EVENT_MIN_DURATION_S * FS)
            ed = []
            for xs, xe in runs(exp_dec[s:e]):
                if (xe - xs) < need:
                    continue
                seg = f[s:e][xs:xe]
                deficit = np.clip(b - seg, 0.0, None)
                ed.append(dict(start=xs, end=xe,
                               nadir=xs + int(np.argmin(seg)),
                               depth_bpm=float(b - seg.min()),
                               duration_s=float((xe - xs) / FS),
                               area_bpm_s=float(deficit.sum() / FS),
                               onset_s=0.0))
            p_exp = pair_to_contractions(ed, uc_pk, FS)
            obs_min = float(vv.sum() / FS / 60.0)

            rows.append(dict(
                n_uc=n_uc,
                # binary criterion, ours vs expert-derived
                rep_ours=int((len(p_ours) / n_uc) > DS.REPETITIVE_FRACTION),
                rep_exp=int((len(p_exp) / n_uc) > DS.REPETITIVE_FRACTION),
                # continuous, ours vs expert-derived
                frac_ours=len(p_ours) / n_uc, frac_exp=len(p_exp) / n_uc,
                area_ours=bur.decel_area_per_min,
                area_exp=float(sum(x["area_bpm_s"] for x in ed) / obs_min)
                if obs_min > 0 else 0.0,
                apc_ours=bur.area_per_contraction,
                apc_exp=float(sum(x["area_bpm_s"] for x in ed) / n_uc),
                run_ours=bur.max_consecutive_uc_with_decel,
                run_exp=_best_run(p_exp, n_uc),
                nd_ours=len(ours), nd_exp=len(ed),
            ))

    R = pd.DataFrame(rows)
    print(f"  {len(R)} analysable 10-min epochs with >=1 contraction, "
          f"{len(files)} recordings")

    print("\n  3.1  BINARY decel_repetitive vs expert-derived repetitiveness")
    tp = int(((R.rep_ours == 1) & (R.rep_exp == 1)).sum())
    fp = int(((R.rep_ours == 1) & (R.rep_exp == 0)).sum())
    fn = int(((R.rep_ours == 0) & (R.rep_exp == 1)).sum())
    tn = int(((R.rep_ours == 0) & (R.rep_exp == 0)).sum())
    sens = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * sens * prec / max(sens + prec, 1e-9)
    agree = (tp + tn) / len(R)
    po, pe = agree, ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / len(R) ** 2
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"    ours positive {100*R.rep_ours.mean():5.1f}%   "
          f"expert-derived positive {100*R.rep_exp.mean():5.1f}%")
    print(f"    sensitivity {sens:.3f}  precision {prec:.3f}  F1 {f1:.3f}")
    print(f"    raw agreement {agree:.3f}   Cohen kappa {kappa:.3f}")
    print("    -> kappa is the number that matters: it is agreement above chance")
    print("       on a BINARY call that one deceleration can flip.")

    print("\n  3.2  CONTINUOUS burden measures vs their expert-derived values")
    print(f"    {'measure':28s}{'Spearman':>10s}{'Pearson':>10s}"
          f"{'ours mean':>11s}{'exp mean':>10s}")
    cont = {}
    for nm, a, b_ in (("frac_contractions_with_decel", "frac_ours", "frac_exp"),
                      ("decel_area_per_min", "area_ours", "area_exp"),
                      ("area_per_contraction", "apc_ours", "apc_exp"),
                      ("max_consecutive_uc_w_decel", "run_ours", "run_exp"),
                      ("n_decels", "nd_ours", "nd_exp")):
        sp = float(spearmanr(R[a], R[b_]).statistic)
        pr = float(np.corrcoef(R[a], R[b_])[0, 1])
        cont[nm] = dict(spearman=sp, pearson=pr,
                        ours_mean=float(R[a].mean()), exp_mean=float(R[b_].mean()))
        print(f"    {nm:28s}{sp:10.3f}{pr:10.3f}{R[a].mean():11.2f}{R[b_].mean():10.2f}")

    print("\n    A continuous measure that correlates well with its expert")
    print("    counterpart while the BINARY call agrees poorly is the signature")
    print("    of information destroyed by thresholding, not by the detector.")

    return dict(n_epochs=len(R), binary=dict(sens=sens, prec=prec, f1=f1,
                                             agreement=agree, kappa=kappa,
                                             ours_pos=float(R.rep_ours.mean()),
                                             exp_pos=float(R.rep_exp.mean())),
                continuous=cont)


def _best_run(pairs, n_uc):
    run = best = 0
    for j in range(n_uc):
        run = run + 1 if j in pairs else 0
        best = max(best, run)
    return int(best)


# ==========================================================================
# PHASE 4
# ==========================================================================
def phase4() -> Dict:
    rule("PHASE 4 -- THRESHOLD INSTABILITY on CTU-UHB")
    z = np.load(os.path.join(DATA, "epochs.npz"), allow_pickle=True)
    names = [str(s) for s in z["descriptor_names"]]
    meta = pd.DataFrame({c: z[c] for c in z.files
                         if c not in ("X", "F", "descriptor_names",
                                      "fhr_valid", "uc_valid")})
    F = z["F"]
    keep = (meta.y_state >= 0).values
    meta, F = meta[keep].reset_index(drop=True), F[keep]
    n_uc = F[:, names.index("n_contractions")].astype(int)
    rep = F[:, names.index("decel_repetitive")].astype(int)

    print(f"  {len(meta)} readable epochs")
    print("\n  4.1  contractions per epoch -- the denominator of the >50% rule")
    vc = pd.Series(n_uc).value_counts().sort_index()
    for k, v in vc.items():
        if v < 20:
            continue
        print(f"    {k:2d} contractions: {v:5d} epochs ({100*v/len(n_uc):5.1f}%)"
              f"   >50% needs >{0.5*k:.1f}, i.e. >={int(np.floor(0.5*k))+1}")
    print(f"    median {int(np.median(n_uc))}   "
          f"<=4 contractions: {100*np.mean(n_uc<=4):.1f}% of epochs")
    print(f"    ZERO contractions: {int((n_uc==0).sum())} epochs "
          f"({100*np.mean(n_uc==0):.1f}%) -- the ratio is undefined and the")
    print("       current code scores them not-repetitive, i.e. benign by default")

    # reconstruct n_hit from the stored ratio is not possible, so recompute
    # the flip test from the criterion's own arithmetic over plausible n_hit
    print("\n  4.2  how close is each epoch to the 50% boundary?")
    # n_hit is recoverable: repetitive iff n_hit/n_uc > 0.5
    # we do not store n_hit in v3, so bound the instability instead:
    # an epoch is FLIPPABLE if adding or removing one paired contraction
    # crosses the threshold. For a given n_uc that is true for n_hit in a
    # narrow band around n_uc/2, which we enumerate.
    flippable = np.zeros(len(n_uc), bool)
    for i, k in enumerate(n_uc):
        if k == 0:
            continue
        crit = 0.5 * k
        # the two integer counts straddling the boundary
        lo, hi = int(np.floor(crit)), int(np.floor(crit)) + 1
        # epoch is flippable if its true n_hit is lo or hi
        # v3 stores only the verdict; the verdict alone tells us which side
        # -- repetitive means n_hit >= hi. Being AT the boundary is the
        # common case for small k, which 4.3 quantifies exactly.
        flippable[i] = (hi - lo) == 1
    print("    for every epoch with k contractions the criterion is decided by")
    print("    a single event whenever n_hit sits at floor(k/2) or that +1.")
    print("    Enumerating how many integer counts are 'safe' per k:")
    print(f"    {'k':>3s}{'counts':>8s}{'flippable':>11s}{'safe':>7s}{'% flippable':>13s}")
    for k in sorted(set(n_uc[n_uc > 0])):
        if int((n_uc == k).sum()) < 20:
            continue
        crit = 0.5 * k
        counts = np.arange(0, k + 1)
        rep_k = counts / k > 0.5
        flip_k = np.zeros(k + 1, bool)
        for c in counts:
            up = ((c + 1) / k > 0.5) if c < k else rep_k[c]
            dn = ((c - 1) / k > 0.5) if c > 0 else False
            flip_k[c] = (up != rep_k[c]) or (dn != rep_k[c])
        print(f"    {k:3d}{k+1:8d}{int(flip_k.sum()):11d}"
              f"{int((~flip_k).sum()):7d}{100*flip_k.mean():13.1f}")

    print("\n  4.3  observed instability, over the epochs as they actually are")
    res = []
    for i, k in enumerate(n_uc):
        if k == 0:
            continue
        # n_hit is not stored; but the SET of n_hit consistent with the
        # observed verdict is known, and we take the worst case within it,
        # which is the count adjacent to the boundary. Reported as an upper
        # bound and labelled as such.
        crit = 0.5 * k
        n_hit = (int(np.floor(crit)) + 1) if rep[i] else int(np.floor(crit))
        res.append(flip_sensitivity(k, n_hit))
    fl = pd.DataFrame(res)
    print(f"    epochs with >=1 contraction: {len(fl)}")
    print(f"    UPPER BOUND on the flippable share: "
          f"{100*fl.flips.mean():.1f}%")
    print(f"    median distance from the boundary: "
          f"{fl.margin_events.median():.2f} deceleration events")
    print("    (n_hit is not stored in v3, so the count nearest the boundary")
    print("     consistent with each epoch's recorded verdict is used. This")
    print("     is an UPPER bound on brittleness, and is labelled as one.)")

    print("\n  4.4  is the continuous ratio less brittle by construction?")
    print("    The ratio itself moves by 1/k when one event changes:")
    for k in (2, 3, 4, 5, 6, 8):
        if int((n_uc == k).sum()) < 20:
            continue
        print(f"      k={k}: one event moves the ratio by {1/k:.3f} "
              f"({100/k:.0f} percentage points) and can cross 0.50")
    print("    A continuous burden in bpm*s/min has no such discontinuity:")
    print("    one borderline deceleration changes it by its own area, which")
    print("    is small when the deceleration is shallow and short -- i.e. the")
    print("    measure degrades gracefully exactly where the binary rule does not.")

    return dict(n_epochs=int(len(meta)),
                median_contractions=int(np.median(n_uc)),
                frac_le4=float(np.mean(n_uc <= 4)),
                frac_zero_uc=float(np.mean(n_uc == 0)),
                flippable_upper_bound=float(fl.flips.mean()),
                median_margin_events=float(fl.margin_events.median()))


def main():
    os.makedirs(OUT, exist_ok=True)
    out = dict(phase3=phase3(), phase4=phase4())
    json.dump(out, open(os.path.join(OUT, "burden_study.json"), "w"),
              indent=2, default=float)
    print(f"\nwrote {OUT}/burden_study.json")


if __name__ == "__main__":
    main()
