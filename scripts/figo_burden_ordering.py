"""
Does the continuous burden preserve clinically meaningful ORDERING and
SEVERITY, or only correlate?

    python scripts/figo_burden_ordering.py

Correlation is a weak claim. A measure can correlate at 0.7 while ordering
individual epochs badly, and ordering is what a severity criterion needs: a
threshold is only defensible if the quantity it cuts ranks epochs the way a
clinician would.

WHAT IS TESTED, ON FHRMA ONLY
------------------------------
  1  GLOBAL ordering      pairwise concordance and Kendall tau-b against the
                          expert-derived value of the same measure
  2  WITHIN-RECORD order  can it rank one mother's own epochs? This is the
                          ordering a monitor actually needs, and it removes
                          between-recording variation that inflates global
                          correlation
  3  SEVERITY TIERS       quartile agreement and quadratic-weighted kappa --
                          does a "high burden" epoch read as high burden
  4  EXTREME DISCRIM.     AUROC for "expert burden in the top decile", the
                          operating question a Pathological threshold asks
  5  BIAS SHAPE           is the systematic over-estimate a MONOTONE
                          (recalibratable) transform, or does it distort
                          order? Monotone bias is fixable by calibration; a
                          non-monotone one is not
  6  FAILURE CASES        what characterises the epochs it gets most wrong

No CTU-UHB outcome, no pH, no early-warning label is read. FHRMA carries no
outcome, so nothing here can be tuned toward prediction.
"""

import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "src", "preprocessing"))

from figo_state import descriptors as DS                         # noqa: E402
from figo_state.burden import (compute_burden, deceleration_events,  # noqa: E402
                               pair_to_contractions)
from filtering import (apply_lowpass_filter, interpolate_missing,  # noqa: E402
                       remove_spikes)

FHRMA = os.path.join(BASE, "data", "raw", "fhrma", "CTGDL_FHRMA_ano_csv")
OUT = os.path.join(BASE, "results", "figo_burden")
FS = 4.0
EPOCH = int(10 * 60 * FS)
SETTLE = int(2.0 * FS)

MEASURES = ["frac_contractions_with_decel", "decel_area_per_min",
            "area_per_contraction", "max_consecutive_uc_with_decel"]


def dilate(m, k):
    if k <= 0 or not m.any():
        return m
    return np.convolve(m.astype(np.int16), np.ones(2 * k + 1, np.int16),
                       mode="same") > 0


def runs(mask):
    m = np.asarray(mask).astype(int)
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def collect() -> pd.DataFrame:
    rows: List[Dict] = []
    files = [f for f in sorted(glob.glob(os.path.join(FHRMA, "*.csv")))
             if not f.endswith(".csv.csv")]
    for path in files:
        rec = os.path.basename(path)[:-4]
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
            ucp = DS.detect_contractions(u[s:e], FS, uc_valid[s:e])
            n_uc = int(ucp.size)
            if n_uc == 0:
                continue
            bur = compute_burden(f[s:e], u[s:e], FS, b, vv, uc_valid[s:e])
            obs = float(vv.sum() / FS / 60.0)

            need = int(DS.EVENT_MIN_DURATION_S * FS)
            ed = []
            for xs, xe in runs(exp_dec[s:e]):
                if (xe - xs) < need:
                    continue
                seg = f[s:e][xs:xe]
                ed.append(dict(nadir=xs + int(np.argmin(seg)),
                               depth_bpm=float(b - seg.min()),
                               duration_s=float((xe - xs) / FS),
                               area_bpm_s=float(np.clip(b - seg, 0, None).sum() / FS)))
            pe = pair_to_contractions(ed, ucp, FS)
            run = best = 0
            for j in range(n_uc):
                run = run + 1 if j in pe else 0
                best = max(best, run)
            area = float(sum(x["area_bpm_s"] for x in ed))

            rows.append(dict(
                record=rec, start=s, n_uc=n_uc, obs_min=obs,
                usable=float(vv.mean()),
                ours_frac_contractions_with_decel=bur.frac_contractions_with_decel,
                exp_frac_contractions_with_decel=len(pe) / n_uc,
                ours_decel_area_per_min=bur.decel_area_per_min,
                exp_decel_area_per_min=area / obs if obs > 0 else 0.0,
                ours_area_per_contraction=bur.area_per_contraction,
                exp_area_per_contraction=area / n_uc,
                ours_max_consecutive_uc_with_decel=float(bur.max_consecutive_uc_with_decel),
                exp_max_consecutive_uc_with_decel=float(best),
            ))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    R = collect()
    print(f"{len(R)} epochs from {R.record.nunique()} FHRMA recordings\n")
    res: Dict[str, Dict] = {}

    def hdr(t):
        print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

    # ------------------------------------------------------------------ 1
    hdr("1. GLOBAL ORDERING  (concordance = P(we order a random pair correctly))")
    print(f"  {'measure':34s}{'Spearman':>10s}{'tau-b':>9s}{'concord':>9s}")
    for mname in MEASURES:
        o, x = R[f"ours_{mname}"].values, R[f"exp_{mname}"].values
        sp = float(spearmanr(o, x).statistic)
        tb = float(kendalltau(o, x, variant="b").statistic)
        conc = (tb + 1) / 2          # concordance implied by tau-b
        res.setdefault(mname, {}).update(spearman=sp, tau_b=tb, concordance=conc)
        print(f"  {mname:34s}{sp:10.3f}{tb:9.3f}{conc:9.3f}")
    print("\n  concordance 0.50 = coin flip, 1.00 = perfect ordering.")

    # ------------------------------------------------------------------ 2
    hdr("2. WITHIN-RECORD ORDERING  (rank one mother's own epochs)")
    print("  Global correlation is inflated by between-recording spread. A")
    print("  monitor must rank THIS labour's epochs, which is this test.\n")
    print(f"  {'measure':34s}{'median rho':>12s}{'IQR':>18s}{'records':>9s}"
          f"{'rho<=0':>8s}")
    for mname in MEASURES:
        rhos = []
        for rec, g in R.groupby("record"):
            if len(g) < 3:
                continue
            o, x = g[f"ours_{mname}"].values, g[f"exp_{mname}"].values
            if len(np.unique(o)) < 2 or len(np.unique(x)) < 2:
                continue
            r = spearmanr(o, x).statistic
            if np.isfinite(r):
                rhos.append(float(r))
        rhos = np.array(rhos)
        q1, q3 = np.percentile(rhos, [25, 75])
        res[mname].update(within_median_rho=float(np.median(rhos)),
                          within_q1=float(q1), within_q3=float(q3),
                          within_n=int(len(rhos)),
                          within_frac_nonpositive=float((rhos <= 0).mean()))
        print(f"  {mname:34s}{np.median(rhos):12.3f}"
              f"   [{q1:+.2f}, {q3:+.2f}]{len(rhos):9d}"
              f"{100*(rhos<=0).mean():7.0f}%")

    # ------------------------------------------------------------------ 3
    hdr("3. SEVERITY TIERS  (expert quartiles; do we assign the same tier?)")
    print(f"  {'measure':34s}{'exact':>8s}{'+/-1 tier':>11s}{'wkappa':>9s}")
    for mname in MEASURES:
        o, x = R[f"ours_{mname}"].values, R[f"exp_{mname}"].values
        try:
            tx = pd.qcut(x, 4, labels=False, duplicates="drop")
            to = pd.qcut(o, 4, labels=False, duplicates="drop")
        except ValueError:
            print(f"  {mname:34s}   too few distinct values")
            continue
        ok = ~(pd.isna(tx) | pd.isna(to))
        tx, to = tx[ok].astype(int), to[ok].astype(int)
        exact = float((tx == to).mean())
        near = float((np.abs(tx - to) <= 1).mean())
        wk = float(cohen_kappa_score(tx, to, weights="quadratic"))
        res[mname].update(tier_exact=exact, tier_within1=near, tier_wkappa=wk)
        print(f"  {mname:34s}{exact:8.3f}{near:11.3f}{wk:9.3f}")

    # ------------------------------------------------------------------ 4
    hdr("4. EXTREME DISCRIMINATION  (can we find the expert's worst epochs?)")
    print("  The question a Pathological threshold actually asks.\n")
    print(f"  {'measure':34s}{'top decile':>12s}{'top quartile':>14s}")
    for mname in MEASURES:
        o, x = R[f"ours_{mname}"].values, R[f"exp_{mname}"].values
        line = {}
        for q, tag in ((0.90, "top decile"), (0.75, "top quartile")):
            lab = (x >= np.quantile(x, q)).astype(int)
            line[tag] = float(roc_auc_score(lab, o)) if len(set(lab)) > 1 else np.nan
        res[mname].update(auroc_top_decile=line["top decile"],
                          auroc_top_quartile=line["top quartile"])
        print(f"  {mname:34s}{line['top decile']:12.3f}{line['top quartile']:14.3f}")

    # ------------------------------------------------------------------ 5
    hdr("5. BIAS SHAPE  (is the over-estimate recalibratable?)")
    print("  If the bias is a MONOTONE transform, a calibration curve fixes it")
    print("  and ordering is intact. Compare correlation on RAW values against")
    print("  correlation on RANKS: if ranks agree much better, the bias is")
    print("  monotone; if they agree similarly badly, ordering is genuinely lost.\n")
    print(f"  {'measure':34s}{'Pearson raw':>13s}{'Pearson rank':>14s}{'ratio o/e':>11s}")
    for mname in MEASURES:
        o, x = R[f"ours_{mname}"].values, R[f"exp_{mname}"].values
        pr = float(np.corrcoef(o, x)[0, 1])
        prk = float(np.corrcoef(pd.Series(o).rank(), pd.Series(x).rank())[0, 1])
        ratio = float(o.mean() / x.mean()) if x.mean() else np.nan
        res[mname].update(pearson_raw=pr, pearson_rank=prk, mean_ratio=ratio)
        print(f"  {mname:34s}{pr:13.3f}{prk:14.3f}{ratio:11.2f}")

    # ------------------------------------------------------------------ 6
    hdr("6. FAILURE CASES  (worst-ordered decile, on area_per_contraction)")
    mname = "area_per_contraction"
    o, x = R[f"ours_{mname}"].values, R[f"exp_{mname}"].values
    ro, rx = pd.Series(o).rank(pct=True), pd.Series(x).rank(pct=True)
    err = (ro - rx).abs()
    bad = R[err >= err.quantile(0.90)]
    good = R[err <= err.quantile(0.50)]
    print(f"  {'':22s}{'worst decile':>14s}{'best half':>12s}")
    for c, lab in (("n_uc", "contractions"), ("usable", "usable fraction"),
                   ("obs_min", "observation min")):
        print(f"  {lab:22s}{bad[c].mean():14.2f}{good[c].mean():12.2f}")
    print(f"  {'expert burden':22s}{bad[f'exp_{mname}'].mean():14.1f}"
          f"{good[f'exp_{mname}'].mean():12.1f}")
    res["failure_cases"] = dict(
        worst_decile_n_uc=float(bad.n_uc.mean()),
        best_half_n_uc=float(good.n_uc.mean()),
        worst_decile_usable=float(bad.usable.mean()),
        best_half_usable=float(good.usable.mean()))

    json.dump(res, open(os.path.join(OUT, "ordering_analysis.json"), "w"),
              indent=2, default=float)
    print(f"\nwrote {OUT}/ordering_analysis.json")


if __name__ == "__main__":
    main()
