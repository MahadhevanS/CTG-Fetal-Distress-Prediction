"""
Phase 20 C5 feature builder (docs/phase20_redesign_protocol.md section 4, Amendments 1-2).

  C5a  the 11 FHR-only descriptors on the last 5 min and last 10 min of each 20-min window          (22)
  C5b  the same 11 on the 40-min span ending at the window (window + the 20 min before it);
       where the span is not fully available (window starts < 20 min into the retained hour) the
       20-min value is used, no availability flag                                                     (11)
  C5c  6 decel-morphology features from event spans re-derived with the chain's own run rule          (6)

Everything is built with the SAME per-window chain as the locked 19 descriptors (pipeline_clinical): remove_spikes ->
interpolate -> lowpass -> iterative baseline -> calculate_variability / detect_accelerations / detect_decelerations,
plus extended_features (via the same z-score float32 round trip; verified in G3(a)).
Causality: a window's features only ever read raw samples up to that window's end; --selftest checks this by
perturbing every later sample (G3(b)) and also asserts, on every window, that the re-derived decel spans equal the
chain's decel count (Amendment 2).
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import find_peaks

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (BASE_DIR, os.path.join(BASE_DIR, "src", "preprocessing"), os.path.join(BASE_DIR, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ingestion import load_ctu_chb_record
from filtering import remove_spikes, interpolate_missing, apply_lowpass_filter
from baseline import calculate_iterative_baseline
from features import calculate_variability, detect_accelerations, detect_decelerations
from knowledge.extended_features import extract_extended_features

RAW_DIR = "data/raw/ctu-chb-intrapartum"
ROLLING = "results/phase8_rolling/rolling_predictions.csv"
OUT = "results/phase20_redesign"
FS = 4.0; LAST_HOUR = 14400; WIN = 4800
FHR_MIN, FHR_MAX = 50.0, 240.0
FHR_ONLY = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13]     # baseline, STV, LTV, accels, prolonged, decel depth/area/burden/longest, baseline & variability slope
BASE19 = ["baseline", "stv", "ltv", "accels", "early", "late", "variable", "prolonged", "decel_max_depth", "decel_area",
          "decel_burden", "decel_longest_sec", "baseline_slope", "variability_slope", "uc_count", "uc_tachysystole",
          "uc_mean_amp", "fhr_uc_lag", "fhr_uc_coupling"]
_sc = np.load("data/processed_clinical/ctu_signal_scaler.npz")
SM, SS = _sc["mean"].astype(np.float32), _sc["std"].astype(np.float32)
MORPH_NAMES = ["recovery_mean_s", "recovery_max_s", "overshoot_mean", "overshoot_max", "nadir_depth_trend", "frac_contractions_slow_recovery"]
NAMES = ([f"c5a_{n}_{m}m" for m in (5, 10) for n in [BASE19[i] for i in FHR_ONLY]] +
         [f"c5b_{BASE19[i]}_40m" for i in FHR_ONLY] + [f"c5c_{n}" for n in MORPH_NAMES])


def clean(raw_f, raw_u):
    f = remove_spikes(raw_f.copy(), fs=FS)
    f = interpolate_missing(f, clip_min=FHR_MIN, clip_max=FHR_MAX)
    return apply_lowpass_filter(f, fs=FS), apply_lowpass_filter(raw_u.copy(), fs=FS)


def desc19(f, u):
    """19 descriptors of one cleaned signal, exactly as pipeline_clinical (see G3(a))."""
    baseline = calculate_iterative_baseline(f)
    stv, ltv = calculate_variability(f, fs=FS, baseline=baseline)
    accels = detect_accelerations(f, baseline, fs=FS)
    dec = detect_decelerations(f, baseline, u, fs=FS)
    bv = float(np.mean(baseline))
    y8 = [bv, stv, ltv, float(accels), float(dec["early"]), float(dec["late"]), float(dec["variable"]), float(dec["prolonged"])]
    f32 = np.float32
    x0 = ((f - baseline).astype(f32) - SM[0]) / SS[0]; x1 = (u.astype(f32) - SM[1]) / SS[1]
    ext = extract_extended_features(x0.astype(f32) * SS[0] + SM[0], x1.astype(f32) * SS[1] + SM[1], bv, fs=FS)
    return np.array(y8 + [float(v) for v in ext], dtype=np.float64), baseline, dec


def decel_events(f, baseline):
    """Spans of the chain's decel rule: >=15 bpm below baseline for >=15 s (identical to detect_decelerations)."""
    is_d = (baseline - f) >= 15.0
    ch = np.diff(is_d.astype(int)); st = np.where(ch == 1)[0] + 1; en = np.where(ch == -1)[0] + 1
    if len(is_d) and is_d[0]: st = np.insert(st, 0, 0)
    if len(is_d) and is_d[-1]: en = np.append(en, len(f))
    return [(int(a), int(b)) for a, b in zip(st, en) if (b - a) >= int(15 * FS)]


def morphology(f, baseline, u, dec):
    ev = decel_events(f, baseline)
    assert len(ev) == sum(dec.values()), "decel spans do not match detect_decelerations counts"
    dev = f - baseline
    rec, over, depth, rec_by_nadir = [], [], [], []
    for s, e in ev:
        nad = s + int(np.argmin(f[s:e])); depth.append(float(baseline[nad] - f[nad]))
        back = np.where(dev[nad:] >= -5.0)[0]
        if len(back):
            r = int(back[0]); rec.append(r / FS); ridx = nad + r; rec_by_nadir.append((nad, r / FS))
            if ridx + int(60 * FS) <= len(f):
                over.append(max(0.0, float(dev[ridx:ridx + int(60 * FS)].max())))
    trend = float(np.polyfit(np.arange(len(depth)), depth, 1)[0]) if len(depth) >= 3 else 0.0
    peaks, _ = find_peaks(u, distance=int(30 * FS), prominence=10)
    frac = 0.0
    if len(peaks):
        hit = sum(1 for p in peaks if any(0 <= nad - p <= 90 * FS and r > 60.0 for nad, r in rec_by_nadir))
        frac = hit / len(peaks)
    return np.array([np.mean(rec) if rec else 0.0, np.max(rec) if rec else 0.0, np.mean(over) if over else 0.0,
                     np.max(over) if over else 0.0, trend, frac])


def window_features(seg_f, seg_u, s):
    """C5 features of the window [s, s+WIN) of the retained-hour segment. Reads samples < s+WIN only."""
    e = s + WIN
    f, u = clean(seg_f[s:e], seg_u[s:e])
    d20, baseline, dec = desc19(f, u)
    a = [desc19(f[-n:], u[-n:])[0][FHR_ONLY] for n in (1200, 2400)]
    if s >= WIN:
        f40, u40 = clean(seg_f[s - WIN:e], seg_u[s - WIN:e]); b = desc19(f40, u40)[0][FHR_ONLY]
    else:
        b = d20[FHR_ONLY]
    return np.concatenate(a + [b, morphology(f, baseline, u, dec)])


def patient_job(args):
    pid, starts, rows = args
    fhr, uc, _ = load_ctu_chb_record(os.path.join(RAW_DIR, str(pid)))
    if len(fhr) > LAST_HOUR:
        fhr, uc = fhr[-LAST_HOUR:], uc[-LAST_HOUR:]
    return rows, np.array([window_features(fhr, uc, int(s)) for s in starts])


def selftest(df, n_pat=12):
    """G3(b): perturbing every sample after a window's end must leave its features bit-identical."""
    rng = np.random.default_rng(0); worst = 0.0; n = 0
    for pid in rng.choice(sorted(df["patient_id"].unique()), n_pat, replace=False):
        fhr, uc, _ = load_ctu_chb_record(os.path.join(RAW_DIR, str(pid)))
        if len(fhr) > LAST_HOUR: fhr, uc = fhr[-LAST_HOUR:], uc[-LAST_HOUR:]
        for s in df[df["patient_id"] == pid]["start_sample"].values[:3]:
            s = int(s); f0 = window_features(fhr, uc, s)
            f2, u2 = fhr.copy(), uc.copy(); f2[s + WIN:] = rng.uniform(60, 200, len(f2[s + WIN:])); u2[s + WIN:] = rng.uniform(0, 100, len(u2[s + WIN:]))
            worst = max(worst, float(np.max(np.abs(window_features(f2, u2, s) - f0)))); n += 1
    print(f"  G3(b) causality: {n} windows, max |feature change| after perturbing all later samples = {worst:.1e}  [{'PASS' if worst <= 1e-9 else 'FAIL'}]")
    return worst <= 1e-9, worst, n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_csv(ROLLING); df["patient_id"] = df["patient_id"].astype(str)
    if a.selftest:
        ok, worst, n = selftest(df)
        json.dump({"n_windows": n, "max_abs_change": worst, "pass": bool(ok)}, open(os.path.join(OUT, "stage0_G3b.json"), "w"), indent=2)
        return
    pids = sorted(df["patient_id"].unique())
    if a.limit: pids = pids[:a.limit]
    jobs = [(p, df[df["patient_id"] == p]["start_sample"].values, np.where(df["patient_id"].values == p)[0]) for p in pids]
    F = np.full((len(df), len(NAMES)), np.nan); t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (rows, feats) in enumerate(ex.map(patient_job, jobs)):
            F[rows] = feats
            if k % 25 == 0: print(f"  {k}/{len(jobs)} patients  {time.time() - t0:.0f}s", flush=True)
    n_nan = int(np.isnan(F[np.isin(df["patient_id"].values, pids)]).any(1).sum())
    print(f"  built {F.shape}; rows with NaN: {n_nan}; time {time.time() - t0:.0f}s")
    np.savez(os.path.join(OUT, "c5_features.npz" if not a.limit else "c5_features_partial.npz"), F=F, names=np.array(NAMES))
    print("  names:", len(NAMES))


if __name__ == "__main__":
    main()
