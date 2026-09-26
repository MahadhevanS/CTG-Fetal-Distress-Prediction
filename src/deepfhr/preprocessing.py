"""
Zhao et al. 2019 (DeepFHR)'s own preprocessing, reconstructed from the paper's Methods text
(docs/phaseD_deepfhr_reproduction_protocol.md section 0). Deliberately NOT this project's own
src/preprocessing/filtering.py (different thresholds) -- this module exists only so Phase D1 is a
faithful reproduction of the paper's specific pipeline, not this project's own.
"""
import numpy as np
from scipy.interpolate import CubicSpline

FS = 4.0


def _spline_fill(fhr, mask_fill):
    """Cubic-spline interpolate the samples where mask_fill is True, using the surrounding valid samples."""
    idx = np.arange(len(fhr))
    valid = ~mask_fill
    if valid.sum() < 4:
        return fhr
    cs = CubicSpline(idx[valid], fhr[valid], extrapolate=True)
    out = fhr.copy()
    out[mask_fill] = cs(idx[mask_fill])
    return out


def fill_gaps(fhr, gap_fill_max_s=15.0, fs=FS):
    """Spline-fill FHR==0 runs. Runs <= gap_fill_max_s are a direct fill; longer runs are also spline-filled
    here (the paper says 'removed directly', which would break the fixed 4800-sample length required
    downstream -- documented deviation, docs/deepfhr_original.yaml 'gap_long_s')."""
    zero = fhr == 0.0
    return _spline_fill(fhr, zero) if zero.any() else fhr.copy()


def despike(fhr, spike_diff_bpm=25.0, stable_window_n=5, stable_diff_bpm=10.0):
    """Zhao et al.'s despiking pass: wherever two adjacent samples differ by more than spike_diff_bpm,
    interpolate from that point until reaching a 'stable section' (stable_window_n consecutive samples each
    differing from their neighbours by less than stable_diff_bpm)."""
    out = fhr.copy()
    n = len(out)
    diffs = np.abs(np.diff(out))
    i = 0
    while i < n - 1:
        if diffs[i] > spike_diff_bpm:
            j = i + 1
            while j < n - stable_window_n:
                window = out[j:j + stable_window_n]
                if np.all(np.abs(np.diff(window)) < stable_diff_bpm):
                    break
                j += 1
            j = min(j + 1, n)
            mask = np.zeros(n, dtype=bool)
            mask[i + 1:j] = True
            if mask.any() and (~mask).sum() >= 4:
                out = _spline_fill(out, mask)
                diffs = np.abs(np.diff(out))
            i = j
        else:
            i += 1
    return out


def clip_extremes(fhr, clip_min_bpm=50.0, clip_max_bpm=200.0):
    """Cubic-spline replace physiologically implausible values."""
    bad = (fhr < clip_min_bpm) | (fhr > clip_max_bpm)
    return _spline_fill(fhr, bad) if bad.any() else fhr.copy()


def preprocess_segment(raw_fhr, cfg):
    """The full Zhao et al. chain on one already-extracted 20-min (4800-sample) FHR segment."""
    f = fill_gaps(raw_fhr, cfg["preprocessing"]["gap_fill_max_s"])
    f = despike(f, cfg["preprocessing"]["spike_diff_bpm"], cfg["preprocessing"]["stable_window_n"], cfg["preprocessing"]["stable_diff_bpm"])
    f = clip_extremes(f, cfg["preprocessing"]["clip_min_bpm"], cfg["preprocessing"]["clip_max_bpm"])
    return f


def extract_last_segment(fhr_full, segment_samples):
    """docs/deepfhr_original.yaml 'segment_choice: last_20_minutes' (ASSUMPTION -- paper does not state which
    segment). Pads at the front with the first valid value if the recording is shorter than one segment."""
    if len(fhr_full) >= segment_samples:
        return fhr_full[-segment_samples:].astype(np.float64)
    pad = np.full(segment_samples - len(fhr_full), fhr_full[0] if len(fhr_full) else 140.0, dtype=np.float64)
    return np.concatenate([pad, fhr_full.astype(np.float64)])
