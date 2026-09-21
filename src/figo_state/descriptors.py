"""
Per-epoch FIGO characteristics from raw FHR + UC.

WHY THIS EXISTS, AND WHY IT IS NOT src/preprocessing/features.py
----------------------------------------------------------------
features.py was written to feed the pH-outcome model. Its outputs are kept
here only as a comparison arm, never as the label source, because two of its
behaviours make it unusable as a *label-definition* instrument:

  1. detect_decelerations() types any deceleration reaching its nadir in
     under 30 s as "variable", and everything else by UC-peak lag alone. It
     fires on 81.4% of 20-minute windows at a mean of 3.44 per window.
     Phase 9A (docs/phase9a_detector_validation.md) measured deceleration
     detection against FHRMA expert consensus at precision 0.338 -- i.e.
     over-detected 1.8x. A label built on "has >=1 variable deceleration" is
     therefore a label about detector noise, and that is exactly the rule
     that produced the 66.6%-Suspicious distribution in y_figo.

  2. calculate_variability() returns the MEAN over 1-minute detrended
     p95-p5 ranges. One artefactual minute moves the whole epoch. 44.2% of
     windows read >25 bpm, which features.py's own docstring flags as a
     known, unresolved calibration gap.

Everything below is written against the FIGO 2015 consensus guideline
(Ayres-de-Campos D, Spong CY, Chandraharan E. FIGO consensus guidelines on
intrapartum fetal monitoring: Cardiotocography. Int J Gynecol Obstet
2015;131:13-24). Every threshold the guideline states is a named module
constant carrying the clause it comes from, so the rule engine can be
audited against the text rather than against this code.

MEASUREMENT vs LABELLING. This module measures. It assigns no state. The
state rules live in rules.py and consume only this module's output, so the
measurement layer can be revalidated against FHRMA expert annotation
without touching the label definition, and vice versa.
"""

from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

# --------------------------------------------------------------------------
# FIGO 2015 thresholds. Each carries the clause it comes from.
# --------------------------------------------------------------------------
BASELINE_NORMAL_LOW = 110.0     # "normal baseline 110-160 bpm"
BASELINE_NORMAL_HIGH = 160.0
BASELINE_BRADY_PATH = 100.0     # "baseline below 100 bpm" -> pathological
BASELINE_ROUND_BPM = 5.0        # baselines are read to the nearest 5 bpm

VARIABILITY_NORMAL_LOW = 5.0    # "normal variability 5-25 bpm"
VARIABILITY_NORMAL_HIGH = 25.0

EVENT_AMPLITUDE_BPM = 15.0      # accel/decel: ">=15 bpm from baseline"
EVENT_MIN_DURATION_S = 15.0     # "...lasting >=15 s"

# "A deceleration lasting more than 3 minutes is a prolonged deceleration."
DECEL_PROLONGED_S = 180.0
# "...lasting more than 5 minutes, with FHR maintained below 80 bpm" is a
# single pathological feature on its own (acute fetal hypoxia).
DECEL_ACUTE_HYPOXIA_S = 300.0
DECEL_ACUTE_HYPOXIA_BPM = 80.0

# Late decelerations have "a gradual onset and/or a gradual return to
# baseline" and their nadir lags the contraction peak. 30 s separates
# gradual from rapid onset; 20 s of lag marks the nadir as late.
DECEL_RAPID_ONSET_S = 30.0
DECEL_LATE_LAG_S = 20.0

# "Repetitive decelerations: occurring with more than 50% of contractions."
REPETITIVE_FRACTION = 0.50

# Uterine contraction detection. Normal labour runs <=5 contractions per
# 10 minutes and each lasts 45-120 s, so a 60 s refractory floor between
# peaks sits well inside physiology and rejects toco sensor jitter.
UC_MIN_SEPARATION_S = 60.0
UC_MIN_PROMINENCE = 10.0

# Variability is an amplitude read off a trace, not a sample statistic.
# FIGO oscillations run 3-5 cycles/min (0.05-0.083 Hz); a 2.5 s moving
# average (~0.4 Hz) is five times above that, so it removes sample-level
# residue without touching the oscillation being measured.
VARIABILITY_SMOOTH_S = 2.5
VARIABILITY_SEGMENT_S = 60.0
VARIABILITY_MIN_CLEAN_FRACTION = 0.40

# A deceleration belongs to a contraction if its nadir falls within 2 min of
# that contraction's peak.
DECEL_UC_PAIRING_TOLERANCE_S = 120.0

# --------------------------------------------------------------------------
# Event delineation. NOT guideline thresholds -- the guideline defines an
# event by a 15 bpm / 15 s core but does not say how to bound it, and these
# two constants supply that.
#
# WHY IT MATTERS. Detecting only the part of an excursion deeper than 15 bpm
# splits one expert-annotated deceleration into several fragments, and each
# extra fragment scores as a false positive. Measured against FHRMA expert
# consensus, 1021 epochs, 1375 expert decelerations
# (scripts/figo_diagnose_decel_detection.py).
#
# HOW THESE WERE CHOSEN -- and the metric that matters.
# Event matching is by ANY temporal overlap, so one bloated span overlapping
# one expert deceleration still scores a true positive. Optimising event F1
# alone is therefore blind to span inflation, and doing exactly that produced
# a first attempt (edge 2 bpm, merge 30 s) whose decelerations covered 38.2%
# of every epoch against the experts' 23.1%. Sample-level DICE is reported
# alongside, and the selection rule is stated in advance:
#
#     maximise event F1 subject to
#       (a) merge gap < 60 s, the minimum contraction separation
#       (b) Dice >= 0.60
#       (c) fraction of time in deceleration within 1.5x the expert's 0.231
#
#   edge  merge      n    sens   prec   evF1   Dice   frac_time
#     15      0   2187   0.588  0.370  0.454  0.591     0.167   as first written
#      2     30   1905   0.751  0.542  0.629  0.605     0.382   fails (c)
#     10     45   1694   0.696  0.565  0.624  0.590     0.344   fails (b)
#    7.5     30   1999   0.740  0.509  0.603  0.619     0.329   CHOSEN
#      -      -   1375     -      -      -      -       0.231   expert
#
# 7.5 bpm is half the 15 bpm FIGO event amplitude, which is why that value
# rather than a fitted one. 30 s is half the 60 s minimum contraction
# separation: a longer merge can fuse decelerations belonging to ADJACENT
# contractions, and a fused span has one nadir, so it pairs with one
# contraction and suppresses the "repetitive decelerations" test -- the exact
# test the pathological criterion depends on.
#
# Note this also retires a number from Phase 9A: it reported 0.616 as the
# expert-baseline UPPER BOUND for deceleration F1. That bound was measured
# without delineation, and the same rule on our own baseline now reaches
# 0.603 -- so 0.616 was an artefact of fragmentation, not a ceiling.
#
# The 15 bpm / 15 s core is NOT tuned. A sweep found 18 bpm / 10 s marginally
# better on F1 and it was not adopted: those two numbers are the guideline's
# definition of the event, and fitting them to a corpus would make the label
# a fitted object rather than FIGO 2015.
DECEL_EDGE_BPM = 7.5        # event extends out to here, from the >=15 bpm core
DECEL_MERGE_GAP_S = 30.0    # spans closer than this are one event


@dataclass
class EpochDescriptors:
    """Everything rules.py is allowed to see about one epoch."""
    # baseline
    baseline_bpm: float
    baseline_from_carry: int        # 1 if seeded from the previous epoch
    # variability
    variability_bpm: float          # bandwidth amplitude, accel/decel excluded
    stv_bpm: float                  # mean |beat-to-beat diff|, for reference
    variability_measurable: int     # 0 if too little clean signal to read
    # accelerations
    n_accels: int
    # contractions
    n_contractions: int
    # decelerations
    n_decels: int
    n_decel_early: int
    n_decel_late: int
    n_decel_variable: int
    n_decel_prolonged: int
    decel_repetitive: int           # decels with >50% of contractions
    longest_decel_s: float
    deepest_decel_bpm: float        # max drop below baseline
    frac_time_in_decel: float
    has_acute_hypoxia_decel: int    # >5 min sustained below 80 bpm
    # signal quality
    quality: float                  # fraction of samples actually MEASURED
    usable: float                   # fraction USABLE for analysis (measured +
                                    # short gaps that interpolation repaired)

    @staticmethod
    def names() -> List[str]:
        return [f.name for f in fields(EpochDescriptors)]

    def to_vector(self) -> np.ndarray:
        return np.array([float(getattr(self, n)) for n in self.names()],
                        dtype=np.float32)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous True runs of a boolean array as [start, end) index pairs."""
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(int(mask.size))
    return list(zip(starts, ends))


def default_valid(fhr: np.ndarray) -> np.ndarray:
    """Fallback validity mask when a caller supplies none: everything is real."""
    return np.ones(fhr.shape, dtype=bool)


def _merge_spans(spans: List[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
    """Merge spans separated by fewer than `gap` samples."""
    if not spans:
        return []
    out = [list(spans[0])]
    for s, e in spans[1:]:
        if s - out[-1][1] < gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(int(a), int(b)) for a, b in out]


def delineate_events(fhr: np.ndarray, baseline: float, fs: float,
                     valid: np.ndarray, sign: int) -> List[Tuple[int, int]]:
    """
    Bound accelerations (sign=+1) or decelerations (sign=-1) as whole events.

    A FIGO event is defined by a >=15 bpm, >=15 s core, but a clinician reads
    it from ONSET to RETURN TO BASELINE -- the shoulders where the excursion
    is real but shallower than 15 bpm belong to the same event. Detecting
    only the core splits one expert-annotated deceleration into several
    fragments, each of which scores as a false positive; that alone accounted
    for most of the disagreement with FHRMA (precision 0.370 -> 0.542, see
    DECEL_EDGE_BPM).

    So: find every excursion reaching DECEL_EDGE_BPM, keep those containing a
    >=15 bpm core, merge what is closer than DECEL_MERGE_GAP_S, and let the
    caller apply the duration criterion to the resulting whole event.
    """
    dev = (fhr - baseline) if sign > 0 else (baseline - fhr)
    core = (dev >= EVENT_AMPLITUDE_BPM) & valid
    if not core.any():
        return []
    edge = (dev >= DECEL_EDGE_BPM) & valid
    spans = [(s, e) for s, e in _runs(edge) if core[s:e].any()]
    return _merge_spans(spans, int(DECEL_MERGE_GAP_S * fs))


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or x.size < n:
        return x
    return np.convolve(x, np.ones(n) / n, mode="same")


def estimate_baseline(fhr: np.ndarray, valid: Optional[np.ndarray] = None,
                      prev_baseline: Optional[float] = None) -> Tuple[float, bool]:
    """
    One baseline value for the epoch, which is FIGO's own definition.

    "The baseline is the mean level of the most horizontal and least
    oscillatory FHR segments, estimated over 10 minutes" -- so a constant per
    10-minute epoch is what the guideline asks for, not an approximation to
    something better. This matters: Phase 9A showed a drift-tracking baseline
    absorbs sustained FHR drops into itself and erases prolonged
    decelerations (dec_prolonged fell 98%), which is precisely the feature
    the pathological rules depend on.

    Estimation is the standard iterative +/-15 bpm exclusion, with two
    additions over calculate_iterative_baseline():

    1. `prev_baseline` seeds the iteration. When an epoch is dominated by one
       long excursion, the median starts inside it and the iteration
       converges onto the deceleration rather than onto the baseline. A
       clinician reads such a trace by carrying the level forward from before
       the event, which is what this does. Strictly causal: the previous
       epoch only, never the next one.
    2. If, after convergence, too little of the epoch sits within the +/-15
       band for the estimate to mean anything, the previous baseline is used
       outright and the caller is told so via the returned flag.

    `valid` marks samples that are real or short-gap-repaired signal. Samples
    outside it are unrecoverable dropout, which CTU-UHB encodes as 0.0 bpm and
    which interpolate_missing() deliberately leaves in place for gaps over 15 s
    so that it does not fabricate decelerations. Left in the arithmetic, a 0.0
    sample sits ~140 bpm below any real baseline and drags the estimate down;
    excluding it is the only correct handling.

    Returns (baseline_bpm rounded to 5, seeded_from_previous).
    """
    if valid is None:
        valid = default_valid(fhr)
    x = fhr[valid]
    if x.size == 0:
        return (float(prev_baseline) if prev_baseline is not None
                else float("nan"), True)

    b = float(prev_baseline) if prev_baseline is not None else float(np.median(x))
    for _ in range(8):
        m = (x >= b - EVENT_AMPLITUDE_BPM) & (x <= b + EVENT_AMPLITUDE_BPM)
        if not np.any(m):
            break
        nb = float(np.mean(x[m]))
        if abs(nb - b) < 0.05:
            b = nb
            break
        b = nb

    within = float(np.mean((x >= b - EVENT_AMPLITUDE_BPM) &
                           (x <= b + EVENT_AMPLITUDE_BPM)))
    carried = False
    if within < 0.30 and prev_baseline is not None:
        b, carried = float(prev_baseline), True

    return float(np.round(b / BASELINE_ROUND_BPM) * BASELINE_ROUND_BPM), carried


def measure_variability(fhr: np.ndarray, baseline: float, fs: float,
                        event_mask: np.ndarray,
                        valid: Optional[np.ndarray] = None) -> Tuple[float, bool]:
    """
    Bandwidth amplitude of the baseline oscillation, in bpm.

    FIGO: "the oscillation of the FHR signal, evaluated as the amplitude of
    the bandwidth", read over segments free of accelerations and
    decelerations. Three deliberate choices, each fixing a defect measured in
    features.py:

    * accel/decel samples are excluded (`event_mask`) before anything else --
      a 15 bpm excursion left inside a 1-minute range trivially reads as >25
      bpm "increased variability" whatever the real oscillation is doing;
    * the signal is smoothed to 2.5 s first, removing sample-level residue
      that a p95-p5 range on raw 4 Hz data counts as oscillation;
    * segments are combined with a MEDIAN, not a mean. features.py's mean
      lets one artefactual minute set the epoch's value, and this dataset has
      a median 18.3% missing-sample rate feeding interpolation.

    Returns (variability_bpm, measurable). `measurable` is False when too
    little clean signal survives exclusion for the reading to mean anything;
    a value is still returned, but callers must not threshold it.
    """
    if fhr.size < 2:
        return 0.0, False
    if valid is None:
        valid = default_valid(fhr)

    # Smooth only across real signal. A moving average run over a dropout
    # would pull the gap's 0.0 samples into their neighbours and depress the
    # measured oscillation of otherwise clean minutes next to a gap.
    sm = _moving_average(fhr, max(1, int(VARIABILITY_SMOOTH_S * fs)))
    seg = max(1, int(VARIABILITY_SEGMENT_S * fs))
    n_seg = max(1, fhr.size // seg)

    vals = []
    for i in range(n_seg):
        s, e = i * seg, min((i + 1) * seg, fhr.size)
        keep = (~event_mask[s:e]) & valid[s:e]
        clean = sm[s:e][keep]
        if clean.size < max(8, int(VARIABILITY_MIN_CLEAN_FRACTION * (e - s))):
            continue
        t = np.arange(clean.size)
        slope, icept = np.polyfit(t, clean, 1)
        det = clean - (slope * t + icept)
        vals.append(float(np.percentile(det, 95) - np.percentile(det, 5)))

    if not vals:
        return 0.0, False
    return float(np.median(vals)), True


def detect_contractions(uc: np.ndarray, fs: float,
                        uc_valid: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Contraction peak indices from the tocodynamometer channel.

    The toco channel has its own dropout, also encoded as 0.0. A peak whose
    neighbourhood is mostly dropout is a sensor edge, not a contraction, and
    counting it would inflate the denominator of the "repetitive
    decelerations" test (>50% of contractions) and so suppress a real
    pathological finding.
    """
    if uc.size == 0 or np.allclose(uc, uc[0]):
        return np.array([], dtype=int)
    peaks, _ = find_peaks(uc,
                          distance=max(1, int(UC_MIN_SEPARATION_S * fs)),
                          prominence=UC_MIN_PROMINENCE)
    if uc_valid is None or peaks.size == 0:
        return peaks
    half = max(1, int(30.0 * fs))
    keep = [p for p in peaks
            if uc_valid[max(0, p - half):min(uc.size, p + half)].mean() > 0.5]
    return np.array(keep, dtype=int)


def event_mask(fhr: np.ndarray, baseline: float, fs: float,
               valid: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Samples inside a qualifying acceleration or deceleration.

    An excursion is only counted where the signal is real. A run broken by a
    dropout is split rather than bridged, which under-counts long events
    rather than inventing them -- the conservative direction, and the one
    that matters when the prolonged-deceleration criterion is a single
    sufficient condition for Pathological.
    """
    if valid is None:
        valid = default_valid(fhr)
    need = int(EVENT_MIN_DURATION_S * fs)
    out = np.zeros(fhr.shape, dtype=bool)
    for sign in (+1, -1):
        for s, e in delineate_events(fhr, baseline, fs, valid, sign):
            if (e - s) >= need:
                out[s:e] = True
    return out


def classify_decelerations(fhr: np.ndarray, baseline: float, fs: float,
                           uc_peaks: np.ndarray,
                           valid: Optional[np.ndarray] = None) -> Dict[str, object]:
    """
    Detect decelerations and type them by FIGO 2015 morphology.

    Typing order follows the guideline's own precedence:
      prolonged (>3 min) > variable (rapid onset, <30 s to nadir)
      > late (gradual, nadir >=20 s after the contraction peak)
      > early (gradual, nadir coincident with the peak).

    An unpaired gradual deceleration -- no contraction anywhere near it -- is
    NOT counted as early. features.py defaulted those to early, which
    silently converts toco dropout into a benign finding. They are counted in
    n_decels and left untyped.
    """
    need = int(EVENT_MIN_DURATION_S * fs)
    prolonged_n = int(DECEL_PROLONGED_S * fs)
    rapid_n = int(DECEL_RAPID_ONSET_S * fs)
    late_lag_n = int(DECEL_LATE_LAG_S * fs)
    acute_n = int(DECEL_ACUTE_HYPOXIA_S * fs)
    pair_tol_n = int(DECEL_UC_PAIRING_TOLERANCE_S * fs)

    if valid is None:
        valid = default_valid(fhr)

    out: Dict[str, object] = dict(
        n_decels=0, early=0, late=0, variable=0, prolonged=0,
        longest_s=0.0, deepest_bpm=0.0, samples_in_decel=0,
        acute_hypoxia=0, paired_contractions=set())

    for s, e in delineate_events(fhr, baseline, fs, valid, -1):
        dur = e - s
        if dur < need:
            continue
        seg = fhr[s:e]
        nadir_i = s + int(np.argmin(seg))
        depth = float(baseline - seg.min())

        out["n_decels"] += 1
        out["samples_in_decel"] += dur
        out["longest_s"] = max(out["longest_s"], dur / fs)
        out["deepest_bpm"] = max(out["deepest_bpm"], depth)

        if dur >= acute_n and float(np.mean(seg < DECEL_ACUTE_HYPOXIA_BPM)) > 0.8:
            out["acute_hypoxia"] = 1

        if dur >= prolonged_n:
            out["prolonged"] += 1
            continue

        # which contraction, if any, does this deceleration belong to
        if uc_peaks.size:
            j = int(np.argmin(np.abs(uc_peaks - nadir_i)))
            lag = nadir_i - int(uc_peaks[j])
            paired = abs(lag) <= pair_tol_n
        else:
            j, lag, paired = -1, 0, False
        if paired:
            out["paired_contractions"].add(j)

        if (nadir_i - s) < rapid_n:
            out["variable"] += 1
        elif paired and lag >= late_lag_n:
            out["late"] += 1
        elif paired:
            out["early"] += 1
        # else: gradual but unpaired -> counted in n_decels, deliberately untyped

    return out


def compute_descriptors(fhr: np.ndarray, uc: np.ndarray, fs: float,
                        quality: float,
                        valid: Optional[np.ndarray] = None,
                        uc_valid: Optional[np.ndarray] = None,
                        prev_baseline: Optional[float] = None) -> EpochDescriptors:
    """
    Full FIGO characteristic set for one epoch of cleaned FHR + UC.

    `fhr` must already be spike-removed, gap-interpolated and low-passed, and
    must be in bpm -- NOT baseline-corrected, NOT z-scored, because every
    threshold above is in clinical units.

    `quality` is the fraction of samples actually MEASURED, before any
    reconstruction. `valid` marks samples usable for analysis: real samples
    plus short gaps that interpolate_missing() repaired. The two differ
    because that function fills gaps up to 15 s and deliberately leaves
    longer ones at 0.0 bpm rather than fabricate a deceleration. Passing
    `valid` is what stops those 0.0 samples being read as ~140 bpm
    decelerations; omit it only when the signal is known to be gap-free.

    `prev_baseline` is the previous epoch's baseline, or None for the first
    epoch. It is used only to rescue a degenerate estimate, and never looks
    forward.
    """
    if valid is None:
        valid = default_valid(fhr)

    baseline, carried = estimate_baseline(fhr, valid, prev_baseline)
    ev = event_mask(fhr, baseline, fs, valid)
    variability, measurable = measure_variability(fhr, baseline, fs, ev, valid)
    dv = np.diff(fhr)
    dv_ok = valid[1:] & valid[:-1]
    stv = float(np.mean(np.abs(dv[dv_ok]))) if np.any(dv_ok) else 0.0

    need = int(EVENT_MIN_DURATION_S * fs)
    n_accels = sum(1 for s, e in delineate_events(fhr, baseline, fs, valid, +1)
                   if (e - s) >= need)

    uc_peaks = detect_contractions(uc, fs, uc_valid)
    d = classify_decelerations(fhr, baseline, fs, uc_peaks, valid)

    n_uc = int(uc_peaks.size)
    repetitive = int(n_uc > 0 and
                     (len(d["paired_contractions"]) / n_uc) > REPETITIVE_FRACTION)

    return EpochDescriptors(
        baseline_bpm=float(baseline),
        baseline_from_carry=int(carried),
        variability_bpm=float(variability),
        stv_bpm=stv,
        variability_measurable=int(measurable),
        n_accels=int(n_accels),
        n_contractions=n_uc,
        n_decels=int(d["n_decels"]),
        n_decel_early=int(d["early"]),
        n_decel_late=int(d["late"]),
        n_decel_variable=int(d["variable"]),
        n_decel_prolonged=int(d["prolonged"]),
        decel_repetitive=repetitive,
        longest_decel_s=float(d["longest_s"]),
        deepest_decel_bpm=float(d["deepest_bpm"]),
        frac_time_in_decel=float(d["samples_in_decel"] / max(1, fhr.size)),
        has_acute_hypoxia_decel=int(d["acute_hypoxia"]),
        quality=float(quality),
        usable=float(np.mean(valid)),
    )
