"""
THE FROZEN FIGO 2015 STATE CLASSIFIER -- this file defines the label.

FREEZE CONTRACT
---------------
Nothing in this module may be changed once a model has been trained against
it. The label must not be tuned toward a model, or the chain

    descriptor -> label definition -> model -> descriptor

closes into a circle and every number downstream becomes meaningless. If a
rule genuinely has to change, bump RULESET_VERSION, regenerate the dataset,
and re-run every experiment. The version string is written into every dataset
file so a mismatch is detectable rather than silent.

WHY THIS REPLACES knowledge/figo.py::classify_figo
--------------------------------------------------
classify_figo assigns Suspicious to any window containing a single variable
deceleration. That is not in FIGO 2015 -- the guideline treats isolated
variable decelerations as a normal finding and only escalates on REPETITIVE
decelerations. On the existing 8,517-window substrate that one clause fires
on 81.4% of windows and is the direct cause of the 66.6%-Suspicious
distribution measured before this work started. classify_figo also uses a
2-minute prolonged-deceleration threshold (the older NICHD convention); FIGO
2015 uses 3 minutes.

THE TABLE, AS WRITTEN
---------------------
FIGO 2015 classifies on three characteristics and a set of duration
qualifiers:

                baseline          variability        decelerations
  Normal        110-160 bpm       5-25 bpm           no repetitive decels
  Suspicious    lacking at least one characteristic of normality,
                with no pathological features
  Pathological  <100 bpm          reduced <5 for     repetitive late or
                                  >50 min, or        prolonged decels for
                                  increased >25      >30 min (>20 min if
                                  for >30 min, or    variability reduced),
                                  sinusoidal >30min  or one prolonged decel
                                                     >5 min

THE DURATION QUALIFIERS ARE THE REASON THIS PROJECT USES EPOCH SEQUENCES
------------------------------------------------------------------------
Every pathological variability/deceleration criterion carries a persistence
requirement of 20-50 minutes. A single 10- or 20-minute window structurally
cannot observe any of them -- a point the existing repo already reached and
documented in knowledge/figo.py, which responded by dropping the variability
flags entirely.

This engine instead evaluates persistence across a patient's CONSECUTIVE
EPOCHS, which is what the guideline actually describes. classify_sequence()
walks a recording forward and, at each epoch, looks back over the trailing
epochs already seen. It never looks forward. The state at epoch t is
therefore computable in real time from data available at epoch t, which is a
hard requirement for the early-warning task and the reason the previous
horizon label had to be abandoned (docs/auroc_ceiling_analysis.md).

Sinusoidal pattern is NOT implemented. It requires a smooth 3-5 cycles/min
undulation of 5-15 bpm sustained >30 min; we have no validated detector for
it and inventing one here would put an unvalidated component inside the
label. Its absence makes the Pathological class slightly under-inclusive,
which is stated in the audit rather than hidden.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .descriptors import (
    BASELINE_BRADY_PATH,
    BASELINE_NORMAL_HIGH,
    BASELINE_NORMAL_LOW,
    VARIABILITY_NORMAL_HIGH,
    VARIABILITY_NORMAL_LOW,
    EpochDescriptors,
)

# HISTORY. The RULE TEXT below has never changed. What changes between
# versions is the MEASUREMENT feeding it, and since that changes the LABEL, a
# dataset built under one version is not comparable with another.
#
#   v1  original -- events detected only where the excursion exceeded 15 bpm
#   v2  gap-aware descriptors: unrepaired dropout no longer read as a
#       deceleration (docs/figo_state_gate1_gate2.md section 1)
#   v3  event delineation from onset to return to baseline
#       (DECEL_EDGE_BPM 7.5, DECEL_MERGE_GAP_S 30). Deceleration F1 against
#       FHRMA expert consensus 0.454 -> 0.603.
RULESET_VERSION = "figo2015-seq-v3"

NORMAL, SUSPICIOUS, PATHOLOGICAL = 0, 1, 2
STATE_NAMES = {NORMAL: "Normal", SUSPICIOUS: "Suspicious",
               PATHOLOGICAL: "Pathological"}

# Persistence requirements, in MINUTES, straight out of the guideline.
PERSIST_REDUCED_VARIABILITY_MIN = 50.0
PERSIST_INCREASED_VARIABILITY_MIN = 30.0
PERSIST_REPETITIVE_DECEL_MIN = 30.0
PERSIST_REPETITIVE_DECEL_REDUCED_VAR_MIN = 20.0

# An epoch too degraded to read is not classified at all. The gate is on
# USABLE signal (measured samples plus short gaps that interpolation
# repaired), not on raw measured fraction: what determines whether a FIGO
# characteristic can be read is how much analysable signal there is, and
# descriptors.py excludes unrepaired dropout from every measurement. 0.50
# matches the CTU-UHB database's own selection criterion and
# pipeline_clinical.py's window gate.
MIN_EPOCH_QUALITY = 0.50

UNREADABLE = -1     # sentinel state: signal too poor to assign a FIGO class


@dataclass
class StateDecision:
    """The state plus every criterion that produced it -- never just the int."""
    state: int
    # normality criteria (all three must hold for Normal)
    baseline_normal: bool
    variability_normal: bool
    no_repetitive_decels: bool
    # pathological criteria (any one is sufficient)
    path_bradycardia: bool
    path_reduced_variability_sustained: bool
    path_increased_variability_sustained: bool
    path_repetitive_decels_sustained: bool
    path_single_prolonged_decel: bool
    # bookkeeping
    readable: bool
    reasons: List[str]

    @property
    def pathological_any(self) -> bool:
        return (self.path_bradycardia
                or self.path_reduced_variability_sustained
                or self.path_increased_variability_sustained
                or self.path_repetitive_decels_sustained
                or self.path_single_prolonged_decel)


def _sustained_minutes(flags: Sequence[bool], epoch_minutes: float) -> float:
    """
    Length in minutes of the run of True ending at the LAST element.

    `flags` is ordered oldest -> newest and must end at the epoch being
    classified. Returns 0.0 if the newest epoch is False. This is the only
    place persistence is computed, and it is backward-looking by
    construction, which is what makes the whole engine causal.
    """
    n = 0
    for f in reversed(flags):
        if not f:
            break
        n += 1
    return n * epoch_minutes


def classify_epoch(d: EpochDescriptors,
                   history: Optional[List[EpochDescriptors]] = None,
                   epoch_minutes: float = 10.0) -> StateDecision:
    """
    FIGO state for one epoch, given the epochs that preceded it.

    Args:
        d: the epoch being classified.
        history: epochs BEFORE `d`, oldest first. Pass [] or None for the
            first epoch of a recording. Only used to evaluate the guideline's
            persistence qualifiers; never looks forward.
        epoch_minutes: length of one epoch. Persistence thresholds are
            evaluated in whole epochs, so a 30-minute requirement needs 3
            consecutive 10-minute epochs.

    Returns a StateDecision, not an int, so that every downstream artefact
    (audit tables, the model's explanation layer, the clinician-facing
    report) can say WHY without recomputing the rules.
    """
    hist = list(history or [])
    reasons: List[str] = []

    if d.usable < MIN_EPOCH_QUALITY:
        return StateDecision(
            state=UNREADABLE, baseline_normal=False, variability_normal=False,
            no_repetitive_decels=False, path_bradycardia=False,
            path_reduced_variability_sustained=False,
            path_increased_variability_sustained=False,
            path_repetitive_decels_sustained=False,
            path_single_prolonged_decel=False, readable=False,
            reasons=[f"usable {d.usable:.2f} < {MIN_EPOCH_QUALITY}"])

    # ---------------------------------------------------------------- normality
    baseline_normal = bool(BASELINE_NORMAL_LOW <= d.baseline_bpm
                           <= BASELINE_NORMAL_HIGH)
    # An unmeasurable variability reading cannot establish normality, but
    # neither may it be thresholded into a pathological finding.
    variability_normal = bool(
        d.variability_measurable
        and VARIABILITY_NORMAL_LOW <= d.variability_bpm <= VARIABILITY_NORMAL_HIGH)
    no_repetitive_decels = not bool(d.decel_repetitive)

    # ------------------------------------------------------------ persistence
    def flag_seq(fn):
        # An epoch too degraded to classify also cannot contribute to a
        # persistence run: 30 minutes of "increased variability" bridged by
        # 10 minutes of sensor dropout is not a 30-minute run, and scoring it
        # as one would manufacture a pathological finding out of missing
        # signal. Unreadable epochs evaluate False and so break the run.
        return [bool(h.usable >= MIN_EPOCH_QUALITY) and fn(h) for h in hist] + [fn(d)]

    reduced = flag_seq(lambda x: bool(x.variability_measurable)
                       and x.variability_bpm < VARIABILITY_NORMAL_LOW)
    increased = flag_seq(lambda x: bool(x.variability_measurable)
                         and x.variability_bpm > VARIABILITY_NORMAL_HIGH)
    rep_bad = flag_seq(lambda x: bool(x.decel_repetitive)
                       and (x.n_decel_late > 0 or x.n_decel_prolonged > 0))

    mins_reduced = _sustained_minutes(reduced, epoch_minutes)
    mins_increased = _sustained_minutes(increased, epoch_minutes)
    mins_rep_bad = _sustained_minutes(rep_bad, epoch_minutes)

    # ------------------------------------------------------------ pathological
    path_brady = bool(d.baseline_bpm < BASELINE_BRADY_PATH)
    path_reduced = mins_reduced >= PERSIST_REDUCED_VARIABILITY_MIN
    path_increased = mins_increased >= PERSIST_INCREASED_VARIABILITY_MIN
    # The guideline shortens the deceleration requirement when variability is
    # also reduced, so the applicable threshold depends on the current epoch.
    decel_threshold = (PERSIST_REPETITIVE_DECEL_REDUCED_VAR_MIN if reduced[-1]
                       else PERSIST_REPETITIVE_DECEL_MIN)
    path_repdecel = mins_rep_bad >= decel_threshold
    path_single_prolonged = bool(d.has_acute_hypoxia_decel)

    if path_brady:
        reasons.append(f"baseline {d.baseline_bpm:.0f} < {BASELINE_BRADY_PATH:.0f} bpm")
    if path_reduced:
        reasons.append(f"variability <5 bpm for {mins_reduced:.0f} min")
    if path_increased:
        reasons.append(f"variability >25 bpm for {mins_increased:.0f} min")
    if path_repdecel:
        reasons.append(f"repetitive late/prolonged decels for {mins_rep_bad:.0f} min "
                       f"(threshold {decel_threshold:.0f})")
    if path_single_prolonged:
        reasons.append(f"prolonged deceleration {d.longest_decel_s / 60:.1f} min below 80 bpm")

    dec = StateDecision(
        state=NORMAL,
        baseline_normal=baseline_normal,
        variability_normal=variability_normal,
        no_repetitive_decels=no_repetitive_decels,
        path_bradycardia=path_brady,
        path_reduced_variability_sustained=path_reduced,
        path_increased_variability_sustained=path_increased,
        path_repetitive_decels_sustained=path_repdecel,
        path_single_prolonged_decel=path_single_prolonged,
        readable=True,
        reasons=reasons,
    )

    if dec.pathological_any:
        dec.state = PATHOLOGICAL
        return dec

    if baseline_normal and variability_normal and no_repetitive_decels:
        dec.state = NORMAL
        return dec

    if not baseline_normal:
        reasons.append(f"baseline {d.baseline_bpm:.0f} outside "
                       f"{BASELINE_NORMAL_LOW:.0f}-{BASELINE_NORMAL_HIGH:.0f}")
    if not variability_normal:
        reasons.append("variability unmeasurable" if not d.variability_measurable
                       else f"variability {d.variability_bpm:.1f} outside "
                            f"{VARIABILITY_NORMAL_LOW:.0f}-{VARIABILITY_NORMAL_HIGH:.0f}")
    if not no_repetitive_decels:
        reasons.append("repetitive decelerations (>50% of contractions)")
    dec.state = SUSPICIOUS
    return dec


def classify_sequence(descriptors: Sequence[EpochDescriptors],
                      epoch_minutes: float = 10.0) -> List[StateDecision]:
    """
    Classify a whole recording, epoch by epoch, forward in time.

    Each epoch sees only the epochs before it. Unreadable epochs stay in the
    history so that persistence runs are broken by them rather than silently
    bridged -- a 30-minute run of increased variability interrupted by 10
    minutes of unreadable signal is not a 30-minute run, and treating it as
    one would invent a pathological finding out of a sensor dropout.
    """
    out: List[StateDecision] = []
    hist: List[EpochDescriptors] = []
    for d in descriptors:
        out.append(classify_epoch(d, hist, epoch_minutes))
        hist.append(d)
    return out


def states_of(decisions: Sequence[StateDecision]) -> np.ndarray:
    return np.array([d.state for d in decisions], dtype=np.int64)
