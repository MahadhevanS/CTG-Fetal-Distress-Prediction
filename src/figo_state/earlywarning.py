"""
Early-warning task construction: anchors, history features, leakage rules.

THE TASK
--------
From an epoch currently classified NORMAL, predict whether the CTG becomes
ABNORMAL (Suspicious or Pathological) within the next 30 minutes.

WHY THIS TARGET AND NOT THE PATHOLOGICAL ONE
---------------------------------------------
docs/figo_state_gate1_gate2.md measured the Pathological-within-30-min target
at 36 positive anchors from 18 patients, and repairing the deceleration
detector made it worse rather than better. That is a base-rate property of
CTU-UHB, not a modelling failure. Normal -> Abnormal is the same question one
severity step earlier, and it is well powered.

THE THREE LEAKAGE RULES, ENFORCED HERE AND NOWHERE ELSE
--------------------------------------------------------
1. A NEGATIVE REQUIRES A FULLY OBSERVED FUTURE. All 3 horizon epochs must
   exist in the recording AND be readable. Otherwise "did not deteriorate"
   can mean "we stopped recording first", and since recordings end at
   delivery that is the exact confound that gave AUROC 0.84 from a clock
   alone (docs/auroc_ceiling_analysis.md).

2. INCLUSION MUST NOT DEPEND ON THE OUTCOME. The same observability test is
   applied to positives. An earlier version of the deterioration target kept
   positives on truncated horizons and censored only negatives; the epoch
   index alone then scored 0.8321. Both classes draw from one anchor pool.

3. NOTHING AFTER THE ANCHOR MAY ENTER A FEATURE. History is epochs strictly
   BEFORE the anchor, in the same recording. `n_epochs_in_record` and
   `minutes_before_end` are never features: a labour ward does not know when
   labour will end, and that quantity produced this project's original time
   confound.

WHY THE COHORT IS FIXED BY HISTORY DEPTH
-----------------------------------------
Anchors deep enough to have k prior epochs are systematically LATER in
labour, and later anchors deteriorate more often: measured, P(deteriorate)
rises 57.2% at epoch 0 to 74.2% at epoch 4, and the anchor pool's prevalence
rises 62.7% (>=0 prior) to 72.9% (>=3 prior). So comparing "no history" on
1,122 anchors against "20 min of history" on 575 anchors would confound
context length with cohort severity, and would report the cohort shift as a
modelling gain. Every context-length comparison therefore runs on ONE fixed
anchor set, chosen by the deepest context being tested.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HORIZON_EPOCHS = 3          # 30 minutes at 10 minutes per epoch
NORMAL, SUSPICIOUS, PATHOLOGICAL, UNREADABLE = 0, 1, 2, -1


def build_anchors(meta: pd.DataFrame,
                  horizon: int = HORIZON_EPOCHS) -> pd.DataFrame:
    """
    One row per valid anchor.

    `meta` must be the full epoch table, sorted by (record_id, epoch_index).
    Returns a frame carrying the anchor's row index into the epoch table, its
    label, and how many prior epochs are available to it.
    """
    meta = meta.sort_values(["record_id", "epoch_index"]).reset_index(drop=True)
    out: List[Dict] = []
    for rid, g in meta.groupby("record_id", sort=False):
        s = g.y_state.values
        rows = g.index.values
        n = len(s)
        for t in range(n):
            if s[t] != NORMAL:
                continue                                   # anchor must be Normal
            fut = s[t + 1: t + 1 + horizon]
            if len(fut) < horizon:
                continue                                   # rule 1: truncated
            if np.any(fut == UNREADABLE):
                continue                                   # rule 1: unobserved
            out.append(dict(
                anchor_row=int(rows[t]), record_id=str(rid),
                epoch_index=int(t), n_hist=int(t),
                # rule 3: history rows are strictly before the anchor
                hist_rows=rows[:t].tolist(),
                y=int(np.any(fut >= SUSPICIOUS))))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# feature construction
# --------------------------------------------------------------------------
def _trend(v: np.ndarray) -> float:
    """Least-squares slope per epoch. 0.0 when a slope is undefined."""
    if len(v) < 2:
        return 0.0
    t = np.arange(len(v), dtype=float)
    return float(np.polyfit(t, v, 1)[0])


def history_features(F: np.ndarray, states: np.ndarray,
                     anchors: pd.DataFrame, context: int,
                     names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Descriptor features for each anchor, using `context` prior epochs.

    context = 0 gives the anchor epoch only (Stage 1's "instantaneous state").
    context = k adds, per descriptor, four summaries over the k epochs before
    the anchor plus the anchor itself:

        level        mean over the window
        trend        least-squares slope per epoch
        instability  standard deviation over the window
        delta        anchor value minus the oldest value in the window

    These four are the vocabulary the plan asks for (level, trend,
    instability, persistence) and they are computed explicitly rather than
    learned, so a gain over context=0 is attributable to trajectory
    information and not to extra capacity.

    Plus FIGO-state history over the same window: how much of the recent past
    was Suspicious, Pathological or unreadable, how many state changes
    occurred, and how long the current Normal run has lasted. The anchor is
    Normal by construction, so these describe warning signs BEFORE it.
    """
    cols: List[str] = []
    rows: List[np.ndarray] = []

    for _, a in anchors.iterrows():
        ar = int(a.anchor_row)
        hist = list(a.hist_rows)[-context:] if context > 0 else []
        win_all = hist + [ar]                   # oldest -> anchor

        # An UNREADABLE epoch has no measured descriptors -- 41 epochs in the
        # corpus carry a NaN baseline because there was no usable signal and
        # no previous level to carry forward. Averaging a baseline that was
        # never measured would invent a value, so descriptor summaries are
        # computed over READABLE epochs only. The unreadable ones are not
        # discarded: they are counted in hist_frac_unreadable below, which is
        # where signal loss belongs as a feature in its own right.
        win = [r for r in win_all if states[r] != UNREADABLE]
        if ar not in win:                       # the anchor is always readable
            win.append(ar)
        Fw = F[win]

        feat = [F[ar]]                          # anchor value, always
        if context > 0:
            feat.append(Fw.mean(axis=0))
            feat.append(np.array([_trend(Fw[:, j]) for j in range(Fw.shape[1])]))
            feat.append(Fw.std(axis=0))
            feat.append(F[ar] - Fw[0])
            sh = states[hist] if hist else np.array([], dtype=int)
            n_run = 0
            for s in reversed(sh):              # length of the Normal run
                if s != NORMAL:
                    break
                n_run += 1
            changes = int((np.diff(sh) != 0).sum()) if len(sh) > 1 else 0
            feat.append(np.array([
                float(np.mean(sh == SUSPICIOUS)) if len(sh) else 0.0,
                float(np.mean(sh == PATHOLOGICAL)) if len(sh) else 0.0,
                float(np.mean(sh == UNREADABLE)) if len(sh) else 0.0,
                float(changes),
                float(n_run),
                float(len(sh)),
            ]))
        rows.append(np.concatenate(feat))

    if not cols:
        cols = [f"{n}__now" for n in names]
        if context > 0:
            cols += [f"{n}__level" for n in names]
            cols += [f"{n}__trend" for n in names]
            cols += [f"{n}__instab" for n in names]
            cols += [f"{n}__delta" for n in names]
            cols += ["hist_frac_suspicious", "hist_frac_pathological",
                     "hist_frac_unreadable", "hist_n_state_changes",
                     "hist_normal_run_len", "hist_len"]
    return np.vstack(rows).astype(np.float32), cols


def fixed_cohort(anchors: pd.DataFrame, min_hist: int) -> pd.DataFrame:
    """
    Anchors with at least `min_hist` prior epochs.

    Used to hold the cohort constant across a context-length sweep. See the
    module docstring: deeper-history anchors are later in labour and
    deteriorate more often, so varying the cohort with the context would
    report a severity shift as a modelling gain.
    """
    return anchors[anchors.n_hist >= min_hist].reset_index(drop=True)
