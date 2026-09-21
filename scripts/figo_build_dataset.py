"""
Build the FIGO-state / early-warning substrate from raw CTU-UHB.

    python scripts/figo_build_dataset.py

Writes data/processed_figo/:
    epochs.npz          signals, descriptors, states, all targets
    manifest.json       cohort counts, ruleset version, every constant used

WHAT ONE ROW IS
---------------
One non-overlapping 10-minute epoch of one recording. Rows are ordered by
(record_id, epoch_index) so a patient's sequence is contiguous and sorted.

TARGETS ON EACH ROW
-------------------
    y_state       FIGO state now: 0 Normal, 1 Suspicious, 2 Pathological,
                  -1 unreadable. This is Approach 1's label.
    y_deteriorate_{20,30,40}
                  1 if a PATHOLOGICAL epoch occurs within the next N minutes,
                  0 if it does not, -1 if the answer is not determinable.
                  This is Approach 2's label.

WHY THERE IS A -1 AND NOT JUST A 0
-----------------------------------
-1 means the anchor is not in the risk set. Two things put it there, and
scoring either as 0 would fabricate a result:

  * RIGHT-CENSORED. The recording ends before the horizon closes. Absence of
    a pathological epoch in the remaining 10 minutes is not evidence that
    none would have occurred in the next 30, and recordings end at delivery,
    so scoring those as 0 loads the negative class with end-of-recording
    anchors -- the exact time confound that produced AUROC 0.84 from a clock
    alone (docs/auroc_ceiling_analysis.md).
  * UNOBSERVED HORIZON. An epoch inside the horizon is quality-gated out, so
    the outcome was not observed.

BOTH CLASSES ARE CENSORED THE SAME WAY. It is tempting to keep a positive
whenever a pathological epoch is seen, even on a truncated horizon, since
the event is not in doubt. That makes INCLUSION depend on the OUTCOME, and
on the first build it did exactly what you would expect: y=0 anchors had
mean epoch index 1.53, y=1 anchors 3.55, and the epoch index alone scored
AUROC 0.8321. See build_deterioration_targets() for the full account.

ELIGIBILITY IS SEPARATE FROM THE LABEL
--------------------------------------
`eligible_ew` marks anchors where the CURRENT epoch is Normal or Suspicious.
An anchor that is already Pathological is recognition, not prediction (the
user's Phase 2B). It is stored, not dropped, so the persistence baseline can
be computed on the full set and the restriction can be shown to matter.
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "src", "preprocessing"))

from figo_state import descriptors as DS          # noqa: E402
from figo_state import rules as RL                # noqa: E402
from figo_state.epochs import (                   # noqa: E402
    EPOCH_MINUTES,
    EPOCH_SAMPLES,
    PATIENT_MAX_MISSING_RATIO,
    describe_recording,
)
from ingestion import TARGET_FS, load_clinical_metadata, load_ctu_chb_record  # noqa: E402

RAW = os.path.join(BASE, "data", "raw", "ctu-chb-intrapartum")
OUT = os.path.join(BASE, "data", "processed_figo")

HORIZONS_MIN = (20.0, 30.0, 40.0)

# pH thresholds carried over unchanged from pipeline_clinical.py so Task 3
# (does predicted deterioration relate to acidaemia) stays comparable with
# every earlier result in this repo.
PH_PRIMARY = 7.15


def _col(md: pd.DataFrame, name: str):
    def norm(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())
    want = norm(name)
    for c in md.columns:
        if norm(c) == want:
            return md[c]
    return None


def build_deterioration_targets(states: np.ndarray, horizon_epochs: int) -> np.ndarray:
    """
    For each epoch t: does a PATHOLOGICAL epoch occur in t+1 .. t+horizon?

    1  = a pathological epoch is observed inside the horizon
    0  = none is, and the whole horizon was observed and readable
    -1 = the anchor is not in the risk set at all

    THE INCLUSION CRITERION MUST NOT DEPEND ON THE OUTCOME.
    An earlier version of this function scored a positive whenever a
    pathological epoch was seen, even if the horizon ran off the end of the
    recording, and required full observation only for negatives. That is the
    natural survival-analysis instinct and it is wrong here, because it makes
    inclusion outcome-dependent: negatives could only be anchored where three
    further epochs existed, positives anywhere. Measured on the first build,
    y=0 anchors had mean epoch_index 1.53 and y=1 anchors 3.55, and a
    predictor consisting of NOTHING BUT the epoch index scored AUROC 0.8321.

    That is the same failure this project already diagnosed once: the horizon
    label of pipeline_mil.py, where window position alone reached 0.84 and
    beat every trained model (docs/auroc_ceiling_analysis.md). It is not
    allowed to come back through the risk set.

    So an anchor is in the risk set if and only if the full horizon exists in
    the recording AND every epoch of it is readable -- a condition evaluated
    without reference to what the horizon contains. Both classes then draw
    from the same anchor pool. The clock predictor falls from 0.8321 to
    0.7078, and the residue is real physiology (labour genuinely deteriorates
    toward delivery) rather than an artefact of how the label was built. It
    stays in the audit as a permanent control.

    `states` is the epoch state sequence for ONE patient, in order, with -1
    for unreadable epochs.
    """
    n = len(states)
    y = np.full(n, -1, dtype=np.int64)
    for t in range(n):
        fut = states[t + 1: t + 1 + horizon_epochs]
        if len(fut) < horizon_epochs:
            continue                       # horizon runs off the recording
        if np.any(fut == RL.UNREADABLE):
            continue                       # horizon not fully observed
        y[t] = int(np.any(fut == RL.PATHOLOGICAL))
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch_minutes", type=float, default=EPOCH_MINUTES)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    epoch_minutes = float(args.epoch_minutes)
    epoch_samples = int(epoch_minutes * 60 * TARGET_FS)
    os.makedirs(args.out, exist_ok=True)

    md = load_clinical_metadata(os.path.join(RAW, "clinical_metadata.csv"))
    if "record_id" in md.columns:
        md = md.set_index("record_id")
    ph = pd.to_numeric(_col(md, "ph"), errors="coerce")
    bdecf = pd.to_numeric(_col(md, "bdecf"), errors="coerce")
    apgar5 = pd.to_numeric(_col(md, "apgar5"), errors="coerce")

    record_ids = sorted(
        f[:-4] for f in os.listdir(RAW) if f.endswith(".dat"))
    print(f"{len(record_ids)} raw records | epoch = {epoch_minutes:.0f} min "
          f"({epoch_samples} samples at {TARGET_FS:g} Hz)")
    print(f"ruleset = {RL.RULESET_VERSION}\n")

    SIG: List[np.ndarray] = []
    VF: List[np.ndarray] = []
    VU: List[np.ndarray] = []
    DESC: List[np.ndarray] = []
    rows: List[Dict] = []
    dropped: List[Dict] = []

    for k, rid in enumerate(record_ids, 1):
        if k % 100 == 0:
            print(f"  ...{k}/{len(record_ids)}")
        fhr, uc, fs = load_ctu_chb_record(os.path.join(RAW, rid))
        if len(fhr) == 0:
            dropped.append(dict(record_id=rid, reason="load_failed"))
            continue
        miss = float(np.mean(fhr == 0.0))
        if miss > PATIENT_MAX_MISSING_RATIO:
            dropped.append(dict(record_id=rid, reason="patient_quality",
                                missing=round(miss, 3)))
            continue

        eps, descs, decisions = describe_recording(
            fhr, uc, fs, epoch_samples, epoch_minutes)
        if not eps:
            dropped.append(dict(record_id=rid, reason="no_epochs"))
            continue

        states = RL.states_of(decisions)
        det = {h: build_deterioration_targets(states, int(round(h / epoch_minutes)))
               for h in HORIZONS_MIN}

        y_ph = float(ph.get(rid, np.nan)) if rid in ph.index else float("nan")
        y_bd = float(bdecf.get(rid, np.nan)) if rid in bdecf.index else float("nan")
        y_ap = float(apgar5.get(rid, np.nan)) if rid in apgar5.index else float("nan")

        for i, (ep, d, dec) in enumerate(zip(eps, descs, decisions)):
            # channel 0 FHR in bpm, 1 UC, 2 pre-interpolation missingness.
            # Stored unnormalised; z-scoring is fit on training patients only
            # at experiment time, never here, so the split cannot leak in.
            SIG.append(np.vstack((ep["fhr"], ep["uc"], ep["mask"])
                                 ).astype(np.float32))
            # Per-sample validity, stored SEPARATELY rather than folded into X
            # so that X stays byte-identical and any model already frozen
            # against it stays exactly reproducible. `mask` (X channel 2) is
            # raw pre-interpolation missingness; these two are what is
            # actually USABLE after short-gap repair, dilated by the filter
            # settling time -- the masks a model's attention must respect.
            VF.append(ep["valid"].astype(np.uint8))
            VU.append(ep["uc_valid"].astype(np.uint8))
            DESC.append(d.to_vector())
            rows.append(dict(
                record_id=rid, epoch_index=i,
                start_sample=int(ep["start"]), end_sample=int(ep["end"]),
                minutes_from_start=float(i * epoch_minutes),
                n_epochs_in_record=len(eps),
                y_state=int(states[i]),
                readable=int(dec.readable),
                quality=float(d.quality),
                y_det20=int(det[20.0][i]), y_det30=int(det[30.0][i]),
                y_det40=int(det[40.0][i]),
                eligible_ew=int(states[i] in (RL.NORMAL, RL.SUSPICIOUS)),
                path_bradycardia=int(dec.path_bradycardia),
                path_reduced_var=int(dec.path_reduced_variability_sustained),
                path_increased_var=int(dec.path_increased_variability_sustained),
                path_rep_decels=int(dec.path_repetitive_decels_sustained),
                path_prolonged=int(dec.path_single_prolonged_decel),
                baseline_normal=int(dec.baseline_normal),
                variability_normal=int(dec.variability_normal),
                no_repetitive_decels=int(dec.no_repetitive_decels),
                ph=y_ph, bdecf=y_bd, apgar5=y_ap,
                y_acidemic=int(y_ph <= PH_PRIMARY) if np.isfinite(y_ph) else -1,
            ))

    meta = pd.DataFrame(rows)
    X = np.stack(SIG).astype(np.float32)
    F = np.stack(DESC).astype(np.float32)

    np.savez_compressed(
        os.path.join(args.out, "epochs.npz"),
        X=X, F=F,
        fhr_valid=np.stack(VF).astype(np.uint8),
        uc_valid=np.stack(VU).astype(np.uint8),
        descriptor_names=np.array(DS.EpochDescriptors.names()),
        **{c: meta[c].values for c in meta.columns})

    manifest = dict(
        ruleset_version=RL.RULESET_VERSION,
        epoch_minutes=epoch_minutes,
        epoch_samples=epoch_samples,
        fs=TARGET_FS,
        horizons_min=list(HORIZONS_MIN),
        n_records_raw=len(record_ids),
        n_records_kept=int(meta.record_id.nunique()),
        n_epochs=int(len(meta)),
        n_dropped=len(dropped),
        dropped=dropped,
        patient_max_missing_ratio=PATIENT_MAX_MISSING_RATIO,
        min_epoch_quality=RL.MIN_EPOCH_QUALITY,
        thresholds={k: getattr(DS, k) for k in dir(DS)
                    if k.isupper() and isinstance(getattr(DS, k), float)},
        persistence_min={k: getattr(RL, k) for k in dir(RL)
                         if k.startswith("PERSIST_")},
        signal_channels=["fhr_bpm", "uc", "missing_mask"],
        aux_arrays={"fhr_valid": "per-sample usable-signal mask for FHR",
                    "uc_valid": "per-sample usable-signal mask for UC"},
        signal_note="unnormalised; z-score must be fit on training patients only",
    )
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nwrote {args.out}/epochs.npz")
    print(f"  X {X.shape}  F {F.shape}  meta {meta.shape}")
    print(f"  {meta.record_id.nunique()} patients, {len(meta)} epochs, "
          f"{len(dropped)} records dropped")
    vc = meta.y_state.value_counts().sort_index()
    for s, n in vc.items():
        nm = "UNREADABLE" if s == -1 else RL.STATE_NAMES[s]
        print(f"  {nm:12s} {n:5d}  ({100 * n / len(meta):5.1f}%)")


if __name__ == "__main__":
    main()
