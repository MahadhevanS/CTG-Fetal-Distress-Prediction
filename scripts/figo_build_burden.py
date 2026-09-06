"""
Recompute the continuous deceleration burden for every CTU-UHB epoch.

    python scripts/figo_build_burden.py

Writes data/processed_figo/burden.npz, aligned row-for-row with epochs.npz.

WHY THIS IS A SEPARATE BUILD
-----------------------------
v3's epochs.npz stores the VERDICT (`decel_repetitive`) but not the counts
behind it, so the brittleness of the >50% rule cannot be measured from it --
only bounded, and the bound is vacuous (every epoch is adjacent to the
boundary for SOME consistent n_hit). Recomputing from the raw signal gives
`n_contractions_with_decel` directly and turns the instability claim into a
measurement.

The deceleration delineation, contraction detection and baseline estimator
are imported unchanged from descriptors.py, so this is the same instrument
that produced v3 -- only more of its internals are retained.

epochs.npz is NOT modified. v3 labels are frozen.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "src", "preprocessing"))

from figo_state.burden import DecelBurden, compute_burden        # noqa: E402
from figo_state.epochs import (EPOCH_SAMPLES,                    # noqa: E402
                               PATIENT_MAX_MISSING_RATIO,
                               segment_recording)
from figo_state import descriptors as DS                         # noqa: E402
from ingestion import load_ctu_chb_record                        # noqa: E402

RAW = os.path.join(BASE, "data", "raw", "ctu-chb-intrapartum")
DATA = os.path.join(BASE, "data", "processed_figo")


def main():
    z = np.load(os.path.join(DATA, "epochs.npz"), allow_pickle=True)
    meta = pd.DataFrame({c: z[c] for c in z.files
                         if c not in ("X", "F", "descriptor_names",
                                      "fhr_valid", "uc_valid")})
    key = {(str(r), int(i)): n
           for n, (r, i) in enumerate(zip(meta.record_id, meta.epoch_index))}
    B = np.full((len(meta), len(DecelBurden.names())), np.nan, np.float32)

    rids = sorted({str(r) for r in meta.record_id})
    print(f"{len(rids)} records, {len(meta)} epochs")
    t0 = time.time()
    for n, rid in enumerate(rids, 1):
        if n % 100 == 0:
            print(f"  ...{n}/{len(rids)}  ({time.time()-t0:.0f}s)")
        fhr, uc, fs = load_ctu_chb_record(os.path.join(RAW, rid))
        if len(fhr) == 0 or float(np.mean(fhr == 0.0)) > PATIENT_MAX_MISSING_RATIO:
            continue
        eps, _ = segment_recording(fhr, uc, fs, EPOCH_SAMPLES)
        prev = None
        for ep in eps:
            k = key.get((rid, int(ep["index"])))
            b, _ = DS.estimate_baseline(ep["fhr"], ep["valid"], prev)
            if float(np.mean(ep["valid"])) >= 0.5 and np.isfinite(b):
                prev = b
            if k is None or not np.isfinite(b):
                continue
            B[k] = compute_burden(ep["fhr"], ep["uc"], fs, b,
                                  ep["valid"], ep["uc_valid"]).to_vector()

    np.savez_compressed(os.path.join(DATA, "burden.npz"),
                        B=B, burden_names=np.array(DecelBurden.names()),
                        record_id=meta.record_id.values,
                        epoch_index=meta.epoch_index.values)
    ok = np.isfinite(B).all(1)
    print(f"\nwrote {DATA}/burden.npz  {B.shape}  ({int(ok.sum())} complete rows)")


if __name__ == "__main__":
    main()
