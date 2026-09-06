"""
Clinically-valid pipeline -- removes the horizon label, the time confound, and
the time-biased quality gate.

WHY THIS EXISTS (2026-09-02). docs/auroc_ceiling_analysis.md showed that the
window label produced by pipeline_mil.py is largely a statement about WHEN a
window occurred rather than what is in it:

  * time alone (window position, no signal at all) reaches AUROC 0.84 on
    y_primary -- beating every trained model (CrossFormer 0.812, CNN1D 0.821,
    MSLSTM 0.839);
  * the delivered model scores only 0.6086 on "is this baby acidotic";
  * in NORMAL patients, whose windows are all label 0, risk still doubles from
    the first third of the hour to the last (rho=+0.35, p=1.8e-27);
  * the last negative and first positive window of every acidotic patient share
    87.5% of their signal and carry opposite labels.

Root cause, pipeline_mil.py:190-192:

    within_horizon = (start >= signal_length - PREDICTION_HORIZON_SAMPLES)
    window_label   = int(is_distress and within_horizon)

The horizon is measured from the END of the recording, and `sig2birth == 0` for
every CTU-UHB record, so recordings end at delivery. The label is therefore
computed from the future and cannot be reproduced at inference time, when
remaining labour duration is unknown. No paper reviewed in
docs/literature_preprocessing_comparison.md does this; all apply the patient
outcome to every window.

WHAT CHANGES vs pipeline_mil.py
  1. Horizon rule deleted. y_primary = the patient outcome on every window.
  2. Secondary composite target y_adverse, plus continuous ph/bdecf/apgar5,
     because 47% of the pH<=7.15 positives had a 5-minute Apgar >= 9.
  3. Window quality gate relaxed 0.30 -> 0.50 (matching Dang et al. and the
     database's own selection criterion). The old 0.30 per-window gate removed
     23.6pp more horizon windows than pre-horizon windows in acidotic patients
     and left 27/108 of them with no positive window at all.
  4. Channel 3 = the missingness mask, taken BEFORE interpolation, so the model
     can tell measured samples from reconstructed ones.
  5. minutes_before_end / is_second_stage / quality emitted as METADATA ONLY --
     never labels. They exist so the time-confound audit can be re-run.

WHAT IS DELIBERATELY UNCHANGED
  The signal chain (spike removal -> cubic interpolation -> lowpass -> iterative
  baseline -> baseline-corrected channel 0 -> train-fit z-score), the 20-minute
  window, the 2.5-minute stride, the 60-minute truncation, and the 70/15/15
  patient split with random_state=42. pipeline.py and pipeline_mil.py are not
  touched, so every existing result stays reproducible.

ACCEPTANCE TEST (docs/preprocessing_redesign.md): after this pipeline runs,
AUROC(window position -> y_primary) must fall from 0.84 to ~0.50. Run
scripts/audit_time_confound.py. If it does not, do not train on this data.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from ingestion import load_ctu_chb_record, load_clinical_metadata, TARGET_FS
from signal_quality import get_valid_windows, assess_patient_missing_ratio
from filtering import (
    remove_spikes,
    interpolate_missing,
    apply_lowpass_filter,
    zscore_normalize_channels,
)
from baseline import calculate_iterative_baseline
from features import calculate_variability, detect_accelerations, detect_decelerations

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from knowledge.figo import classify_figo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_MINUTES = 20
LAST_HOUR_MINUTES = 60
STRIDE_MINUTES = 2.5

WINDOW_SAMPLES = int(WINDOW_MINUTES * 60 * TARGET_FS)        # 4800
LAST_HOUR_SAMPLES = int(LAST_HOUR_MINUTES * 60 * TARGET_FS)  # 14400
STRIDE_SAMPLES = int(STRIDE_MINUTES * 60 * TARGET_FS)        # 600

# CHANGED vs pipeline_mil.py: 0.30 -> 0.50. See module docstring.
PATIENT_MAX_MISSING_RATIO = 0.5
WINDOW_MAX_MISSING_RATIO = 0.5

FHR_MIN_BPM = 50.0
FHR_MAX_BPM = 240.0

# Targets
PH_PRIMARY = 7.15      # field convention -- keeps comparability with the benchmark
PH_SEVERE = 7.05       # composite arm 1
BDECF_ACIDEMIA = 12.0  # composite arm 2 -- metabolic (not respiratory) acidemia
APGAR5_LOW = 7         # composite arm 3 -- depressed newborn


def _col(md: pd.DataFrame, name: str):
    """
    Resolve a metadata column by name.

    The CTU-UHB header is messy -- 'pos. ii.st.' arrives as 'pos._ii.st.' after
    the loader's whitespace handling, and case varies. Compare on alphanumerics
    only so spaces, underscores and punctuation cannot cause a silent miss.
    """
    def norm(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    want = norm(name)
    for c in md.columns:
        if norm(c) == want:
            return md[c]
    return None


def build_targets(md: pd.DataFrame) -> pd.DataFrame:
    """Primary pH label, composite adverse outcome, and the continuous markers."""
    ph = pd.to_numeric(_col(md, 'ph'), errors='coerce')
    bd = pd.to_numeric(_col(md, 'bdecf'), errors='coerce')
    a5 = pd.to_numeric(_col(md, 'apgar5'), errors='coerce')

    out = pd.DataFrame(index=md.index)
    out['ph'] = ph
    out['bdecf'] = bd
    out['apgar5'] = a5
    out['y_primary'] = (ph <= PH_PRIMARY).astype(int)
    out['y_adverse'] = (
        (ph <= PH_SEVERE) | (bd >= BDECF_ACIDEMIA) | (a5 < APGAR5_LOW)
    ).fillna(False).astype(int)
    return out


def process_pipeline_clinical(raw_data_dir: str, metadata_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("Loading metadata...")
    metadata = load_clinical_metadata(metadata_path)
    if 'record_id' in metadata.columns:
        metadata = metadata.set_index('record_id')
    tgt = build_targets(metadata)
    _p2 = _col(metadata, 'pos. ii.st.')
    if _p2 is None:
        print("  [WARN] 'pos. ii.st.' not found -- is_second_stage will be -1 (unknown)")
        pos_ii = pd.Series(-1.0, index=metadata.index)
    else:
        pos_ii = pd.to_numeric(_p2, errors='coerce').fillna(-1.0)
    print(f"  second-stage onset known for {int((pos_ii >= 0).sum())}/{len(pos_ii)} records")

    print(f"  {len(tgt)} records | pH<={PH_PRIMARY}: {int(tgt['y_primary'].sum())} "
          f"| composite adverse: {int(tgt['y_adverse'].sum())}")

    # Stratify on the PRIMARY target, same 70/15/15 / random_state=42 as
    # pipeline_mil.py so the patient cohorts stay comparable.
    records = list(metadata.index)
    strat = tgt['y_primary'].values

    train_ids, tv_ids, _, y_tv = train_test_split(
        records, strat, test_size=0.3, stratify=strat, random_state=42)
    val_ids, test_ids, _, _ = train_test_split(
        tv_ids, y_tv, test_size=0.5, stratify=y_tv, random_state=42)
    splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}

    dropped_patients = []

    for split_name, ids in splits.items():
        print(f"\nProcessing {split_name} split ({len(ids)} patients)...")
        X, Yp, Yadv, Yfigo, Yfeat, Meta = [], [], [], [], [], []
        Wend, Wsecond, Wqual, Yph, Ybd, Yap = [], [], [], [], [], []
        n_kept = n_quality = n_missing = 0

        for rid in ids:
            path = os.path.join(raw_data_dir, str(rid))
            if not os.path.exists(path + '.dat'):
                n_missing += 1
                continue
            fhr, uc, fs = load_ctu_chb_record(path)
            if len(fhr) == 0:
                n_missing += 1
                continue
            if not assess_patient_missing_ratio(fhr, max_missing_ratio=PATIENT_MAX_MISSING_RATIO):
                n_quality += 1
                dropped_patients.append((str(rid), split_name, 'patient_quality'))
                continue

            full_len = len(fhr)
            if full_len > LAST_HOUR_SAMPLES:
                fhr = fhr[-LAST_HOUR_SAMPLES:]
                uc = uc[-LAST_HOUR_SAMPLES:]
            seg_len = len(fhr)
            offset = full_len - seg_len          # where the segment starts in full coords

            # Second-stage onset, translated into segment coordinates. -1 = unknown.
            p2 = float(pos_ii.get(rid, -1))
            second_start = (p2 - offset) if p2 >= 0 else -1.0

            starts = get_valid_windows(fhr, WINDOW_SAMPLES, STRIDE_SAMPLES,
                                       max_missing_ratio=WINDOW_MAX_MISSING_RATIO)
            if len(starts) == 0:
                dropped_patients.append((str(rid), split_name, 'no_valid_windows'))
                continue
            n_kept += 1

            row = tgt.loc[rid]
            y_primary = int(row['y_primary'])
            y_adverse = int(row['y_adverse'])

            for start in starts:
                end = start + WINDOW_SAMPLES
                raw_fhr = fhr[start:end].copy()
                raw_uc = uc[start:end].copy()

                # (4) missingness mask BEFORE any reconstruction
                mask = (raw_fhr == 0.0).astype(np.float32)
                quality = float(1.0 - mask.mean())

                f = remove_spikes(raw_fhr.copy(), fs=fs)
                f = interpolate_missing(f, clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)
                f = apply_lowpass_filter(f, fs=fs)
                u = apply_lowpass_filter(raw_uc.copy(), fs=fs)

                baseline = calculate_iterative_baseline(f)
                stv, ltv = calculate_variability(f, fs=fs, baseline=baseline)
                accels = detect_accelerations(f, baseline, fs=fs)
                decels = detect_decelerations(f, baseline, u, fs=fs)
                base_val = float(np.mean(baseline))

                # channel 0 baseline-corrected FHR, 1 UC, 2 missingness mask
                X.append(np.vstack((f - baseline, u, mask)))

                # (1) NO horizon rule -- the patient outcome, on every window
                Yp.append(y_primary)
                Yadv.append(y_adverse)
                Yfigo.append(int(classify_figo(base_val, ltv, accels, decels)))
                Yfeat.append([base_val, stv, ltv, float(accels),
                              float(decels['early']), float(decels['late']),
                              float(decels['variable']), float(decels['prolonged'])])
                Meta.append((str(rid), int(start), int(end)))

                # (5) metadata only -- NEVER labels
                Wend.append(float((seg_len - end) / (TARGET_FS * 60.0)))
                Wsecond.append(float(1.0) if (second_start >= 0 and start >= second_start)
                               else (0.0 if second_start >= 0 else -1.0))
                Wqual.append(quality)
                Yph.append(float(row['ph']) if pd.notna(row['ph']) else float('nan'))
                Ybd.append(float(row['bdecf']) if pd.notna(row['bdecf']) else float('nan'))
                Yap.append(float(row['apgar5']) if pd.notna(row['apgar5']) else float('nan'))

        if not X:
            print(f"  [WARNING] no windows for {split_name}")
            continue

        ds = {
            'X': torch.tensor(np.array(X), dtype=torch.float32),
            'y_primary': torch.tensor(np.array(Yp), dtype=torch.long),
            'y_patient': torch.tensor(np.array(Yp), dtype=torch.long),  # identical by design
            'y_adverse': torch.tensor(np.array(Yadv), dtype=torch.long),
            'y_figo': torch.tensor(np.array(Yfigo), dtype=torch.long),
            'y_features': torch.tensor(np.array(Yfeat), dtype=torch.float32),
            'metadata': Meta,
            'w_minutes_before_end': torch.tensor(np.array(Wend), dtype=torch.float32),
            'w_is_second_stage': torch.tensor(np.array(Wsecond), dtype=torch.float32),
            'w_quality': torch.tensor(np.array(Wqual), dtype=torch.float32),
            'y_ph': torch.tensor(np.array(Yph), dtype=torch.float32),
            'y_bdecf': torch.tensor(np.array(Ybd), dtype=torch.float32),
            'y_apgar5': torch.tensor(np.array(Yap), dtype=torch.float32),
        }
        torch.save(ds, os.path.join(output_dir, f'{split_name}_dataset.pt'))
        n = len(X)
        print(f"  Saved {split_name}_dataset.pt -- {n} windows from {n_kept} patients "
              f"({n_missing} missing files, {n_quality} failed patient quality)")
        print(f"    y_primary (pH<={PH_PRIMARY}) positive : {int(ds['y_primary'].sum())} "
              f"({100*float(ds['y_primary'].float().mean()):.1f}% of windows)")
        print(f"    y_adverse (composite)      positive : {int(ds['y_adverse'].sum())} "
              f"({100*float(ds['y_adverse'].float().mean()):.1f}%)")
        print(f"    mean window quality                 : {float(ds['w_quality'].mean()):.3f}")

    # -----------------------------------------------------------------------
    # Z-score channels 0 and 1 on TRAIN only. Channel 2 is a 0/1 mask and is
    # deliberately left unnormalised -- z-scoring it would destroy its meaning.
    # -----------------------------------------------------------------------
    print("\nNormalising channels 0-1 (fit on train; channel 2 mask left as 0/1)...")
    train_path = os.path.join(output_dir, 'train_dataset.pt')
    tr = torch.load(train_path, weights_only=False)
    _, mean, std = zscore_normalize_channels(tr['X'].numpy()[:, :2, :])
    np.savez(os.path.join(output_dir, 'ctu_signal_scaler.npz'), mean=mean, std=std)
    print(f"  Channel means: FHR={mean[0]:.4f} UC={mean[1]:.4f} | "
          f"stds: FHR={std[0]:.4f} UC={std[1]:.4f}")

    for split_name in ('train', 'val', 'test'):
        p = os.path.join(output_dir, f'{split_name}_dataset.pt')
        if not os.path.exists(p):
            continue
        ds = torch.load(p, weights_only=False)
        arr = ds['X'].numpy()
        arr[:, :2, :], _, _ = zscore_normalize_channels(arr[:, :2, :], mean=mean, std=std)
        ds['X'] = torch.tensor(arr, dtype=torch.float32)
        torch.save(ds, p)
        print(f"  Normalised {split_name}_dataset.pt  shape={tuple(ds['X'].shape)}")

    tr = torch.load(train_path, weights_only=False)
    yf = tr['y_features'].numpy()
    np.savez(os.path.join(output_dir, 'feature_scaler.npz'),
             feature_means=yf.mean(axis=0),
             feature_stds=np.clip(yf.std(axis=0), 1e-6, None))

    if dropped_patients:
        import json
        with open(os.path.join(output_dir, 'dropped_patients.json'), 'w') as fh:
            json.dump([{'record_id': r, 'split': s, 'reason': why}
                       for r, s, why in dropped_patients], fh, indent=2)
        print(f"\n  {len(dropped_patients)} patients dropped -- recorded in "
              f"dropped_patients.json (pipeline_mil.py dropped these silently)")

    print("\nClinical pipeline complete.")
    print("NEXT (mandatory gate): python scripts/audit_time_confound.py "
          f"--data_dir {output_dir}")
    print("  AUROC(window position -> y_primary) must be ~0.50. Do not train otherwise.")


if __name__ == "__main__":
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    process_pipeline_clinical(
        raw_data_dir=os.path.join(BASE, 'data', 'raw', 'ctu-chb-intrapartum'),
        metadata_path=os.path.join(BASE, 'data', 'raw', 'ctu-chb-intrapartum',
                                   'clinical_metadata.csv'),
        output_dir=os.path.join(BASE, 'data', 'processed_clinical'),
    )
