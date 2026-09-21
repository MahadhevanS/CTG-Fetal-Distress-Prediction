"""
Build Clinician Review Packet
==============================
Selects a stratified, patient-diverse sample of CTG windows from the corrected
(post-repair) dataset, renders each as a clean, unlabeled CTG-style trace
(FHR + UC panels, standard grid, no classification info), and writes:

    docs/clinician_review/case_images/case_C##.png   -- blinded trace images
    docs/clinician_review/REVEAL_SHEET.md            -- researcher-only answer key
                                                          (pipeline outputs per case)
    docs/clinician_review/case_manifest.json         -- machine-readable manifest

IMPORTANT: REVEAL_SHEET.md and case_manifest.json contain the automated
pipeline's classification for every case and must NOT be shown to the
reviewing clinician before they submit their independent (blinded) read.
Only the images in case_images/ and the clinician-facing packet (built
separately) are safe to share prior to that.

Usage:
    python scripts/build_clinician_review_packet.py
"""

import json
import os
import random
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR = os.path.join(BASE_DIR, "docs", "clinician_review")
IMG_DIR = os.path.join(OUT_DIR, "case_images")
FS = 4.0
SEED = 42

PER_CLASS = 8          # cases per FIGO class (0=Normal, 1=Suspicious, 2=Pathological)
N_BORDERLINE = 6        # extra cases near FIGO decision thresholds
FIGO_NAMES = {0: "Normal", 1: "Suspicious", 2: "Pathological"}
FEATURE_NAMES = ["Baseline (bpm)", "STV (bpm)", "LTV (bpm)", "Accel Count",
                  "Early Decels", "Late Decels", "Variable Decels", "Prolonged Decels"]


FHR_GAP_FLOOR_BPM = 40.0  # below the 50 bpm interpolation clamp -- only unfilled
                            # long (>15s) missing-data gaps can land under this


def reconstruct_absolute_signals(X: np.ndarray, y_features: np.ndarray,
                                  mean: np.ndarray, std: np.ndarray):
    """Exactly inverts Z-score normalization and re-adds baseline to recover
    absolute-unit FHR (bpm) and UC signals for clinical-style plotting.

    Gaps longer than 15s are left as literal 0.0 by the production pipeline
    (interpolate_missing only fills shorter gaps) -- reconstructed naively
    this shows up as FHR plunging to ~0 bpm, which reads as a terminal
    deceleration to a clinical reviewer instead of what it actually is
    (signal loss / sensor dropout). Masked to NaN here so the plotted line
    breaks cleanly, matching how a real CTG monitor displays signal loss.
    """
    raw_ch0 = X[:, 0, :] * std[0] + mean[0]           # baseline-corrected FHR (bpm)
    raw_ch1 = X[:, 1, :] * std[1] + mean[1]            # UC (raw units)
    baseline = y_features[:, 0:1]                      # (N, 1)
    abs_fhr = raw_ch0 + baseline                        # (N, 4800), absolute bpm
    abs_fhr = np.where(abs_fhr < FHR_GAP_FLOOR_BPM, np.nan, abs_fhr)
    return abs_fhr, raw_ch1


def _shade_missing_regions(ax, t: np.ndarray, is_missing: np.ndarray) -> None:
    """Draws an explicit light-gray band over each contiguous missing-data
    run, so signal loss reads unambiguously as 'no data' rather than being
    confused with the axes' own bottom border or a real flat trace."""
    if not is_missing.any():
        return
    changes = np.diff(is_missing.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if is_missing[0]:
        starts = np.insert(starts, 0, 0)
    if is_missing[-1]:
        ends = np.append(ends, len(is_missing))
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[min(e, len(t) - 1)], facecolor="#d9d9d9", alpha=0.6,
                   zorder=0, hatch="///", edgecolor="#bbbbbb", linewidth=0)


def render_case(case_id: str, fhr: np.ndarray, uc: np.ndarray, out_path: str) -> None:
    """Renders a blinded, CTG-paper-style two-panel trace: no classification
    info, no shaded reference bands -- just the grid a clinician would expect.
    Signal-loss gaps are explicitly shaded (see _shade_missing_regions) so
    they cannot be misread as a real flat/terminal trace."""
    t = np.arange(len(fhr)) / (FS * 60.0)  # minutes
    is_missing = np.isnan(fhr)

    fig, (ax_fhr, ax_uc) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    ax_fhr.set_ylim(50, 210)
    _shade_missing_regions(ax_fhr, t, is_missing)
    ax_fhr.plot(t, fhr, color="#1a1a1a", linewidth=0.7, zorder=2)
    ax_fhr.yaxis.set_major_locator(MultipleLocator(20))
    ax_fhr.yaxis.set_minor_locator(MultipleLocator(10))
    ax_fhr.set_ylabel("FHR (bpm)")
    ax_fhr.set_title(f"Case {case_id}", fontsize=12, loc="left")
    ax_fhr.grid(which="major", color="#b0b0b0", linewidth=0.8)
    ax_fhr.grid(which="minor", color="#e0e0e0", linewidth=0.4)

    ax_uc.set_ylim(min(0, np.nanmin(uc) - 5), max(100, np.nanmax(uc) + 5))
    _shade_missing_regions(ax_uc, t, is_missing)
    ax_uc.plot(t, uc, color="#1a1a1a", linewidth=0.7, zorder=2)
    ax_uc.set_ylabel("UC")
    ax_uc.set_xlabel("Time (minutes)")
    ax_uc.xaxis.set_major_locator(MultipleLocator(5))
    ax_uc.xaxis.set_minor_locator(MultipleLocator(1))
    ax_uc.set_xlim(0, 20)
    ax_uc.grid(which="major", color="#b0b0b0", linewidth=0.8)
    ax_uc.grid(which="minor", color="#e0e0e0", linewidth=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


MAX_GAP_RUNS = 3  # exclude windows with more than this many distinct signal-loss
                    # runs from the review pool -- a few clean, clearly-shaded gaps
                    # read fine; many scattered short gaps (plus any residual spline
                    # ringing between them) produce a visually noisy trace that would
                    # confuse a clinical reviewer without testing anything useful.


def count_gap_runs(missing_row: np.ndarray) -> int:
    """Number of contiguous True runs in a 1D boolean missing-data mask."""
    d = np.diff(missing_row.astype(int))
    return int((d == 1).sum() + (1 if missing_row[0] else 0))


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    rng = random.Random(SEED)

    scaler = np.load(os.path.join(DATA_DIR, "ctu_signal_scaler.npz"))
    mean, std = scaler["mean"], scaler["std"]

    ds = torch.load(os.path.join(DATA_DIR, "train_dataset.pt"), weights_only=False)
    X = ds["X"].numpy()
    y_primary = ds["y_primary"].numpy()
    y_figo = ds["y_figo"].numpy()
    y_features = ds["y_features"].numpy()
    record_ids = np.array([m[0] for m in ds["metadata"]])
    window_bounds = [(m[1], m[2]) for m in ds["metadata"]]

    abs_fhr_all, abs_uc_all = reconstruct_absolute_signals(X, y_features, mean, std)

    # --- Signal-quality gate: cap on number of distinct missing-data runs ---
    is_missing_all = np.isnan(abs_fhr_all)
    n_gap_runs = np.array([count_gap_runs(is_missing_all[i]) for i in range(len(is_missing_all))])
    quality_ok = n_gap_runs <= MAX_GAP_RUNS
    print(f"[Quality gate] {quality_ok.sum()}/{len(quality_ok)} windows have "
          f"<= {MAX_GAP_RUNS} signal-loss runs and are eligible for selection.")

    selected_idx = []
    used_patients = set()

    # --- Stratified selection: PER_CLASS cases per FIGO class, one per patient ---
    for cls in [0, 1, 2]:
        cls_idx = np.where((y_figo == cls) & quality_ok)[0].tolist()
        rng.shuffle(cls_idx)
        count = 0
        for i in cls_idx:
            pid = record_ids[i]
            if pid in used_patients:
                continue
            selected_idx.append(i)
            used_patients.add(pid)
            count += 1
            if count >= PER_CLASS:
                break

    # --- Borderline selection: windows near FIGO decision thresholds ---
    baseline = y_features[:, 0]
    ltv = y_features[:, 2]
    dist_to_threshold = np.minimum.reduce([
        np.abs(baseline - 110), np.abs(baseline - 160),
        np.abs(ltv - 5), np.abs(ltv - 25),
    ])
    dist_to_threshold = np.where(quality_ok, dist_to_threshold, np.inf)
    border_order = np.argsort(dist_to_threshold)
    count = 0
    for i in border_order:
        pid = record_ids[i]
        if pid in used_patients:
            continue
        selected_idx.append(int(i))
        used_patients.add(pid)
        count += 1
        if count >= N_BORDERLINE:
            break

    # --- Shuffle final case order so class grouping isn't inferable from ID order ---
    rng.shuffle(selected_idx)

    manifest = []
    reveal_lines = [
        "# REVEAL SHEET -- RESEARCHER ONLY",
        "",
        "**Do not share this file (or its contents) with the reviewing clinician",
        "before they submit their independent, blinded read for every case.**",
        "",
        "| Case | Record ID | Window (start-end, samples) | FIGO Class (pipeline) | "
        "Distress Label (y_primary) | Baseline | STV | LTV | Accels | Early | Late | "
        "Variable | Prolonged |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | "
        ":--- | :--- | :--- |",
    ]

    for case_num, idx in enumerate(selected_idx, 1):
        case_id = f"C{case_num:02d}"
        fhr = abs_fhr_all[idx]
        uc = abs_uc_all[idx]
        img_path = os.path.join(IMG_DIR, f"case_{case_id}.png")
        render_case(case_id, fhr, uc, img_path)

        feat = y_features[idx]
        start, end = window_bounds[idx]
        manifest.append({
            "case_id": case_id,
            "record_id": str(record_ids[idx]),
            "window_start": int(start),
            "window_end": int(end),
            "figo_class": int(y_figo[idx]),
            "figo_name": FIGO_NAMES[int(y_figo[idx])],
            "y_primary": int(y_primary[idx]),
            "features": {name: float(val) for name, val in zip(FEATURE_NAMES, feat)},
            "image": os.path.relpath(img_path, OUT_DIR),
        })

        reveal_lines.append(
            f"| {case_id} | {record_ids[idx]} | {start}-{end} | "
            f"{FIGO_NAMES[int(y_figo[idx])]} | {int(y_primary[idx])} | "
            f"{feat[0]:.1f} | {feat[1]:.2f} | {feat[2]:.1f} | {feat[3]:.0f} | "
            f"{feat[4]:.0f} | {feat[5]:.0f} | {feat[6]:.0f} | {feat[7]:.0f} |"
        )

    with open(os.path.join(OUT_DIR, "case_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(OUT_DIR, "REVEAL_SHEET.md"), "w") as f:
        f.write("\n".join(reveal_lines) + "\n")

    class_counts = {FIGO_NAMES[c]: int((y_figo[selected_idx] == c).sum()) for c in [0, 1, 2]}
    print(f"[Done] {len(selected_idx)} cases selected ({len(used_patients)} distinct patients).")
    print(f"       Class breakdown: {class_counts}")
    print(f"       Images  -> {IMG_DIR}")
    print(f"       Manifest -> {os.path.join(OUT_DIR, 'case_manifest.json')}")
    print(f"       Reveal sheet (researcher only) -> {os.path.join(OUT_DIR, 'REVEAL_SHEET.md')}")


if __name__ == "__main__":
    main()
