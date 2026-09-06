"""
Acceptance gate for a preprocessed dataset: can the label be predicted from a
window's POSITION IN THE RECORDING, with no access to the signal?

On data/processed_mil/ this audit fails badly -- time alone reaches AUROC 0.84
on y_primary, beating every trained model (see docs/auroc_ceiling_analysis.md).
That is the horizon rule leaking the label. A dataset whose window label is a
patient-level outcome should score ~0.50 here, because position within a
recording carries no information about the patient's outcome.

Run this BEFORE training on any new preprocessed dataset.

    python scripts/audit_time_confound.py --data_dir data/processed_clinical

PASS  time-alone AUROC within [0.45, 0.55] on every split
WARN  within [0.40, 0.60]
FAIL  outside that -- the label is contaminated by time; do not train.
"""
import argparse
import os
import sys

import numpy as np
import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def auprc(y, s):
    pr, rc, _ = precision_recall_curve(y, s)
    return auc(rc, pr)


def relative_position(pid, st):
    """0 = earliest window of that patient, 1 = latest."""
    rel = np.zeros(len(st), dtype=float)
    for q in set(pid):
        m = pid == q
        s = st[m]
        rel[m] = (s - s.min()) / max(s.max() - s.min(), 1)
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_clinical")
    ap.add_argument("--targets", nargs="*", default=["y_primary", "y_adverse"])
    a = ap.parse_args()

    print("=" * 76)
    print(f" TIME-CONFOUND AUDIT -- {a.data_dir}")
    print("=" * 76)
    print("  A window's position must NOT predict its label. ~0.50 = clean.\n")

    worst = 0.5
    any_split = False
    for split in ("train", "val", "test"):
        p = os.path.join(a.data_dir, f"{split}_dataset.pt")
        if not os.path.exists(p):
            continue
        any_split = True
        d = torch.load(p, map_location="cpu", weights_only=False)
        meta = d["metadata"]
        pid = np.array([m[0] for m in meta])
        st = np.array([m[1] for m in meta])
        rel = relative_position(pid, st)

        print(f"  --- {split} ({len(st)} windows, {len(set(pid))} patients) ---")
        for t in a.targets:
            if t not in d:
                continue
            y = d[t].numpy()
            if len(set(y.tolist())) < 2:
                print(f"    {t:<12} single-class, skipped")
                continue
            au = roc_auc_score(y, rel)
            worst = max(worst, abs(au - 0.5) + 0.5)
            flag = "PASS" if abs(au - 0.5) <= 0.05 else ("WARN" if abs(au - 0.5) <= 0.10 else "FAIL")
            print(f"    {t:<12} positives {int(y.sum()):>5}/{len(y):<5} "
                  f"({y.mean()*100:>4.1f}%)  time-alone AUROC {au:.4f}  AUPRC {auprc(y,rel):.4f}   [{flag}]")
        print()

    if not any_split:
        sys.exit(f"[ABORT] no *_dataset.pt found in {a.data_dir}")

    print("=" * 76)
    if worst <= 0.55:
        print(f" RESULT: PASS -- worst deviation {worst:.4f}. The time confound is gone.")
        rc = 0
    elif worst <= 0.60:
        print(f" RESULT: WARN -- worst {worst:.4f}. Residual time signal; inspect before training.")
        rc = 0
    else:
        print(f" RESULT: FAIL -- worst {worst:.4f}. The label is still contaminated by time.")
        print(" Do NOT train on this dataset. See docs/auroc_ceiling_analysis.md.")
        rc = 1
    print("=" * 76)
    sys.exit(rc)


if __name__ == "__main__":
    main()
