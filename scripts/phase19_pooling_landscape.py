"""Phase 19 -- ORACLE landscape (descriptive only, NOT a result): CV AUROC of the recency x risk pooling family over a (lam, kap) grid at each horizon. Evaluated on the same folds it is plotted on, so no point on it may be reported as a performance claim; it only shows where good settings live and how horizon-dependent they are."""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scripts.phase19_parametric_pooling import *
with open(FOLDS_PATH) as fh: fb = json.load(fh)
pids = sorted(fb["assignment"].keys())
df = pd.read_csv(ROLLING_PATH); df["patient_id"] = df["patient_id"].astype(str)
y_lookup = {p: int(df[df["patient_id"] == p]["primary_label_715"].iloc[0]) for p in pids}
y = np.array([y_lookup[p] for p in pids])
p6 = np.load(P6_PRED_PATH, allow_pickle=True)
pd_, tdel = build_patient_data(p6["pred_unweighted_cv"], p6["patient_ids"], df, pids, y_lookup)
R, TAU, L, Y = pad_patients(pids, pd_)
ks = {h: np.array([eligible_prefix_length(tdel[p], h, pd_[p][3]) - 1 for p in pids]) for h in HORIZONS}
lams = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0]
kaps = [0, 1, 2, 4, 8, 16]
for h in HORIZONS:
    print(f"\n[oracle landscape, CV AUROC at h={h}m]  rows=lam(/min), cols=kap {kaps}")
    for lam in lams:
        row = []
        for kap in kaps:
            z = all_prefix_scores(R, TAU, L, lam, kap)
            s = z[np.arange(len(pids)), ks[h]]
            row.append(roc_auc_score(y, s))
        print(f"lam={lam:<5} " + " ".join(f"{v:.4f}" for v in row))
