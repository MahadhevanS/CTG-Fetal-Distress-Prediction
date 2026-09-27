"""
Phase 19 Step 3 -- where in labor is the window-level signal strongest, and how noisy is it?

Window-level AUROC of the frozen PRS (P6) score, split by how many minutes
before delivery the window ends (each patient contributes at most one window
per 2.5-min bin), plus the mean window-to-window jump |r_t - r_(t-1)| as a
noise proxy. Explains why smoothing over recent windows (TAM) helps at the
delivery-time query but not at earlier lead times.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

p6 = np.load("results/phase13/audit/p6_predictions.npz", allow_pickle=True)
d = pd.DataFrame({"pid": p6["patient_ids"].astype(str), "t": p6["t_del"], "r": p6["pred_unweighted_cv"], "y": p6["y_715"]})
d["tb"] = (d["t"] / 2.5).round() * 2.5
d = d.sort_values(["pid", "t"], ascending=[True, False])
d["dr"] = d.groupby("pid")["r"].diff().abs()
rows = []
for tb in sorted(d.tb.unique()):
    s = d[d.tb == tb]
    if tb > 40 or s.y.nunique() < 2 or len(s) < 100:
        continue
    rows.append({"minutes_before_delivery": tb, "n_patients": len(s), "n_positive": int(s.y.sum()),
                 "window_auroc": round(roc_auc_score(s.y, s.r), 4), "sd_score": round(s.r.std(), 4),
                 "mean_abs_window_to_window_jump": round(s.dr.mean(), 4)})
out = pd.DataFrame(rows)
out.to_csv("results/phase19_attention/information_profile_by_lead_time.csv", index=False)
print(out.to_string(index=False))
