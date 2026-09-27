"""
Phase 19 Step 1 -- what did Model 3 (TAM)'s attention MLP actually learn?

Exploratory, no new training: loads the five frozen fold checkpoints and
inspects the learned scoring function e(r, elapsed) and the attention it
induces on real patient sequences. Questions:
  1. Is e monotone in the window risk r? (i.e. is TAM a learned "soft-max"
     pooling that interpolates between mean and max?) What is its effective
     temperature (de/dr)?
  2. Does elapsed time change the weight (recency / early-window bias)?
  3. How concentrated is the attention (effective number of windows)?
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.phase16_causal_attention import score_full_sequence, pooled_prediction_from_logits
from src.models.phase16_checkpoint_utils import load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
CHECKPOINT_DIR = "results/phase16/checkpoints"
OUT_DIR = "results/phase19_attention"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    fold_of = {p: folds_blob["assignment"][p][0] for p in clean_pids}

    df = pd.read_csv(ROLLING_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    y_lookup = {p: int(df[df["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}
    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_data, t_del = build_patient_data(p6["pred_unweighted_cv"], p6["patient_ids"], df, clean_pids, y_lookup)

    completed = load_completed_units(CHECKPOINT_DIR)
    scorers = {f: load_scorer_checkpoint(completed[f"Model_3_magnitude_position_fold{f}"], in_dim=2, hidden=8) for f in range(5)}

    all_r = np.concatenate([patient_data[p][0].numpy() for p in clean_pids])
    print(f"P6 window-score range: min={all_r.min():.3f} p5={np.percentile(all_r,5):.3f} median={np.median(all_r):.3f} "
          f"p95={np.percentile(all_r,95):.3f} max={all_r.max():.3f}")

    # ---------- 1/2: shape of e(r, elapsed) per fold ----------
    r_grid = np.linspace(np.percentile(all_r, 2), np.percentile(all_r, 98), 25)
    el_grid_min = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    rows = []
    for f, sc in scorers.items():
        sc.eval()
        with torch.no_grad():
            X = torch.tensor([[r, el / 60.0] for el in el_grid_min for r in r_grid], dtype=torch.float32)
            e = sc(X).numpy().reshape(len(el_grid_min), len(r_grid))
        # de/dr (mean slope over r-range, at each elapsed) and de/d(elapsed) (mean over r)
        slope_r = [(e[i, -1] - e[i, 0]) / (r_grid[-1] - r_grid[0]) for i in range(len(el_grid_min))]
        slope_el = (e[-1, :] - e[0, :]).mean() / (el_grid_min[-1] - el_grid_min[0])
        # monotonicity in r at each elapsed
        mono = [bool(np.all(np.diff(e[i]) > 0)) for i in range(len(el_grid_min))]
        rows.append({"fold": f, "de_dr_at_el0": round(slope_r[0], 2), "de_dr_at_el20": round(slope_r[2], 2),
                     "de_dr_at_el40": round(slope_r[4], 2), "de_per_min_elapsed": round(float(slope_el), 4),
                     "monotone_increasing_in_r_all_elapsed": all(mono),
                     "e_range_over_r_at_el20": round(float(e[2].max() - e[2].min()), 3)})
    df_shape = pd.DataFrame(rows)
    df_shape.to_csv(os.path.join(OUT_DIR, "attention_function_shape.csv"), index=False)
    print("\n--- Learned scoring function e(r, elapsed): shape per fold ---")
    print(df_shape.to_string(index=False))

    # ---------- 3: induced attention on real sequences (each patient scored by own fold's scorer) ----------
    eff_n, alpha_r_corr, alpha_recency_corr, top1_mass, top1_is_max, T_list, y_list = [], [], [], [], [], [], []
    from scipy import stats
    for pid in clean_pids:
        r_seq, el_seq, y, T = patient_data[pid]
        sc = scorers[fold_of[pid]]
        with torch.no_grad():
            logits = score_full_sequence(sc, r_seq, el_seq, True)
            _, alpha = pooled_prediction_from_logits(logits, r_seq, T)
        a = alpha.numpy()
        eff_n.append(1.0 / np.sum(a ** 2))
        top1_mass.append(a.max())
        top1_is_max.append(int(np.argmax(a) == np.argmax(r_seq.numpy())))
        T_list.append(T); y_list.append(int(y.item()))
        if T >= 4 and np.std(r_seq.numpy()) > 0:
            alpha_r_corr.append(stats.spearmanr(a, r_seq.numpy())[0])
            alpha_recency_corr.append(stats.spearmanr(a, np.arange(T))[0])
    eff_n = np.array(eff_n); T_arr = np.array(T_list)
    print("\n--- Induced attention on real patient sequences (CV, own-fold scorer) ---")
    print(f"  effective #windows attended (1/sum a^2): median={np.median(eff_n):.2f}  (of median T={np.median(T_arr):.0f})")
    print(f"  effective fraction of sequence: median={np.median(eff_n / T_arr):.3f}  (uniform mean-pooling = 1.0, pure max = {1/np.median(T_arr):.3f})")
    print(f"  top-1 window carries median {np.median(top1_mass):.3f} of the weight; is the max-risk window in {np.mean(top1_is_max):.1%} of patients")
    print(f"  within-patient Spearman(alpha, r):       median={np.median(alpha_r_corr):+.3f}")
    print(f"  within-patient Spearman(alpha, recency): median={np.median(alpha_recency_corr):+.3f}  (>0 = later windows weighted more)")
    pd.DataFrame([{
        "median_eff_windows": float(np.median(eff_n)), "median_T": float(np.median(T_arr)),
        "median_eff_fraction": float(np.median(eff_n / T_arr)), "median_top1_mass": float(np.median(top1_mass)),
        "frac_top1_is_max_window": float(np.mean(top1_is_max)),
        "median_spearman_alpha_r": float(np.median(alpha_r_corr)), "median_spearman_alpha_recency": float(np.median(alpha_recency_corr)),
    }]).to_csv(os.path.join(OUT_DIR, "attention_induced_summary.csv"), index=False)
    print(f"\nSaved -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
