"""
Phase 26, Problem 9 -- patient-level clinical explanation for the LOCKED URM.

Produces, per patient, a structured explanation tied to the clinically-recognisable variables already in the
locked pipeline (FHR baseline/STV/LTV/accelerations, deceleration burden and duration, the FIGO-style trajectory
state sequence, and the parity contribution) -- not raw SHAP values on a screen. Directionality is z-scored
against the canonical-fold TRAINING population only (never the patient's own fold), so the reference is honest.
"""
import os, sys, json
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from scripts.phase26_clinical_evaluation import build_urm_sequences
from src.models.phase16_causal_attention import eligible_prefix_length

OUT = "results/phase26_clinical_evaluation"; os.makedirs(OUT, exist_ok=True)
X40_PATH = "results/phase9c_state_trajectory/state_trajectory_features.npz"
X40_NAMES_PATH = "results/phase9c_state_trajectory/state_trajectory_feature_names.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
STATE_LABELS = {0: "State 0 (reassuring)", 1: "State 1 (mildly abnormal)", 2: "State 2 (suspicious)",
               3: "State 3 (pathological)", 4: "State 4 (severely abnormal)"}
CLINICAL_LABELS = {"raw_0": "Baseline FHR", "raw_1": "Short-term variability (STV)", "raw_2": "Long-term variability (LTV)",
                   "raw_3": "Accelerations", "raw_4": "Early decelerations", "raw_5": "Late decelerations",
                   "raw_6": "Variable decelerations", "raw_7": "Prolonged decelerations", "raw_8": "Deceleration depth",
                   "raw_9": "Deceleration area", "raw_10": "Deceleration burden", "raw_11": "Longest deceleration",
                   "dom_sev_baseline": "Baseline severity (domain score)", "dom_sev_variability": "Variability severity (domain score)",
                   "dom_sev_deceleration": "Deceleration severity (domain score)", "dom_sev_uterine": "Uterine-activity severity (domain score)"}
# direction where HIGHER is worse (used for the up/down clinical arrow); False = lower is worse
HIGHER_IS_WORSE = {"raw_0": None, "raw_1": False, "raw_2": False, "raw_3": False, "raw_4": None, "raw_5": True,
                   "raw_6": True, "raw_7": True, "raw_8": True, "raw_9": True, "raw_10": True, "raw_11": True,
                   "dom_sev_baseline": True, "dom_sev_variability": True, "dom_sev_deceleration": True, "dom_sev_uterine": True}


def zscore_direction(val, mu, sd, higher_worse):
    z = (val - mu) / max(sd, 1e-9)
    if higher_worse is None:
        arrow = "->" if abs(z) < 0.5 else ("UP" if z > 0 else "DOWN")
    elif higher_worse:
        arrow = "UP (worse)" if z > 0.5 else ("DOWN (better)" if z < -0.5 else "-> (near normal)")
    else:
        arrow = "DOWN (worse)" if z < -0.5 else ("UP (better)" if z > 0.5 else "-> (near normal)")
    return round(float(z), 2), arrow


def main():
    d = build_urm_sequences()
    clean_pids, fold_of, fused_seq, t_del_cv, y_lookup = d["clean_pids"], d["fold_of"], d["fused_seq_cv"], d["t_del_cv"], d["y_lookup"]
    X40 = np.load(X40_PATH)["X_state_trajectory"]; names = json.load(open(X40_NAMES_PATH))
    idx_of = {n: i for i, n in enumerate(names)}
    df = pd.read_csv(ROLLING_PATH); df["patient_id"] = df["patient_id"].astype(str)
    ss = df["start_sample"].values; pid_w = df["patient_id"].values
    win_idx = {p: np.where(pid_w == p)[0] for p in clean_pids}

    op = json.load(open(os.path.join(OUT, "operating_point_80pct.json")))
    thr = op["mean_threshold"]

    # training-population mean/std for z-scores, per canonical fold (train = all patients NOT in that fold)
    train_stats = {}
    for f in range(5):
        tr_idx = np.concatenate([win_idx[p] for p in clean_pids if fold_of[p] != f])
        train_stats[f] = {n: (float(X40[tr_idx, idx_of[n]].mean()), float(X40[tr_idx, idx_of[n]].std())) for n in CLINICAL_LABELS}

    def explain(pid):
        f = fold_of[pid]; idx = win_idx[pid]; o = idx[np.argsort(ss[idx])]; last = o[-1]
        s = fused_seq[pid]; risk = float(s[-1]); y = y_lookup[pid]
        k20 = eligible_prefix_length(t_del_cv[pid], 20, len(s)) - 1
        warned_ge20 = bool(s[k20] >= thr)
        feats = {}
        for n, label in CLINICAL_LABELS.items():
            mu, sd = train_stats[f][n]; val = float(X40[last, idx_of[n]])
            z, arrow = zscore_direction(val, mu, sd, HIGHER_IS_WORSE[n])
            feats[label] = {"value": round(val, 2), "population_mean": round(mu, 2), "z": z, "direction": arrow}
        states = [STATE_LABELS.get(int(X40[i, idx_of["current_state"]]), "?") for i in o]
        traj = [states[0]] if states else []
        for st in states[1:]:
            if st != traj[-1]: traj.append(st)
        return {"patient_id": pid, "outcome_pH_le_715": bool(y), "risk_score": round(risk, 3),
               "risk_band": "HIGH" if risk >= thr else ("BORDERLINE" if risk >= thr * 0.7 else "LOW"),
               "operating_threshold": round(thr, 3), "warned_20min_before_delivery": warned_ge20,
               "fhr_and_uterine_features": feats, "trajectory_state_sequence": traj,
               "parity_raw": None}   # filled below

    meta = pd.read_csv("data/raw/ctu-chb-intrapartum/clinical_metadata.csv").set_index("record_id")
    for pid in clean_pids:
        pass  # parity filled per-patient below

    # pick 3 illustrative patients: a clear true positive, a clear true negative, and a borderline/missed case
    y = np.array([y_lookup[p] for p in clean_pids]); delivery = np.array([fused_seq[p][-1] for p in clean_pids])
    order_pos = [p for p, yy, s in zip(clean_pids, y, delivery) if yy == 1]
    order_neg = [p for p, yy, s in zip(clean_pids, y, delivery) if yy == 0]
    tp_candidates = sorted(order_pos, key=lambda p: -fused_seq[p][-1])
    tn_candidates = sorted(order_neg, key=lambda p: fused_seq[p][-1])
    fn_candidates = sorted(order_pos, key=lambda p: fused_seq[p][-1])
    picks = {"clear_true_positive": tp_candidates[2], "clear_true_negative": tn_candidates[2], "missed_case_false_negative": fn_candidates[2]}

    out = {}
    for tag, pid in picks.items():
        rec = explain(pid); rec["parity_raw"] = float(meta.loc[int(pid), "parity"])
        out[tag] = rec
        print(f"\n=== {tag}: patient {pid} (pH<=7.15 = {rec['outcome_pH_le_715']}) ===")
        print(f"  risk={rec['risk_score']} band={rec['risk_band']} threshold={rec['operating_threshold']} warned>=20min={rec['warned_20min_before_delivery']}")
        for k, v in rec["fhr_and_uterine_features"].items():
            print(f"    {k:<32} {v['value']:>7}  (pop mean {v['population_mean']:>6}, z={v['z']:+.2f})  {v['direction']}")
        print("    Trajectory:", " -> ".join(rec["trajectory_state_sequence"]))
        print("    Parity:", rec["parity_raw"])

    json.dump(out, open(os.path.join(OUT, "explainability_examples.json"), "w"), indent=2)
    print(f"\nSaved -> {OUT}/explainability_examples.json")


if __name__ == "__main__":
    main()
