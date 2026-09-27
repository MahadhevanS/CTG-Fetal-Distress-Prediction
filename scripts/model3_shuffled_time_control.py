"""
Tier-3 hardening item 5 (docs/pre_external_validation_hardening_plan.md #5)
-- the shuffled-time control.

WHY THIS EXISTS. Model 3's only evidence that it uses "something beyond
magnitude" is that its attention argmax agrees with the pure-magnitude
argmax in only 31.6% of patients (vs. Model 2's 100%). That is suggestive,
not dispositive: a model with an extra, uninformative input dimension can
show reduced argmax agreement purely from added noise, without genuinely
exploiting temporal order. This script is the actual control.

THE CONTROL. Retrain Model 3's identical architecture (same 2-input
AttentionScorer, same hidden=8, same train_scorer procedure and
hyperparameters, same canonical folds, same inner-validation-carving
seeds) with elapsed_t VALUES RANDOMLY PERMUTED WITHIN EACH PATIENT'S OWN
WINDOW SEQUENCE. This breaks the true correspondence between a window's
identity/chronological order and its elapsed-time value while preserving
the exact same per-patient value distribution, sequence length, and
parameter count as real Model 3. r_t and the causal eligible-prefix
determination (which depends on time-before-delivery, not elapsed_t) are
completely untouched -- only the (r_t, elapsed_t) pairing at each position
is scrambled. The permutation is fixed once per patient (not re-drawn
every epoch), deterministically seeded so the whole run is reproducible.

Three independent shuffle draws (seeds 777/888/999) are trained, not one --
a shuffle-draw-sensitivity check directly analogous to Model 3's own
fold-resplit-sensitivity verification (phase16_model3_verification.py),
run because a conclusion-relevant control deserves the same scrutiny this
project already applies to its headline results, not less.

THE DECISION RULE -- FIXED BEFORE ANY RESULT IS COMPUTED, NOT ADJUSTED
AFTER LOOKING (this is the whole point of a pre-registered control):
  - "Genuine temporal-order use" if shuffled-time Model 3 is SIGNIFICANTLY
    WORSE than real-time Model 3 (CV paired bootstrap p<0.05, delivery)
    AND shuffled-time Model 3 sits close to Model 2 (the zero-time-info
    endpoint), not close to real Model 3.
  - "Capacity artifact" if shuffled-time Model 3 performs COMPARABLY to
    real-time Model 3 (p>=0.05, similar magnitude, CI including zero) --
    the extra input dimension bought noise-driven flexibility, not signal.
  - Anything else (e.g. shuffled beats Model 2 but doesn't reach real
    Model 3, or the three seeds disagree) is reported as genuinely
    ambiguous, not forced into either bucket.
This rule is applied MECHANICALLY at the end of this script from the
computed numbers -- it is not a judgment call made after seeing results.

Checkpointed the same resumable pattern as phase16_model3_verification.py:
each of the (up to) 18 individual fold-trainings (3 seeds x 6 units) this
script runs is its own resumable unit; interrupting and re-running this
script skips whatever has already completed.

Reported regardless of outcome, per this project's standing rule against
running a control and discarding it if inconvenient.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.phase13_common import get_patient_scores_at_horizon_corrected
from src.models.phase16_causal_attention import train_scorer, predict_at_horizon_for_patients
from src.models.phase16_checkpoint_utils import get_or_train, load_completed_units, load_scorer_checkpoint
from scripts.phase16_temporal_attention_model import build_patient_data, carve_inner_validation
from scripts.phase11_bootstrap import paired_patient_bootstrap
from scripts.delong_test import delong_roc_test

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
P6_PRED_PATH = "results/phase13/audit/p6_predictions.npz"
DATA_DIR = "data/processed_clinical"
OUT_DIR = "results/phase16"
REAL_CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints")            # real Model 2 / Model 3 (already committed)
SHUFFLED_CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints_shuffled_time")  # new, this script's own units

HORIZONS = [0, 10, 20, 30]
SHUFFLE_SEEDS = [777, 888, 999]   # 777 = primary/headline seed; 888, 999 = sensitivity replicates
PRIMARY_SEED = 777
REAL_MODEL3_NAME = "Model_3_magnitude_position"
REAL_MODEL2_NAME = "Model_2_magnitude_only"


def build_shuffled_patient_data(patient_data, clean_pids_for_perm, shuffle_seed):
    """
    Returns a new patient_data dict where elapsed_seq is permuted WITHIN
    each patient's own sequence (r_seq, y, T_i unchanged). One fixed
    permutation per (shuffle_seed, patient_id), independent of dict
    iteration order or whether the patient appears in the CV or test data --
    so the same patient gets the identical permutation in both, exactly
    mirroring how the same patient's real elapsed_t values are identical
    across the CV and test data dicts (only the r_t source differs).
    """
    pid_order = sorted(clean_pids_for_perm)
    pid_index = {p: i for i, p in enumerate(pid_order)}
    shuffled = {}
    for pid, (r_seq, elapsed_seq, y, T_i) in patient_data.items():
        seed_for_pid = shuffle_seed * 100003 + pid_index[pid]
        rng = np.random.default_rng(seed_for_pid)
        perm = rng.permutation(T_i)
        elapsed_shuffled = elapsed_seq[torch.as_tensor(perm, dtype=torch.long)]
        shuffled[pid] = (r_seq, elapsed_shuffled, y, T_i)
    return shuffled


def train_shuffled_fold(checkpoint_dir, unit_id, inner_tr, inner_val, patient_data):
    return get_or_train(
        checkpoint_dir, unit_id, in_dim=2, hidden=8,
        train_fn=lambda: train_scorer(inner_tr, inner_val, patient_data, use_elapsed=True, seed=42),
    )


def run_shuffled_seed(shuffle_seed, clean_pids, folds_blob, train_val_pids, test_pids,
                       patient_data_cv, t_del_cv, patient_data_test, t_del_test, y_lookup):
    print(f"\n{'='*90}\n  SHUFFLED-TIME MODEL 3 -- shuffle_seed={shuffle_seed}\n{'='*90}")

    shuf_cv = build_shuffled_patient_data(patient_data_cv, clean_pids, shuffle_seed)
    shuf_test = build_shuffled_patient_data(patient_data_test, clean_pids, shuffle_seed)

    cv_preds = {h: {} for h in HORIZONS}
    pids_arr = np.array(clean_pids)
    for f_idx in range(5):
        te_pids = [p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx]
        tr_pids_all = [p for p in clean_pids if p not in te_pids]
        inner_tr, inner_val = carve_inner_validation(tr_pids_all, y_lookup, seed=42 + f_idx)

        unit_id = f"Model_3_shuffled_seed{shuffle_seed}_fold{f_idx}"
        scorer, val_loss, n_epochs, resumed = train_shuffled_fold(
            SHUFFLED_CHECKPOINT_DIR, unit_id, inner_tr, inner_val, shuf_cv)
        tag = "[resumed]" if resumed else "[trained]"
        print(f"  fold {f_idx} {tag}: {n_epochs} epochs, val_loss={val_loss:.4f}")

        for h in HORIZONS:
            out = predict_at_horizon_for_patients(scorer, te_pids, shuf_cv, t_del_cv, True, h)
            te_mask = np.isin(pids_arr, te_pids)
            for pid in pids_arr[te_mask]:
                cv_preds[h][pid] = out[pid][0]

    inner_tr_tv, inner_val_tv = carve_inner_validation(train_val_pids, y_lookup, seed=123)
    test_unit_id = f"Model_3_shuffled_seed{shuffle_seed}_testmodel"
    scorer_test, val_loss_test, n_epochs_test, resumed_test = train_shuffled_fold(
        SHUFFLED_CHECKPOINT_DIR, test_unit_id, inner_tr_tv, inner_val_tv, shuf_cv)
    tag = "[resumed]" if resumed_test else "[trained]"
    print(f"  test-model {tag}: {n_epochs_test} epochs, val_loss={val_loss_test:.4f}")

    test_preds = {h: {} for h in HORIZONS}
    for h in HORIZONS:
        out = predict_at_horizon_for_patients(scorer_test, test_pids, shuf_test, t_del_test, True, h)
        for pid in test_pids:
            test_preds[h][pid] = out[pid][0]

    return cv_preds, test_preds


def load_real_predictions(model_name, in_dim, clean_pids, folds_blob, test_pids,
                           patient_data_cv, t_del_cv, patient_data_test, t_del_test):
    """Pure inference on already-committed real Model 2 / Model 3 checkpoints
    -- identical pattern to scripts/model3_direct_comparisons.py."""
    completed = load_completed_units(REAL_CHECKPOINT_DIR)
    pids_arr = np.array(clean_pids)
    cv_preds = {h: np.zeros(len(clean_pids)) for h in HORIZONS}
    for f_idx in range(5):
        te_pids = [p for p in clean_pids if folds_blob["assignment"][p][0] == f_idx]
        rec = completed[f"{model_name}_fold{f_idx}"]
        scorer = load_scorer_checkpoint(rec, in_dim=in_dim, hidden=8)
        for h in HORIZONS:
            out = predict_at_horizon_for_patients(scorer, te_pids, patient_data_cv, t_del_cv, in_dim == 2, h)
            te_mask = np.isin(pids_arr, te_pids)
            cv_preds[h][te_mask] = [out[p][0] for p in pids_arr[te_mask]]

    rec_test = completed[f"{model_name}_testmodel"]
    scorer_test = load_scorer_checkpoint(rec_test, in_dim=in_dim, hidden=8)
    test_preds = {h: np.zeros(len(test_pids)) for h in HORIZONS}
    for h in HORIZONS:
        out = predict_at_horizon_for_patients(scorer_test, test_pids, patient_data_test, t_del_test, in_dim == 2, h)
        test_preds[h] = np.array([out[p][0] for p in test_pids])
    return cv_preds, test_preds


def paired_row(y_pat, base, cand, y_test, base_test, cand_test):
    boot_cv = paired_patient_bootstrap(y_pat, base, cand, n_boot=2000, seed=42)
    pdel_cv, _, _ = delong_roc_test(y_pat, cand, base)
    boot_test = paired_patient_bootstrap(y_test, base_test, cand_test, n_boot=2000, seed=42)
    pdel_test, _, _ = delong_roc_test(y_test, cand_test, base_test)
    return {
        "cv_auroc_cand": round(roc_auc_score(y_pat, cand), 4), "cv_auroc_base": round(roc_auc_score(y_pat, base), 4),
        "cv_delta": round(roc_auc_score(y_pat, cand) - roc_auc_score(y_pat, base), 4),
        "cv_boot_p": round(boot_cv["p_value"], 4), "cv_delong_p": round(pdel_cv, 4),
        "cv_ci_low": round(boot_cv["ci_95_low"], 4), "cv_ci_high": round(boot_cv["ci_95_high"], 4),
        "test_auroc_cand": round(roc_auc_score(y_test, cand_test), 4), "test_auroc_base": round(roc_auc_score(y_test, base_test), 4),
        "test_delta": round(roc_auc_score(y_test, cand_test) - roc_auc_score(y_test, base_test), 4),
        "test_boot_p": round(boot_test["p_value"], 4), "test_delong_p": round(pdel_test, 4),
        "test_ci_low": round(boot_test["ci_95_low"], 4), "test_ci_high": round(boot_test["ci_95_high"], 4),
    }


def main():
    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())

    p6 = np.load(P6_PRED_PATH, allow_pickle=True)
    patient_ids_arr = p6["patient_ids"]
    t_del_raw = p6["t_del"]
    pred_cv_window = p6["pred_unweighted_cv"]
    pred_test_window = p6["pred_unweighted_test"]

    df = pd.read_csv(ROLLING_PATH)
    df["patient_id"] = df["patient_id"].astype(str)
    y_lookup = {p: int(df[df["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}
    y_pat = np.array([y_lookup[p] for p in clean_pids])

    test_pt = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"), weights_only=False)
    test_pids = sorted(set(str(m[0]) for m in test_pt["metadata"]))
    train_val_pids = [p for p in clean_pids if p not in test_pids]
    y_test_pat = np.array([y_lookup[p] for p in test_pids])

    print("================================================================================")
    print("  MODEL 3 -- SHUFFLED-TIME CONTROL (Tier-3 hardening item #5)                      ")
    print("================================================================================")

    patient_data_cv, t_del_cv = build_patient_data(pred_cv_window, patient_ids_arr, df, clean_pids, y_lookup)
    patient_data_test, t_del_test = build_patient_data(pred_test_window, patient_ids_arr, df, test_pids, y_lookup)

    sw_cv_by_h = {h: get_patient_scores_at_horizon_corrected(pred_cv_window, patient_ids_arr, clean_pids, df["time_before_delivery_min"].values, h) for h in HORIZONS}
    sw_test_by_h = {h: get_patient_scores_at_horizon_corrected(pred_test_window, patient_ids_arr, test_pids, df["time_before_delivery_min"].values, h) for h in HORIZONS}

    # ---------------- train/reload shuffled-time Model 3, all 3 seeds ----------------
    shuffled_results_by_seed = {}
    for seed in SHUFFLE_SEEDS:
        cv_preds, test_preds = run_shuffled_seed(seed, clean_pids, folds_blob, train_val_pids, test_pids,
                                                  patient_data_cv, t_del_cv, patient_data_test, t_del_test, y_lookup)
        rows = []
        for h in HORIZONS:
            cv_arr = np.array([cv_preds[h][p] for p in clean_pids])
            test_arr = np.array([test_preds[h][p] for p in test_pids])
            row = paired_row(y_pat, sw_cv_by_h[h], cv_arr, y_test_pat, sw_test_by_h[h], test_arr)
            row["horizon_min"] = h
            row["cv_auprc"] = round(average_precision_score(y_pat, cv_arr), 4)
            rows.append(row)
            print(f"  seed={seed} h={h:>3}m  CV: sw={row['cv_auroc_base']:.4f} shuffled={row['cv_auroc_cand']:.4f} "
                  f"(d{row['cv_delta']:+.4f} p={row['cv_boot_p']:.3f})  TEST: sw={row['test_auroc_base']:.4f} "
                  f"shuffled={row['test_auroc_cand']:.4f} (d{row['test_delta']:+.4f} p={row['test_boot_p']:.3f})")
        shuffled_results_by_seed[seed] = {"rows": rows, "cv_preds": cv_preds, "test_preds": test_preds}
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"Model_3_shuffled_time_seed{seed}_auroc_results.csv"), index=False)

    # ---------------- reload real Model 3 and Model 2 (pure inference, already committed) ----------------
    print("\nReloading real Model 3 and Model 2 checkpoints for direct comparison (pure inference)...")
    real_m3_cv, real_m3_test = load_real_predictions(REAL_MODEL3_NAME, 2, clean_pids, folds_blob, test_pids,
                                                      patient_data_cv, t_del_cv, patient_data_test, t_del_test)
    real_m2_cv, real_m2_test = load_real_predictions(REAL_MODEL2_NAME, 1, clean_pids, folds_blob, test_pids,
                                                      patient_data_cv, t_del_cv, patient_data_test, t_del_test)

    # ---------------- primary-seed direct comparisons: shuffled vs real Model 3, shuffled vs Model 2 ----------------
    primary_cv_preds = shuffled_results_by_seed[PRIMARY_SEED]["cv_preds"]
    primary_test_preds = shuffled_results_by_seed[PRIMARY_SEED]["test_preds"]

    ablation_rows = []
    for h in HORIZONS:
        shuf_cv_arr = np.array([primary_cv_preds[h][p] for p in clean_pids])
        shuf_test_arr = np.array([primary_test_preds[h][p] for p in test_pids])

        vs_real = paired_row(y_pat, real_m3_cv[h], shuf_cv_arr, y_test_pat, real_m3_test[h], shuf_test_arr)
        vs_model2 = paired_row(y_pat, real_m2_cv[h], shuf_cv_arr, y_test_pat, real_m2_test[h], shuf_test_arr)

        row = {
            "horizon_min": h,
            "auroc_model2_cv": round(roc_auc_score(y_pat, real_m2_cv[h]), 4),
            "auroc_shuffled_model3_cv": round(roc_auc_score(y_pat, shuf_cv_arr), 4),
            "auroc_real_model3_cv": round(roc_auc_score(y_pat, real_m3_cv[h]), 4),
            "auroc_model2_test": round(roc_auc_score(y_test_pat, real_m2_test[h]), 4),
            "auroc_shuffled_model3_test": round(roc_auc_score(y_test_pat, shuf_test_arr), 4),
            "auroc_real_model3_test": round(roc_auc_score(y_test_pat, real_m3_test[h]), 4),
            "shuffled_vs_real_cv_delta": round(vs_real["cv_delta"], 4), "shuffled_vs_real_cv_p": vs_real["cv_boot_p"],
            "shuffled_vs_real_cv_ci_low": vs_real["cv_ci_low"], "shuffled_vs_real_cv_ci_high": vs_real["cv_ci_high"],
            "shuffled_vs_real_test_delta": round(vs_real["test_delta"], 4), "shuffled_vs_real_test_p": vs_real["test_boot_p"],
            "shuffled_vs_model2_cv_delta": round(vs_model2["cv_delta"], 4), "shuffled_vs_model2_cv_p": vs_model2["cv_boot_p"],
            "shuffled_vs_model2_cv_ci_low": vs_model2["cv_ci_low"], "shuffled_vs_model2_cv_ci_high": vs_model2["cv_ci_high"],
            "shuffled_vs_model2_test_delta": round(vs_model2["test_delta"], 4), "shuffled_vs_model2_test_p": vs_model2["test_boot_p"],
        }
        ablation_rows.append(row)
        print(f"\n--- Ablation ladder, h={h}m (seed={PRIMARY_SEED}) ---")
        print(f"  Model2(zero-time)={row['auroc_model2_cv']:.4f}  ShuffledM3(broken-time)={row['auroc_shuffled_model3_cv']:.4f}  "
              f"RealM3(true-time)={row['auroc_real_model3_cv']:.4f}   [CV]")
        print(f"  Shuffled vs Real:  CV delta={row['shuffled_vs_real_cv_delta']:+.4f} p={row['shuffled_vs_real_cv_p']:.3f} "
              f"CI=[{row['shuffled_vs_real_cv_ci_low']:+.4f},{row['shuffled_vs_real_cv_ci_high']:+.4f}]   "
              f"TEST delta={row['shuffled_vs_real_test_delta']:+.4f} p={row['shuffled_vs_real_test_p']:.3f}")
        print(f"  Shuffled vs Model2: CV delta={row['shuffled_vs_model2_cv_delta']:+.4f} p={row['shuffled_vs_model2_cv_p']:.3f} "
              f"CI=[{row['shuffled_vs_model2_cv_ci_low']:+.4f},{row['shuffled_vs_model2_cv_ci_high']:+.4f}]   "
              f"TEST delta={row['shuffled_vs_model2_test_delta']:+.4f} p={row['shuffled_vs_model2_test_p']:.3f}")

    pd.DataFrame(ablation_rows).to_csv(os.path.join(OUT_DIR, "model3_shuffled_time_ablation_ladder.csv"), index=False)

    # ---------------- shuffle-draw sensitivity across all 3 seeds, delivery ----------------
    print(f"\n{'='*90}\nSHUFFLE-DRAW SENSITIVITY (delivery, all {len(SHUFFLE_SEEDS)} seeds):")
    sensitivity_rows = []
    for seed in SHUFFLE_SEEDS:
        r0 = shuffled_results_by_seed[seed]["rows"][0]  # horizon_min == 0 is first in HORIZONS
        assert r0["horizon_min"] == 0
        sensitivity_rows.append({"shuffle_seed": seed, "cv_auroc": r0["cv_auroc_cand"], "cv_delta_vs_sw": r0["cv_delta"],
                                  "cv_boot_p": r0["cv_boot_p"], "test_auroc": r0["test_auroc_cand"], "test_delta_vs_sw": r0["test_delta"]})
        print(f"  seed={seed}: CV AUROC={r0['cv_auroc_cand']:.4f} (delta_vs_sw={r0['cv_delta']:+.4f}, p={r0['cv_boot_p']:.3f})  "
              f"TEST AUROC={r0['test_auroc_cand']:.4f} (delta_vs_sw={r0['test_delta']:+.4f})")
    pd.DataFrame(sensitivity_rows).to_csv(os.path.join(OUT_DIR, "model3_shuffled_time_seed_sensitivity.csv"), index=False)

    # ---------------- mechanical verdict (decision rule fixed in the docstring above, applied here) ----------------
    delivery = ablation_rows[0]
    shuffled_vs_real_sig_worse = (delivery["shuffled_vs_real_cv_p"] < 0.05) and (delivery["shuffled_vs_real_cv_delta"] < 0)
    shuffled_close_to_model2 = abs(delivery["shuffled_vs_model2_cv_delta"]) < abs(delivery["shuffled_vs_real_cv_delta"])
    shuffled_comparable_to_real = (delivery["shuffled_vs_real_cv_p"] >= 0.05) and (delivery["shuffled_vs_real_cv_ci_low"] < 0 < delivery["shuffled_vs_real_cv_ci_high"])

    seeds_agree_direction = len(set(np.sign([shuffled_results_by_seed[s]["rows"][0]["cv_delta"] for s in SHUFFLE_SEEDS]).tolist())) <= 1

    if shuffled_vs_real_sig_worse and shuffled_close_to_model2:
        verdict = ("GENUINE TEMPORAL-ORDER USE: shuffled-time Model 3 is significantly worse than real-time Model 3 "
                    "at delivery, and sits closer to Model 2's zero-time-information endpoint than to real Model 3. "
                    "Model 3's use of elapsed_t appears to reflect real information, not capacity noise.")
    elif shuffled_comparable_to_real:
        verdict = ("CAPACITY ARTIFACT: shuffled-time Model 3 performs comparably to real-time Model 3 at delivery "
                    "(not significantly different, CI includes zero). The extra input dimension appears to have "
                    "bought noise-driven flexibility rather than genuine temporal-order exploitation. This "
                    "meaningfully undercuts Model 3's current interpretation, though not necessarily its raw "
                    "predictive value against the P6 baseline.")
    else:
        verdict = ("AMBIGUOUS: the result does not cleanly match either pre-registered pattern -- reported as such, "
                    "not forced into a bucket. See the full ablation ladder and seed-sensitivity tables.")

    print(f"\n{'='*90}\nMECHANICAL VERDICT (delivery horizon, decision rule fixed before computation):\n  {verdict}")
    print(f"  Seeds agree on direction of shuffled_delta_vs_sw: {seeds_agree_direction}")

    summary = {
        "shuffle_seeds": SHUFFLE_SEEDS, "primary_seed": PRIMARY_SEED,
        "ablation_ladder": ablation_rows, "seed_sensitivity": sensitivity_rows,
        "verdict_delivery": verdict, "seeds_agree_direction": bool(seeds_agree_direction),
    }
    with open(os.path.join(OUT_DIR, "model3_shuffled_time_control_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved -> {OUT_DIR}/model3_shuffled_time_control_summary.json")
    print(f"Saved -> {OUT_DIR}/model3_shuffled_time_ablation_ladder.csv")
    print(f"Saved -> {OUT_DIR}/model3_shuffled_time_seed_sensitivity.csv")


if __name__ == "__main__":
    main()
