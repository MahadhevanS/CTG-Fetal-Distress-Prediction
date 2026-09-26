"""
Phase D2/D3/D4-D6 combined -- the metric that settles the URM-vs-DeepFHR comparison
(docs/phaseD_deepfhr_reproduction_protocol.md, Amendment 1).

Same reconstructed CWT+CNN pipeline as D1 (with both verified fixes: class-weighted loss, zero-centering), but:
  - patient-grouped canonical folds (this project's own 547/110 cohort, folds.json) instead of image-random
  - one image set per CAUSAL ROLLING WINDOW (results/phase8_rolling/) instead of one per patient
  - patient-level AUROC at horizons 0/10/20/30 min via eligible_prefix_length, exactly as URM is scored
  - paired patient bootstrap against URM's own exact, gate-verified scores (scripts/phase26_clinical_evaluation.py)

  python scripts/phaseD2_patient_causal_comparison.py --build-images   # cache all window images (slower: ~51k images)
  python scripts/phaseD2_patient_causal_comparison.py --train          # patient-grouped 5-fold training + comparison
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (BASE_DIR, os.path.join(BASE_DIR, "src", "preprocessing")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ingestion import load_ctu_chb_record
from src.deepfhr.preprocessing import preprocess_segment
from src.deepfhr.cwt_image import build_images
from src.deepfhr.cnn import DeepFHRNet
from src.models.phase16_causal_attention import eligible_prefix_length
import scripts.phaseD1_deepfhr_paperstyle as d1
import scripts.phase20_harness as hz
from scripts.phase26_clinical_evaluation import build_urm_sequences
from scripts.phase11_bootstrap import paired_patient_bootstrap, fast_auc

FOLDS_PATH = "data/processed_clinical/folds.json"
ROLLING_PATH = "results/phase8_rolling/rolling_predictions.csv"
RAW_DIR = "data/raw/ctu-chb-intrapartum"
CFG_PATH = "docs/deepfhr_original.yaml"
OUT_DIR = "results/phaseD_deepfhr"
IMG_CACHE = os.path.join(OUT_DIR, "d2_window_images.npz")
LAST_HOUR_SAMPLES = 14400   # this project's own retained-recording convention (phase20_g3_feature_reproduction.py)
H = [0, 10, 20, 30]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cfg():
    import yaml
    return yaml.safe_load(open(CFG_PATH))


def build_all_window_images(cfg):
    df = pd.read_csv(ROLLING_PATH); df["patient_id"] = df["patient_id"].astype(str)
    pids = sorted(df["patient_id"].unique())
    all_imgs, all_win_idx, all_pid, all_tdel = [], [], [], []
    t0 = time.time()
    for k, pid in enumerate(pids):
        fhr, uc, fs = load_ctu_chb_record(os.path.join(RAW_DIR, str(pid)))
        if len(fhr) == 0:
            print(f"  [WARN] {pid} failed to load, skipped"); continue
        if len(fhr) > LAST_HOUR_SAMPLES:
            fhr = fhr[-LAST_HOUR_SAMPLES:]
        sub = df[df["patient_id"] == pid]
        for wi, (s, e, tdel) in enumerate(zip(sub["start_sample"], sub["end_sample"], sub["time_before_delivery_min"])):
            seg = fhr[int(s):int(e)].astype(np.float64)
            if len(seg) < 100:
                continue
            seg = preprocess_segment(seg, cfg)
            imgs = build_images(seg, wavelets=cfg["cwt"]["mother_wavelets"], scales=cfg["cwt"]["scales"],
                                size=cfg["cwt"]["image_size"][0], cmap_name=cfg["cwt"]["colormap"])
            for im in imgs:
                all_imgs.append(im); all_win_idx.append(wi); all_pid.append(pid); all_tdel.append(float(tdel))
        if k % 50 == 0:
            print(f"  {k}/{len(pids)} patients  ({time.time() - t0:.0f}s)", flush=True)
    X = np.stack(all_imgs).astype(np.float32)
    pid_arr = np.array(all_pid); win_arr = np.array(all_win_idx, dtype=np.int32); tdel_arr = np.array(all_tdel, dtype=np.float32)
    print(f"  built {X.shape[0]} images from {len(pids)} patients in {time.time() - t0:.0f}s")
    np.savez_compressed(IMG_CACHE, X=X, pid=pid_arr, win=win_arr, tdel=tdel_arr)
    print(f"  cached -> {IMG_CACHE}")


def train_fold_probs(Xtr, ytr, Xte, cfg):
    """Identical recipe to D1's train_fold (class-weighted loss + zero-centering), returns held-out probs only."""
    torch.manual_seed(42)
    model = DeepFHRNet(in_channels=3, conv_filters=cfg["cnn"]["conv_filters"], kernel=cfg["cnn"]["conv_kernel"][0],
                       fc_hidden=cfg["cnn"]["fc_hidden"], dropout=cfg["cnn"]["dropout"], img_size=cfg["cwt"]["image_size"][0]).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["training"]["lr_init"], momentum=cfg["training"]["momentum"],
                          weight_decay=cfg["training"]["l2_regularizer"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg["training"]["lr_drop_period_epochs"], gamma=cfg["training"]["lr_drop_factor"])
    n_pos = int((ytr == 1).sum()); n_neg = int((ytr == 0).sum()); n_tot = n_pos + n_neg
    class_weights = torch.tensor([n_tot / (2.0 * n_neg), n_tot / (2.0 * n_pos)], dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    # Memory fix (2026-09-26, diagnosed after an OOM on fold 1 with ~40k training images): a full-array
    # .permute(0,3,1,2).contiguous() materialises a SECOND full-size float32 copy of the training set (>2GB at
    # this scale). Keep both splits in their original NHWC numpy layout and permute only per mini-batch instead
    # -- negligible memory, no behaviour change.
    ch_mean = Xtr.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)   # (1,1,1,3), training fold only
    ytr_t = torch.from_numpy(ytr)
    n = len(Xtr); bs = cfg["training"]["batch_size"]
    pad = 4; size = Xtr.shape[1]

    def to_batch(x_np, device, augment):
        xb = torch.from_numpy(x_np - ch_mean).permute(0, 3, 1, 2).contiguous().to(device)
        if augment and xb.shape[0] > 4 and cfg["training"]["augmentation"] == "random_crop":
            xb = nn.functional.pad(xb, (pad, pad, pad, pad), mode="reflect")
            top = np.random.randint(0, 2 * pad + 1); left = np.random.randint(0, 2 * pad + 1)
            xb = xb[:, :, top:top + size, left:left + size]
        return xb

    for ep in range(cfg["training"]["epochs"]):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = to_batch(Xtr[idx], DEVICE, augment=True); yb = ytr_t[idx].to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
    model.eval()
    probs = np.zeros(len(Xte), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(Xte), bs):
            xb = to_batch(Xte[i:i + bs], DEVICE, augment=False)
            probs[i:i + bs] = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-images", action="store_true")
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg()

    if a.build_images:
        build_all_window_images(cfg)
        return

    if not a.train:
        return

    with open(FOLDS_PATH) as fh:
        folds_blob = json.load(fh)
    clean_pids = sorted(folds_blob["assignment"].keys())
    fold_of = {p: folds_blob["assignment"][p][0] for p in clean_pids}
    df = pd.read_csv(ROLLING_PATH); df["patient_id"] = df["patient_id"].astype(str)
    y_lookup = {p: int(df[df["patient_id"] == p]["primary_label_715"].iloc[0]) for p in clean_pids}

    d = np.load(IMG_CACHE)
    X, pid_arr, win_arr, tdel_arr = d["X"], d["pid"], d["win"], d["tdel"]
    print(f"  loaded {X.shape[0]} window-images from {len(np.unique(pid_arr))} patients")

    # image-level label = that image's patient's outcome (unchanged across a patient's own windows/images)
    y_img = np.array([y_lookup[p] for p in pid_arr])
    fold_img = np.array([fold_of[p] for p in pid_arr])

    # per-image OOF probability, patient-grouped 5-fold (canonical folds.json)
    oof = np.zeros(len(X), dtype=np.float32)
    for f in range(5):
        t0 = time.time()
        tr_mask = fold_img != f; te_mask = fold_img == f
        probs = train_fold_probs(X[tr_mask], y_img[tr_mask], X[te_mask], cfg)
        oof[te_mask] = probs
        print(f"  fold {f}: {te_mask.sum()} images, {(fold_img == f).sum()} -- window AUROC (sanity) "
              f"{hz.fast_auc(y_img[te_mask], probs):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        np.save(os.path.join(OUT_DIR, f"d2_oof_partial_fold{f}.npy"), oof)   # checkpoint in case of interruption

    # per-window score = mean over that window's 6 image variants
    key = pd.DataFrame({"pid": pid_arr, "win": win_arr, "tdel": tdel_arr, "prob": oof})
    window_scores = key.groupby(["pid", "win"], sort=False).agg(prob=("prob", "mean"), tdel=("tdel", "first")).reset_index()

    # per-patient chronological sequence (ascending elapsed = descending time_before_delivery, project convention)
    seq, tdel_seq = {}, {}
    for pid, g in window_scores.groupby("pid"):
        g = g.sort_values("tdel", ascending=False)
        seq[pid] = g["prob"].values.astype(np.float64)
        tdel_seq[pid] = g["tdel"].values.astype(np.float64)

    y = np.array([y_lookup[p] for p in clean_pids])
    deepfhr_scores = {h_: np.zeros(len(clean_pids)) for h_ in H}
    for i, p in enumerate(clean_pids):
        s = seq[p]; T = len(s)
        for h_ in H:
            k = eligible_prefix_length(tdel_seq[p], h_, T) - 1
            deepfhr_scores[h_][i] = s[k]

    print("\n=== DeepFHR-reconstruction (causal, patient-grouped canonical folds) vs URM ===")
    urm = build_urm_sequences()
    assert urm["clean_pids"] == clean_pids, "cohort mismatch vs URM's own build"
    urm_horiz = {}
    for h_ in H:
        urm_horiz[h_] = np.array([urm["fused_seq_cv"][p][eligible_prefix_length(urm["t_del_cv"][p], h_, len(urm["fused_seq_cv"][p])) - 1] for p in clean_pids])

    rows = []
    for h_ in H:
        auc_deepfhr = fast_auc(y, deepfhr_scores[h_]); auc_urm = fast_auc(y, urm_horiz[h_])
        boot = paired_patient_bootstrap(y, deepfhr_scores[h_], urm_horiz[h_], n_boot=2000, seed=42)
        rows.append({"horizon_min": h_, "deepfhr_reconstruction_auroc": round(auc_deepfhr, 4), "urm_auroc": round(auc_urm, 4),
                    "delta_urm_minus_deepfhr": round(auc_urm - auc_deepfhr, 4), "ci95_low": round(boot["ci_95_low"], 4),
                    "ci95_high": round(boot["ci_95_high"], 4), "p_value": round(boot["p_value"], 4)})
        print(f"  h={h_:>2}m: DeepFHR-recon {auc_deepfhr:.4f}  URM {auc_urm:.4f}  Delta(URM-DeepFHR) "
              f"{auc_urm - auc_deepfhr:+.4f}  95% CI [{boot['ci_95_low']:.4f}, {boot['ci_95_high']:.4f}]  p={boot['p_value']:.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "d2_urm_comparison.csv"), index=False)
    json.dump(rows, open(os.path.join(OUT_DIR, "d2_urm_comparison.json"), "w"), indent=2, default=float)
    print(f"\nSaved -> {OUT_DIR}/d2_urm_comparison.csv")


if __name__ == "__main__":
    main()
