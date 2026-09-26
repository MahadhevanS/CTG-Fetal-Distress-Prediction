"""
Phase D1 -- faithful, paper-style reproduction of Zhao et al. 2019 (DeepFHR), per
docs/phaseD_deepfhr_reproduction_protocol.md section 3 and docs/deepfhr_original.yaml.

We do NOT change DeepFHR before establishing whether we can reproduce the original result: image-level
random 10-fold CV, on all 552 raw CTU-UHB records, exactly the paper's own protocol.

  python scripts/phaseD1_deepfhr_paperstyle.py --build-images   # cache the 3312 images (slow, run once)
  python scripts/phaseD1_deepfhr_paperstyle.py --train          # 10-fold CV training + evaluation (fast, GPU)
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (BASE_DIR, os.path.join(BASE_DIR, "src", "preprocessing")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ingestion import load_ctu_chb_record
from src.deepfhr.preprocessing import preprocess_segment, extract_last_segment
from src.deepfhr.cwt_image import build_images
from src.deepfhr.cnn import DeepFHRNet

RAW_DIR = "data/raw/ctu-chb-intrapartum"
META_PATH = "data/raw/ctu-chb-intrapartum/clinical_metadata.csv"
CFG_PATH = "docs/deepfhr_original.yaml"
OUT_DIR = "results/phaseD_deepfhr"
IMG_CACHE = os.path.join(OUT_DIR, "d1_images.npz")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cfg():
    import yaml
    return yaml.safe_load(open(CFG_PATH))


def build_all_images(cfg):
    meta = pd.read_csv(META_PATH).set_index("record_id")
    rids = sorted(meta.index.tolist())
    seg_samples = int(cfg["dataset"]["segment_minutes"] * 60 * cfg["dataset"]["fs_hz"])
    all_imgs = []
    all_labels = []   # 1 = Normal (paper's positive class), 0 = Pathological
    all_rids = []
    t0 = time.time()
    for k, rid in enumerate(rids):
        fhr, uc, fs = load_ctu_chb_record(os.path.join(RAW_DIR, str(rid)))
        if len(fhr) == 0:
            print(f"  [WARN] record {rid} failed to load, skipped"); continue
        seg = extract_last_segment(fhr, seg_samples)
        seg = preprocess_segment(seg, cfg)
        imgs = build_images(seg, wavelets=cfg["cwt"]["mother_wavelets"], scales=cfg["cwt"]["scales"],
                            size=cfg["cwt"]["image_size"][0], cmap_name=cfg["cwt"]["colormap"])
        ph = float(meta.loc[rid, "ph"])
        label = 1 if ph >= cfg["dataset"]["ph_threshold"] else 0   # Normal = positive, per the paper
        for im in imgs:
            all_imgs.append(im); all_labels.append(label); all_rids.append(rid)
        if k % 50 == 0:
            print(f"  {k}/{len(rids)} records  ({time.time() - t0:.0f}s)", flush=True)
    X = np.stack(all_imgs).astype(np.float32)          # (N, 64, 64, 3)
    y = np.array(all_labels, dtype=np.int64)
    rid_arr = np.array(all_rids)
    print(f"  built {X.shape[0]} images ({int((y == 1).sum())} normal / {int((y == 0).sum())} pathological) in {time.time() - t0:.0f}s")
    np.savez_compressed(IMG_CACHE, X=X, y=y, rid=rid_arr)
    print(f"  cached -> {IMG_CACHE}")


def train_fold(Xtr, ytr, Xte, yte, cfg):
    torch.manual_seed(42)
    model = DeepFHRNet(in_channels=3, conv_filters=cfg["cnn"]["conv_filters"], kernel=cfg["cnn"]["conv_kernel"][0],
                       fc_hidden=cfg["cnn"]["fc_hidden"], dropout=cfg["cnn"]["dropout"], img_size=cfg["cwt"]["image_size"][0]).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["training"]["lr_init"], momentum=cfg["training"]["momentum"],
                          weight_decay=cfg["training"]["l2_regularizer"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg["training"]["lr_drop_period_epochs"], gamma=cfg["training"]["lr_drop_factor"])

    # FIX (diagnosed 2026-09-26): plain CrossEntropyLoss on this ~4.3:1 imbalanced set collapses to predicting the
    # majority class (Se=99.9%/Sp=2.5% in the first run) even though a trivial pixel-level logistic regression under
    # the identical split reaches AUC 0.93 -- the images do carry the leakage signal, the unweighted loss just never
    # uses it. Inverse-frequency class weights, computed from the TRAINING fold only (never the held-out fold).
    n_pos = int((ytr == 1).sum()); n_neg = int((ytr == 0).sum()); n_tot = n_pos + n_neg
    class_weights = torch.tensor([n_tot / (2.0 * n_neg), n_tot / (2.0 * n_pos)], dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    Xtr_t = torch.tensor(Xtr).permute(0, 3, 1, 2).contiguous()
    ytr_t = torch.tensor(ytr)
    Xte_t = torch.tensor(Xte).permute(0, 3, 1, 2).contiguous()

    # FIX 2 (diagnosed 2026-09-26): Table 2 Layer 1 specifies "Data normalization: Zero center", which the first
    # run omitted entirely -- images were fed in raw [0,1] colormap range. Per-channel mean computed on the
    # TRAINING fold only, applied to both splits.
    if cfg["training"].get("data_normalization", "zero_center") == "zero_center":
        ch_mean = Xtr_t.mean(dim=(0, 2, 3), keepdim=True)
        Xtr_t = Xtr_t - ch_mean; Xte_t = Xte_t - ch_mean
    Xte_t = Xte_t.to(DEVICE)

    bs = cfg["training"]["batch_size"]; n = len(Xtr_t)
    for ep in range(cfg["training"]["epochs"]):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr_t[idx].to(DEVICE); yb = ytr_t[idx].to(DEVICE)
            if xb.shape[0] > 4 and cfg["training"]["augmentation"] == "random_crop":
                # random_crop augmentation: pad then random-crop back to the original size
                pad = 4
                xb = nn.functional.pad(xb, (pad, pad, pad, pad), mode="reflect")
                size = Xtr_t.shape[-1]
                top = np.random.randint(0, 2 * pad + 1); left = np.random.randint(0, 2 * pad + 1)
                xb = xb[:, :, top:top + size, left:left + size]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        probs_tr = torch.softmax(model(Xtr_t.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
        probs_te = torch.softmax(model(Xte_t), dim=1)[:, 1].cpu().numpy()   # P(class = Normal = positive)
    return probs_te, probs_tr


def youden_threshold(y_true, probs):
    """Threshold maximising Se+Sp-1 (Youden's J), chosen on TRAINING predictions only, then frozen for held-out use."""
    best_t, best_j = 0.5, -1.0
    for t in np.unique(probs):
        pred = (probs >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum(); fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum(); fp = ((pred == 1) & (y_true == 0)).sum()
        se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1)
        j = se + sp - 1
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def evaluate(y_true, probs, thresh=0.5):
    pred = (probs >= thresh).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum()); fn = int(((pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(len(y_true), 1)
    se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1)
    qi = float(np.sqrt(max(se, 0) * max(sp, 0)))
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    return dict(acc=acc, se=se, sp=sp, qi=qi, auc=auc, tp=tp, fp=fp, tn=tn, fn=fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-images", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="override docs/deepfhr_original.yaml epochs (diagnostic only)")
    a = ap.parse_args()
    cfg = load_cfg()
    if a.epochs is not None:
        cfg["training"]["epochs"] = a.epochs

    if a.build_images:
        build_all_images(cfg)
        return

    if a.train:
        d = np.load(IMG_CACHE)
        X, y, rid = d["X"], d["y"], d["rid"]
        print(f"  loaded {X.shape[0]} images, {int((y == 1).sum())} normal / {int((y == 0).sum())} pathological")
        n_folds = cfg["evaluation"]["n_folds"]
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_metrics_fixed, fold_metrics_youden = [], []
        oof_probs = np.zeros(len(y)); oof_seen = np.zeros(len(y), dtype=bool)
        for f, (tr_idx, te_idx) in enumerate(kf.split(X)):
            t0 = time.time()
            probs_te, probs_tr = train_fold(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], cfg)
            oof_probs[te_idx] = probs_te; oof_seen[te_idx] = True
            thr = youden_threshold(y[tr_idx], probs_tr)   # frozen from training predictions only, never test
            m_fixed = evaluate(y[te_idx], probs_te, thresh=0.5); m_fixed["fold"] = f; fold_metrics_fixed.append(m_fixed)
            m_yj = evaluate(y[te_idx], probs_te, thresh=thr); m_yj["fold"] = f; m_yj["threshold"] = thr; fold_metrics_youden.append(m_yj)
            print(f"  fold {f} (thr=0.5): Acc={m_fixed['acc']:.4f} Se={m_fixed['se']:.4f} Sp={m_fixed['sp']:.4f} QI={m_fixed['qi']:.4f} AUC={m_fixed['auc']:.4f}"
                  f"  | (thr={thr:.3f} train-Youden): Acc={m_yj['acc']:.4f} Se={m_yj['se']:.4f} Sp={m_yj['sp']:.4f} QI={m_yj['qi']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

        def summarize(fold_metrics, label):
            df = pd.DataFrame(fold_metrics)
            means = df[["acc", "se", "sp", "qi", "auc"]].mean(); stds = df[["acc", "se", "sp", "qi", "auc"]].std()
            print(f"\n  [{label}] Mean +/- SD across folds:")
            for k in ["acc", "se", "sp", "qi", "auc"]:
                tgt = cfg['evaluation']['published_target'][{'acc': 'accuracy', 'se': 'sensitivity', 'sp': 'specificity', 'qi': 'quality_index', 'auc': 'auroc'}[k]]
                print(f"    {k:<4} {means[k]*100:.2f} +/- {stds[k]*100:.2f}  (published target {tgt})")
            return df, {k: float(means[k]) for k in means.index}, {k: float(stds[k]) for k in stds.index}

        df_fixed, mean_fixed, std_fixed = summarize(fold_metrics_fixed, "fixed threshold 0.5")
        df_yj, mean_yj, std_yj = summarize(fold_metrics_youden, "train-Youden threshold")
        overall_auc = evaluate(y[oof_seen], oof_probs[oof_seen])["auc"]
        print(f"\n  Pooled AUC (all folds' out-of-fold predictions, threshold-independent): {overall_auc:.4f}")

        out = {"fixed_threshold_0.5": {"per_fold": fold_metrics_fixed, "mean": mean_fixed, "std": std_fixed},
               "train_youden_threshold": {"per_fold": fold_metrics_youden, "mean": mean_yj, "std": std_yj},
               "pooled_auc": overall_auc, "published_target": cfg["evaluation"]["published_target"],
               "fixes_applied": ["inverse-frequency class-weighted CrossEntropyLoss (computed per training fold)",
                                "per-channel zero-centering (mean computed per training fold), per Table 2 Layer 1"]}
        json.dump(out, open(os.path.join(OUT_DIR, "d1_results.json"), "w"), indent=2, default=float)
        df_fixed.to_csv(os.path.join(OUT_DIR, "d1_fold_metrics_fixed_threshold.csv"), index=False)
        df_yj.to_csv(os.path.join(OUT_DIR, "d1_fold_metrics_youden_threshold.csv"), index=False)
        print(f"\nSaved -> {OUT_DIR}/d1_results.json")


if __name__ == "__main__":
    main()
