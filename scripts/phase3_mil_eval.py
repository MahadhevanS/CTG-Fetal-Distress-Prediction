"""
Phase 3: CTU-UHB Patient-Level Learning & 1D Multi-Instance Aggregation.

Evaluates Fixed Aggregation (Max, P90, Mean, Top-3), Mean Embedding Pooling,
Max Embedding Pooling, Attention MIL, Fixed-K Attention, and Negative Controls
under the frozen 5-fold patient-grouped protocol.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.protocol import Protocol
from src.models.cnn1d_encoder import CNN1DEncoder

FOLDS_PATH = "data/processed_clinical/folds.json"
DATA_DIR = "data/phase1_candidates"
OUT_DIR = "results/phase3_mil"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_ENCODER = 25
EPOCHS_MIL = 50
BATCH_SIZE = 64
LR_MIL = 2e-3


# ---------------------------------------------------------------------------
# 1D ResNet Window Encoder
# ---------------------------------------------------------------------------
class WindowEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 128):
        super().__init__()
        self.encoder = CNN1DEncoder(in_channels=in_channels, seq_len=4800, latent_dim=latent_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)
        logits = self.head(feat).squeeze(-1)
        return feat, logits


# ---------------------------------------------------------------------------
# Patient Bag Dataset
# ---------------------------------------------------------------------------
class PatientBagDataset(Dataset):
    def __init__(self, embeddings_by_patient: dict, labels_by_patient: dict,
                 window_scores_by_patient: dict = None, fixed_k: int = None, shuffle_order: bool = False):
        self.pids = sorted(embeddings_by_patient.keys())
        self.bags = []
        self.labels = []
        self.scores = []

        for p in self.pids:
            emb = embeddings_by_patient[p] # (N_p, D)
            if shuffle_order:
                idx = np.random.permutation(len(emb))
                emb = emb[idx]
            if fixed_k is not None:
                # Take the last fixed_k windows (or pad if fewer)
                if len(emb) >= fixed_k:
                    emb = emb[-fixed_k:]
                else:
                    pad = np.repeat(emb[-1:], fixed_k - len(emb), axis=0)
                    emb = np.vstack([emb, pad])
            self.bags.append(torch.tensor(emb, dtype=torch.float32))
            self.labels.append(labels_by_patient[p])
            if window_scores_by_patient:
                self.scores.append(window_scores_by_patient[p])

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        return self.bags[idx], torch.tensor(self.labels[idx], dtype=torch.float32), self.pids[idx]


# ---------------------------------------------------------------------------
# MIL Aggregation Modules
# ---------------------------------------------------------------------------
class MeanPoolingMIL(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, bag: torch.Tensor):
        # bag: (N_p, D)
        h_p = bag.mean(dim=0, keepdim=True) # (1, D)
        logit = self.classifier(h_p)
        return logit.view(-1), torch.ones(len(bag), device=bag.device) / len(bag)


class MaxPoolingMIL(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, bag: torch.Tensor):
        # bag: (N_p, D)
        h_p = bag.amax(dim=0, keepdim=True) # (1, D)
        logit = self.classifier(h_p)
        return logit.view(-1), torch.ones(len(bag), device=bag.device) / len(bag)


class AttentionMIL(nn.Module):
    """
    Gated / Tanh Attention Multi-Instance Learning (Ilse et al.).
    Computes a_i = softmax(w^T tanh(V h_i)) -> h_p = sum a_i h_i.
    """
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 32):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, bag: torch.Tensor):
        # bag: (N_p, D)
        att_weights = self.attention(bag) # (N_p, 1)
        att_norm = F.softmax(att_weights, dim=0) # (N_p, 1)
        h_p = torch.sum(att_norm * bag, dim=0, keepdim=True) # (1, D)
        logit = self.classifier(h_p)
        return logit.view(-1), att_norm.view(-1)


# ---------------------------------------------------------------------------
# Training and Extraction Functions
# ---------------------------------------------------------------------------
def extract_embeddings_and_train_encoder(X, y, pid, prot):
    cache_path = os.path.join(OUT_DIR, "extracted_embeddings.npz")
    if os.path.exists(cache_path):
        print(f"\n--- Loading cached embeddings from {cache_path} ---")
        cached = np.load(cache_path)
        return cached["embeddings"], cached["window_probs"]

    print("\n--- Training 1D Encoders & Extracting Latent Embeddings across 5 Folds ---")
    window_embeddings = np.zeros((len(y), 128), dtype=np.float32)
    window_probs = np.zeros(len(y), dtype=np.float32)

    for fold, (tr_mask, te_patients) in enumerate(prot.folds(repeat=0), 1):
        te_mask = ~tr_mask
        te_idx = np.where(te_mask)[0]

        X_tr = X[tr_mask].copy()
        X_te = X[te_mask].copy()

        # Training-set normalization
        means = X_tr[:, 0:1, :].mean(axis=(0, 2), keepdims=True)
        stds = X_tr[:, 0:1, :].std(axis=(0, 2), keepdims=True) + 1e-6
        X_tr = (X_tr - means) / stds
        X_te = (X_te - means) / stds

        n_pos = int(y[tr_mask].sum())
        n_neg = len(y[tr_mask]) - n_pos
        pos_weight = float(n_neg / max(1, n_pos))

        tr_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y[tr_mask], dtype=torch.float32)),
            batch_size=BATCH_SIZE, shuffle=True
        )
        te_loader = DataLoader(
            TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(y[te_mask], dtype=torch.float32)),
            batch_size=BATCH_SIZE, shuffle=False
        )

        torch.manual_seed(SEED + fold)
        model = WindowEncoder(in_channels=1, latent_dim=128).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_ENCODER)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))

        for ep in range(EPOCHS_ENCODER):
            model.train()
            for x_b, y_b in tr_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                _, logits = model(x_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Extract embeddings and predictions on test fold
        model.eval()
        feats, probs = [], []
        with torch.no_grad():
            for x_b, _ in te_loader:
                x_b = x_b.to(DEVICE)
                f, logit = model(x_b)
                p = torch.sigmoid(logit).cpu().numpy()
                feats.append(f.cpu().numpy())
                probs.extend(p.tolist())

        window_embeddings[te_idx] = np.vstack(feats)
        window_probs[te_idx] = np.array(probs)
        print(f"  Fold {fold} Encoder trained: extracted {len(te_idx)} test embeddings.")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=window_embeddings, window_probs=window_probs)
    return window_embeddings, window_probs


def train_and_eval_mil_model(model_cls, pids_by_split, embs_by_p, labs_by_p, prot,
                             fixed_k=None, shuffle_order=False, permute_train_labels=False,
                             shuffle_across_patients=False):
    oof_patient_scores = {}
    attention_records = {}

    for fold, (tr_mask, te_patients) in enumerate(prot.folds(repeat=0), 1):
        tr_patients = [p for p in prot.patients if p not in te_patients]

        # Inner split for validation
        tr_fit, tr_val = prot.inner_split(tr_mask, frac=0.2, seed=SEED + fold)
        val_patients = np.unique(prot.pid[tr_val])
        fit_patients = [p for p in tr_patients if p not in val_patients]

        # Handle controls
        fit_embs = {p: embs_by_p[p].copy() for p in fit_patients}
        val_embs = {p: embs_by_p[p].copy() for p in val_patients}
        te_embs = {p: embs_by_p[p].copy() for p in te_patients}

        fit_labs = {p: labs_by_p[p] for p in fit_patients}
        val_labs = {p: labs_by_p[p] for p in val_patients}
        te_labs = {p: labs_by_p[p] for p in te_patients}

        if permute_train_labels:
            shuffled_labs = np.random.permutation(list(fit_labs.values()))
            fit_labs = {p: shuffled_labs[i] for i, p in enumerate(fit_patients)}

        if shuffle_across_patients:
            # Pool all windows and randomly assign back
            all_embs = np.vstack(list(fit_embs.values()) + list(val_embs.values()) + list(te_embs.values()))
            np.random.shuffle(all_embs)
            ptr = 0
            for p in fit_patients:
                n = len(fit_embs[p])
                fit_embs[p] = all_embs[ptr:ptr + n]
                ptr += n
            for p in val_patients:
                n = len(val_embs[p])
                val_embs[p] = all_embs[ptr:ptr + n]
                ptr += n
            for p in te_patients:
                n = len(te_embs[p])
                te_embs[p] = all_embs[ptr:ptr + n]
                ptr += n

        fit_ds = PatientBagDataset(fit_embs, fit_labs, fixed_k=fixed_k, shuffle_order=shuffle_order)
        val_ds = PatientBagDataset(val_embs, val_labs, fixed_k=fixed_k, shuffle_order=shuffle_order)
        te_ds = PatientBagDataset(te_embs, te_labs, fixed_k=fixed_k, shuffle_order=shuffle_order)

        # Train MIL model
        torch.manual_seed(SEED + fold)
        model = model_cls().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MIL, weight_decay=1e-4)
        pos_weight = float((len(fit_patients) - sum(fit_labs.values())) / max(1, sum(fit_labs.values())))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))

        best_val_auc = -1.0
        best_state = None

        for ep in range(EPOCHS_MIL):
            model.train()
            # Mini-batch over patients
            indices = np.random.permutation(len(fit_ds))
            for idx in indices:
                bag, lab, _ = fit_ds[idx]
                bag, lab = bag.to(DEVICE), lab.to(DEVICE)
                optimizer.zero_grad()
                logit, _ = model(bag)
                loss = criterion(logit, lab.view(-1))
                loss.backward()
                optimizer.step()

            # Eval on val patients
            model.eval()
            val_scores, val_targets = [], []
            with torch.no_grad():
                for i in range(len(val_ds)):
                    bag, lab, _ = val_ds[i]
                    logit, _ = model(bag.to(DEVICE))
                    val_scores.append(torch.sigmoid(logit).item())
                    val_targets.append(lab.item())
            val_auc = roc_auc_score(val_targets, val_scores)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

        # Evaluate on Test Patients
        model.eval()
        with torch.no_grad():
            for i in range(len(te_ds)):
                bag, lab, p = te_ds[i]
                logit, att = model(bag.to(DEVICE))
                score = torch.sigmoid(logit).item()
                oof_patient_scores[p] = score
                attention_records[p] = att.cpu().numpy()

    # Form ordered arrays
    y_test_all = np.array([labs_by_p[p] for p in prot.patients])
    scores_all = np.array([oof_patient_scores[p] for p in prot.patients])
    return scores_all, y_test_all, attention_records


def paired_bootstrap_test(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    diffs = []
    idx = np.arange(len(labels))
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(labels[b])) < 2:
            continue
        auc_a = roc_auc_score(labels[b], scores_a[b])
        auc_b = roc_auc_score(labels[b], scores_b[b])
        diffs.append(auc_b - auc_a)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_val = float(np.mean(diffs <= 0.0)) if np.mean(diffs) > 0 else float(np.mean(diffs >= 0.0))
    return float(np.mean(diffs)), (float(lo), float(hi)), p_val


def run_phase3():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== STARTING PHASE 3: PATIENT-LEVEL LEARNING & MULTI-INSTANCE AGGREGATION ===")

    # 1. Load P2 Quality-Aware dataset
    p2_data = torch.load(os.path.join(DATA_DIR, "p2_dataset.pt"), weights_only=False)
    X = p2_data["X"].numpy()[:, 0:1, :] # Raw FHR 1D (8517, 1, 4800)
    y_arr = p2_data["y"].numpy()
    pid_arr = p2_data["pid"]

    prot = Protocol.load_or_create(pid_arr, y_arr, path=FOLDS_PATH)

    # 2. Extract 1D window embeddings & predictions
    embeddings, window_probs = extract_embeddings_and_train_encoder(X, y_arr, pid_arr, prot)

    # Group embeddings and scores by patient
    embs_by_p = {}
    scores_by_p = {}
    labs_by_p = {}
    for p in prot.patients:
        idx = prot.pidx[p]
        embs_by_p[p] = embeddings[idx]
        scores_by_p[p] = window_probs[idx]
        labs_by_p[p] = int(y_arr[idx].max())

    results = {}

    # -------------------------------------------------------------
    # Diagnostic: Bag-Size & Duration Baseline
    # -------------------------------------------------------------
    bag_sizes = np.array([len(embs_by_p[p]) for p in prot.patients])
    lab_patients = np.array([labs_by_p[p] for p in prot.patients])
    auc_bag_size = roc_auc_score(lab_patients, bag_sizes)
    print(f"\n[Diagnostic] Bag Size Alone (N_p) -> AUROC: {auc_bag_size:.4f}")

    # -------------------------------------------------------------
    # Exp 3.0: Fixed Heuristic Aggregations (Max, P90, Mean, Top-3)
    # -------------------------------------------------------------
    print("\n--- Exp 3.0: Fixed Heuristic Aggregations ---")
    lab, sc_max = prot.to_patient(window_probs, how="max")
    _, sc_mean = prot.to_patient(window_probs, how="mean")
    _, sc_p90 = prot.to_patient(window_probs, how="p90")
    sc_top3 = np.array([np.mean(np.sort(scores_by_p[p])[-min(3, len(scores_by_p[p])):]) for p in prot.patients])

    rep_max = prot.report("Fixed_Max", window_probs, how="max")
    rep_mean = prot.report("Fixed_Mean", window_probs, how="mean")
    rep_p90 = prot.report("Fixed_P90", window_probs, how="p90")
    rep_top3 = prot.report_patient_scores("Fixed_Top3", lab, sc_top3)

    results["Fixed_Max"] = rep_max
    results["Fixed_Mean"] = rep_mean
    results["Fixed_P90"] = rep_p90
    results["Fixed_Top3"] = rep_top3

    # -------------------------------------------------------------
    # Exp 3.1: Mean Embedding Pooling
    # -------------------------------------------------------------
    print("\n--- Exp 3.1: Mean Embedding Pooling ---")
    sc_mean_emb, _, _ = train_and_eval_mil_model(MeanPoolingMIL, None, embs_by_p, labs_by_p, prot)
    rep_mean_emb = prot.report_patient_scores("Mean_Embedding_MIL", lab, sc_mean_emb)
    results["Mean_Embedding_MIL"] = rep_mean_emb

    # -------------------------------------------------------------
    # Exp 3.2: Max Embedding Pooling
    # -------------------------------------------------------------
    print("\n--- Exp 3.2: Max Embedding Pooling ---")
    sc_max_emb, _, _ = train_and_eval_mil_model(MaxPoolingMIL, None, embs_by_p, labs_by_p, prot)
    rep_max_emb = prot.report_patient_scores("Max_Embedding_MIL", lab, sc_max_emb)
    results["Max_Embedding_MIL"] = rep_max_emb

    # -------------------------------------------------------------
    # Exp 3.3: Attention MIL (Primary)
    # -------------------------------------------------------------
    print("\n--- Exp 3.3: Attention MIL (Primary) ---")
    sc_att, _, att_records = train_and_eval_mil_model(AttentionMIL, None, embs_by_p, labs_by_p, prot)
    rep_att = prot.report_patient_scores("Attention_MIL", lab, sc_att)
    results["Attention_MIL"] = rep_att

    # -------------------------------------------------------------
    # Exp 3.4: Fixed-Context Attention MIL (K=6 windows = last 32.5 min)
    # -------------------------------------------------------------
    print("\n--- Exp 3.4: Fixed-Context Attention MIL (K=6) ---")
    sc_fixed_k, _, _ = train_and_eval_mil_model(AttentionMIL, None, embs_by_p, labs_by_p, prot, fixed_k=6)
    rep_fixed_k = prot.report_patient_scores("Fixed_K_Attention_MIL", lab, sc_fixed_k)
    results["Fixed_K_Attention_MIL"] = rep_fixed_k

    # -------------------------------------------------------------
    # Exp 3.5: Shuffled-Window Order Control (Permutation Invariance)
    # -------------------------------------------------------------
    print("\n--- Exp 3.5: Shuffled-Window Order Control ---")
    sc_shuf_order, _, _ = train_and_eval_mil_model(AttentionMIL, None, embs_by_p, labs_by_p, prot, shuffle_order=True)
    rep_shuf_order = prot.report_patient_scores("Shuffled_Order_MIL", lab, sc_shuf_order)
    results["Shuffled_Order_MIL"] = rep_shuf_order

    # -------------------------------------------------------------
    # Exp 3.6: Patient-Window Shuffle Leakage Control
    # -------------------------------------------------------------
    print("\n--- Exp 3.6: Patient-Window Shuffle Leakage Control ---")
    sc_leak_shuf, _, _ = train_and_eval_mil_model(AttentionMIL, None, embs_by_p, labs_by_p, prot, shuffle_across_patients=True)
    rep_leak_shuf = prot.report_patient_scores("Leakage_Shuffle_Control", lab, sc_leak_shuf)
    results["Leakage_Shuffle_Control"] = rep_leak_shuf

    # -------------------------------------------------------------
    # Exp 3.7: Label Permutation Control
    # -------------------------------------------------------------
    print("\n--- Exp 3.7: Label Permutation Control ---")
    sc_label_perm, _, _ = train_and_eval_mil_model(AttentionMIL, None, embs_by_p, labs_by_p, prot, permute_train_labels=True)
    rep_label_perm = prot.report_patient_scores("Label_Permutation_Control", lab, sc_label_perm)
    results["Label_Permutation_Control"] = rep_label_perm

    # -------------------------------------------------------------
    # Paired Statistical Comparisons vs Fixed Max
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("PAIRED STATISTICAL COMPARISONS vs Fixed Max Control")
    print("=======================================================")
    mil_comparisons = [
        ("Fixed_P90", sc_p90),
        ("Fixed_Mean", sc_mean),
        ("Fixed_Top3", sc_top3),
        ("Mean_Embedding_MIL", sc_mean_emb),
        ("Max_Embedding_MIL", sc_max_emb),
        ("Attention_MIL", sc_att),
        ("Fixed_K_Attention_MIL", sc_fixed_k),
        ("Shuffled_Order_MIL", sc_shuf_order),
        ("Leakage_Shuffle_Control", sc_leak_shuf),
        ("Label_Permutation_Control", sc_label_perm),
    ]

    paired_summary = []
    for tag, sc_comp in mil_comparisons:
        diff, ci, pval = paired_bootstrap_test(lab, sc_max, sc_comp)
        print(f"  {tag:26s}: dAUROC = {diff:+.4f} [{ci[0]:.3f}, {ci[1]:.3f}] | p={pval:.3f}")
        paired_summary.append({
            "model": tag,
            "delta_auroc_vs_max": diff,
            "ci_95": list(ci),
            "p_val": pval
        })

    # Save summary results
    final_payload = {
        "bag_size_diagnostic_auroc": float(auc_bag_size),
        "benchmark_clinical_lr": {
            "auroc": 0.7271,
            "ci": [0.670, 0.779],
            "auprc": 0.409
        },
        "results": {k: {
            "name": v["name"],
            "auroc": float(v["auroc"]),
            "auprc": float(v["auprc"]),
            "ci": [float(v["ci_lo"]), float(v["ci_hi"])],
        } for k, v in results.items()},
        "paired_comparisons_vs_max": paired_summary,
        "patient_scores": {
            "labels": [int(x) for x in lab],
            "fixed_max": [float(x) for x in sc_max],
            "fixed_mean": [float(x) for x in sc_mean],
            "fixed_p90": [float(x) for x in sc_p90],
            "fixed_top3": [float(x) for x in sc_top3],
            "mean_embedding": [float(x) for x in sc_mean_emb],
            "max_embedding": [float(x) for x in sc_max_emb],
            "attention_mil": [float(x) for x in sc_att],
            "fixed_k_attention": [float(x) for x in sc_fixed_k]
        },
        "attention_weights": {p: [float(w) for w in att_records[p]] for p in prot.patients}
    }

    with open(os.path.join(OUT_DIR, "phase3_results.json"), "w") as fh:
        json.dump(final_payload, fh, indent=2)

    print(f"\nPhase 3 execution complete. Saved results to {OUT_DIR}/phase3_results.json")
    return final_payload


if __name__ == "__main__":
    run_phase3()
