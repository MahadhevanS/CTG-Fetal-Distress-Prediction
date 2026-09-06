"""
GATE 2 -- is the FIGO state learnable, and from what?

    python scripts/figo_gate2_state_detection.py

THE CIRCULARITY, STATED BEFORE THE NUMBERS
------------------------------------------
The FIGO state is a DETERMINISTIC function of the descriptors in
src/figo_state/descriptors.py. Feeding those same descriptors to a
classifier and predicting the state is not a prediction problem; it is a
function-recovery problem, and a good score means the pipeline is wired
correctly, nothing more. It is reported here as ARM 0 and labelled a wiring
check. A score near 1.0 is the expected, uninteresting outcome; a score below
~0.95 would mean something is broken and is the only reason to look at it.

ARM 0 MUST USE A TREE, NOT LOGISTIC REGRESSION.
The rule is a conjunction of INTERVALS -- "110 <= baseline <= 160", "5 <=
variability <= 25". A linear model cannot express an interval: it can only
push a coefficient in one direction, so it must fail at one end of every
band. Measured here, on the same columns and the same folds:

    logistic regression   AUROC 0.8371
    decision tree (d=8)   AUROC 0.9998

The 0.84 is a statement about logistic regression, not about the task. This
matters beyond the wiring check, because "LR on the clinical descriptors" is
the natural first baseline to reach for and it would have been read as "the
task is only moderately learnable" when in fact the label is exactly
recoverable. Both are reported so the gap is visible.

The real question is ARM 2: can the state be recovered from the RAW SIGNAL,
with no descriptor ever shown to the model? That is a genuine task -- the
network has to learn baseline estimation, bandwidth amplitude and event
morphology end to end -- and it is what "automated CTG interpretation"
actually means. ARM 1 sits between them: descriptors the rules DO NOT use.

ARMS
----
  0  rule descriptors -> tree        wiring check, expect ~1.0, not a result
  0b rule descriptors -> LR          NOT a fair check -- see below
  1  non-rule descriptors -> LR      information outside the rule inputs
  2  raw FHR + UC -> small 1D CNN    THE RESULT
  2a raw FHR only -> same CNN        how much the UC channel contributes
  3  epoch index alone               time confound control, must be ~0.50

TASKS
-----
  A1 binary       Normal vs Abnormal (Suspicious or Pathological)
  A2 three-class  Normal / Suspicious / Pathological, macro-averaged

A1 is primary. A2 is reported because a three-class score can be poor purely
because Suspicious sits between the other two, which is a property of the
ordinal scale rather than of the model.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))

from figo_state import rules as RL                                # noqa: E402
from figo_state.protocol import FigoProtocol, summarise           # noqa: E402

DATA = os.path.join(BASE, "data", "processed_figo")
OUT = os.path.join(BASE, "results", "figo_state")

# Descriptors the rule engine reads. Everything else is "non-rule".
RULE_INPUTS = ["baseline_bpm", "variability_bpm", "variability_measurable",
               "decel_repetitive", "n_decel_late", "n_decel_prolonged",
               "has_acute_hypoxia_decel", "usable"]


# --------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """
    A deliberately small 1D CNN. ~60k parameters.

    Sized against the data, not against a leaderboard: 3,497 readable epochs
    from 547 patients. This project has already measured what happens when
    capacity outruns supervision -- Model 9 (KG-MIL) put 2.5M parameters on
    ~435 bags and collapsed (docs/model8_crossformer_run_history.md, Part E).
    The user's plan says start small and only reach for a Transformer if the
    small model shows the signal is there. This is that small model.

    Stride-4 first conv downsamples 4 Hz to 1 Hz, which is above every
    frequency FIGO describes (oscillations run 3-5 cycles/min, events last
    >=15 s), so nothing the label depends on is aliased away.
    """

    def __init__(self, in_ch: int = 3, n_out: int = 1, width: int = 32):
        super().__init__()
        def block(i, o, k, s):
            return nn.Sequential(
                nn.Conv1d(i, o, k, stride=s, padding=k // 2),
                nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
        self.net = nn.Sequential(
            block(in_ch, width, 15, 4),      # 2400 -> 300
            block(width, width * 2, 9, 1),   # 300 -> 150
            block(width * 2, width * 2, 7, 1),   # 150 -> 75
            block(width * 2, width * 2, 5, 1),   # 75 -> 37
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(width * 2, n_out))

    def forward(self, x):
        return self.head(self.net(x))


def train_cnn(Xtr, ytr, Xva, yva, in_ch, n_out, device, epochs=30, bs=64,
              lr=1e-3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SmallCNN(in_ch=in_ch, n_out=n_out).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    if n_out == 1:
        pos = float(ytr.sum())
        w = torch.tensor([(len(ytr) - pos) / max(pos, 1.0)], device=device)
        lossfn = nn.BCEWithLogitsLoss(pos_weight=w)
        yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
        yv = torch.tensor(yva, dtype=torch.float32).unsqueeze(1)
    else:
        cnt = np.bincount(ytr.astype(int), minlength=n_out).astype(float)
        w = torch.tensor(len(ytr) / (n_out * np.maximum(cnt, 1.0)),
                         dtype=torch.float32, device=device)
        lossfn = nn.CrossEntropyLoss(weight=w)
        yt = torch.tensor(ytr, dtype=torch.long)
        yv = torch.tensor(yva, dtype=torch.long)

    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Xv = torch.tensor(Xva, dtype=torch.float32).to(device)
    yv_d = yv.to(device)

    best, best_state, patience = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            loss = lossfn(model(Xt[j].to(device)), yt[j].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            lo = model(Xv)
            score = -float(lossfn(lo, yv_d))
        # Selection is on an INNER validation split carved from training
        # patients (protocol rule 3), never on the reported fold.
        if score > best:
            best, patience = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, X, device, n_out, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            b = torch.tensor(X[i:i + bs], dtype=torch.float32).to(device)
            lo = model(b)
            out.append((torch.sigmoid(lo)[:, 0] if n_out == 1
                        else torch.softmax(lo, 1)).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------
def zscore_fit(X, mask):
    """Fit per-channel z-score on TRAINING rows only; leave the mask channel."""
    m = X[mask][:, :2].mean(axis=(0, 2), keepdims=True)
    s = X[mask][:, :2].std(axis=(0, 2), keepdims=True)
    return m, np.clip(s, 1e-6, None)


def zscore_apply(X, m, s):
    Z = X.copy()
    Z[:, :2] = (Z[:, :2] - m) / s
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=0,
                    help="which fold repeat to run (0-4)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--skip_cnn", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    z = np.load(os.path.join(DATA, "epochs.npz"), allow_pickle=True)
    names = [str(s) for s in z["descriptor_names"]]
    meta = pd.DataFrame({c: z[c] for c in z.files
                         if c not in ("X", "F", "descriptor_names")})
    F = pd.DataFrame(z["F"], columns=names)
    X = z["X"]

    keep = (meta.y_state >= 0).values
    meta, F, X = meta[keep].reset_index(drop=True), F[keep].reset_index(drop=True), X[keep]
    pid = meta.record_id.values.astype(str)
    state = meta.y_state.values.astype(int)
    y_bin = (state >= 1).astype(int)
    print(f"{len(meta)} readable epochs, {len(set(pid))} patients | "
          f"Normal {int((state == 0).sum())}  Suspicious {int((state == 1).sum())}"
          f"  Pathological {int((state == 2).sum())}")

    # Stratify folds on each patient's WORST state so the 34 patients who ever
    # reach Pathological are spread across folds rather than clumped.
    worst = meta.groupby("record_id").y_state.max().to_dict()
    P = FigoProtocol.load_or_create(pid, {str(k): int(v) for k, v in worst.items()},
                                    path=os.path.join(DATA, "folds.json"))

    non_rule = [c for c in names if c not in RULE_INPUTS]
    results = {}

    # ------------------------------------------------------------ tabular
    def run_tab(cols, y, kind="lr", multi=False):
        Xf = F[cols].values
        n_cls = int(y.max()) + 1 if multi else 2
        oof = (np.zeros((len(y), n_cls)) if multi else np.zeros(len(y)))
        seen = np.zeros(len(y), bool)
        for tr, te in P.folds(repeat=args.repeat):
            if kind == "lr":
                sc = StandardScaler().fit(Xf[tr])
                m = LogisticRegression(max_iter=3000, class_weight="balanced")
                m.fit(sc.transform(Xf[tr]), y[tr])
                p = m.predict_proba(sc.transform(Xf[te]))
            else:
                m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
                m.fit(Xf[tr], y[tr])
                p = m.predict_proba(Xf[te])
            oof[te] = p if multi else p[:, 1]
            seen[te] = True
        assert seen.all(), "some epochs never appeared in a test fold"
        return oof

    print("\n" + "=" * 74)
    print("TASK A1 -- BINARY: Normal vs Abnormal (Suspicious or Pathological)")
    print("=" * 74)
    print(f"  prevalence {100 * y_bin.mean():.1f}%  "
          f"({int(y_bin.sum())} abnormal / {len(y_bin)})")

    print("\nARM 0  rule descriptors -> tree   [WIRING CHECK, NOT A RESULT]")
    results["A1_arm0_rule_tree"] = summarise(
        "rule descriptors, tree (circular)", y_bin,
        run_tab(RULE_INPUTS, y_bin, "tree"), pid, P)
    print("       expected ~1.0 by construction: the label is a deterministic")
    print("       function of these columns. Below ~0.95 would mean a bug.")

    print("\nARM 0b rule descriptors -> LR   [same columns, linear model]")
    results["A1_arm0b_rule_lr"] = summarise(
        "rule descriptors, LR (circular)", y_bin,
        run_tab(RULE_INPUTS, y_bin, "lr"), pid, P)
    print("       the gap to ARM 0 is what a linear model costs on an interval")
    print("       rule. It is a fact about LR, not about the task.")

    print("\nARM 1  non-rule descriptors")
    results["A1_arm1_non_rule_tree"] = summarise(
        f"non-rule descriptors, tree (n={len(non_rule)})", y_bin,
        run_tab(non_rule, y_bin, "tree"), pid, P)
    results["A1_arm1_non_rule_lr"] = summarise(
        f"non-rule descriptors, LR (n={len(non_rule)})", y_bin,
        run_tab(non_rule, y_bin, "lr"), pid, P)

    print("\nARM 3  epoch index alone   [TIME-CONFOUND CONTROL]")
    results["A1_arm3_clock"] = summarise(
        "epoch index (no signal)", y_bin,
        meta.epoch_index.values.astype(float), pid, P)

    # ---------------------------------------------------------------- CNN
    if not args.skip_cnn:
        for tag, chans, label in (("arm2_fhr_uc", [0, 1, 2], "raw FHR+UC+mask -> CNN"),
                                  ("arm2a_fhr", [0, 2], "raw FHR+mask -> CNN")):
            print(f"\nARM 2  {label}   [THE RESULT]")
            t0 = time.time()
            oof = np.zeros(len(y_bin))
            for k, (tr, te) in enumerate(P.folds(repeat=args.repeat), 1):
                fit, val = P.inner_split(tr)
                m, s = zscore_fit(X, fit)
                Xz = zscore_apply(X, m, s)[:, chans]
                model = train_cnn(Xz[fit], y_bin[fit], Xz[val], y_bin[val],
                                  len(chans), 1, device, epochs=args.epochs,
                                  seed=k)
                oof[te] = predict(model, Xz[te], device, 1)
                print(f"    fold {k}/5 done ({time.time() - t0:.0f}s)")
            results[f"A1_{tag}"] = summarise(label, y_bin, oof, pid, P)
            np.save(os.path.join(OUT, f"oof_A1_{tag}_r{args.repeat}.npy"), oof)

    # -------------------------------------------------------- three-class
    print("\n" + "=" * 74)
    print("TASK A2 -- THREE-CLASS: Normal / Suspicious / Pathological")
    print("=" * 74)
    p3 = run_tab(RULE_INPUTS, state, "tree", multi=True)
    pred = p3.argmax(1)
    present = sorted(set(state))
    print("  ARM 0, tree (circular wiring check):")
    print(f"    macro AUROC       {roc_auc_score(state, p3, multi_class='ovr', average='macro'):.4f}")
    print(f"    macro F1          {f1_score(state, pred, average='macro'):.4f}")
    print(f"    balanced accuracy {balanced_accuracy_score(state, pred):.4f}")
    cm = confusion_matrix(state, pred, labels=present)
    print("    confusion (rows true, cols pred):")
    print("      " + "".join(f"{RL.STATE_NAMES[c][:6]:>10s}" for c in present))
    for i, c in enumerate(present):
        print(f"      {RL.STATE_NAMES[c][:10]:10s}" +
              "".join(f"{cm[i, j]:10d}" for j in range(len(present))))
    results["A2_arm0_macro_auroc"] = float(
        roc_auc_score(state, p3, multi_class="ovr", average="macro"))

    with open(os.path.join(OUT, f"gate2_state_detection_r{args.repeat}.json"),
              "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nwrote {OUT}/gate2_state_detection_r{args.repeat}.json")


if __name__ == "__main__":
    main()
