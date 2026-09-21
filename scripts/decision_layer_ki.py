"""
Decision-layer knowledge infusion for the inverse_freq deliverable model.

MOTIVATION. Seven training-time knowledge-infusion mechanisms delivered ~0,
because the network already learns what the FIGO rules encode (7 binary flags
alone reach AUROC 0.762 vs the network's 0.777). But a screening device does not
operate at "average AUROC" -- it operates at HIGH SENSITIVITY, where it flags
~52% of windows and precision is the binding problem. Knowledge may still help
THERE even though it does not help on the global average.

So this evaluates knowledge fusion against the decision-relevant metrics --
specificity at fixed sensitivity, and precision at the operating point -- rather
than AUROC.

Schemes:
  (a) model alone                      -- baseline
  (b) LR(model, 7 FIGO flags)          -- learned combination
  (c) suppress-if-clean                -- damp the score when NO concerning FIGO
                                          criterion fired (clinically: the model
                                          is alarmed but the trace shows no
                                          recognised pathology)
  (d) escalate-if-pathological         -- boost the score when >=2 concerning
                                          criteria fired

(c) and (d) are interpretable rules a clinician can audit, which matters more
for a device than a marginal metric gain from a black-box combiner.

Developed on CV out-of-fold predictions; the held-out test set is touched ONCE
at the end, for the single scheme selected on CV.
"""
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.knowledge.figo import derive_figo_criteria_flags_torch
from src.models.ctg_crossformer import CTGCrossformerEncoder, CTGCrossformerForClassification
from src.training.train import create_patient_level_folds

CKPT_DIR = "checkpoints/ctg_crossformer_invfreq"
# Indices into FIGO_CRITERIA_NAMES that are CONCERNING (the rest are reassuring)
CONCERNING_IDX = [0, 2, 4, 5, 6]  # baseline_low, baseline_high, late/variable/prolonged decel


def build(dev):
    e = CTGCrossformerEncoder(in_channels=2, seq_len=4800, cnn_channels=128, n_heads_cross=4,
                              n_heads_tf=8, n_tf_layers=4, d_ff=512, dropout=0.1, latent_dim=128)
    return CTGCrossformerForClassification(encoder=e, hidden_dim=128, dropout=0.3).to(dev)


@torch.no_grad()
def score_windows(model, X, idx, dev, bs=64):
    out = np.zeros(len(idx), dtype=np.float32)
    for s in range(0, len(idx), bs):
        c = idx[s:s + bs]
        out[s:s + len(c)] = torch.sigmoid(model(X[c].to(dev)).squeeze(-1)).cpu().numpy()
    return out


def spec_at_sens(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    i = np.where(tpr >= target)[0]
    return float(1.0 - fpr[i].min()) if len(i) else float("nan")


def prec_at_sens(y, s, target):
    fpr, tpr, thr = roc_curve(y, s)
    i = np.where(tpr >= target)[0]
    if not len(i):
        return float("nan")
    t = thr[i[0]]
    pred = s >= t
    return float(y[pred].mean()) if pred.any() else float("nan")


def evaluate(y, s, name):
    return {
        "scheme": name,
        "auroc": roc_auc_score(y, s),
        "auprc": average_precision_score(y, s),
        "spec@90sens": spec_at_sens(y, s, 0.90),
        "spec@80sens": spec_at_sens(y, s, 0.80),
        "prec@80sens": prec_at_sens(y, s, 0.80),
    }


def print_table(rows, title):
    print(f"\n{'=' * 88}\n {title}\n{'=' * 88}")
    print(f"  {'scheme':<28}{'AUROC':<9}{'AUPRC':<9}{'spec@90s':<11}{'spec@80s':<11}{'prec@80s':<10}")
    print("  " + "-" * 84)
    for r in rows:
        print(f"  {r['scheme']:<28}{r['auroc']:<9.4f}{r['auprc']:<9.4f}"
              f"{r['spec@90sens']:<11.4f}{r['spec@80sens']:<11.4f}{r['prec@80sens']:<10.4f}")


def build_schemes(prob, flags, y=None, fit_idx=None, apply_idx=None,
                  suppress=0.35, escalate=1.25):
    """Returns dict of scheme-name -> scores on apply_idx."""
    ncon = flags[:, CONCERNING_IDX].sum(axis=1)
    out = {"(a) model alone": prob[apply_idx]}

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    F = np.column_stack([prob, flags])
    lr.fit(F[fit_idx], y[fit_idx])
    out["(b) LR(model + FIGO flags)"] = lr.predict_proba(F[apply_idx])[:, 1]

    s = prob.copy()
    s[ncon == 0] *= suppress
    out["(c) suppress-if-clean"] = s[apply_idx]

    s2 = prob.copy()
    s2[ncon >= 2] = np.clip(s2[ncon >= 2] * escalate, 0, 1)
    out["(d) escalate-if-pathological"] = s2[apply_idx]

    s3 = prob.copy()
    s3[ncon == 0] *= suppress
    s3[ncon >= 2] = np.clip(s3[ncon >= 2] * escalate, 0, 1)
    out["(e) suppress + escalate"] = s3[apply_idx]
    return out


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = torch.load("data/processed_mil/train_dataset.pt", map_location="cpu", weights_only=False)
    X, y, yf = d["X"], d["y_primary"].numpy(), d["y_features"]
    pids = np.array([m[0] for m in d["metadata"]])
    folds = create_patient_level_folds(list(pids), torch.as_tensor(y), k_folds=5,
                                       secondary_labels=d["y_figo"])
    flags = derive_figo_criteria_flags_torch(yf.float()).numpy()

    prob = np.full(len(y), np.nan, np.float32)
    for k, (_, vi) in enumerate(folds, 1):
        m = build(dev)
        m.load_state_dict(torch.load(f"{CKPT_DIR}/ctg_crossformer_fold_{k}_best.pth",
                                     map_location=dev, weights_only=True))
        m.eval()
        prob[vi] = score_windows(m, X, vi, dev)
        del m
        torch.cuda.empty_cache()
    assert not np.isnan(prob).any()

    # Per-fold evaluation so nothing is fit and scored on the same windows
    acc = {}
    for tr, va in folds:
        sch = build_schemes(prob, flags, y=y, fit_idx=tr, apply_idx=va)
        for name, s in sch.items():
            acc.setdefault(name, []).append(evaluate(y[va], s, name))
    rows = []
    for name, lst in acc.items():
        rows.append({"scheme": name,
                     **{k: float(np.mean([r[k] for r in lst]))
                        for k in ("auroc", "auprc", "spec@90sens", "spec@80sens", "prec@80sens")}})
    print_table(rows, "CROSS-VALIDATION (per-fold mean) -- development")

    base = next(r for r in rows if r["scheme"].startswith("(a)"))
    print(f"\n  deltas vs model alone (spec@80sens is the device-relevant one):")
    best, best_gain = None, -1e9
    for r in rows:
        if r["scheme"].startswith("(a)"):
            continue
        g = r["spec@80sens"] - base["spec@80sens"]
        print(f"    {r['scheme']:<30} spec@80s {g:+.4f}   spec@90s "
              f"{r['spec@90sens']-base['spec@90sens']:+.4f}   AUROC {r['auroc']-base['auroc']:+.4f}")
        if g > best_gain:
            best, best_gain = r["scheme"], g
    print(f"\n  -> best on CV by spec@80sens: {best} ({best_gain:+.4f})")


if __name__ == "__main__":
    main()
