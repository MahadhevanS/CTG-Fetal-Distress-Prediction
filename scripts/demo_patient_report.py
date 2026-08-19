"""
Patient-level clinical reports from the DELIVERED model (CRP-pretrained).

Scores the held-out test set with the 5-fold ensemble, builds a per-labour
report, and prints worked examples: a true positive, a false positive, and a
true negative -- so the failure modes are visible alongside the successes.
"""
import os
import sys

import numpy as np
import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.explainability.ctg_explainer import CTGExplainer
from src.explainability.patient_report import build_patient_report
from src.models.ctg_crossformer import CTGCrossformerEncoder, CTGCrossformerForClassification

CK_DIR = "checkpoints/ctg_crossformer_crp"
THRESHOLD = 0.30   # ~80% sensitivity operating point on the delivered model


def build(dev):
    e = CTGCrossformerEncoder(in_channels=2, seq_len=4800, cnn_channels=128, n_heads_cross=4,
                              n_heads_tf=8, n_tf_layers=4, d_ff=512, dropout=0.1, latent_dim=128)
    return CTGCrossformerForClassification(encoder=e, hidden_dim=128, dropout=0.3).to(dev)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load("data/processed_mil/test_dataset.pt", map_location="cpu", weights_only=False)
    X, y, yf = d["X"], d["y_primary"].numpy(), d["y_features"]
    ypat = d["y_patient"].numpy()
    pids = np.array([m[0] for m in d["metadata"]])
    starts = np.array([m[1] for m in d["metadata"]])

    # Ensemble score across folds, then explanations from fold 1 (the ensemble
    # has no single set of weights to attribute through, so the narrative comes
    # from one member while the risk score is the ensemble's).
    ens = np.zeros(len(y))
    for k in range(1, 6):
        m = build(dev)
        m.load_state_dict(torch.load(f"{CK_DIR}/ctg_crossformer_fold_{k}_best.pth",
                                     map_location=dev, weights_only=True))
        m.eval()
        with torch.no_grad():
            for s in range(0, len(y), 64):
                c = np.arange(s, min(s + 64, len(y)))
                ens[c] += torch.sigmoid(m(X[c].to(dev)).squeeze(-1)).cpu().numpy()
        del m
        torch.cuda.empty_cache()
    ens /= 5.0

    m1 = build(dev)
    m1.load_state_dict(torch.load(f"{CK_DIR}/ctg_crossformer_fold_1_best.pth",
                                  map_location=dev, weights_only=True))
    with CTGExplainer(m1, dev, threshold=THRESHOLD) as ex:
        exps = [ex.explain(X[i], yf[i]) for i in range(len(y))]
    # Replace the single-model risk with the ensemble risk -- the reported score
    # must be the one the device would actually act on.
    for i in range(len(y)):
        exps[i]["risk_score"] = float(ens[i])
        exps[i]["flagged"] = bool(ens[i] >= THRESHOLD)

    # Pick illustrative patients
    uniq = sorted(set(pids))
    plabel = {p: int(ypat[pids == p].max()) for p in uniq}
    pmax = {p: float(ens[pids == p].max()) for p in uniq}
    pos = [p for p in uniq if plabel[p] == 1]
    neg = [p for p in uniq if plabel[p] == 0]

    picks = []
    if pos:
        picks.append(("TRUE POSITIVE - acidotic, highest peak risk",
                      max(pos, key=lambda p: pmax[p])))
        picks.append(("MISSED - acidotic, lowest peak risk",
                      min(pos, key=lambda p: pmax[p])))
    if neg:
        picks.append(("FALSE POSITIVE - normal outcome, highest peak risk",
                      max(neg, key=lambda p: pmax[p])))
        picks.append(("TRUE NEGATIVE - normal outcome, lowest peak risk",
                      min(neg, key=lambda p: pmax[p])))

    for title, p in picks:
        sel = np.where(pids == p)[0]
        rep = build_patient_report(str(p), [exps[i] for i in sel], starts[sel], THRESHOLD)
        outcome = "pH <= 7.15 (acidotic)" if plabel[p] == 1 else "pH > 7.15 (normal)"
        print(f"\n\n########## {title} ##########")
        print(f"########## actual outcome: {outcome} ##########\n")
        print(rep.render())


if __name__ == "__main__":
    main()
