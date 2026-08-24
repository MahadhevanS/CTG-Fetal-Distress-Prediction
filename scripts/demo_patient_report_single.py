"""Same as demo_patient_report.py, but for one chosen patient only --
filters to that patient's windows before running the model, instead of
scoring the whole test set to then print one case."""
import os, sys
import numpy as np
import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.explainability.ctg_explainer import CTGExplainer
from src.explainability.patient_report import build_patient_report
from src.models.ctg_crossformer import CTGCrossformerEncoder, CTGCrossformerForClassification

CK_DIR = "checkpoints/ctg_crossformer_crp"
THRESHOLD = 0.30

PATIENT_ID = sys.argv[1] if len(sys.argv) > 1 else "2045"


def build(dev):
    e = CTGCrossformerEncoder(in_channels=2, seq_len=4800, cnn_channels=128, n_heads_cross=4,
                              n_heads_tf=8, n_tf_layers=4, d_ff=512, dropout=0.1, latent_dim=128)
    return CTGCrossformerForClassification(encoder=e, hidden_dim=128, dropout=0.3).to(dev)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load("data/processed_mil/test_dataset.pt", map_location="cpu", weights_only=False)
    pids = np.array([m[0] for m in d["metadata"]])

    sel = np.where(pids == PATIENT_ID)[0]
    if len(sel) == 0:
        print(f"Patient {PATIENT_ID} not in the test split. "
              f"Available IDs: {sorted(set(pids))[:15]} ...")
        return

    X, yf = d["X"][sel], d["y_features"][sel]
    ypat = d["y_patient"].numpy()[sel]
    starts = np.array([m[1] for m in d["metadata"]])[sel]

    ens = np.zeros(len(sel))
    for k in range(1, 6):
        m = build(dev)
        m.load_state_dict(torch.load(f"{CK_DIR}/ctg_crossformer_fold_{k}_best.pth",
                                     map_location=dev, weights_only=True))
        m.eval()
        with torch.no_grad():
            ens += torch.sigmoid(m(X.to(dev)).squeeze(-1)).cpu().numpy()
        del m

    ens /= 5.0

    m1 = build(dev)
    m1.load_state_dict(torch.load(f"{CK_DIR}/ctg_crossformer_fold_1_best.pth",
                                  map_location=dev, weights_only=True))
    with CTGExplainer(m1, dev, threshold=THRESHOLD) as ex:
        exps = [ex.explain(X[i], yf[i]) for i in range(len(sel))]
    for i in range(len(sel)):
        exps[i]["risk_score"] = float(ens[i])
        exps[i]["flagged"] = bool(ens[i] >= THRESHOLD)

    rep = build_patient_report(PATIENT_ID, exps, starts, THRESHOLD)
    outcome = "pH <= 7.15 (acidotic)" if ypat.max() == 1 else "pH > 7.15 (normal)"
    print(f"\n########## patient {PATIENT_ID} ##########")
    print(f"########## actual outcome: {outcome} ##########\n")
    print(rep.render())


if __name__ == "__main__":
    main()
