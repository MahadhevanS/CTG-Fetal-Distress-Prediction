"""
Demonstrate + VALIDATE the clinical explainability module on held-out test data.

Runs the DELIVERED model (clinical-relational pretrained, "CRP") over the test
set, produces an explanation for every window, prints worked examples, and then
reports whether the explanations are FAITHFUL -- i.e. whether the FIGO findings
the narrative cites actually track the model's risk score, or are merely
plausible-looking decoration presented alongside an unrelated number.

Defaults to fold 1 of checkpoints/ctg_crossformer_crp/ (seed 42, the delivered
CRP run), which reproduces rho=+0.405. Pass --ckpt to explain a different
model; the pre-CRP inverse_freq baseline reproduces rho=+0.461, e.g.

    python scripts/demo_explainability.py \
        --ckpt checkpoints/ctg_crossformer_invfreq/ctg_crossformer_fold_1_best.pth \
        --thresh 0.365

NOTE ON --thresh: it affects the narrative text only ("-> flagged" vs
"-> not flagged") and does NOT affect the headline rho, which is a Spearman
correlation over raw risk scores. faithfulness_report() additionally splits
mean_concerning_when_flagged at a fixed 0.5 internally, independent of this
flag. This is a SINGLE-FOLD score, so it is not on the same scale as the 0.30
operating point used by the 5-fold ensemble in demo_inference.py -- per-fold
cut-points for this model span 0.0004 to 0.4951 for the same sensitivity. A
per-fold operating point should be derived on validation data, not here.
"""
import argparse
import os, sys
import numpy as np, torch

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.models.ctg_crossformer import CTGCrossformerEncoder, CTGCrossformerForClassification
from src.explainability.ctg_explainer import CTGExplainer, faithfulness_report

CKPT = "checkpoints/ctg_crossformer_crp/ctg_crossformer_fold_1_best.pth"
THRESH = 0.30   # narrative display only -- see NOTE ON --thresh in the docstring


def build(dev):
    e = CTGCrossformerEncoder(in_channels=2, seq_len=4800, cnn_channels=128, n_heads_cross=4,
                              n_heads_tf=8, n_tf_layers=4, d_ff=512, dropout=0.1, latent_dim=128)
    return CTGCrossformerForClassification(encoder=e, hidden_dim=128, dropout=0.3).to(dev)


def main(ckpt=None, thresh=None):
    ckpt = ckpt or CKPT
    thresh = THRESH if thresh is None else thresh
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load("data/processed_mil/test_dataset.pt", map_location="cpu", weights_only=False)
    X, y, yf = d["X"], d["y_primary"].numpy(), d["y_features"]
    meta = d["metadata"]

    print(f"model     : {ckpt}")
    print(f"threshold : {thresh}  (narrative display only)")

    model = build(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))

    exps = []
    with CTGExplainer(model, dev, threshold=thresh) as ex:
        for i in range(len(y)):
            exps.append(ex.explain(X[i], yf[i]))

        risk = np.array([e["risk_score"] for e in exps])
        print("=" * 74)
        print(" WORKED EXAMPLES (held-out test set)")
        print("=" * 74)
        # highest-risk true positive, and highest-risk false positive
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        picks = [("TRUE POSITIVE (highest risk)", pos[np.argmax(risk[pos])]),
                 ("FALSE POSITIVE (highest risk negative)", neg[np.argmax(risk[neg])]),
                 ("LOWEST-RISK POSITIVE", pos[np.argmin(risk[pos])])]
        for title, i in picks:
            pid, st, _ = meta[i]
            print(f"\n--- {title} | patient {pid}, window @ {st/(4*60):.0f} min ---")
            print(ex.narrate(exps[i]))

    print("\n" + "=" * 74)
    print(" FAITHFULNESS CHECK -- are the explanations real, or decoration?")
    print("=" * 74)
    rep = faithfulness_report(exps, y)
    for k, v in rep.items():
        print(f"  {k:<42}: {v}")
    rho = rep["spearman_risk_vs_concerning_criteria"]
    print()
    if abs(rho) < 0.1:
        print("  VERDICT: correlation is near zero -- the narrative and the risk score")
        print("  are describing DIFFERENT things. The FIGO findings must NOT be presented")
        print("  as the model's reason for firing.")
    elif rho > 0:
        print(f"  VERDICT: risk score rises with concerning FIGO findings (rho={rho:+.3f}).")
        print("  The narrative is consistent with the model's behaviour.")
    else:
        print(f"  VERDICT: risk score FALLS as concerning findings rise (rho={rho:+.3f}) --")
        print("  the explanation contradicts the model. Do not ship as-is.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=CKPT, help="checkpoint to explain (default: delivered CRP fold 1)")
    ap.add_argument("--thresh", type=float, default=THRESH, help="narrative display threshold")
    a = ap.parse_args()
    main(a.ckpt, a.thresh)
