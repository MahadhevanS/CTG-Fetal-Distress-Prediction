"""Evaluate the delivered CRP ensemble on the held-out test set.

Loads every fold checkpoint in checkpoints/ctg_crossformer_crp/, averages the
per-fold probabilities, and reports metrics at both the naive 0.5 cut-off and
the deployed 0.30 operating point.
"""
import sys, glob, os
import numpy as np, torch

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.models.ctg_crossformer import CTGCrossformerEncoder, CTGCrossformerForClassification
from src.models.train_ctg_crossformer import compute_metrics

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = torch.load("data/processed_mil/test_dataset.pt", map_location="cpu", weights_only=False)
X, y = d["X"], d["y_primary"].numpy()
print(f"test set: {tuple(X.shape)}  positives {y.sum()}/{len(y)} ({y.mean()*100:.1f}%)  device {dev}")

probs = []
for ckpt in sorted(glob.glob("checkpoints/ctg_crossformer_crp/ctg_crossformer_fold_*_best.pth")):
    enc = CTGCrossformerEncoder(in_channels=2, seq_len=4800, cnn_channels=128,
                                n_heads_cross=4, n_heads_tf=8, n_tf_layers=4,
                                d_ff=512, dropout=0.1, latent_dim=128)
    m = CTGCrossformerForClassification(encoder=enc, hidden_dim=128, dropout=0.3).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            out.append(torch.sigmoid(m(X[i:i+64].to(dev))).cpu().numpy().ravel())
    p = np.concatenate(out)
    probs.append(p)
    print(f"  loaded {os.path.basename(ckpt)}  mean_prob={p.mean():.4f}")

ens = np.mean(probs, axis=0)
for thresh in (0.5, 0.30):
    print(f"\n--- threshold={thresh} ---")
    for k, v in compute_metrics(y, ens, threshold=thresh).items():
        print(f"  {k:<16}: {v:.4f}" if isinstance(v, float) else f"  {k:<16}: {v}")
