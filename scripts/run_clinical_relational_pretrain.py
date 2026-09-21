"""
Run clinical-relational pretraining, then hand the encoder to fine-tuning.

Trains ONLY on the CV pool (train_dataset.pt). The validation and test splits
are never seen during pretraining, so no information from them can leak into the
representation -- important because the pretraining objective needs no outcome
labels and would otherwise be tempting to run on everything.

Saves a bare encoder state_dict, loadable by train_ctg_crossformer.py with
--pretrained_encoder. Works with any encoder in src/models/encoder_registry.py
(crossformer, crossformer_latent, cnn1d, mslstm) -- the relational objective
only requires a (B,2,4800) -> (B,128) encoder, nothing CrossFormer-specific.

The SAME --encoder must be passed to both this script and the fine-tuner, or
the state_dict will not load.
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.models.encoder_registry import ENCODER_NAMES, build_encoder, param_count
from src.training.clinical_relational_pretrain import (
    ClinicalRelationalPretrainer, build_clinical_space, relational_loss,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_mil/")
    ap.add_argument("--out", default="checkpoints/clinical_relational/encoder_pretrained.pth")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--in_channels", type=int, default=2, choices=[2, 3],
                    help="must match the --in_channels used for fine-tuning")
    ap.add_argument("--encoder", default="crossformer", choices=ENCODER_NAMES,
                    help="temporal encoder to pretrain. The saved state_dict must be "
                         "loaded back with the SAME --encoder in train_ctg_crossformer.py.")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    d = torch.load(os.path.join(a.data_dir, "train_dataset.pt"), map_location="cpu", weights_only=False)
    # data/processed_clinical ships 3 channels (FHR, UC, missingness mask); the
    # encoders are built for --in_channels, so slice to match.
    X = d["X"][:, :a.in_channels, :]
    yf = d["y_features"].numpy()
    ext = np.load(os.path.join(a.data_dir, "train_extended_features.npy"))
    pids = np.array([m[0] for m in d["metadata"]])

    clin, names = build_clinical_space(yf, ext)
    print(f"[pretrain] {len(X)} windows | clinical space {clin.shape[1]}-dim")
    print(f"[pretrain] features: {', '.join(names)}")

    # Patient-level holdout purely to watch for overfitting of the objective
    uniq = np.array(sorted(set(pids)))
    rng = np.random.default_rng(a.seed)
    ho = set(rng.choice(uniq, size=max(int(len(uniq) * a.holdout_frac), 1), replace=False).tolist())
    is_ho = np.array([p in ho for p in pids])
    tr_idx, ho_idx = np.where(~is_ho)[0], np.where(is_ho)[0]
    print(f"[pretrain] fit {len(tr_idx)} windows | holdout {len(ho_idx)} ({len(ho)} patients)")

    clin_t = torch.tensor(clin)
    train_ds = TensorDataset(X[tr_idx], clin_t[tr_idx])
    ho_ds = TensorDataset(X[ho_idx], clin_t[ho_idx])
    # drop_last: a final tiny batch gives a near-empty distance matrix and a
    # very noisy relational target.
    train_ld = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, drop_last=True)
    ho_ld = DataLoader(ho_ds, batch_size=a.batch_size, shuffle=False, drop_last=True)

    enc = build_encoder(a.encoder, m_cfg={"latent_dim": 128})
    print(f"[pretrain] encoder: {a.encoder} ({param_count(enc):,} params)")
    model = ClinicalRelationalPretrainer(encoder=enc).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=max(a.epochs * max(len(train_ld), 1), 10), pct_start=0.1)
    amp = torch.amp.GradScaler(dev.type) if dev.type == "cuda" else None

    best, best_ep = float("inf"), -1
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = n = 0
        for xb, cb in train_ld:
            xb, cb = xb.to(dev, non_blocking=True), cb.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=amp is not None):
                loss = relational_loss(model(xb), cb)
            if amp is not None:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp.step(opt); amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            tot += float(loss.item()); n += 1

        model.eval()
        h = hn = 0
        with torch.no_grad():
            for xb, cb in ho_ld:
                xb, cb = xb.to(dev), cb.to(dev)
                with torch.autocast(device_type=dev.type, enabled=amp is not None):
                    h += float(relational_loss(model(xb), cb).item()); hn += 1
        ho_loss = h / max(hn, 1)
        star = ""
        if ho_loss < best:
            best, best_ep = ho_loss, ep
            torch.save(model.encoder.state_dict(), a.out)
            star = " *"
        if ep % 5 == 0 or ep == 1 or star:
            print(f"  epoch {ep:>3}/{a.epochs} | train {tot/max(n,1):.5f} | holdout {ho_loss:.5f}{star}")

    print(f"\n[pretrain] best holdout {best:.5f} at epoch {best_ep}")
    print(f"[pretrain] encoder saved -> {a.out}")
    print("[pretrain] next: train_ctg_crossformer.py --pretrained_encoder " + a.out)


if __name__ == "__main__":
    main()
