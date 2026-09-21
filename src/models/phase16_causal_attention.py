"""
Phase 16 (docs/phase16_protocol.md) -- causal attention aggregator over the
frozen P6 window-level score sequence.

x_t = (r_t,) for Model 2 or (r_t, elapsed_t/60) for Model 3. Deliberately
tiny (fixed hidden=8, one hidden layer) per protocol Section 0(a)/(c) --
this is not meant to be tuned, it is meant to be the lowest-capacity model
that can express "learn which windows matter."

Trained via causal chronological truncation (protocol Section 4): every
prefix length of a patient's window sequence is a separate training
example sharing that patient's outcome label, which is what makes one
trained-per-fold model usable at every evaluation horizon.

Implementation note: the attention LOGIT e_t = f(x_t) depends only on
window t's own (r_t, elapsed_t) -- never on the truncation point k. So the
MLP scorer is run once per patient over the full sequence to get
e_1..e_T, and each truncation k=1..T reuses those same logits, taking a
softmax over only the first k (a cheap masked-softmax, not a new MLP call).
This is mathematically identical to scoring each prefix independently, just
without the O(T) redundant forward passes that would otherwise cost this
project real wall-clock time for no reason.
"""

import numpy as np
import torch
import torch.nn as nn


class AttentionScorer(nn.Module):
    def __init__(self, in_dim, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (T, in_dim) -> (T,) attention logits
        return self.net(x).squeeze(-1)


def build_input(r_seq, elapsed_seq, use_elapsed):
    if use_elapsed:
        return torch.stack([r_seq, elapsed_seq / 60.0], dim=-1)
    return r_seq.unsqueeze(-1)


def score_full_sequence(scorer, r_seq, elapsed_seq, use_elapsed):
    """One MLP forward pass over the whole sequence -> logits e_1..e_T."""
    x = build_input(r_seq, elapsed_seq, use_elapsed)
    return scorer(x)


def pooled_prediction_from_logits(logits, r_seq, k):
    """z_k = sum_{t<=k} softmax(e_1..e_k)_t * r_t -- cheap, no new MLP call."""
    e_k = logits[:k]
    r_k = r_seq[:k]
    alpha = torch.softmax(e_k, dim=0)
    z = torch.sum(alpha * r_k)
    return z, alpha


def patient_avg_bce_loss(scorer, r_seq, elapsed_seq, use_elapsed, y, T_i):
    """Per-patient loss: BCE averaged across ALL truncation lengths k=1..T_i,
    so a patient with 17 windows contributes the same total gradient weight
    as a patient with 3 -- the same per-patient normalization principle
    established in Phase 13's information-density work, applied here to
    avoid longer recordings dominating training."""
    logits = score_full_sequence(scorer, r_seq, elapsed_seq, use_elapsed)
    losses = []
    for k in range(1, T_i + 1):
        z, _ = pooled_prediction_from_logits(logits, r_seq, k)
        z_clamped = torch.clamp(z, 1e-6, 1 - 1e-6)
        loss = -(y * torch.log(z_clamped) + (1 - y) * torch.log(1 - z_clamped))
        losses.append(loss)
    return torch.stack(losses).mean()


def train_scorer(train_patients, inner_val_patients, patient_data, use_elapsed,
                  hidden=8, lr=0.01, weight_decay=1e-4, max_epochs=200, patience=10, seed=42):
    torch.manual_seed(seed)
    in_dim = 2 if use_elapsed else 1
    scorer = AttentionScorer(in_dim, hidden=hidden)
    optimizer = torch.optim.Adam(scorer.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    n_epochs_run = 0

    for epoch in range(max_epochs):
        n_epochs_run = epoch + 1
        scorer.train()
        optimizer.zero_grad()
        train_losses = []
        for pid in train_patients:
            r_seq, elapsed_seq, y, T_i = patient_data[pid]
            train_losses.append(patient_avg_bce_loss(scorer, r_seq, elapsed_seq, use_elapsed, y, T_i))
        loss = torch.stack(train_losses).mean()
        loss.backward()
        optimizer.step()

        scorer.eval()
        with torch.no_grad():
            val_losses = []
            for pid in inner_val_patients:
                r_seq, elapsed_seq, y, T_i = patient_data[pid]
                val_losses.append(patient_avg_bce_loss(scorer, r_seq, elapsed_seq, use_elapsed, y, T_i))
            val_loss = torch.stack(val_losses).mean().item() if val_losses else loss.item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in scorer.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        scorer.load_state_dict(best_state)
    return scorer, best_val_loss, n_epochs_run


def eligible_prefix_length(t_del_seq, h_val, T_i):
    """
    t_del_seq: chronologically-ordered (ascending elapsed time = descending
    time_before_delivery) array of length T_i. Causally-eligible windows at
    horizon h are {i : t_del_i >= h_val} -- since t_del_seq is monotonically
    descending in chronological order, that set is exactly the prefix
    t_del_seq[:k]. Falls back to the full sequence if none qualify (short
    recording), matching the convention used throughout Phases 13-15.
    """
    if h_val <= 0:
        return T_i
    k = int(np.sum(t_del_seq >= h_val))
    return k if k > 0 else T_i


def predict_all_prefixes(scorer, r_seq, elapsed_seq, use_elapsed, T_i):
    """z_1..z_T for one patient, one MLP forward pass -- used for the rolling
    causal lead-time simulation (protocol Section 7 operational metrics)."""
    scorer.eval()
    with torch.no_grad():
        logits = score_full_sequence(scorer, r_seq, elapsed_seq, use_elapsed)
        zs = []
        for k in range(1, T_i + 1):
            z, _ = pooled_prediction_from_logits(logits, r_seq, k)
            zs.append(float(z.item()))
    return np.array(zs)


def predict_at_horizon_for_patients(scorer, patients, patient_data_by_source, t_del_by_source, use_elapsed, h_val):
    """Returns dict {pid: (prediction, alpha_weights, window_index_argmax_alpha, window_index_argmax_r)}."""
    scorer.eval()
    out = {}
    with torch.no_grad():
        for pid in patients:
            r_seq, elapsed_seq, y, T_i = patient_data_by_source[pid]
            t_del_seq = t_del_by_source[pid]
            k = eligible_prefix_length(t_del_seq, h_val, T_i)
            logits = score_full_sequence(scorer, r_seq, elapsed_seq, use_elapsed)
            z, alpha = pooled_prediction_from_logits(logits, r_seq, k)
            alpha_np = alpha.numpy()
            r_np = r_seq[:k].numpy()
            out[pid] = (float(z.item()), alpha_np, int(np.argmax(alpha_np)), int(np.argmax(r_np)))
    return out
