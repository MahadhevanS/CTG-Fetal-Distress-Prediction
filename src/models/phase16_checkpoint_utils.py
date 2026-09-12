"""
Resumable-training checkpoint utilities for Phase 16.

Same pattern as this project's own prior convention (append-only JSONL of
completed units, skip anything already recorded on restart, one model
checkpoint file per unit): each trained scorer is a "unit" identified by a
string id (e.g. "Model_2_magnitude_only_fold_0"). Completing a unit means:
(1) save its state_dict to <checkpoint_dir>/<unit_id>.pt, (2) append one
line to <checkpoint_dir>/progress.jsonl recording it's done. On restart,
any unit already in progress.jsonl has its model reloaded from disk instead
of being retrained.
"""

import json
import os
import torch

from src.models.phase16_causal_attention import AttentionScorer


def progress_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, "progress.jsonl")


def load_completed_units(checkpoint_dir):
    """Returns {unit_id: record_dict} for every unit already marked done."""
    path = progress_path(checkpoint_dir)
    completed = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                completed[rec["unit_id"]] = rec
    return completed


def mark_unit_complete(checkpoint_dir, unit_id, scorer, extra_fields=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"{unit_id}.pt")
    torch.save(scorer.state_dict(), model_path)

    record = {"unit_id": unit_id, "model_path": model_path}
    if extra_fields:
        record.update(extra_fields)
    with open(progress_path(checkpoint_dir), "a") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def load_scorer_checkpoint(record, in_dim, hidden=8):
    scorer = AttentionScorer(in_dim, hidden=hidden)
    scorer.load_state_dict(torch.load(record["model_path"], weights_only=True))
    scorer.eval()
    return scorer


def get_or_train(checkpoint_dir, unit_id, in_dim, hidden, train_fn):
    """
    train_fn: zero-arg callable returning (scorer, val_loss, n_epochs) --
    only called if unit_id is not already checkpointed. Returns
    (scorer, val_loss, n_epochs, was_resumed: bool).
    """
    completed = load_completed_units(checkpoint_dir)
    if unit_id in completed:
        rec = completed[unit_id]
        scorer = load_scorer_checkpoint(rec, in_dim, hidden)
        print(f"  [resumed from checkpoint] {unit_id}  "
              f"(previously: {rec.get('n_epochs', '?')} epochs, val_loss={rec.get('val_loss', '?')})")
        return scorer, rec.get("val_loss"), rec.get("n_epochs"), True

    scorer, val_loss, n_epochs = train_fn()
    mark_unit_complete(checkpoint_dir, unit_id, scorer, extra_fields={"val_loss": val_loss, "n_epochs": n_epochs})
    return scorer, val_loss, n_epochs, False
