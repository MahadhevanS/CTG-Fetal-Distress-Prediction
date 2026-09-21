"""
Frozen evaluation harness for all patient-level CTG experiments.

WHY THIS EXISTS
---------------
Until now every script built its own folds. `train_ctg_crossformer.py` uses
StratifiedGroupKFold over windows, `train_knowledge_infused.py` uses
create_patient_level_folds() joint-stratified on y_figo, and each probe in
scripts/ calls StratifiedKFold with its own seed. Those produce DIFFERENT
partitions, and per-fold AUROC on this data swings 0.72-0.91 -- far larger than
the effect sizes being compared. Any two numbers from two scripts are therefore
not strictly comparable, however carefully each was measured.

This module fixes one partition, writes it to disk, and hands the same fold
indices to every experiment. Numbers produced through it are comparable by
construction.

THE PROTOCOL (locked -- changing any of it invalidates cross-experiment claims)
------------------------------------------------------------------------------
1. Splits are PATIENT-GROUPED. A patient's windows never straddle a fold.
2. The primary metric is PATIENT-LEVEL AUROC, max-aggregated. Measured
   2026-09-03: max (0.7290) > p90 (0.7174) > mean (0.6741) on identical
   predictions, so max is the default aggregator.
3. Folds and seeds are identical across experiments, loaded from folds.json.
4. NO model selection on the reported fold. Anything needing a validation
   signal must carve it from the training patients (see inner_split).
5. Uncertainty is bootstrapped over PATIENTS, never windows. Resampling windows
   understates the interval by 2.7x because they overlap and cluster
   (scripts/audit_evaluation_protocol.py, `clustering` probe).

Usage:
    from src.training.protocol import Protocol
    P = Protocol.load_or_create(patient_ids, y_window)
    for fold, (tr_mask, te_patients) in enumerate(P.folds(), 1):
        ...
    P.report("my model", oof_window_probs)
"""
import json
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

DEFAULT_PATH = "data/processed_clinical/folds.json"
N_FOLDS = 5
N_REPEATS = 5
SEED = 0
N_BOOT = 2000


class Protocol:
    """A fixed patient-grouped partition plus the metrics computed on it."""

    def __init__(self, patient_ids: np.ndarray, y_window: np.ndarray,
                 assignment: Dict[str, List[int]], n_folds: int = N_FOLDS,
                 n_repeats: int = N_REPEATS, seed: int = SEED):
        self.pid = np.asarray(patient_ids)
        self.y = np.asarray(y_window)
        self.patients = np.array(sorted(assignment))
        self.pidx = {p: np.where(self.pid == p)[0] for p in self.patients}
        self.plab = np.array([int(self.y[self.pidx[p]].max()) for p in self.patients])
        self.assignment = assignment          # patient -> [fold per repeat]
        self.n_folds, self.n_repeats, self.seed = n_folds, n_repeats, seed
        self._rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- build
    @classmethod
    def load_or_create(cls, patient_ids, y_window, path: str = DEFAULT_PATH,
                       n_folds: int = N_FOLDS, n_repeats: int = N_REPEATS,
                       seed: int = SEED, force: bool = False) -> "Protocol":
        pid = np.asarray(patient_ids)
        y = np.asarray(y_window)
        if os.path.exists(path) and not force:
            with open(path) as fh:
                blob = json.load(fh)
            if (blob["n_folds"], blob["n_repeats"], blob["seed"]) != (n_folds, n_repeats, seed):
                raise ValueError(
                    f"{path} was built with n_folds={blob['n_folds']}, "
                    f"n_repeats={blob['n_repeats']}, seed={blob['seed']}, but "
                    f"({n_folds}, {n_repeats}, {seed}) was requested. Refusing to "
                    f"silently mix partitions -- pass force=True to rebuild, which "
                    f"invalidates every number measured against the old one.")
            missing = set(np.unique(pid)) - set(blob["assignment"])
            if missing:
                raise ValueError(f"{len(missing)} patient(s) absent from {path}, e.g. "
                                 f"{sorted(missing)[:3]}. The cohort changed; rebuild "
                                 f"with force=True and re-run every experiment.")
            return cls(pid, y, blob["assignment"], n_folds, n_repeats, seed)

        patients = np.array(sorted(set(pid)))
        plab = np.array([int(y[pid == p].max()) for p in patients])
        assignment = {p: [] for p in patients}
        for rep in range(n_repeats):
            skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed + rep)
            for k, (_, te) in enumerate(skf.split(patients, plab)):
                for p in patients[te]:
                    assignment[p].append(k)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"n_folds": n_folds, "n_repeats": n_repeats, "seed": seed,
                       "n_patients": len(patients), "n_positive": int(plab.sum()),
                       "assignment": {str(k): v for k, v in assignment.items()}}, fh, indent=1)
        print(f"[protocol] wrote {path}: {len(patients)} patients, "
              f"{int(plab.sum())} positive, {n_folds}x{n_repeats} folds, seed {seed}")
        return cls(pid, y, assignment, n_folds, n_repeats, seed)

    # ---------------------------------------------------------------- folds
    def folds(self, repeat: Optional[int] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_window_mask, test_patient_array) for each fold.

        repeat=None walks every repeat; pass an int for a single pass (use this
        for anything expensive, e.g. training a network per fold).
        """
        reps = range(self.n_repeats) if repeat is None else [repeat]
        for rep in reps:
            for k in range(self.n_folds):
                te_p = np.array([p for p in self.patients if self.assignment[p][rep] == k])
                te_set = set(te_p.tolist())
                tr_mask = np.array([q not in te_set for q in self.pid])
                yield tr_mask, te_p

    def inner_split(self, tr_mask: np.ndarray, frac: float = 0.2, seed: int = 42):
        """Carve a validation set out of TRAINING patients only.

        Rule 4 of the protocol: never select on the reported fold. This is the
        only sanctioned way to get a validation signal.
        """
        tr_p = np.array(sorted(set(self.pid[tr_mask])))
        lab = np.array([int(self.y[self.pidx[p]].max()) for p in tr_p])
        strat = lab if len(np.unique(lab)) > 1 and np.bincount(lab).min() >= 2 else None
        fit_p, val_p = train_test_split(tr_p, test_size=frac, stratify=strat,
                                        random_state=seed)
        val_set = set(val_p.tolist())
        is_val = np.array([q in val_set for q in self.pid]) & tr_mask
        return tr_mask & ~is_val, is_val

    # -------------------------------------------------------------- scoring
    def to_patient(self, window_probs: np.ndarray, patients: Optional[np.ndarray] = None,
                   how: str = "max") -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate window probabilities into (labels, scores) per patient."""
        ps = self.patients if patients is None else patients
        fn = {"max": np.max, "mean": np.mean,
              "p90": lambda v: np.percentile(v, 90)}[how]
        scores = np.array([fn(window_probs[self.pidx[p]]) for p in ps])
        labels = np.array([int(self.y[self.pidx[p]].max()) for p in ps])
        return labels, scores

    def bootstrap_ci(self, labels: np.ndarray, scores: np.ndarray,
                     n_boot: int = N_BOOT) -> Tuple[float, float]:
        """95% CI resampled over PATIENTS. Never resample windows -- see rule 5."""
        rng = np.random.default_rng(self.seed)
        vals = []
        idx = np.arange(len(labels))
        for _ in range(n_boot):
            b = rng.choice(idx, len(idx), replace=True)
            if len(np.unique(labels[b])) < 2:
                continue
            vals.append(roc_auc_score(labels[b], scores[b]))
        return tuple(np.percentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan)

    def report(self, name: str, oof_window_probs: np.ndarray,
               how: str = "max", verbose: bool = True) -> Dict[str, float]:
        """Patient-level metrics on out-of-fold window probabilities."""
        lab, sc = self.to_patient(oof_window_probs, how=how)
        auroc = roc_auc_score(lab, sc)
        auprc = average_precision_score(lab, sc)
        lo, hi = self.bootstrap_ci(lab, sc)
        out = {"name": name, "agg": how, "auroc": auroc, "auprc": auprc,
               "ci_lo": lo, "ci_hi": hi, "n_patients": len(lab),
               "n_positive": int(lab.sum())}
        if verbose:
            print(f"  {name:44s} AUROC {auroc:.4f} [{lo:.3f}-{hi:.3f}]  AUPRC {auprc:.4f}")
        return out

    def report_patient_scores(self, name: str, labels: np.ndarray, scores: np.ndarray,
                              verbose: bool = True) -> Dict[str, float]:
        """Same, for models that produce one score per patient directly."""
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)
        lo, hi = self.bootstrap_ci(labels, scores)
        out = {"name": name, "agg": "native", "auroc": auroc, "auprc": auprc,
               "ci_lo": lo, "ci_hi": hi, "n_patients": len(labels),
               "n_positive": int(labels.sum())}
        if verbose:
            print(f"  {name:44s} AUROC {auroc:.4f} [{lo:.3f}-{hi:.3f}]  AUPRC {auprc:.4f}")
        return out

    # ---------------------------------------------------------- oof caching
    @staticmethod
    def cache_path(tag: str, cache_dir: str = "results/oof_cache") -> str:
        return os.path.join(cache_dir, f"{tag}.npy")

    @staticmethod
    def load_oof(tag: str, cache_dir: str = "results/oof_cache") -> Optional[np.ndarray]:
        p = Protocol.cache_path(tag, cache_dir)
        return np.load(p) if os.path.exists(p) else None

    @staticmethod
    def save_oof(tag: str, probs: np.ndarray, cache_dir: str = "results/oof_cache"):
        os.makedirs(cache_dir, exist_ok=True)
        np.save(Protocol.cache_path(tag, cache_dir), probs)


def load_clinical(data_dir: str = "data/processed_clinical/"):
    """Windows, 19 features, labels, patient ids -- the standard experiment input."""
    import torch
    X, Fe, y, pid = [], [], [], []
    for s in ("train", "val", "test"):
        d = torch.load(os.path.join(data_dir, f"{s}_dataset.pt"), weights_only=False)
        X.append(d["X"].numpy())
        Fe.append(np.hstack([d["y_features"].numpy(),
                             np.load(os.path.join(data_dir, f"{s}_extended_features.npy"))]))
        y.append(d["y_primary"].numpy())
        pid.append(np.array([m[0] for m in d["metadata"]]))
    return np.vstack(X), np.vstack(Fe), np.concatenate(y), np.concatenate(pid)


# Feature-group index map, for the Step-5 ablations. The first 8 are the FIGO
# features the pipeline computes; the remaining 11 are extended_features.py.
FEATURE_GROUPS = {
    "baseline": [0],
    "variability": [1, 2],
    "accelerations": [3],
    "decelerations": [4, 5, 6, 7],
    "extended": list(range(8, 19)),
}
