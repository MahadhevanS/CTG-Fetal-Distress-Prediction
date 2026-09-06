"""
Frozen evaluation harness for the FIGO-state / early-warning experiments.

This is a thin re-implementation of src/training/protocol.py's contract for a
DIFFERENT cohort and a DIFFERENT unit of analysis, kept separate on purpose.
src/training/protocol.py is pinned to data/processed_clinical/folds.json --
547 patients, 8,517 overlapping 20-minute windows, patient-level max
aggregation, pH outcome. Reusing that partition here would silently mix two
cohorts; its own load_or_create() raises rather than allow it. So this module
writes data/processed_figo/folds.json and nothing else touches it.

THE RULES, UNCHANGED FROM THE ORIGINAL PROTOCOL
-----------------------------------------------
1. Splits are PATIENT-GROUPED. A patient's epochs never straddle a fold.
2. Folds and seeds are fixed on disk and identical across experiments.
3. NO model selection on the reported fold. Anything needing a validation
   signal carves it from the training patients (inner_split).
4. Uncertainty is bootstrapped over PATIENTS, never epochs.

WHAT IS DIFFERENT, AND WHY
--------------------------
* The primary unit is the EPOCH, not the patient. FIGO state is a property of
  a 10-minute epoch, and "does this patient ever go pathological" is a
  different question that max-aggregation answers. Both are reported;
  epoch-level is primary for state detection, and EVENT-level is primary for
  early warning (see report_event_level).
* Epochs do not overlap, so unlike the 20-minute/2.5-minute-stride substrate
  there are no near-duplicate rows to leak across a fold boundary. Bootstrap
  over patients is still the correct interval because epochs within a patient
  remain correlated.
* Stratification is on the patient's WORST state, so the 34 patients who ever
  reach Pathological are spread evenly. With so few, an unstratified split
  would put wildly different counts in different folds.
"""

import json
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split

DEFAULT_PATH = "data/processed_figo/folds.json"
N_FOLDS = 5
N_REPEATS = 5
SEED = 0
N_BOOT = 2000


class FigoProtocol:
    """A fixed patient-grouped partition over epochs."""

    def __init__(self, patient_ids: np.ndarray, strat_label: np.ndarray,
                 assignment: Dict[str, List[int]],
                 n_folds: int = N_FOLDS, n_repeats: int = N_REPEATS,
                 seed: int = SEED):
        self.pid = np.asarray(patient_ids).astype(str)
        self.patients = np.array(sorted(assignment))
        self.pidx = {p: np.where(self.pid == p)[0] for p in self.patients}
        self.strat = np.asarray(strat_label)
        self.assignment = assignment
        self.n_folds, self.n_repeats, self.seed = n_folds, n_repeats, seed

    # ---------------------------------------------------------------- build
    @classmethod
    def load_or_create(cls, patient_ids, patient_strat: Dict[str, int],
                       path: str = DEFAULT_PATH, n_folds: int = N_FOLDS,
                       n_repeats: int = N_REPEATS, seed: int = SEED,
                       force: bool = False) -> "FigoProtocol":
        pid = np.asarray(patient_ids).astype(str)
        if os.path.exists(path) and not force:
            blob = json.load(open(path))
            if (blob["n_folds"], blob["n_repeats"], blob["seed"]) != (
                    n_folds, n_repeats, seed):
                raise ValueError(
                    f"{path} was built with n_folds={blob['n_folds']}, "
                    f"n_repeats={blob['n_repeats']}, seed={blob['seed']}, but "
                    f"({n_folds}, {n_repeats}, {seed}) was requested. Refusing "
                    f"to mix partitions -- pass force=True to rebuild, which "
                    f"invalidates every number measured against the old one.")
            missing = set(np.unique(pid)) - set(blob["assignment"])
            if missing:
                raise ValueError(
                    f"{len(missing)} patient(s) absent from {path}, e.g. "
                    f"{sorted(missing)[:3]}. The cohort changed; rebuild with "
                    f"force=True and re-run every experiment.")
            return cls(pid, np.array([patient_strat.get(p, 0) for p in pid]),
                       blob["assignment"], n_folds, n_repeats, seed)

        patients = np.array(sorted(set(pid)))
        lab = np.array([patient_strat.get(p, 0) for p in patients])
        assignment = {p: [] for p in patients}
        for rep in range(n_repeats):
            skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed + rep)
            for k, (_, te) in enumerate(skf.split(patients, lab)):
                for p in patients[te]:
                    assignment[p].append(k)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        json.dump({"n_folds": n_folds, "n_repeats": n_repeats, "seed": seed,
                   "n_patients": len(patients),
                   "strat_counts": {int(v): int((lab == v).sum())
                                    for v in np.unique(lab)},
                   "assignment": {str(k): v for k, v in assignment.items()}},
                  open(path, "w"), indent=1)
        print(f"[figo-protocol] wrote {path}: {len(patients)} patients, "
              f"{n_folds}x{n_repeats} folds, seed {seed}")
        return cls(pid, np.array([patient_strat.get(p, 0) for p in pid]),
                   assignment, n_folds, n_repeats, seed)

    # ---------------------------------------------------------------- folds
    def folds(self, repeat: Optional[int] = None
              ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_row_mask, test_row_mask) per fold."""
        reps = range(self.n_repeats) if repeat is None else [repeat]
        for rep in reps:
            for k in range(self.n_folds):
                te_p = {p for p in self.patients if self.assignment[p][rep] == k}
                te = np.array([q in te_p for q in self.pid])
                yield ~te, te

    def inner_split(self, tr_mask: np.ndarray, frac: float = 0.2,
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Carve a validation set out of TRAINING patients only (rule 3)."""
        tr_p = np.array(sorted(set(self.pid[tr_mask])))
        lab = np.array([int(self.strat[self.pidx[p]][0]) for p in tr_p])
        ok = len(np.unique(lab)) > 1 and np.bincount(lab).min() >= 2
        fit_p, val_p = train_test_split(tr_p, test_size=frac,
                                        stratify=lab if ok else None,
                                        random_state=seed)
        vs = set(val_p.tolist())
        is_val = np.array([q in vs for q in self.pid]) & tr_mask
        return tr_mask & ~is_val, is_val

    # -------------------------------------------------------------- scoring
    def bootstrap_ci(self, y: np.ndarray, s: np.ndarray, pid: np.ndarray,
                     metric=roc_auc_score, n_boot: int = N_BOOT
                     ) -> Tuple[float, float]:
        """
        95% CI resampled over PATIENTS, never rows.

        Epochs within a patient share a fetus, a sensor placement and a
        labour, so they are not independent draws. Resampling rows would
        report an interval narrower than the data supports -- measured at 2.7x
        too narrow on the previous substrate
        (scripts/audit_evaluation_protocol.py, `clustering` probe).
        """
        rng = np.random.default_rng(self.seed)
        pats = np.array(sorted(set(pid)))
        idx = {p: np.where(pid == p)[0] for p in pats}
        vals = []
        for _ in range(n_boot):
            b = rng.choice(pats, len(pats), replace=True)
            rows = np.concatenate([idx[p] for p in b])
            if len(np.unique(y[rows])) < 2:
                continue
            vals.append(metric(y[rows], s[rows]))
        return (tuple(np.percentile(vals, [2.5, 97.5])) if vals
                else (float("nan"), float("nan")))


def sens_spec_at_best(y: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    """
    The operating point that maximises min(sensitivity, specificity), plus
    the two threshold-free points this project has always reported.

    max-min is chosen because the stated goal is sensitivity AND specificity
    both >= 0.85. Youden's J would happily trade 0.95/0.70 for a better sum;
    max-min reports the balanced point the goal actually asks about.
    """
    fpr, tpr, thr = roc_curve(y, s)
    spec = 1 - fpr
    j = int(np.argmax(np.minimum(tpr, spec)))
    out = {"sens_at_balanced": float(tpr[j]),
           "spec_at_balanced": float(spec[j]),
           "threshold_balanced": float(thr[j]),
           "min_sens_spec": float(min(tpr[j], spec[j]))}
    for target, key in ((0.80, "sens_at_80spec"), (0.90, "sens_at_90spec")):
        m = spec >= target
        out[key] = float(tpr[m].max()) if m.any() else 0.0
    m = tpr >= 0.90
    out["spec_at_90sens"] = float(spec[m].max()) if m.any() else 0.0
    return out


def summarise(name: str, y: np.ndarray, s: np.ndarray, pid: np.ndarray,
              P: Optional[FigoProtocol] = None, verbose: bool = True
              ) -> Dict[str, float]:
    """AUROC / AUPRC / the balanced operating point, with a patient CI."""
    res = {"name": name, "n": int(len(y)), "n_pos": int(y.sum()),
           "prevalence": float(y.mean()),
           "auroc": float(roc_auc_score(y, s)),
           "auprc": float(average_precision_score(y, s))}
    res.update(sens_spec_at_best(y, s))
    if P is not None:
        lo, hi = P.bootstrap_ci(y, s, pid)
        res["ci_lo"], res["ci_hi"] = float(lo), float(hi)
    if verbose:
        ci = (f" [{res.get('ci_lo', float('nan')):.3f}-"
              f"{res.get('ci_hi', float('nan')):.3f}]" if P is not None else "")
        print(f"  {name:38s} AUROC {res['auroc']:.4f}{ci}  "
              f"AUPRC {res['auprc']:.4f}  "
              f"sens/spec {res['sens_at_balanced']:.3f}/"
              f"{res['spec_at_balanced']:.3f}")
    return res
