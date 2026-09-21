"""
Information-Density Adaptive Weighting Module.
=============================================
Implements the weighting mechanism specified in docs/information_density_weighting_design.md:

1. Patient Normalization (Section 2.1):
   Rescales per-patient window contribution so every patient contributes a fixed
   total vote to the loss regardless of recording length.

2. Causal Novelty Score (Section 2.2):
   Computes consecutive-window cosine distance on z-scored clinical descriptors,
   smoothed via causal EWMA. First window falls back to fold median novelty.

3. Elapsed-Time Empirical Prior / Shrinkage (Section 2.3 & 2.4):
   Binds causal elapsed monitoring time into horizon bands and shrinks individual
   novelty toward the training fold's empirical horizon average:
   novelty_final = beta * novelty_smoothed + (1 - beta) * fold_bin_avg(elapsed)

4. Strict Fold Isolation:
   All calibration statistics (z-score means/stds, median novelty, patient scale K,
   fold_bin_avg) are fit strictly on the training partition and applied unchanged
   to validation/test partitions.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


DEFAULT_BIN_EDGES = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, np.inf]


class InformationDensityWeighter:
    """
    Computes patient normalization and information-density novelty weights
    strictly within training fold boundaries.
    """

    def __init__(
        self,
        span: float = 3.0,
        beta: float = 1.0,
        bin_edges: Optional[List[float]] = None,
        stride_min: float = 2.5,
    ):
        self.span = span
        self.beta = beta
        self.bin_edges = bin_edges if bin_edges is not None else list(DEFAULT_BIN_EDGES)
        self.stride_min = stride_min

        # Fitted training statistics
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_std_: Optional[np.ndarray] = None
        self.K_: Optional[float] = None
        self.fold_median_novelty_: Optional[float] = None
        self.fold_bin_avg_: Optional[np.ndarray] = None
        self.overall_mean_smoothed_novelty_: Optional[float] = None
        self.mean_novelty_final_train_: Optional[float] = None
        self.mean_combined_train_: Optional[float] = None

    def _get_bin_indices(self, elapsed_min: np.ndarray) -> np.ndarray:
        """Assigns each elapsed time to its corresponding bin index [0, n_bins-1]."""
        bins = np.digitize(elapsed_min, self.bin_edges[1:], right=False)
        return np.clip(bins, 0, len(self.bin_edges) - 2)

    def _compute_chronological_ordering(
        self,
        patient_ids: np.ndarray,
        time_or_indices: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Groups window indices by patient, ordered chronologically.
        Returns list of (pid, sorted_indices).
        """
        unique_pids = []
        seen = set()
        for p in patient_ids:
            p_str = str(p)
            if p_str not in seen:
                seen.add(p_str)
                unique_pids.append(p_str)

        ordered_groups = []
        for pid in unique_pids:
            mask = np.where(patient_ids.astype(str) == pid)[0]
            if time_or_indices is not None:
                sort_keys = time_or_indices[mask]
                order = np.argsort(sort_keys)
                sorted_idx = mask[order]
            else:
                sorted_idx = mask
            ordered_groups.append((pid, sorted_idx))
        return ordered_groups

    def _compute_raw_novelties(
        self,
        X_z: np.ndarray,
        ordered_groups: List[Tuple[str, np.ndarray]],
        fallback_novelty: float,
        elapsed_lookup: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes causal 1 - cosine_similarity(X_j, X_{j-1}) per patient.
        First window of each patient uses fallback_novelty.

        Elapsed monitoring time: if `elapsed_lookup` is given (a full-length
        array of real, gap-aware time-since-recording-start values, e.g.
        start_sample / (fs * 60), in the SAME index space as X_z), it is used
        directly, offset so each patient's own first retained window reads 0.
        This correctly reflects quality-gate-dropped windows (non-uniform
        stride). If not given, falls back to the positional approximation
        j * stride_min, which silently assumes no windows were ever dropped.
        """
        n_samples = len(X_z)
        novelty_raw = np.zeros(n_samples, dtype=np.float32)
        elapsed_min = np.zeros(n_samples, dtype=np.float32)

        eps = 1e-8
        norms = np.linalg.norm(X_z, axis=1)

        for pid, idxs in ordered_groups:
            k = len(idxs)
            t0 = float(elapsed_lookup[idxs[0]]) if elapsed_lookup is not None else 0.0
            for j in range(k):
                g_idx = idxs[j]
                if elapsed_lookup is not None:
                    elapsed_min[g_idx] = float(elapsed_lookup[g_idx]) - t0
                else:
                    elapsed_min[g_idx] = float(j * self.stride_min)
                if j == 0:
                    novelty_raw[g_idx] = fallback_novelty
                else:
                    prev_idx = idxs[j - 1]
                    dot_prod = np.dot(X_z[g_idx], X_z[prev_idx])
                    denom = max(norms[g_idx] * norms[prev_idx], eps)
                    cos_sim = np.clip(dot_prod / denom, -1.0, 1.0)
                    novelty_raw[g_idx] = float(1.0 - cos_sim)

        return novelty_raw, elapsed_min

    def _apply_ewma_smoothing(
        self,
        novelty_raw: np.ndarray,
        ordered_groups: List[Tuple[str, np.ndarray]],
    ) -> np.ndarray:
        """
        Applies strictly causal EWMA smoothing across each patient's novelty trajectory.
        """
        n_samples = len(novelty_raw)
        novelty_smoothed = np.zeros(n_samples, dtype=np.float32)
        alpha = 2.0 / (self.span + 1.0)

        for pid, idxs in ordered_groups:
            k = len(idxs)
            s = novelty_raw[idxs[0]]
            novelty_smoothed[idxs[0]] = s
            for j in range(1, k):
                g_idx = idxs[j]
                s = alpha * novelty_raw[g_idx] + (1.0 - alpha) * s
                novelty_smoothed[g_idx] = s

        return novelty_smoothed

    def fit(
        self,
        X: np.ndarray,
        patient_ids: np.ndarray,
        time_or_indices: Optional[np.ndarray] = None,
    ) -> "InformationDensityWeighter":
        """
        Fits all statistics strictly on the training partition.
        """
        patient_ids = np.asarray(patient_ids)
        X = np.asarray(X, dtype=np.float32)

        # 1. Fit feature z-score statistics
        self.feature_mean_ = np.mean(X, axis=0)
        self.feature_std_ = np.std(X, axis=0)
        self.feature_std_[self.feature_std_ < 1e-6] = 1.0

        X_z = (X - self.feature_mean_) / self.feature_std_

        # 2. Chronological groups
        ordered_groups = self._compute_chronological_ordering(patient_ids, time_or_indices)

        # 3. Compute raw novelties for j > 1 to determine fold median novelty
        pairs_novelty = []
        eps = 1e-8
        norms = np.linalg.norm(X_z, axis=1)
        for pid, idxs in ordered_groups:
            k = len(idxs)
            for j in range(1, k):
                cur_idx = idxs[j]
                prev_idx = idxs[j - 1]
                dot_prod = np.dot(X_z[cur_idx], X_z[prev_idx])
                denom = max(norms[cur_idx] * norms[prev_idx], eps)
                cos_sim = np.clip(dot_prod / denom, -1.0, 1.0)
                pairs_novelty.append(float(1.0 - cos_sim))

        if len(pairs_novelty) > 0:
            self.fold_median_novelty_ = float(np.median(pairs_novelty))
        else:
            self.fold_median_novelty_ = 0.05

        # 4. Compute full novelty and EWMA on training fold
        novelty_raw_tr, elapsed_tr = self._compute_raw_novelties(
            X_z, ordered_groups, self.fold_median_novelty_, elapsed_lookup=time_or_indices
        )
        novelty_sm_tr = self._apply_ewma_smoothing(novelty_raw_tr, ordered_groups)

        self.overall_mean_smoothed_novelty_ = float(np.mean(novelty_sm_tr))

        # 5. Fit elapsed-time empirical prior (horizon bin averages)
        bin_idx_tr = self._get_bin_indices(elapsed_tr)
        n_bins = len(self.bin_edges) - 1
        bin_avgs = np.zeros(n_bins, dtype=np.float32)

        for b in range(n_bins):
            in_b = novelty_sm_tr[bin_idx_tr == b]
            if len(in_b) > 0:
                bin_avgs[b] = float(np.mean(in_b))
            else:
                bin_avgs[b] = self.overall_mean_smoothed_novelty_
        self.fold_bin_avg_ = bin_avgs

        # 6. Patient normalization constant K = N_train / N_unique_patients_train
        n_train_samples = len(X)
        n_train_patients = len(ordered_groups)
        self.K_ = float(n_train_samples) / float(max(n_train_patients, 1))

        # 7. Compute training calibration scale for novelty and combined weights
        prior_tr = self.fold_bin_avg_[bin_idx_tr]
        novelty_final_tr = self.beta * novelty_sm_tr + (1.0 - self.beta) * prior_tr
        self.mean_novelty_final_train_ = float(np.mean(novelty_final_tr))
        if self.mean_novelty_final_train_ < 1e-8:
            self.mean_novelty_final_train_ = 1.0

        # Patient normalization weights on train
        p_counts = {pid: len(idxs) for pid, idxs in ordered_groups}
        w_pat_tr = np.array([self.K_ / float(p_counts[str(p)]) for p in patient_ids], dtype=np.float32)
        w_nov_tr = novelty_final_tr / self.mean_novelty_final_train_
        w_comb_raw_tr = w_pat_tr * w_nov_tr
        self.mean_combined_train_ = float(np.mean(w_comb_raw_tr))
        if self.mean_combined_train_ < 1e-8:
            self.mean_combined_train_ = 1.0

        return self

    def transform(
        self,
        X: np.ndarray,
        patient_ids: np.ndarray,
        time_or_indices: Optional[np.ndarray] = None,
        beta: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Computes causal novelty features and weights using fitted training statistics.
        """
        if self.feature_mean_ is None:
            raise RuntimeError("InformationDensityWeighter must be fit before calling transform.")

        effective_beta = self.beta if beta is None else beta
        patient_ids = np.asarray(patient_ids)
        X = np.asarray(X, dtype=np.float32)

        X_z = (X - self.feature_mean_) / self.feature_std_
        ordered_groups = self._compute_chronological_ordering(patient_ids, time_or_indices)

        novelty_raw, elapsed_min = self._compute_raw_novelties(
            X_z, ordered_groups, self.fold_median_novelty_, elapsed_lookup=time_or_indices
        )
        novelty_smoothed = self._apply_ewma_smoothing(novelty_raw, ordered_groups)

        bin_idx = self._get_bin_indices(elapsed_min)
        prior = self.fold_bin_avg_[bin_idx]
        novelty_final = effective_beta * novelty_smoothed + (1.0 - effective_beta) * prior

        # Novelty weight (mean ~ 1.0 on train)
        w_nov = novelty_final / self.mean_novelty_final_train_

        # Patient normalization weight
        p_counts = {pid: len(idxs) for pid, idxs in ordered_groups}
        w_pat = np.array([self.K_ / float(p_counts[str(p)]) for p in patient_ids], dtype=np.float32)

        # Composite weight
        w_combined = (w_pat * w_nov) / self.mean_combined_train_

        return {
            "w_patient": w_pat,
            "w_novelty": w_nov,
            "w_combined": w_combined,
            "novelty_raw": novelty_raw,
            "novelty_smoothed": novelty_smoothed,
            "elapsed_monitoring_min": elapsed_min,
            "prior_bin_avg": prior,
        }

    def extract_novelty_features(
        self,
        X: np.ndarray,
        patient_ids: np.ndarray,
        time_or_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Returns (N, 3) matrix of causal features: [novelty_raw, novelty_smoothed, elapsed_monitoring_min].
        """
        res = self.transform(X, patient_ids, time_or_indices)
        return np.column_stack([
            res["novelty_raw"],
            res["novelty_smoothed"],
            res["elapsed_monitoring_min"]
        ]).astype(np.float32)


def test_weighter_causality():
    """Unit and causal invariance test for InformationDensityWeighter."""
    print("Testing InformationDensityWeighter...")
    np.random.seed(42)
    n_pats = 20
    windows_per_pat = np.random.randint(5, 25, size=n_pats)
    pids = []
    times = []
    features = []

    for i, n_w in enumerate(windows_per_pat):
        pid = f"pat_{i:03d}"
        base_f = np.random.randn(19)
        for j in range(n_w):
            pids.append(pid)
            times.append(j * 2.5)
            # gradual drift + noise
            f = base_f + 0.05 * j + np.random.randn(19) * 0.1
            features.append(f)

    pids = np.array(pids)
    times = np.array(times)
    features = np.array(features, dtype=np.float32)

    weighter = InformationDensityWeighter(span=3.0, beta=0.8)
    weighter.fit(features, pids, times)
    out = weighter.transform(features, pids, times)

    # 1. Check patient normalization mean
    assert abs(np.mean(out["w_patient"]) - 1.0) < 1e-4, f"Patient weight mean != 1.0: {np.mean(out['w_patient'])}"

    # 2. Check composite weight mean
    assert abs(np.mean(out["w_combined"]) - 1.0) < 1e-4, f"Combined weight mean != 1.0: {np.mean(out['w_combined'])}"

    # 3. Check first window of each patient matches median
    first_idxs = [np.where(pids == f"pat_{i:03d}")[0][0] for i in range(n_pats)]
    for idx in first_idxs:
        assert abs(out["novelty_raw"][idx] - weighter.fold_median_novelty_) < 1e-6

    # 4. Check causal invariance under future perturbation
    target_pid = "pat_000"
    target_idx = np.where(pids == target_pid)[0]
    orig_weights = out["w_combined"][target_idx[2]].copy()
    orig_feat = out["novelty_smoothed"][target_idx[2]].copy()

    perturbed_features = features.copy()
    perturbed_features[target_idx[3:]] += np.random.randn(*perturbed_features[target_idx[3:]].shape) * 10.0

    out_perturbed = weighter.transform(perturbed_features, pids, times)
    pert_weight = out_perturbed["w_combined"][target_idx[2]]
    pert_feat = out_perturbed["novelty_smoothed"][target_idx[2]]

    assert abs(orig_weights - pert_weight) < 1e-6, "Causal violation in weight computation!"
    assert abs(orig_feat - pert_feat) < 1e-6, "Causal violation in smoothed feature computation!"

    print("All InformationDensityWeighter tests PASSED successfully.")


if __name__ == "__main__":
    test_weighter_causality()
