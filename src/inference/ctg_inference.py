"""
Deployable inference path: raw CTG signal in, structured clinical output out.

This is the contract the hardware integrates against. It deliberately mirrors
src/preprocessing/pipeline_mil.py step for step -- any divergence between
training-time and inference-time preprocessing silently degrades the model, and
that class of bug is invisible in metrics because both sides look correct in
isolation.

INPUT   raw FHR and UC arrays at 4 Hz, as captured by the monitor
        (missing samples encoded as 0.0, matching the CTU-UHB convention)
OUTPUT  per-window risk scores plus the clinical findings behind them

The model scores 20-minute windows. A monitor supplies a continuous trace, so
windows are taken at a fixed stride and each is scored independently; the
report layer assembles them into a labour-level account.
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
_PP = os.path.join(BASE, "src", "preprocessing")
if _PP not in sys.path:
    sys.path.insert(0, _PP)

from baseline import calculate_iterative_baseline
from features import calculate_variability, detect_accelerations, detect_decelerations
from filtering import apply_lowpass_filter, interpolate_missing, remove_spikes
from signal_quality import assess_signal_quality

from src.knowledge.figo import FIGO_CRITERIA_NAMES, derive_figo_criteria_flags_torch
from src.models.ctg_crossformer import CTGCrossformerEncoder, CTGCrossformerForClassification

FS = 4.0
WINDOW_SAMPLES = 4800          # 20 minutes
STRIDE_SAMPLES = 600           # 2.5 minutes, matching training
FHR_MIN_BPM, FHR_MAX_BPM = 50.0, 240.0
MAX_MISSING_RATIO = 0.30

FEATURE_NAMES = ["baseline_fhr", "stv", "ltv", "accel_count",
                 "early_decel", "late_decel", "variable_decel", "prolonged_decel"]


class CTGInference:
    """Loads the delivered ensemble once; scores recordings repeatedly."""

    def __init__(self, checkpoint_dir: str, scaler_path: str,
                 device: Optional[torch.device] = None, threshold: float = 0.30,
                 n_folds: int = 5):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        sc = np.load(scaler_path)
        self.sig_mean, self.sig_std = sc["mean"], sc["std"]

        self.models = []
        for k in range(1, n_folds + 1):
            m = self._build()
            m.load_state_dict(torch.load(
                os.path.join(checkpoint_dir, f"ctg_crossformer_fold_{k}_best.pth"),
                map_location=self.device, weights_only=True))
            m.eval()
            self.models.append(m)

    def _build(self):
        e = CTGCrossformerEncoder(in_channels=2, seq_len=4800, cnn_channels=128,
                                  n_heads_cross=4, n_heads_tf=8, n_tf_layers=4,
                                  d_ff=512, dropout=0.1, latent_dim=128)
        return CTGCrossformerForClassification(encoder=e, hidden_dim=128, dropout=0.3).to(self.device)

    # ------------------------------------------------------------------ #

    def preprocess_window(self, fhr_win: np.ndarray, uc_win: np.ndarray):
        """One 20-minute window -> (model input tensor, clinical features).

        Same chain as pipeline_mil.py. Features are computed from the FILTERED
        but NOT baseline-corrected FHR, while the model input channel 0 IS
        baseline-corrected -- that asymmetry is intentional and must be
        preserved, since the feature extractors expect absolute bpm.
        """
        fhr = remove_spikes(fhr_win.copy(), fs=FS)
        fhr = interpolate_missing(fhr, clip_min=FHR_MIN_BPM, clip_max=FHR_MAX_BPM)
        fhr = apply_lowpass_filter(fhr, fs=FS)
        uc = apply_lowpass_filter(uc_win.copy(), fs=FS)

        baseline = calculate_iterative_baseline(fhr)
        stv, ltv = calculate_variability(fhr, fs=FS, baseline=baseline)
        accels = detect_accelerations(fhr, baseline, fs=FS)
        decels = detect_decelerations(fhr, baseline, uc, fs=FS)

        feats = np.array([float(np.mean(baseline)), stv, ltv, float(accels),
                          float(decels["early"]), float(decels["late"]),
                          float(decels["variable"]), float(decels["prolonged"])],
                         dtype=np.float32)

        x = np.vstack((fhr - baseline, uc)).astype(np.float32)
        x = (x - self.sig_mean[:, None]) / self.sig_std[:, None]
        return x, feats

    @torch.no_grad()
    def _score(self, x: np.ndarray) -> float:
        xb = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        return float(np.mean([torch.sigmoid(m(xb).squeeze()).item() for m in self.models]))

    # ------------------------------------------------------------------ #

    def analyse(self, fhr: np.ndarray, uc: np.ndarray,
                patient_id: str = "unknown") -> Dict:
        """
        Score a full recording. Returns the structured output a device consumes.
        Windows failing the >30%-missing quality gate are reported as skipped
        rather than silently dropped -- a clinician must be able to see that a
        stretch of trace was not assessed.
        """
        assert len(fhr) == len(uc), "FHR and UC must be the same length"
        results, skipped = [], []

        for start in range(0, max(len(fhr) - WINDOW_SAMPLES + 1, 0), STRIDE_SAMPLES):
            end = start + WINDOW_SAMPLES
            fw, uw = fhr[start:end], uc[start:end]
            if not assess_signal_quality(fw, max_missing_ratio=MAX_MISSING_RATIO):
                skipped.append(round(start / (FS * 60), 1))
                continue

            x, feats = self.preprocess_window(fw, uw)
            risk = self._score(x)
            flags = derive_figo_criteria_flags_torch(
                torch.tensor(feats).unsqueeze(0))[0].numpy().astype(bool)
            criteria = {n: bool(f) for n, f in zip(FIGO_CRITERIA_NAMES, flags)}
            concerning = [n for n, f in criteria.items()
                          if f and n in ("baseline_low", "baseline_high", "has_late_decel",
                                         "has_variable_decel", "has_prolonged_decel")]

            results.append({
                "start_min": round(start / (FS * 60), 1),
                "end_min": round(end / (FS * 60), 1),
                "risk": round(risk, 4),
                "flagged": bool(risk >= self.threshold),
                "concerning_findings": concerning,
                "measurements": {k: round(float(v), 2) for k, v in zip(FEATURE_NAMES, feats)},
            })

        return {
            "patient_id": patient_id,
            "threshold": self.threshold,
            "duration_min": round(len(fhr) / (FS * 60), 1),
            "windows_analysed": len(results),
            "windows_skipped_poor_quality": skipped,
            "windows": results,
            "summary": self._summarise(results),
        }

    @staticmethod
    def _summarise(results: List[Dict]) -> Dict:
        if not results:
            return {"status": "no analysable windows"}
        risks = np.array([r["risk"] for r in results])
        t = np.array([r["start_min"] for r in results]) / 60.0
        trend = 0.0
        if len(t) >= 3:
            tc = t - t.mean()
            d = float((tc ** 2).sum())
            trend = float((tc * (risks - risks.mean())).sum() / d) if d > 0 else 0.0
        peak = int(np.argmax(risks))
        run = best = 0
        for r in results:
            run = run + 1 if r["flagged"] else 0
            best = max(best, run)
        counts: Dict[str, int] = {}
        for r in results:
            if r["flagged"]:
                for c in r["concerning_findings"]:
                    counts[c] = counts.get(c, 0) + 1
        return {
            "peak_risk": round(float(risks.max()), 4),
            "peak_at_min": results[peak]["start_min"],
            "mean_risk": round(float(risks.mean()), 4),
            "n_flagged": int(sum(r["flagged"] for r in results)),
            "longest_sustained_windows": best,
            "risk_trend_per_hour": round(trend, 4),
            "recurring_findings": sorted(counts.items(), key=lambda kv: -kv[1]),
        }
