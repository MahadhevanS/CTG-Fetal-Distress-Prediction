"""
Clinician-facing explanations for CTG-CrossFormer predictions.

WHY THIS EXISTS: for a device that replaces a CTG monitor, a bare risk score is
not actionable -- a clinician will not act on a black-box alarm, and
explainability is a regulatory expectation for clinical decision support. This
module turns one 20-minute window into a structured, clinically-phrased account
of WHAT the model saw and WHERE.

Three layers, deliberately ordered by how much they can be trusted:

  1. FIGO CRITERIA (fully faithful). Derived deterministically from the
     clinical features already computed by the preprocessing pipeline
     (baseline, STV/LTV, deceleration counts). These are facts about the trace,
     not inferences about the model, so they cannot mislead. They are what a
     clinician would independently verify against the strip.

  2. TEMPORAL SALIENCY (model-derived). Gradient-of-output w.r.t. input,
     aggregated per minute, showing which parts of the window moved the score.
     Standard saliency, with the usual caveats: gradients are noisy and
     indicate local sensitivity, not causation.

  3. FHR<->UC CROSS-ATTENTION (model-derived). Captured by forward hooks on the
     bidirectional cross-attention module -- the architecture's own account of
     which uterine-contraction positions each FHR position attended to. This is
     the mechanism the model uses for deceleration timing, so it is the natural
     place to look for a late-deceleration signature.

Layers 2 and 3 are explanations OF THE MODEL and are only as trustworthy as the
model. `faithfulness_report()` exists to check whether they actually track the
predictions rather than merely looking plausible -- run it before presenting any
of this to a clinician.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.knowledge.figo import FIGO_CRITERIA_NAMES, derive_figo_criteria_flags_torch

# Clinically meaningful direction for each derived FIGO flag: True means the
# flag firing is REASSURING, False means it is CONCERNING. Used to phrase the
# narrative correctly -- "variability normal" firing is good news, "late
# decelerations present" is not.
_CRITERION_IS_REASSURING = {
    "baseline_low": False,
    "baseline_normal": True,
    "baseline_high": False,
    "variability_normal": True,
    "has_late_decel": False,
    "has_variable_decel": False,
    "has_prolonged_decel": False,
}

_CRITERION_PHRASE = {
    "baseline_low": "baseline bradycardia (<100 bpm)",
    "baseline_normal": "baseline within normal range (110-160 bpm)",
    "baseline_high": "baseline tachycardia (>160 bpm)",
    "variability_normal": "long-term variability normal (5-25 bpm)",
    "has_late_decel": "late decelerations present",
    "has_variable_decel": "variable decelerations present",
    "has_prolonged_decel": "prolonged decelerations present",
}

FEATURE_NAMES = [
    "baseline_fhr", "stv", "ltv", "accel_count",
    "early_decel", "late_decel", "variable_decel", "prolonged_decel",
]

# Reference ranges for narrative context only (FIGO 2015 simplified). These are
# for display; the binary judgements come from derive_figo_criteria_flags_torch
# so the two can never disagree.
FEATURE_REFERENCE = {
    "baseline_fhr": (110.0, 160.0, "bpm"),
    "stv": (1.0, 25.0, "bpm"),
    "ltv": (5.0, 25.0, "bpm"),
}


class CTGExplainer:
    """
    Wraps a trained CTGCrossformerForClassification and produces explanations.

    The model is NOT modified -- cross-attention weights are captured with
    forward hooks, so behaviour and checkpoints are untouched.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 threshold: float = 0.5, fs: float = 4.0):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.fs = fs
        self._attn: Dict[str, torch.Tensor] = {}
        self._handles: List = []
        self._register_hooks()

    def _register_hooks(self):
        """Capture cross-attention weights without touching the model class."""
        enc = getattr(self.model, "encoder", None)
        cross = getattr(enc, "cross_attn", None) if enc is not None else None
        if cross is None:
            return

        def mk(name):
            def hook(_module, _inp, out):
                # nn.MultiheadAttention returns (output, weights); weights are
                # computed by default (need_weights=True) and discarded by the
                # encoder, so the hook is the cheapest way to reach them.
                if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                    self._attn[name] = out[1].detach()
            return hook

        if hasattr(cross, "mha_fhr_uc"):
            self._handles.append(cross.mha_fhr_uc.register_forward_hook(mk("fhr_attends_uc")))
        if hasattr(cross, "mha_uc_fhr"):
            self._handles.append(cross.mha_uc_fhr.register_forward_hook(mk("uc_attends_fhr")))

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ------------------------------------------------------------------ #

    def explain(self, x: torch.Tensor, y_features: torch.Tensor) -> Dict:
        """
        Args:
            x:          (2, 4800) one window, normalized exactly as in training.
            y_features: (8,) that window's clinical features in RAW units.
        Returns a structured explanation dict.
        """
        self.model.eval()
        self._attn.clear()
        xb = x.unsqueeze(0).to(self.device).requires_grad_(True)

        logit = self.model(xb).squeeze()
        prob = torch.sigmoid(logit)

        # Layer 2: gradient saliency w.r.t. the input trace
        self.model.zero_grad(set_to_none=True)
        logit.backward()
        grad = xb.grad.detach()[0].abs()               # (2, 4800)
        sal = (grad * xb.detach()[0].abs()).cpu().numpy()  # gradient x input

        samples_per_min = int(60 * self.fs)
        n_min = sal.shape[1] // samples_per_min
        per_min = sal[:, :n_min * samples_per_min].reshape(2, n_min, samples_per_min).sum(axis=2)
        total = per_min.sum()
        per_min_norm = per_min / total if total > 0 else per_min

        # Layer 1: FIGO criteria (deterministic, faithful)
        feats = y_features.detach().cpu().float().view(1, -1)
        flags = derive_figo_criteria_flags_torch(feats)[0].cpu().numpy().astype(bool)
        criteria = {n: bool(f) for n, f in zip(FIGO_CRITERIA_NAMES, flags)}
        concerning = [n for n, f in criteria.items() if f and not _CRITERION_IS_REASSURING[n]]
        reassuring = [n for n, f in criteria.items() if f and _CRITERION_IS_REASSURING[n]]

        fvals = {n: float(v) for n, v in zip(FEATURE_NAMES, feats[0].numpy())}

        # Layer 3: cross-attention -- for each FHR position, where in the UC
        # trace it looked. Summed over query positions gives, per UC minute,
        # how much total FHR attention it received.
        cross = None
        if "fhr_attends_uc" in self._attn:
            a = self._attn["fhr_attends_uc"][0].cpu().numpy()   # (150, 150)
            uc_attention = a.sum(axis=0)
            uc_attention = uc_attention / (uc_attention.sum() + 1e-8)
            tok_per_min = max(len(uc_attention) // n_min, 1)
            usable = (len(uc_attention) // tok_per_min) * tok_per_min
            cross = uc_attention[:usable].reshape(-1, tok_per_min).sum(axis=1)

        return {
            "risk_score": float(prob.item()),
            "flagged": bool(prob.item() >= self.threshold),
            "threshold": self.threshold,
            "criteria": criteria,
            "concerning_criteria": concerning,
            "reassuring_criteria": reassuring,
            "features": fvals,
            "saliency_per_minute_fhr": per_min_norm[0],
            "saliency_per_minute_uc": per_min_norm[1],
            "uc_attention_per_minute": cross,
            "peak_saliency_minute": int(np.argmax(per_min_norm[0])) if n_min else None,
        }

    # ------------------------------------------------------------------ #

    def narrate(self, e: Dict) -> str:
        """Render an explanation as clinician-readable text."""
        L = []
        verdict = "FLAGGED" if e["flagged"] else "not flagged"
        L.append(f"Risk score {e['risk_score']:.3f} (threshold {e['threshold']:.2f}) -> {verdict}")

        if e["concerning_criteria"]:
            L.append("Concerning findings:")
            for c in e["concerning_criteria"]:
                L.append(f"  - {_CRITERION_PHRASE[c]}")
        else:
            L.append("Concerning findings: none of the derived FIGO criteria fired")

        if e["reassuring_criteria"]:
            L.append("Reassuring findings:")
            for c in e["reassuring_criteria"]:
                L.append(f"  - {_CRITERION_PHRASE[c]}")

        f = e["features"]
        L.append("Measurements:")
        for k in ("baseline_fhr", "stv", "ltv"):
            lo, hi, unit = FEATURE_REFERENCE[k]
            mark = "" if lo <= f[k] <= hi else "  <-- outside normal range"
            L.append(f"  {k:<14} {f[k]:7.2f} {unit}  (normal {lo:g}-{hi:g}){mark}")
        decels = {k: f[k] for k in ("early_decel", "late_decel", "variable_decel", "prolonged_decel")}
        if any(v > 0 for v in decels.values()):
            L.append("  decelerations: " + ", ".join(f"{k.replace('_decel','')}={int(v)}"
                                                     for k, v in decels.items() if v > 0))
        else:
            L.append("  decelerations: none detected")
        L.append(f"  accelerations: {int(f['accel_count'])}")

        if e["peak_saliency_minute"] is not None:
            m = e["peak_saliency_minute"]
            L.append(f"Model attended most to minute {m}-{m+1} of the 20-minute window "
                     f"(FHR channel).")
        return "\n".join(L)


def faithfulness_report(explanations: List[Dict], y_true: np.ndarray) -> Dict:
    """
    Do the explanations actually track the model, or are they decoration?

    Checks whether the count of concerning FIGO criteria correlates with the
    model's risk score. A near-zero correlation would mean the narrative and
    the prediction are describing different things -- which must be disclosed
    rather than glossed over, since a clinician would reasonably assume the
    stated findings are WHY the model fired.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    risk = np.array([e["risk_score"] for e in explanations])
    n_concern = np.array([len(e["concerning_criteria"]) for e in explanations], dtype=float)

    rho, p = spearmanr(risk, n_concern)
    out = {
        "n": len(explanations),
        "spearman_risk_vs_concerning_criteria": float(rho),
        "p_value": float(p),
        "mean_concerning_when_flagged": float(n_concern[risk >= 0.5].mean()) if (risk >= 0.5).any() else float("nan"),
        "mean_concerning_when_not": float(n_concern[risk < 0.5].mean()) if (risk < 0.5).any() else float("nan"),
    }
    if len(set(y_true.tolist())) > 1:
        out["auroc_model"] = float(roc_auc_score(y_true, risk))
        out["auroc_criteria_count_alone"] = float(roc_auc_score(y_true, n_concern))
    return out
