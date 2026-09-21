"""
Knowledge-infused FIGO detection: KI-1, KI-2, KI-3.

THE QUESTION
------------
Raw-signal detection sits at AUROC 0.7559 (arm Z). The same label is
recoverable from the descriptors at 0.9997 by a decision tree. So the ~0.24
gap is entirely the network's failure to compute the descriptors from signal
and then apply the rule. The FHR-UC interaction hypothesis was tested and
falsified (docs/figo_detection_interaction_experiment.md). This tests the
next one:

    does explicitly teaching the network the clinical concepts, and/or the
    FIGO reasoning over them, recover that gap?

WHAT MAKES THIS TESTABLE RATHER THAN VAGUE
-------------------------------------------
The binary label is not merely "related to" the descriptors, it is EXACTLY

    Abnormal = NOT (baseline_normal AND variability_normal AND
                    no_repetitive_decels)

verified on all 3,497 readable epochs: 100.0000% match, zero mismatches, and
zero pathological epochs where all three normality flags hold. So the soft
rule layer in KI-2 is not an approximation of the label -- it is the label,
with the hard thresholds replaced by sigmoids. If the concepts can be
estimated from signal, KI-2 must work; if KI-2 fails, the failure is
localised to concept estimation and is measurable as such.

THE BACKBONE IS ARM Z, UNCHANGED
---------------------------------
All three variants use the SmallCNN convolutional stack and its pooled
64-d feature, identical to the frozen 0.7559 control. Only the heads differ.
A KI variant beating Z therefore cannot be explained by a better encoder,
and the capacity finding from the interaction experiment (the 69k model beat
the 452k one by 0.0231, CI excluding zero) says not to reach for one.

THE CONCEPTS ARE TRAINING-TIME KNOWLEDGE
-----------------------------------------
Descriptors are targets, never inputs. At inference the model sees signal and
nothing else, exactly as Z does. This is what separates knowledge infusion
from the circular descriptor arms in Gate 2.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .descriptors import (BASELINE_NORMAL_HIGH, BASELINE_NORMAL_LOW,
                          VARIABILITY_NORMAL_HIGH, VARIABILITY_NORMAL_LOW)

# Concepts supervised in L_clinical. Continuous ones are z-scored on the
# training fold; binary ones are used as-is.
CONCEPTS_CONT: List[str] = [
    "baseline_bpm", "variability_bpm", "stv_bpm", "n_accels",
    "n_contractions", "n_decels", "n_decel_late", "n_decel_prolonged",
    "frac_time_in_decel", "longest_decel_s", "deepest_decel_bpm",
]
CONCEPTS_BIN: List[str] = [
    "decel_repetitive", "variability_measurable",
]
# has_acute_hypoxia_decel was supervised in the first KI run and is REMOVED
# from L_clinical here. It has ONE positive epoch in 3,497 (prevalence 0.03%)
# and its out-of-fold AUROC of 0.071 is noise, not a measurement. It remains a
# sufficient pathological criterion in rules.py -- a single deceleration over
# 5 minutes below 80 bpm must still escalate -- but supervising an almost
# empty target only injects gradient noise into the shared trunk.
CONCEPTS_BIN_DROPPED: List[str] = ["has_acute_hypoxia_decel"]

# The three the rule actually reads. baseline and variability come from the
# continuous head (in raw bpm after un-normalising); repetitive from the
# binary head.
RULE_CONT = ["baseline_bpm", "variability_bpm"]
RULE_BIN = ["decel_repetitive"]


class KIBackbone(nn.Module):
    """
    The arm-Z convolutional stack, verbatim, exposing its pooled feature.

    Kept numerically identical to legacy_cnn.SmallCNN so that "KI vs Z" is a
    statement about the heads and the loss, not about the encoder.
    """

    OUT_DIM = 64

    def __init__(self, width: int = 32, use_uc: bool = False):
        super().__init__()
        self.use_uc = use_uc

        def block(i, o, k, s):
            return nn.Sequential(
                nn.Conv1d(i, o, k, stride=s, padding=k // 2),
                nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))

        # use_uc adds two input channels (uc and its validity mask) to the
        # FIRST convolution and changes nothing else. That is deliberately the
        # smallest possible intervention: the FHR-UC interaction experiment
        # already showed that giving a CLASSIFIER cross-attention over the two
        # channels buys nothing (C - B = +0.0004, CI crossing zero), so this
        # must not reintroduce an attention mechanism. The claim under test is
        # different -- that UC is needed to ESTIMATE n_contractions, and hence
        # decel_repetitive, which then enters an exact rule.
        self.net = nn.Sequential(
            block(4 if use_uc else 2, width, 15, 4),
            block(width, width * 2, 9, 1),
            block(width * 2, width * 2, 7, 1),
            block(width * 2, width * 2, 5, 1),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())

    def forward(self, fhr, fhr_valid, uc=None, uc_valid=None):
        ch = [fhr, (~fhr_valid).float()]
        if self.use_uc:
            ch += [uc, (~uc_valid).float()]
        return self.pool(self.net(torch.stack(ch, dim=1)))


class SoftFigoRule(nn.Module):
    """
    The FIGO normality conjunction, made differentiable.

        P(Normal) = inband(baseline, 110, 160)
                  * inband(variability, 5, 25)
                  * (1 - P(repetitive decelerations))

    with each hard interval test replaced by a product of two sigmoids:

        inband(x, lo, hi) = sigmoid((x - lo)/tau) * sigmoid((hi - x)/tau)

    This is the specific structure a plain classifier cannot express. Gate 2
    measured the cost of that directly: on the SAME descriptor columns and
    the SAME folds, a decision tree scored 0.9997 and logistic regression
    0.8371, because a linear model must fail at one end of every band. The
    conjunction matters for the same reason -- "all three hold" is a product,
    not a weighted sum.

    tau is learnable per concept and initialised at 2 bpm, roughly the
    resolution at which a clinician reads a baseline (they are rounded to
    5 bpm). It is bounded below so the rule cannot degenerate into a step
    function and kill its own gradient.
    """

    TAU_MIN = 0.25

    def __init__(self):
        super().__init__()
        self.log_tau_b = nn.Parameter(torch.tensor(0.6931))   # log(2)
        self.log_tau_v = nn.Parameter(torch.tensor(0.6931))

    @staticmethod
    def _inband(x, lo, hi, tau):
        return torch.sigmoid((x - lo) / tau) * torch.sigmoid((hi - x) / tau)

    def forward(self, baseline_bpm, variability_bpm, p_repetitive):
        tau_b = self.log_tau_b.exp().clamp(min=self.TAU_MIN)
        tau_v = self.log_tau_v.exp().clamp(min=self.TAU_MIN)
        b_ok = self._inband(baseline_bpm, BASELINE_NORMAL_LOW,
                            BASELINE_NORMAL_HIGH, tau_b)
        v_ok = self._inband(variability_bpm, VARIABILITY_NORMAL_LOW,
                            VARIABILITY_NORMAL_HIGH, tau_v)
        p_normal = (b_ok * v_ok * (1.0 - p_repetitive)).clamp(1e-6, 1 - 1e-6)
        # logit of ABNORMAL
        return torch.log((1.0 - p_normal) / p_normal)


class KIModel(nn.Module):
    """
    mode = "ki1"  concept supervision, state from a normal classifier head
    mode = "ki2"  state comes ONLY through the soft FIGO rule
    mode = "ki3"  both, combined by a learned 2-input linear layer

    `set_scaler` must be called once per fold with the training-fold mean and
    std of the continuous concepts, so the head's z-scored predictions can be
    returned to bpm before the rule applies clinical thresholds to them. This
    mirrors figo_rule_loss_normalized() in src/knowledge/figo.py, which exists
    because an earlier version of this project applied FIGO thresholds
    directly to normalised outputs -- a unit mismatch that silently made the
    rule meaningless.
    """

    def __init__(self, mode: str = "ki1", width: int = 32, p_drop: float = 0.3,
                 use_uc: bool = False):
        super().__init__()
        assert mode in ("ki1", "ki2", "ki3")
        self.mode = mode
        self.backbone = KIBackbone(width, use_uc=use_uc)
        d = KIBackbone.OUT_DIM

        self.cont_head = nn.Sequential(
            nn.Dropout(p_drop), nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, len(CONCEPTS_CONT)))
        self.bin_head = nn.Sequential(
            nn.Dropout(p_drop), nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, len(CONCEPTS_BIN)))

        if mode in ("ki1", "ki3"):
            self.state_head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(d, 1))
        if mode in ("ki2", "ki3"):
            self.rule = SoftFigoRule()
        if mode == "ki3":
            # free to weight the two routes, and to ignore either
            self.combine = nn.Linear(2, 1)
            nn.init.constant_(self.combine.weight, 0.5)
            nn.init.zeros_(self.combine.bias)

        self.register_buffer("cont_mean", torch.zeros(len(CONCEPTS_CONT)))
        self.register_buffer("cont_std", torch.ones(len(CONCEPTS_CONT)))
        self._i_base = CONCEPTS_CONT.index("baseline_bpm")
        self._i_var = CONCEPTS_CONT.index("variability_bpm")
        self._i_rep = CONCEPTS_BIN.index("decel_repetitive")

    def set_scaler(self, mean: torch.Tensor, std: torch.Tensor):
        self.cont_mean.copy_(mean)
        self.cont_std.copy_(std.clamp(min=1e-6))

    def forward(self, fhr, uc, fhr_valid, uc_valid) -> Dict[str, torch.Tensor]:
        h = self.backbone(fhr, fhr_valid, uc, uc_valid)
        cont = self.cont_head(h)                       # z-scored
        binl = self.bin_head(h)                        # logits
        out = {"cont": cont, "bin": binl}

        rule_logit = None
        if self.mode in ("ki2", "ki3"):
            raw = cont * self.cont_std + self.cont_mean          # back to bpm
            rule_logit = self.rule(raw[:, self._i_base],
                                   raw[:, self._i_var],
                                   torch.sigmoid(binl[:, self._i_rep]))
            out["rule_logit"] = rule_logit

        if self.mode == "ki1":
            out["logit"] = self.state_head(h).squeeze(1)
        elif self.mode == "ki2":
            out["logit"] = rule_logit
        else:
            direct = self.state_head(h).squeeze(1)
            out["direct_logit"] = direct
            out["logit"] = self.combine(
                torch.stack([rule_logit, direct], dim=1)).squeeze(1)
        return out


def concept_loss(out: Dict[str, torch.Tensor], t_cont: torch.Tensor,
                 t_bin: torch.Tensor,
                 bin_pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    L_clinical: MSE on the z-scored continuous concepts, BCE on the binary
    ones. Both averaged over concepts so neither group dominates by count.
    """
    l_cont = F.mse_loss(out["cont"], t_cont)
    l_bin = F.binary_cross_entropy_with_logits(out["bin"], t_bin,
                                               pos_weight=bin_pos_weight)
    return l_cont + l_bin


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
