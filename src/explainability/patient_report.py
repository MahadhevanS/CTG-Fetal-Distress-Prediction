"""
Patient-level clinical report -- what the device would actually display.

A per-window risk score is not what a clinician reads. During labour they see a
continuous trace and ask three questions: is this fetus at risk, when did that
change, and what in the trace says so. This module answers those by assembling
per-window explanations into one account of a labour.

Design decisions worth stating:

  * RISK TIMELINE, not a single number. Compressing a labour to one score
    discards the thing clinicians most need -- whether risk is rising. Patient
    aggregation was also measured to be actively worse than window-level scoring
    on this data (AUROC 0.79 -> 0.66), so this module deliberately does NOT
    produce a patient-level prediction. It summarises window-level evidence.

  * FINDINGS COME FROM THE FIGO LAYER ONLY. Of the three explanation layers,
    only the criteria are deterministic facts about the trace. Saliency and
    attention are model-derived and, in attention's case, contested as
    explanations (Jain & Wallace, 2019). They belong in a visualisation, not in
    a clinical summary that reads as fact.

  * THE THRESHOLD IS SHOWN, NOT HIDDEN. Sensitivity at a fixed 0.5 cut ranged
    29-94% across folds in this project. Any flag count is meaningless without
    the operating point that produced it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.explainability.ctg_explainer import _CRITERION_PHRASE


@dataclass
class WindowEntry:
    start_sample: int
    minutes_from_start: float
    risk: float
    flagged: bool
    concerning: List[str]
    features: Dict[str, float]


@dataclass
class PatientReport:
    patient_id: str
    threshold: float
    windows: List[WindowEntry] = field(default_factory=list)

    # ---------------------------------------------------------------- #
    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def n_flagged(self) -> int:
        return sum(w.flagged for w in self.windows)

    @property
    def peak(self) -> Optional[WindowEntry]:
        return max(self.windows, key=lambda w: w.risk) if self.windows else None

    @property
    def risk_trend(self) -> float:
        """Slope of risk over time (per hour). Positive = deteriorating."""
        if len(self.windows) < 3:
            return 0.0
        t = np.array([w.minutes_from_start for w in self.windows]) / 60.0
        r = np.array([w.risk for w in self.windows])
        t = t - t.mean()
        denom = float((t ** 2).sum())
        return float((t * (r - r.mean())).sum() / denom) if denom > 0 else 0.0

    @property
    def sustained_run(self) -> int:
        """Longest run of consecutive flagged windows."""
        best = run = 0
        for w in self.windows:
            run = run + 1 if w.flagged else 0
            best = max(best, run)
        return best

    def recurring_findings(self, min_share: float = 0.25) -> List[tuple]:
        """Criteria appearing in at least min_share of FLAGGED windows."""
        flagged = [w for w in self.windows if w.flagged]
        if not flagged:
            return []
        counts: Dict[str, int] = {}
        for w in flagged:
            for c in w.concerning:
                counts[c] = counts.get(c, 0) + 1
        out = [(c, n, n / len(flagged)) for c, n in counts.items()
               if n / len(flagged) >= min_share]
        return sorted(out, key=lambda x: -x[1])

    # ---------------------------------------------------------------- #
    def render(self, width: int = 78) -> str:
        L = []
        L.append("=" * width)
        L.append(f" CTG REVIEW -- patient {self.patient_id}")
        L.append("=" * width)

        if not self.windows:
            L.append("  No analysable windows.")
            return "\n".join(L)

        peak = self.peak
        L.append(f"  {self.n_windows} windows analysed | "
                 f"{self.n_flagged} flagged at threshold {self.threshold:.2f}")
        L.append(f"  Peak risk {peak.risk:.3f} at minute "
                 f"{peak.minutes_from_start:.0f}-{peak.minutes_from_start + 20:.0f}")

        trend = self.risk_trend
        direction = ("rising" if trend > 0.02 else
                     "falling" if trend < -0.02 else "stable")
        L.append(f"  Risk trend: {direction} ({trend:+.3f} per hour)")
        if self.sustained_run >= 2:
            L.append(f"  Longest sustained elevation: {self.sustained_run} consecutive windows")

        # --- timeline -------------------------------------------------
        L.append("")
        L.append("  Timeline (each row = one 20-minute window)")
        L.append(f"  {'start':>7}  {'risk':>6}  {'':<24} findings")
        L.append("  " + "-" * (width - 4))
        for w in self.windows:
            bar_len = int(round(w.risk * 20))
            bar = ("#" * bar_len).ljust(20)
            mark = ">>" if w.flagged else "  "
            names = ", ".join(_CRITERION_PHRASE[c].split(" (")[0] for c in w.concerning) or "-"
            L.append(f"  {w.minutes_from_start:>5.0f}m {mark} {w.risk:>6.3f}  [{bar}] {names}")

        # --- recurring findings --------------------------------------
        rec = self.recurring_findings()
        L.append("")
        if rec:
            L.append("  Findings recurring across flagged windows:")
            for c, n, share in rec:
                L.append(f"    - {_CRITERION_PHRASE[c]} ({n}/{self.n_flagged} flagged windows)")
        elif self.n_flagged:
            L.append("  No FIGO criterion recurs across the flagged windows -- the model is")
            L.append("  responding to signal features the criteria do not capture. Review the")
            L.append("  trace directly.")
        else:
            L.append("  No windows flagged at this threshold.")

        # --- measurements at peak ------------------------------------
        L.append("")
        L.append(f"  Measurements at peak-risk window (minute {peak.minutes_from_start:.0f}):")
        f = peak.features
        L.append(f"    baseline {f['baseline_fhr']:.0f} bpm | STV {f['stv']:.2f} | LTV {f['ltv']:.1f} bpm")
        dec = {k: f[k] for k in ("early_decel", "late_decel", "variable_decel", "prolonged_decel")}
        shown = ", ".join(f"{k.replace('_decel','')} {int(v)}" for k, v in dec.items() if v > 0)
        L.append(f"    decelerations: {shown if shown else 'none'} | "
                 f"accelerations: {int(f['accel_count'])}")

        L.append("")
        L.append("  Decision support only. Findings are computed from the trace by FIGO")
        L.append("  2015 definitions and should be verified against the strip.")
        L.append("=" * width)
        return "\n".join(L)


def build_patient_report(patient_id: str, explanations: List[Dict],
                         starts: np.ndarray, threshold: float,
                         fs: float = 4.0) -> PatientReport:
    """Assemble per-window explanations for one patient into a report, in time order."""
    order = np.argsort(starts)
    rep = PatientReport(patient_id=patient_id, threshold=threshold)
    for i in order:
        e = explanations[i]
        rep.windows.append(WindowEntry(
            start_sample=int(starts[i]),
            minutes_from_start=float(starts[i]) / (fs * 60.0),
            risk=e["risk_score"],
            flagged=e["risk_score"] >= threshold,
            concerning=e["concerning_criteria"],
            features=e["features"],
        ))
    return rep
