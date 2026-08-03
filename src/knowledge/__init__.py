"""
Knowledge module for CTG fetal distress prediction.
Responsible for:
- FIGO 2015 clinical rule engine (scalar + vectorized)
- Knowledge-guided differentiable loss functions
"""

from .figo import (
    classify_figo,
    vectorized_classify_figo,
    figo_rule_loss,
    figo_rule_loss_normalized,
)

__all__ = [
    "classify_figo",
    "vectorized_classify_figo",
    "figo_rule_loss",
    "figo_rule_loss_normalized",
]


