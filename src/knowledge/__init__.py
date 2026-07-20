"""
Knowledge module for CTG fetal distress prediction.
Responsible for:
- FIGO rules
- NICHD rules
- Knowledge-guided loss
"""

from .figo import classify_figo, figo_rule_loss

__all__ = ["classify_figo", "figo_rule_loss"]

