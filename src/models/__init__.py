"""
Models module for CTG fetal distress prediction.
Responsible for:
- Temporal encoders (PatchTST, CNN, BiLSTM, etc.)
- Multi-task heads
- Clinical prior network
"""

from .patchtst import PatchTSTEncoder, PatchTSTForClassification

__all__ = [
    "PatchTSTEncoder",
    "PatchTSTForClassification",
]
