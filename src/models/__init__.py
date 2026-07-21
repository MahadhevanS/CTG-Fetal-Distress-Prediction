"""
Models module for CTG fetal distress prediction.
Responsible for:
- Temporal encoders (PatchTST, 1D CNN, BiLSTM, etc.)
- Multi-task heads
- Multi-task framework architectures
"""

from .patchtst import PatchTSTEncoder, PatchTSTForClassification

__all__ = [
    "PatchTSTEncoder",
    "PatchTSTForClassification",
]
