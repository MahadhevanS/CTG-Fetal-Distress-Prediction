"""
Models module for CTG fetal distress prediction.
Responsible for:
- Temporal encoders (1D CNN, BiLSTM, GRU, TCN, MultiScaleLSTM, PatchCTG, PatchTST, etc.)
- Multi-task heads (Distress Head, FIGO Head, Clinical Feature Head)
- Multi-task framework architectures
"""

from src.models.multiscale_lstm import MultiScaleLSTMEncoder, MultiScaleLSTMForClassification
from src.models.patchctg import PatchCTGEncoder, PatchCTGForClassification

__all__ = [
    "MultiScaleLSTMEncoder",
    "MultiScaleLSTMForClassification",
    "PatchCTGEncoder",
    "PatchCTGForClassification",
]
