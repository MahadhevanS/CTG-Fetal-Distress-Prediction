"""
Models module for CTG fetal distress prediction.
Responsible for:
- Temporal encoders (1D CNN, BiLSTM, GRU, TCN, MultiScaleLSTM, PatchCTG, PatchTST, etc.)
- Multi-task heads (Distress Head, FIGO Head, Clinical Feature Head)
- Multi-task framework architectures
"""

from .cnn1d_encoder import CNN1DEncoder
from .bilstm_encoder import BiLSTMEncoder
from src.models.gru_encoder import GRUEncoder
from src.models.tcn_encoder import TCNEncoder
from src.models.classifier import UniversalClassifier    
from src.models.multiscale_lstm import MultiScaleLSTMEncoder, MultiScaleLSTMForClassification
from src.models.patchctg import PatchCTGEncoder, PatchCTGForClassification
from src.models.patchtst import PatchTSTEncoder, PatchTSTForClassification

__all__ = [
    "CNN1DEncoder",
    "BiLSTMEncoder",
    "MultiScaleLSTMEncoder",
    "MultiScaleLSTMForClassification",
    "PatchCTGEncoder",
    "PatchCTGForClassification",
    "PatchTSTEncoder",
    "PatchTSTForClassification",
    "GRUEncoder",
    "TCNEncoder",
    "UniversalClassifier",
]

