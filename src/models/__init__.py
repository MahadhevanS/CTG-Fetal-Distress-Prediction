"""
Models module for CTG fetal distress prediction.
Responsible for:
- Temporal encoders (1D CNN, BiLSTM, GRU, TCN, MultiScaleLSTM, PatchCTG, PatchTST)
- Multi-task heads (Distress Head, FIGO Head, Clinical Feature Head)
- Multi-task framework architectures (KnowledgeInfusedFramework)
"""

from .cnn1d_encoder import CNN1DEncoder
from .bilstm_encoder import BiLSTMEncoder
from src.models.gru_encoder import GRUEncoder
from src.models.tcn_encoder import TCNEncoder
from src.models.classifier import UniversalClassifier
from src.models.multiscale_lstm import MultiScaleLSTMEncoder, MultiScaleLSTMForClassification
from src.models.patchctg import PatchCTGEncoder, PatchCTGForClassification
from src.models.patchtst import PatchTSTEncoder, PatchTSTForClassification
from src.models.knowledge_infused_framework import (
    KnowledgeInfusedFramework,
    DistressHead,
    FIGOHead,
    ClinicalFeatureHead,
)

__all__ = [
    # Phase 3 Encoders
    "CNN1DEncoder",
    "BiLSTMEncoder",
    "GRUEncoder",
    "TCNEncoder",
    "MultiScaleLSTMEncoder",
    "MultiScaleLSTMForClassification",
    "PatchCTGEncoder",
    "PatchCTGForClassification",
    "PatchTSTEncoder",
    "PatchTSTForClassification",
    "UniversalClassifier",
    # Phase 4 Multi-Task Framework
    "KnowledgeInfusedFramework",
    "DistressHead",
    "FIGOHead",
    "ClinicalFeatureHead",
]


