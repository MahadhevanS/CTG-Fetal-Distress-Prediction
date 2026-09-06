"""
The Gate-2 SmallCNN, unchanged, so the frozen 0.7518 control can be re-scored
through the same harness as every new arm.

Kept in its own file and never edited. It is a fixed point of comparison; if
it is modified, the number it produced stops meaning anything. Its forward
signature is adapted to the (fhr, uc, fhr_valid, uc_valid) call the runner
uses, but the network is identical: it sees FHR plus the raw missingness
channel, exactly as in scripts/figo_gate2_state_detection.py.
"""

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, width: int = 32, n_out: int = 1):
        super().__init__()

        def block(i, o, k, s):
            return nn.Sequential(
                nn.Conv1d(i, o, k, stride=s, padding=k // 2),
                nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))

        self.net = nn.Sequential(
            block(2, width, 15, 4),
            block(width, width * 2, 9, 1),
            block(width * 2, width * 2, 7, 1),
            block(width * 2, width * 2, 5, 1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(width * 2, n_out))

    def forward(self, fhr, uc, fhr_valid, uc_valid):
        x = torch.stack([fhr, (~fhr_valid).float()], dim=1)
        return self.head(self.net(x))
