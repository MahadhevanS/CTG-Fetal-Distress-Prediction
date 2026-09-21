"""
Compact 1D CNN baseline for CTU-UHB intrapartum CTG.
"""
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Sequential() if (cin == cout and stride == 1)
                     else nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                        nn.BatchNorm1d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return self.act(h + self.skip(x))


class SmallCNN1D(nn.Module):
    """
    Compact 1D ResNet baseline (~200k parameters).
    """
    def __init__(self, in_channels: int = 1, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        self.blocks = nn.Sequential(
            ResBlock1D(width, width * 2, stride=2),
            ResBlock1D(width * 2, width * 4, stride=2),
            ResBlock1D(width * 4, width * 4, stride=2)
        )
        self.head = nn.Sequential(
            nn.Linear(width * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1) # (B, width * 8)
        return self.head(h)
