"""
Compact 2D CNN models for CTU-UHB intrapartum CTG time-frequency scalograms and recurrence plots.
"""
import torch
import torch.nn as nn
from .wavelet import ContinuousWaveletTransform
from .recurrence import RecurrencePlot


class ResBlock2D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = (nn.Sequential() if (cin == cout and stride == 1)
                     else nn.Sequential(nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False),
                                        nn.BatchNorm2d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return self.act(h + self.skip(x))


class SmallCNN2D(nn.Module):
    """
    Compact 2D ResNet architecture (~250k - 400k parameters).
    Operates on 2D image representations (CWT scalograms or RP matrices).
    """
    def __init__(self, in_channels: int = 1, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.blocks = nn.Sequential(
            ResBlock2D(width, width * 2, stride=2),
            ResBlock2D(width * 2, width * 4, stride=2),
            ResBlock2D(width * 4, width * 4, stride=2)
        )
        self.head = nn.Sequential(
            nn.Linear(width * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        # Global Average + Global Max Pooling across both spatial dimensions (H, W)
        avg_pool = h.mean(dim=(-2, -1))
        max_pool = h.amax(dim=(-2, -1))
        feat = torch.cat([avg_pool, max_pool], dim=1) # (B, width * 8)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.head(feat)


class EndToEndCWTCNN(nn.Module):
    """
    End-to-End model: 1D Raw Waveforms -> CWT Transform -> 2D CNN -> Prediction.
    """
    def __init__(self, in_channels: int = 1, num_scales: int = 64, time_pool: int = 8,
                 width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.cwt = ContinuousWaveletTransform(num_scales=num_scales, time_pool=time_pool)
        self.cnn2d = SmallCNN2D(in_channels=in_channels, width=width, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, in_channels, 4800)
        scalogram = self.cwt(x) # (B, in_channels, num_scales, 4800 // time_pool)
        return self.cnn2d(scalogram)


class EndToEndRPCNN(nn.Module):
    """
    End-to-End model: 1D Raw Waveforms -> Recurrence Plot -> 2D CNN -> Prediction.
    """
    def __init__(self, dimension: int = 3, time_delay: int = 4, target_size: int = 128,
                 width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.rp = RecurrencePlot(dimension=dimension, time_delay=time_delay, target_size=target_size)
        self.cnn2d = SmallCNN2D(in_channels=1, width=width, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, 4800)
        rp_img = self.rp(x) # (B, 1, 128, 128)
        return self.cnn2d(rp_img)
