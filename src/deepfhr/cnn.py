"""
The exact 8-layer CNN from Zhao et al. 2019, Fig. 4 / Table 2 (docs/deepfhr_original.yaml):
input -> conv(5x5, 15 filters) -> ReLU -> batch-norm -> max-pool(2x2, stride 2) -> FC(256) -> dropout(0.5) -> FC(2) -> softmax.
"""
import torch
import torch.nn as nn


class DeepFHRNet(nn.Module):
    def __init__(self, in_channels=3, conv_filters=15, kernel=5, fc_hidden=256, dropout=0.5, n_classes=2, img_size=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, conv_filters, kernel_size=kernel, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(conv_filters)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        out_hw = (img_size - kernel + 1) // 2
        self.flat_dim = conv_filters * out_hw * out_hw
        self.fc1 = nn.Linear(self.flat_dim, fc_hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, n_classes)

    def forward(self, x):
        x = self.pool(self.bn(torch.relu(self.conv(x))))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)   # logits; softmax applied via CrossEntropyLoss
