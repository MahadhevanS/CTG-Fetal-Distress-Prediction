import torch
import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder

    Input:
        (Batch, 2, 4800)

    Output:
        (Batch, 128)
    """

    def __init__(
        self,
        input_size=2,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, 128)

    def forward(self, x):
        # Input shape:
        # (Batch, 2, 4800)

        # Convert to:
        # (Batch, 4800, 2)
        x = x.permute(0, 2, 1)

        _, (hidden, _) = self.lstm(x)

        # Forward hidden state
        forward_hidden = hidden[-2]

        # Backward hidden state
        backward_hidden = hidden[-1]

        # Concatenate
        x = torch.cat((forward_hidden, backward_hidden), dim=1)

        # Project to 128-dimensional latent vector
        x = self.fc(x)

        return x