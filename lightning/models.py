from pathlib import Path

import numpy as np
import torch


class SimpleCNN(torch.nn.Module):
    """A simple CNN model with one convolutional layer."""

    def __init__(self, in_channels: int = 3):
        super().__init__()        
        self.cnn1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=2*in_channels,
            kernel_size=3,
            padding=1
        )
        self.relu = torch.nn.ReLU()
        self.cnn2 = torch.nn.Conv2d(
            in_channels=2*in_channels,
            out_channels=1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        y = self.relu(self.cnn1(x))
        y = self.cnn2(y)
        return y.squeeze(1)


class SimpleLSTM(torch.nn.Module):
    """A simple LSTM model."""

    def __init__(self, input_size: int = 1, output_size: int = 1, hidden_size: int = 10, num_layers: int = 1):
        super().__init__()        
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def normalize(self, x):
        """Min-max normalize x per variable per batch item.

        Args:
            x: Tensor of shape (batch, seq_len, input_size)

        Returns:
            Normalized tensor with values in [0, 1] along the sequence dimension.
        """
        # x shape: (batch, seq_len, input_size)
        x = x.clone()
        mins, maxs = [], []
        for ibatch in range(x.shape[0]):
            min_ivar, max_ivar = [], []
            for ivar in range(x.shape[2]):
                x_min = x[ibatch, :, ivar].min()
                x_max = x[ibatch, :, ivar].max()
                x[ibatch, :, ivar] = (x[ibatch, :, ivar] - x_min) / (x_max - x_min + 1e-8)
                min_ivar.append(x_min)
                max_ivar.append(x_max)
            mins.append(min_ivar)
            maxs.append(max_ivar)
        return x, mins, maxs

    def denormalize(self, y, mins, maxs):
        """Undo min-max normalization on the output.

        Args:
            y:    Tensor of shape (batch, output_size)
            mins: List of per-(batch, variable) minimums from normalize()
            maxs: List of per-(batch, variable) maximums from normalize()

        Returns:
            Denormalized tensor in the original input scale.
        """
        y = y.clone()
        for ibatch in range(y.shape[0]):
            for ivar in range(y.shape[1]):
                y[ibatch, ivar] = y[ibatch, ivar] * (maxs[ibatch][ivar] - mins[ibatch][ivar] + 1e-8) + mins[ibatch][ivar]
        return y

    def forward(self, x):
        x, mins, maxs = self.normalize(x)
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out[:, -1, :])  # Use the last output of the LSTM
        return self.denormalize(output, mins, maxs)
