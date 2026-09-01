import numpy as np
import torch
from util import load_data

#example:  data_dict = {"t_ref": "data/atmos.192101-201012.t_ref.nc"}

class TrainingDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for NetCDF data."""

    def __init__(
        self,
        data: np.ndarray|torch.Tensor|list = None,
        sequence_length: int = 3,
    ):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        if self.data.ndim != 2:
            raise ValueError("TrainingLSTM1DDataset expects data with shape (n_timesteps, n_features).")
        self.sequence_length = sequence_length

    def __len__(self):
        return self.data.shape[0] - self.sequence_length

    def __getitem__(self, idx):
        """
        for example sequence_length=3, idx=3, returns
        inputs = [
          [var1[3], var2[3], var3[3]],
          [var1[4], var2[4], var3[4]],
          [var1[5], var2[5], var3[5]]
        ]
        target = [var1[6], var2[6], var3[6]]
        """
        inputs = self.data[idx:idx+self.sequence_length]
        target = self.data[idx+self.sequence_length]
        return inputs, target


class PredictDataset:
    """A PyTorch Dataset for NetCDF data."""

    def __init__(
        self,
        data_dict: dict,
        sequence_length: int = 3,
        test_with_global_average: bool = False
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.test_with_global_average = test_with_global_average

        self.data_dict = data_dict
        self.data, self.ntimes = load_data(self.data_dict, self.test_with_global_average)

        # initial predictions as the first sequence_length
        self.predictions = torch.tensor(self.data[:self.sequence_length], dtype=torch.float32)

    def get_inputs(self):
        """
        Returns the last sequence_length predictions as input.
        self.predictions.shape = [n_timesteps, n_features]
            pytorch modules expect [batch, sequence_length, n_features]
        """
        return self.predictions[-self.sequence_length:].unsqueeze(0)

    def add(self, value):
        """
        Adds a new prediction to the dataset.
        value.shape = [batch, n_features]
        self.predictions.shape = [n_timesteps, n_features]
        """        
        self.predictions = torch.cat((self.predictions, value.detach()), dim=0)
    
    def plot(self, tb_logger=None):
        """Plot the predictions."""
        from matplotlib import pyplot as plt
        for ivar in range(self.predictions.shape[1]):
            fig, ax = plt.subplots()
            target_mean = [self.data[itime, ivar].mean() for itime in range(self.ntimes)]
            predictions_mean = [self.predictions[itime, ivar].detach().cpu().mean() for itime in range(self.predictions.shape[0])]
            ax.plot(target_mean, color='black', label=f'actual {ivar}')
            ax.plot(predictions_mean, color='pink', label=f'predicted {ivar}')
            ax.legend()
            if tb_logger is not None:
                tb_logger.experiment.add_figure(f"evaluation {ivar}", fig)
            plt.show()

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
