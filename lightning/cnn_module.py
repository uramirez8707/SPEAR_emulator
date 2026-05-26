import numpy as np
import torch
from util import load_data

class TrainingDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for NetCDF data.
    data = [time, n_features, height, width]
    """

    def __init__(
        self,
        data: np.ndarray|torch.Tensor|list = None,
        sequence_length: int = 3,
    ):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.nfeatures = self.data.shape[1]
        self.sequence_length = sequence_length

    def __len__(self):
        return self.data.shape[0] - self.sequence_length

    def __getitem__(self, idx):
        """
        for example sequence_length=3, idx=3, returns
        inputs = [
          var1[3], var1[4], var1[5],
          var2[3], var2[4], var2[5],
          var3[3], var3[4], var3[5]
        ]
        target = [var1[6], var2[6], var3[6]]
        """
        inputs = torch.cat(
            [self.data[idx:idx+self.sequence_length, ivar] for ivar in range(self.nfeatures)]
        )
        target = torch.stack(
            [self.data[idx+self.sequence_length, ivar] for ivar in range(self.nfeatures)], dim=0
        )
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
        self.nfeatures = self.data.shape[1]

        # initial predictions as the first sequence_length
        self.predictions = torch.tensor(self.data[:self.sequence_length], dtype=torch.float32)

    def get_inputs(self):
        """
        Returns the last sequence_length predictions as input.
        """
        return torch.cat(
            [self.predictions[-self.sequence_length:, ivar] for ivar in range(self.nfeatures)]
        )
    
    def add(self, value):
        """
        Adds a new prediction to the dataset.
        value.shape = [batch, n_features]
        self.predictions.shape = [n_timesteps, n_features]
        """
        self.predictions = torch.cat([self.predictions, value.detach().unsqueeze(0)])
    
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
            out_channels=2,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        y = self.relu(self.cnn1(x))
        y = self.cnn2(y)
        return y
