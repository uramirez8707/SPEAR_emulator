import numpy as np
from pathlib import Path

import lightning as pl
import torch
import xarray as xr
from sklearn.model_selection import train_test_split


def load_variable(datafile: str | Path, variable: str) -> tuple[np.ndarray, int, list[int]]:
    """Open a NetCDF file and return (values, ntimes, time) for the given variable."""
    with xr.open_dataset(datafile, decode_timedelta=True) as ds:
        data = ds[variable].values
        ntimes = len(data)
        time = list(range(ntimes))
    return data, ntimes, time


def normalize(data):
    """Normalize data by mean"""
    data = np.array(data)
    return data/data.mean()


class TrainingAutoregressiveDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for NetCDF data."""

    def __init__(
        self,
        data: np.ndarray = None,
        sequence_length: int = 3,
    ):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        """
        for example sequence_length=3, idx=3, returns 
        inputs = [data[3], data[4], data[5]]
        target = data[6] 
        """
        return self.data[idx:idx+self.sequence_length], self.data[idx+self.sequence_length]


class TrainingLSTMDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for LSTM training on 1D time series (e.g. global means).

    Returns:
        inputs: shape (sequence_length, 1) — sequence of scalar values as LSTM features
        target: scalar float — next timestep value
    """

    def __init__(
        self,
        data: np.ndarray = None,
        sequence_length: int = 3,
    ):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        """
        for example, sequence_length=3, idx=3, returns
        inputs = [[data[3]], [data[4]], [data[5]]]  # shape (sequence_length, 1)
        target = data[6]  # scalar
        """
        return self.data[idx:idx+self.sequence_length].unsqueeze(-1), self.data[idx+self.sequence_length]


class AutoDataModule(pl.LightningDataModule):
    """Data module for LSTM training on 1D global-mean time series."""

    def __init__(self, sequence_length: int = 3, train_size: float = 0.8, val_size: float = 0.3, TrainingDatasetClass: torch.utils.data.Dataset = TrainingLSTMDataset):
        super().__init__()        
        self.sequence_length = sequence_length
        self.train_size = train_size
        self.val_size = val_size
        self.TrainingDatasetClass = TrainingDatasetClass
        self.train_dataset = None
        self.val_dataset = None
        self.train_data = None
        self.val_data = None

    def prepare(self, data):
        """prepare training and val set"""
        self.train_data, self.val_data = train_test_split(
            train_test_split(data, train_size=self.train_size, shuffle=False)[0],
            test_size=self.val_size,
            shuffle=False,
        )
        
        self.setup()

        print(f"Training size: {len(self.train_data)}")
        print(f"Validation size: {len(self.val_data)}")

        return self
        
    def setup(self, stage = None):
        """setup data"""

        self.train_dataset = self.TrainingDatasetClass(self.train_data, sequence_length=self.sequence_length)
        self.val_dataset = self.TrainingDatasetClass(self.val_data, sequence_length=self.sequence_length)

    def train_dataloader(self, batch_size: int = 32):
        """Load training data onto DataLoader"""
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)

    def val_dataloader(self, batch_size: int = 32):
        """Load validation data onto DataLoader"""
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)


class PredictAutoregressiveDataset():
    """A PyTorch Dataset for NetCDF data."""

    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 3,
    ):
        super().__init__()

        self.sequence_length = sequence_length

        self.data = data
        self.ntimes = len(self.data)
        self.time = list(range(self.ntimes))

        # initial predictions as the first sequence_length
        self.predictions = torch.tensor(self.data[:self.sequence_length], dtype=torch.float32)

    def get_inputs(self):
        """Returns the last sequence_length predictions as input."""
        return self.predictions[-self.sequence_length:]

    def add(self, value):
        """Adds a new prediction to the dataset."""
        self.predictions = torch.cat((self.predictions, value.unsqueeze(0)), dim=0)


class PredictLSTMDataset():
    """A dataset for autoregressive prediction with an LSTM model on 1D time series (e.g. global means)."""

    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 3,
    ):
        super().__init__()

        self.sequence_length = sequence_length

        self.data = data
        self.ntimes = len(self.data)
        self.time = list(range(self.ntimes))

        # initial predictions seeded from the first sequence_length timesteps
        self.predictions = torch.tensor(self.data[:self.sequence_length], dtype=torch.float32)

    def get_inputs(self):
        """Returns the last sequence_length predictions as input of shape (1, sequence_length, 1)."""
        return self.predictions[-self.sequence_length:].unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)

    def add(self, value):
        """Adds a new scalar prediction to the dataset."""
        self.predictions = torch.cat((self.predictions, value.detach().reshape(1)), dim=0)


