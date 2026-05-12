from pathlib import Path
from types import SimpleNamespace

import lightning as pl
import numpy as np
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
    return data / data.mean()


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
        datalist: np.ndarray = None,
        sequence_length: int = 3,
    ):
        super().__init__()
        self.datalist = torch.tensor(datalist, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.datalist) - self.sequence_length

    def __getitem__(self, idx):
        """
        for example, sequence_length=3, idx=3, returns
        inputs = [
            [input_data1[3], input_data2[3]], 
            [input_data1[4], input_data2[4]], 
            [input_data1[5], input_data2[5]]
        ]         
        target = input_data1[6], input_data2[6]
        """
        return self.datalist[idx:idx+self.sequence_length,:], self.datalist[idx+self.sequence_length,:]


class AutoDataModule(pl.LightningDataModule):
    """Data module for LSTM training on 1D global-mean time series."""

    def __init__(
        self,
        sequence_length: int = 3,
        TrainingDatasetClass: torch.utils.data.Dataset = TrainingLSTMDataset,
    ):
        super().__init__()
        self.sequence_length = sequence_length        
        self.TrainingDatasetClass = TrainingDatasetClass

        self.sizes = SimpleNamespace(train = None, val = None)
        self.data = SimpleNamespace(train = None, val = None)
        self.dataset = SimpleNamespace(train = None, val = None)

    def prepare(self, data, train_size: float = 0.8, val_size: float = 0.2):
        """prepare training and val set"""

        train, val = train_test_split(
            train_test_split(data, train_size=train_size, shuffle=False)[0],
            test_size=val_size,
            shuffle=False,
        )    

        self.data.train = torch.tensor(train, dtype=torch.float32)
        self.data.val = torch.tensor(val, dtype=torch.float32)

        self.sizes.train = len(self.data.train)
        self.sizes.val = len(self.data.val)
        
        self.setup()

        print(f"Training size: {self.sizes.train}")
        print(f"Validation size: {self.sizes.val}")

        return self
        
    def setup(self, stage: str = None):
        """setup data"""

        self.dataset.train = self.TrainingDatasetClass(self.data.train, sequence_length=self.sequence_length)
        self.dataset.val = self.TrainingDatasetClass(self.data.val, sequence_length=self.sequence_length)

    def train_dataloader(self, batch_size: int = 32):
        """Load training data onto DataLoader"""
        returnme = torch.utils.data.DataLoader(self.dataset.train, batch_size=batch_size, shuffle=False)
        return returnme

    def val_dataloader(self, batch_size: int = 32):
        """Load validation data onto DataLoader"""
        return torch.utils.data.DataLoader(self.dataset.val, batch_size=batch_size, shuffle=False)


class PredictAutoregressiveDataset:
    """A PyTorch Dataset for NetCDF data."""

    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 3,
    ):
        super().__init__()

        self.sequence_length = sequence_length

        self.data = data

        # initial predictions as the first sequence_length
        self.predictions = torch.tensor(self.data[:self.sequence_length], dtype=torch.float32)

    def get_inputs(self):
        """Returns the last sequence_length predictions as input."""
        return self.predictions[-self.sequence_length:]

    def add(self, value):
        """Adds a new prediction to the dataset."""
        self.predictions = torch.cat((self.predictions, value.unsqueeze(0)), dim=0)


class PredictLSTMDataset:
    """A dataset for autoregressive prediction with an LSTM model on 1D time series (e.g. global means)."""

    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 3,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.data = data
        self.ntimes = self.data.shape[0]

        # initial predictions seeded from the first sequence_length timesteps
        self.predictions = torch.tensor(self.data[:self.sequence_length, :], dtype=torch.float32)

    def get_inputs(self):
        """Returns the last sequence_length predictions as input of shape (sequence_length, input_size)."""
        return self.predictions[-self.sequence_length:, :].unsqueeze(0)  # add batch dimension

    def add(self, value):
        """Adds a new scalar prediction to the dataset."""
        self.predictions = torch.cat((self.predictions, value.detach()), dim=0)


