import numpy as np
import torch
import xarray as xr

#example:  data_dict = {"t_ref": "data/atmos.192101-201012.t_ref.nc"}

def load_variable(data_dict: dict) -> dict[str, np.ndarray]:
    """Open NetCDF files and return a mapping of variable names to arrays."""
    
    data = {}
    for variable, datafile in data_dict.items():
        with xr.open_dataset(datafile, decode_timedelta=True) as ds:
            data[variable] = ds[variable].values
    return data
    
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
            raise ValueError("TrainingDataset expects data with shape (n_timesteps, n_features).")
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
        self.data = None
        self.ntimes = None
        self.time = None

        loaded_data = load_variable(self.data_dict)
        
        key = list(loaded_data.keys())[0]
        self.ntimes = loaded_data[key].shape[0]
        
        datalist = []
        if self.test_with_global_average:
            for itime in range(self.ntimes):
                datalist.append([loaded_data[variable][itime].mean() for variable in self.data_dict.keys()])
        else:
            for itime in range(self.ntimes):
                datalist.append([loaded_data[variable][itime] for variable in self.data_dict.keys()])
        
        self.data = np.array(datalist)

        # initial predictions as the first sequence_length
        self.predictions = torch.tensor(self.data[:self.sequence_length], dtype=torch.float32)

    def get_inputs(self):
        """Returns the last sequence_length predictions as input."""
        return self.predictions[-self.sequence_length:]

    def add(self, value):
        """Adds a new prediction to the dataset."""
        self.predictions = torch.cat((self.predictions, value.detach().unsqueeze(0)), dim=0)
    
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

