from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from anemoi.datasets import open_dataset
from utils import configSetUp, log_dataset_info
from torch.utils.data import Dataset, Subset
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(message)s"
)

logger = logging.getLogger(__name__)

class SPEARAutoregressiveDataset(Dataset):
    def __init__(self, anemoi_dataset, dynamic_vars, static_vars, output_vars, nsteps=2):
        self.dataset = anemoi_dataset
        self.nsteps = nsteps

        all_vars = list(self.dataset.variables)

        self.dyn_indices = [all_vars.index(v) for v in dynamic_vars]
        self.stat_indices = [all_vars.index(v) for v in static_vars]
        self.out_indices = [all_vars.index(v) for v in output_vars]

    def __len__(self):
        return len(self.dataset) - self.nsteps

    def __getitem__(self, idx):
        frame_np = self.dataset[idx]
        frame = torch.tensor(frame_np, dtype=torch.float32)
        channels = []

        for var_idx in self.dyn_indices:
            channels.append(frame[var_idx:var_idx+1, ...])

        for stat_idx in self.stat_indices:
            channels.append(frame[stat_idx:stat_idx+1, ...])
        x = torch.cat(channels, dim=0)

        y_sequence = []
        for step in range(1, self.nsteps + 1):
            target_np = self.dataset[idx + step]
            target_frame = torch.tensor(target_np, dtype=torch.float32)
            y_sequence.append(target_frame[self.dyn_indices, ...])
        y = torch.stack(y_sequence, dim=0)

        return x, y


class SPEARLaggedDataset(Dataset):
    def __init__(self, anemoi_dataset, dynamic_vars, static_vars, output_vars, nlags=3):
        self.dataset = anemoi_dataset
        self.nlags = nlags
        
        all_vars = list(self.dataset.variables)
        
        # Find exactly where each variable lives in the channel dimension (dim 0)
        self.dyn_indices = [all_vars.index(v) for v in dynamic_vars]
        self.stat_indices = [all_vars.index(v) for v in static_vars]
        self.out_indices = [all_vars.index(v) for v in output_vars]

    def __len__(self):
        return len(self.dataset) - self.nlags - 1

    def __getitem__(self, idx):
        target_idx = idx + self.nlags 
        
        # 1. Fetch the historical frames from Anemoi once to minimize I/O
        frames = []
        for lag in range(1, self.nlags + 1): # e.g., lags 1, 2, 3
            frame_np = self.dataset[target_idx - lag] 
            frames.append(torch.tensor(frame_np, dtype=torch.float32))

        channels = []
        
        # 2. Extract DYNAMIC variables (Grouped by variable, then by lag)
        for var_idx in self.dyn_indices:
            for frame in frames:
                channels.append(frame[var_idx:var_idx+1, ...])
                
        # 3. Extract STATIC variables
        for stat_idx in self.stat_indices:
            channels.append(frames[0][stat_idx:stat_idx+1, ...])
            
        # 4. Concatenate into your final X tensor
        x = torch.cat(channels, dim=0)

        # 5. Get the Target (Y) tensor for time 't'
        target_np = self.dataset[target_idx]
        target_frame = torch.tensor(target_np, dtype=torch.float32)
        
        y = target_frame[self.out_indices, ...]
        
        return x, y

class SPEARDataStore():
    def __init__(self, tensor, original_database):
        self.tensor = tensor
        testing_dates = original_database.dates
        self.times =  pd.to_datetime(testing_dates)
        self.lat = original_database.latitudes
        self.lon = original_database.longitudes
        self.vars = list(original_database.variables)

def get_updated_channels(config):
    nlags = config.get_nlags()
    dynamics = config.dynamics
    statics = config.statics
    diags = config.diagnostics_only

    in_channels = []
    for var in dynamics:
        for lag in range(1, nlags + 1):
            in_channels.append(f"{var}(t-{lag})")

    # Static variables (no lags)
    for var in statics:
        in_channels.append(var)

    out_channels = []
    for var in config.outputs:
        out_channels.append(f"{var}(t)")

    diag_channels = []
    for var in diags:
        diag_channels.append(f"{var}(t)")

    return in_channels, out_channels, diag_channels

def split_data_set(config, dataset):
    data_type = config.data_config.get("type") # "residual" or "default"
    if data_type not in ["residual", "default"]:
        raise RuntimeError(f"The type must be 'residual' or 'default', but you specified {data_type}")

    method = config.get_data_load_method()
    if method == "lags":
        nlags = config.get_nlags()

        logger.info(f"Setting up the data to use {nlags} {method} for dynamic variables \n")
        return SPEARLaggedDataset(
                    anemoi_dataset=dataset, 
                    dynamic_vars=config.dynamics, 
                    static_vars=config.statics,
                    output_vars=config.outputs,
                    nlags=nlags
                )
    elif method == "autoregressive":
        nsteps = config.get_nregressive_steps()
        logger.info(f"Setting up the data to use {nsteps} {method} steps during training \n")
        return SPEARAutoregressiveDataset(
            anemoi_dataset=dataset,
            dynamic_vars=config.dynamics,
            static_vars=config.statics,
            output_vars=config.outputs,
            nsteps=nsteps
        )

def get_tensor(config:configSetUp, dataset, mode):
    using_gpu = torch.cuda.is_available()
    batch_size = config.batch_size
    split_set = split_data_set(config, dataset)

    num_workers = 16
    if mode == "testing" or not using_gpu:
        num_workers = 1
        batch_size = 1

    data_loader = DataLoader(
        split_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers = num_workers)

    return data_loader

'''
Loads the training, validating, and testing datasets defined in the configuration yaml
Returns an instance of the SPEARDataStore which contains
  - The dataset tensor
  - Information about the dataset (grid size, time periods)
'''
def get_dataloaders(config:configSetUp):
    if config.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Getting the training dataset from file: {config.training}")
    training = open_dataset(config.training, select=config.inputs)
    if sorted(list(training.variables)) != sorted(config.inputs):
        raise RuntimeError(f"Expected inputs and the dataset variables do not match the variables available in the dataset..."
                           f"\n---> Expected inputs: {config.inputs}"
                           f"\n---> Variables available: {list(training.variables)}")

    # Determine the size of the grid
    config.set_grid_size(
        len(np.unique(training.latitudes)),
        len(np.unique(training.longitudes))
    )
    logger.debug(f"Grid size {config.nlon}x{config.nlat}")

    # Determine the variables available in the dataset
    config.inputs = list(training.variables)
    logger.debug(f"Variables: {config.inputs}")

    # Set up the normalization method
    config.set_normalization_info(
            training.statistics['mean'], training.statistics['stdev']
    )
    logger.debug(f"Means:\n"
                 f"    {config.means}")
    logger.debug(f"Standard deviations:\n"
                 f"    {config.stds}")
    log_dataset_info(logger, "Training", training)

    # Construct training dataset
    training_tensor = get_tensor(config, training, "training")
    training_set = SPEARDataStore(training_tensor, training)

    # Construct validation dataset
    logger.info(f"Getting the validating dataset from file: {config.validating}")
    validating = open_dataset(config.validating, select=config.inputs)
    log_dataset_info(logger, "Validating", validating)
    validating_tensor = get_tensor(config, validating, "validating")
    validating_set = SPEARDataStore(validating_tensor, validating)

    # Construct testing dataset
    logger.info(f"Getting the testing dataset from file: {config.testing}")
    testing = open_dataset(config.testing, select=config.inputs)
    log_dataset_info(logger, "Testing", testing)
    testing_tensor = get_tensor(config, testing, "testing")
    testing_set = SPEARDataStore(testing_tensor, testing)

    return training_set, validating_set, testing_set

