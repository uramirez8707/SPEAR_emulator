from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from anemoi.datasets import open_dataset
from utils import configSetUp
from torch.utils.data import Dataset, Subset

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

def get_updated_channels(config):
    nlags = config.data_config.get("method").get("nlags")
    dynamics = config.dynamics
    statics = config.statics

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

    return in_channels, out_channels

def split_data_set(config, dataset):
    data_type = config.data_config.get("type") # "residual" or "default"
    if data_type not in ["residual", "default"]:
        raise RuntimeError(f"The type must be 'residual' or 'default', but you specified {data_type}")

    method = config.data_config.get("method").get("name")
    if method not in ["lags", "autoregressive"]:
        raise RuntimeError(f"The method name must be 'lags' or 'autoregressive', but you specified {method}")

    if method == "lags":
        return SPEARLaggedDataset(
                    anemoi_dataset=dataset, 
                    dynamic_vars=config.dynamics, 
                    static_vars=config.statics,
                    output_vars=config.outputs,
                    nlags=config.data_config.get("method").get("nlags")
                )

def get_tensor(config:configSetUp, dataset, mode):
    batch_size = config.batch_size
    split_set = split_data_set(config, dataset)

    # Only shuffle if we are training
    if mode == "training_shuffle":
        g = torch.Generator()
        g.manual_seed(config.seed)

        indices = torch.randperm(len(split_set), generator=g).tolist()
        final_set = Subset(split_set, indices)
    else:
        # Keep validation/testing in chronological order
        final_set = split_set

    data_loader = DataLoader(
        final_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers = 16)

    return data_loader

def get_dataloaders(config:configSetUp):
    training = open_dataset(config.training)
    if sorted(list(training.variables)) != sorted(config.inputs):
        raise RuntimeError(f"Expected inputs and the dataset variables do not match the variables available in the dataset..."
                           f"\n---> Expected inputs: {config.inputs}"
                           f"\n---> Variables available: {list(training.variables)}")
    config.inputs = list(training.variables)
    config.set_grid_size(
        len(np.unique(training.latitudes)),
        len(np.unique(training.longitudes))
    )

    config.set_normalization_info(
            training.statistics['mean'], training.statistics['stdev']
    )
    print(f"Training shape: {training.shape}")
    print(f"Training actual dates: {training.dates[0]} to {training.dates[-1]}\n")
    training_tensor = get_tensor(config, training, "training")

    validating = open_dataset(config.validating)
    print(f"Validating shape: {validating.shape}")
    print(f"Validating actual dates: {validating.dates[0]} to {validating.dates[-1]}\n")
    validating_tensor = get_tensor(config, validating, "validating")

    testing = open_dataset(config.testing)
    print(f"Testing shape: {testing.shape}")
    print(f"Testing actual dates: {testing.dates[0]} to {testing.dates[-1]}\n")
    testing_tensor = get_tensor(config, testing, "testing")

    if list(training.variables) != config.inputs:
        raise RuntimeError(f"Expected inputs and the dataset variables do not match the variables available in the dataset..."
                           f"\n---> Expected inputs: {config.inputs}"
                           f"\n---> Variables available: {list(training.variables)}")

    return training_tensor, validating_tensor, testing_tensor

