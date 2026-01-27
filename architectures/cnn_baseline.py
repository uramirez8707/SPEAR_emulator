import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from architectures.Models import get_model_archirecture
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
import cftime

import random
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(message)s"
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True  # force deterministic algorithms
    torch.backends.cudnn.benchmark = False     # disable auto-tuner for convolution


class CNN2D(nn.Module):
    def __init__(self, data, num_epochs=50, batch_size=32, lr=1e-3, case=0, label="baseline", debug=True):
        super(CNN2D, self).__init__()

        self.logger = logging.getLogger(label)
        if debug:
           self.logger.setLevel(logging.DEBUG)

        self.in_channels = data.x_train.shape[1]
        self.out_channels = data.Y_train.shape[1]

        self.model = get_model_archirecture(case, self.in_channels, self.out_channels)
        self.label = label
        self.out_path = f".results/{label}.pt"
        self.data = data
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.checkpoint = None
        self.targets = self.data.y_description
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.logger.debug(self.model)
        self.logger.debug(f"Input channels: {self.in_channels}")
        self.logger.debug(f"Output channels: {self.out_channels}")
        self.logger.debug(f"Number of epochs: {self.num_epochs}")
        self.logger.debug(f"Batch Size: {self.batch_size}")
        self.logger.debug(f"Learning Rate: {self.lr}")
        self.logger.debug(f"Inputs: {self.data.x_description}")
        self.logger.debug(f"Targets: {self.targets}")

    def forward(self, x):
        return self.model(x)

    def train_multi_step_loss(self, load_model=False, horizon=3):
        if not load_model:
            path = Path(self.out_path)
            if path.exists():
                self.checkpoint = torch.load(self.out_path, weights_only=False)
                self.model.load_state_dict(self.checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
                self.val_losses = self.checkpoint['val_losses']
                self.train_losses = self.checkpoint['train_losses']
                return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        criterion = nn.MSELoss()
        train_loader, validate_loader = self.create_data_tensors()

        epoch_train_losses = {target: [] for target in self.targets}
        epoch_val_losses = {target: [] for target in self.targets}

        for epoch in range(self.num_epochs):
            # Training Phase
            self.model.train()
            batch_train_losses = {target: 0.0 for target in self.targets}
            total_train_samples = 0

            for x_batch, Y_batch in train_loader:
                x_batch = x_batch.to(device)
                Y_batch = Y_batch.to(device)

                self.optimizer.zero_grad()
                preds = self.model(x_batch)

                total_loss = 0.0
                current_batch_losses = []

                for i, target in enumerate(self.targets):
                    target_loss = criterion(preds[:, i], Y_batch[:, i])
                    current_batch_losses.append(target_loss.item())
                    total_loss += target_loss

                total_loss.backward()
                self.optimizer.step()

                # Accumulate loss and number of samples for each target
                for i, target in enumerate(self.targets):
                    batch_train_losses[target] += current_batch_losses[i] * x_batch.size(0)
                total_train_samples += x_batch.size(0)

            # Calculate average training loss for each target
            for target in self.targets:
                avg_train_loss = batch_train_losses[target] / total_train_samples
                epoch_train_losses[target].append(avg_train_loss)

            # Validation Phase
            self.model.eval()
            batch_val_losses = {target: 0.0 for target in self.targets}
            total_val_samples = 0

            with torch.no_grad():
                for x_val, y_val in validate_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)

                    val_preds = self.model(x_val)

                    for i, target in enumerate(self.targets):
                        specific_val_loss = criterion(val_preds[:, i], y_val[:, i])
                        batch_val_losses[target] += specific_val_loss.item() * x_val.size(0)

                    total_val_samples += x_val.size(0)

            # Calculate average validation loss for each target
            for target in self.targets:
                avg_val_loss = batch_val_losses[target] / total_val_samples
                epoch_val_losses[target].append(avg_val_loss)

            # Logging every 10 epochs
            if (epoch + 1) % 10 == 0:
                for target in self.targets:
                    self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, "
                                     f"Train Loss ({target}): {epoch_train_losses[target][-1]:.4f}, "
                                     f"Validation Loss ({target}): {epoch_val_losses[target][-1]:.4f}")

        self.save_checkpoint(epoch_train_losses, epoch_val_losses)
        self.logger.info("Training completed.")

    def train(self):
        # Check if a model already exists, load it and move on to testing:
        path = Path(self.out_path)
        if path.exists():
            self.checkpoint = torch.load(self.out_path, weights_only=False)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            self.val_losses = self.checkpoint['val_losses']
            self.train_losses = self.checkpoint['train_losses']
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        criterion = nn.MSELoss()

        train_loader, validate_loader = self.create_data_tensors()

        epoch_train_losses = {target: [] for target in self.targets}
        epoch_val_losses = {target: [] for target in self.targets}

        for epoch in range(self.num_epochs):
            # Training Phase
            self.model.train()
            batch_train_losses = {target: 0.0 for target in self.targets}
            total_train_samples = 0

            for x_batch, Y_batch in train_loader:
                x_batch = x_batch.to(device)
                Y_batch = Y_batch.to(device)

                self.optimizer.zero_grad()
                preds = self.model(x_batch)

                total_loss = 0.0
                current_batch_losses = []

                for i, target in enumerate(self.targets):
                    target_loss = criterion(preds[:, i], Y_batch[:, i])
                    current_batch_losses.append(target_loss.item())
                    total_loss += target_loss

                total_loss.backward()
                self.optimizer.step()

                # Accumulate loss and number of samples for each target
                for i, target in enumerate(self.targets):
                    batch_train_losses[target] += current_batch_losses[i] * x_batch.size(0)
                total_train_samples += x_batch.size(0)

            # Calculate average training loss for each target
            for target in self.targets:
                avg_train_loss = batch_train_losses[target] / total_train_samples
                epoch_train_losses[target].append(avg_train_loss)

            # Validation Phase
            self.model.eval()
            batch_val_losses = {target: 0.0 for target in self.targets}
            total_val_samples = 0

            with torch.no_grad():
                for x_val, y_val in validate_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)

                    val_preds = self.model(x_val)

                    for i, target in enumerate(self.targets):
                        specific_val_loss = criterion(val_preds[:, i], y_val[:, i])
                        batch_val_losses[target] += specific_val_loss.item() * x_val.size(0)

                    total_val_samples += x_val.size(0)

            # Calculate average validation loss for each target
            for target in self.targets:
                avg_val_loss = batch_val_losses[target] / total_val_samples
                epoch_val_losses[target].append(avg_val_loss)

            # Logging every 10 epochs
            if (epoch + 1) % 10 == 0:
                for target in self.targets:
                    self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, "
                                     f"Train Loss ({target}): {epoch_train_losses[target][-1]:.4f}, "
                                     f"Validation Loss ({target}): {epoch_val_losses[target][-1]:.4f}")

        self.save_checkpoint(epoch_train_losses, epoch_val_losses)
        self.logger.info("Training completed.")

    def plot_loss_over_epochs(self):
        plt.figure(figsize=(12, 6))

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        for i, target in enumerate(self.targets):
            color = colors[i % len(colors)]

            train_losses = self.train_losses[target]
            val_losses = self.val_losses[target]
            epochs = range(len(train_losses))

            plt.plot(epochs, train_losses, label=f'Training Loss - {target}',
                     color=color, linestyle='-', linewidth=2)

            plt.plot(epochs, val_losses, label=f'Validation Loss - {target}',
                     color=color, linestyle='--', alpha=0.8)

        plt.xlabel("Epoch")
        plt.ylabel("Loss (Log Scale)")
        plt.title("Model Convergence: Loss over Epochs")

        plt.yscale('log')
        plt.ylim(bottom=min(min(self.train_losses[t]) for t in self.targets) * 0.8, top=0.5)
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend(loc='upper right', frameon=True)
        plt.tight_layout()
        plt.show()

    def predict_with_ground_truth(self, lon, lat):
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        x_test = torch.tensor(self.data.x_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred = self.model(x_test).cpu().numpy()

        # Get back to physical units:
        y_pred = self.inverse_transform(y_pred)
        y_test = self.inverse_transform(self.data.Y_test)
        label = self.label
        output = ModelOutput(label, y_test, y_pred,
                            self.targets, self.data.standardization_info,
                            lon, lat)
    
        return output
    
    def forecast_rollout(self, lon, lat):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device)

        y_test = torch.tensor(self.data.Y_test, dtype=torch.float32).to(device) # (physical units)
        x_test = torch.tensor(self.data.x_test, dtype=torch.float32).to(device) # (normalized)
        mappings = self.get_mappings()

        nsteps = y_test.shape[0]
        self.logger.debug(f"Doing inference for {nsteps} time steps")

        y_prediction = torch.zeros_like(y_test)
        with torch.no_grad():
            X = x_test[0:1] # Initial condition

            for t in range(nsteps):
                y = self.model(X)
                y_prediction[t] = y[0] # Save the prediction

                if t == nsteps - 1:
                    break

                # Copy the data for the next time step
                X_next = x_test[t+1:t+2].clone()

                # Overwrite ONLY the targets that were predicted
                for i, target in enumerate(self.targets):
                    lag_indices = mappings[i]

                    for j in range(len(lag_indices) - 1):
                        X_next[:, lag_indices[j]] = X[:, lag_indices[j + 1]]
                    X_next[:, lag_indices[-1]] = y[:, i]
                X = X_next

        y_prediction = y_prediction.numpy()
        
        # Convert back to physical units
        y_prediction = self.inverse_transform(y_prediction)
        y_test = self.inverse_transform(self.data.Y_test)
        label = f"{self.label}"
        output = ModelOutput(label, y_test, y_prediction,
                            self.targets, self.data.standardization_info,
                            lon, lat)
    
        return output

    def get_standardization_info(self, target):
        for info in self.data.standardization_info:
            if target == info['var_name']:
                return info['mean'], info['std']
        raise RuntimeError(f"Unable to find the get_standardization_info for {target}")

    def get_var_units(self, target):
        for info in self.data.standardization_info:
            if target == info['var_name']:
                return info['original_units']

        raise RuntimeError(f"Unable to find orginial units for {target}")

    def inverse_transform(self, y):
        y_inv = y.copy()
        for i, target in enumerate(self.targets):
            mean, std = self.get_standardization_info(target)
            y_inv[:, i, :, :] = y[:, i, :, :] * std + mean
        return y_inv

    def get_mappings(self):
        mappings = []
        for target in self.targets:
            indices = [i for i, s in enumerate(self.data.x_description) if target in s]
            mappings.append(indices)
        return mappings

    def get_persistence_baseline(self, lon, lat):
        y_actual = self.data.Y_test[1:, :, :, :]
        y_persistence = self.data.Y_test[:-1, :, :, :]

        # Get back to physical units:
        y_actual = self.inverse_transform(y_actual)
        y_persistence = self.inverse_transform(y_persistence)

        persistence_results = ModelOutput("Persistence", y_actual, y_persistence,
                                         self.targets, self.data.standardization_info,
                                         lon, lat)
        return persistence_results

    def save_checkpoint(self, train_losses, val_losses):
        self.train_losses = train_losses
        self.val_losses = val_losses
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        torch.save(checkpoint, self.out_path)
        self.logger.info(f"Checkpoint saved at epoch {self.num_epochs}.")

    def create_data_tensors(self):
        g = torch.Generator()
        g.manual_seed(42)

        g2 = torch.Generator()
        g2.manual_seed(42)

        def worker_init_fn(worker_id):
            np.random.seed(42 + worker_id)
            random.seed(42 + worker_id)

        x_train_t = torch.tensor(self.data.x_train, dtype=torch.float32)
        Y_train_t = torch.tensor(self.data.Y_train, dtype=torch.float32)
        x_validate_t = torch.tensor(self.data.x_validate, dtype=torch.float32)
        Y_validate_t = torch.tensor(self.data.Y_validate, dtype=torch.float32)

        train_ds = TensorDataset(x_train_t, Y_train_t)
        validate_ds = TensorDataset(x_validate_t, Y_validate_t)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  generator=g,
                                  worker_init_fn=worker_init_fn,
                                  num_workers=0)
        validate_loader = DataLoader(validate_ds, batch_size=self.batch_size, shuffle=False,
                                 generator=g2,
                                 worker_init_fn=worker_init_fn,
                                 num_workers=0)

        return train_loader, validate_loader


def plot_subplot(ax, lon, lat, data, title, colorbar_label, vmin, vmax, cmap='viridis'):
    """
    Helper function to plot data on a given axis (ax).
    """
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, edgecolor='black')

    im = ax.pcolormesh(lon, lat, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label(colorbar_label)

class ModelOutput:
    def __init__(self, label, Y_test, Y_pred, targets, var_info,
                 lon, lat, debug=True):
        self.logger = logging.getLogger(f"{label}-output")
        if debug:
           self.logger.setLevel(logging.DEBUG)

        self.label = label
        self.Y_test = Y_test
        self.Y_pred = Y_pred
        self.targets = targets
        self.var_info = var_info
        self.RMSE = []

        self.calculate_RMSE(lon, lat)

    def calculate_RMSE(self, lon, lat):
        err = self.Y_pred - self.Y_test

        # Calculating latitude weights
        area_weights = np.cos(np.deg2rad(lat.to_numpy()))[:, None]
        area_weights = np.broadcast_to(area_weights, (lat.size, lon.size))
        area_weights = area_weights / np.sum(area_weights)

        self.logger.debug(f"Size of the area weight:: {area_weights.shape}")

        for i, target in enumerate(self.targets):
            self.logger.debug(f"Calculating RMSE for target: {target}")

            target_err = err[:, i, :, :]
            sq_err = target_err ** 2

            # --- Global RMSE (area + time weighted) ---
            global_mse = np.sum(sq_err * area_weights[None, :, :]) / sq_err.shape[0]
            global_RMSE = np.sqrt(global_mse)

            # --- Spatial RMSE (per grid cell, weighted over time only) ---
            spatial_RMSE = np.sqrt(np.mean(sq_err, axis=0))  # (lat, lon)

            # --- Temporal RMSE (per timestep, area-weighted) ---
            temporal_mse = np.sum(sq_err * area_weights[None, :, :], axis=(1, 2))
            temporal_RMSE = np.sqrt(temporal_mse)

            units = self.get_var_units(target)

            self.RMSE.append(
                {'target': target,
                 'global_RMSE': global_RMSE,
                 'spatial_RMSE': spatial_RMSE,
                 'temporal_RMSE': temporal_RMSE,
                 'regional_RMSE': self.calculate_regional_rmse(sq_err, lat, lon),
                 'units': units}
                )
            self.logger.debug(f"Global RMSE: {target} :: {global_RMSE}")

    def calculate_regional_rmse(self, sq_err, lattitude, longitude):
        regions = {
            "Tropics":  {"lat": (-30, 30)},
            "NH":       {"lat": (0, 90)},
            "SH":       {"lat": (-90, 0)},
            "NH_mid":   {"lat": (30, 60)},
        }

        lat = lattitude.to_numpy()
        lon = longitude.to_numpy()

        area_weights = np.cos(np.deg2rad(lat))[:, None]
        area_weights = np.broadcast_to(area_weights, (lat.size, lon.size))

        regional_rmse = {}

        for region_name, bounds in regions.items():
            mask = np.ones((lat.size, lon.size), dtype=bool)

            if "lat" in bounds:
                lat_min, lat_max = bounds["lat"]
                mask &= (lat[:, None] >= lat_min) & (lat[:, None] <= lat_max)

            if "lon" in bounds:
                lon_min, lon_max = bounds["lon"]
                mask &= (lon[None, :] >= lon_min) & (lon[None, :] <= lon_max)

            w_reg = area_weights * mask
            w_reg = w_reg / np.sum(w_reg)

            mse = np.mean(np.sum(sq_err * w_reg[None, :, :], axis=(1, 2)))
            regional_rmse[region_name] = np.sqrt(mse)

        return regional_rmse

    def get_var_units(self, target):
        for info in self.var_info:
            if target == info['var_name']:
                return info['original_units']

        raise RuntimeError(f"Unable to find orginial units for {target}")
    
class Results:
    def __init__(self, longitude, latitude):
        self.output = None
        self.lat = latitude
        self.lon = longitude

    def add_model_output(self, Model, run_type="ground_truth"):
        if run_type == "ground_truth":
            self.predict_with_groud_truth(Model)
        elif run_type == "forecast_rollout":
            self.forecast_rollout(Model)
        else:
            raise RuntimeError("The run type is not valid!")

    def forecast_rollout(self, Model):
        if self.output is None:
            self.output = []
        self.output.append(Model.forecast_rollout(self.lon, self.lat))

    def predict_with_groud_truth(self, Model):
        if self.output is None:
            self.output = []
            self.output.append(Model.get_persistence_baseline(self.lon, self.lat))
        self.output.append(Model.predict_with_ground_truth(self.lon, self.lat))

    def plot_regional_RMSE(self, target):
        regional_RMSE = []
        model_labels = []
        for Model in self.output:
            found = False
            for variable in Model.RMSE:
                if variable['target'] == target:
                    model_labels.append(Model.label)
                    found = True
                    regional_RMSE.append(variable['regional_RMSE'])
                    break

            if not found:
                raise RuntimeError(f"Unable to determine the results for {target} in model {Model.label}")

        regions = list(regional_RMSE[0].keys())
        n_models = len(model_labels)
        n_regions = len(regions)

        values = np.array([
            [rmse[r] for r in regions]
            for rmse in regional_RMSE
        ])  # shape (models, regions)

        x = np.arange(n_regions)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(10, 5))

        for i, label in enumerate(model_labels):
            ax.bar(
                x + i * width,
                values[i],
                width,
              label=label
            )

        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(regions)
        ax.set_ylabel("RMSE")
        ax.set_title(f"Regional RMSE — {target}")
        ax.legend()

        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_global_RMSE(self, target):
        rmse_values = []
        model_labels = []
        units = 'None'

        for Model in self.output:
            found = False
            for variable in Model.RMSE:
                if variable['target'] == target:
                    rmse_values.append(variable['global_RMSE'])
                    model_labels.append(Model.label)
                    units = variable['units']
                    found = True
                    break
            if not found:
                raise RuntimeError(f"Unable to determine the results for {target} in model {Model.label}")

        plt.figure(figsize=(12, 6))
        x = np.arange(len(model_labels))
        bars = plt.bar(x, rmse_values, color='tab:blue', alpha=0.75, edgecolor='black', linewidth=0.5)
        plt.bar_label(bars, padding=3, fmt='%.3f', fontweight='bold')
    
        plt.xticks(x, model_labels, rotation=30)
        plt.ylabel(f"RMSE ({units})")
        plt.title(f"Global RMSE Comparison by Model and {target}")
        plt.tight_layout()
        plt.show()

    def plot_temporal_RMSE(self, target, time_test, nlags=3):
        time_converted = time_test.to_index().to_datetimeindex()

        l = None

        num_models = len(self.output)
        fig, axes = plt.subplots(nrows=num_models, ncols=1, figsize=(12, 3 * num_models), sharex=True)
        for idx, Model in enumerate(self.output):
            ax = axes[idx]
            i = nlags
            if Model.label == "Persistence":
                i = nlags + 1
    
            found = False
            for variable in Model.RMSE:
                if variable['target'] == target:
                    found = True
                    units = variable['units']
                    if l is None:
                        l = min(variable['temporal_RMSE'])
                        u = max(variable['temporal_RMSE'])

                    ax.set_title(f"{target}: Global RMSE for {Model.label} Model")
                    ax.set_ylim(l, u)
                    ax.grid(True, alpha=0.3)
                    ax.set_ylabel(f"RMSE ({units})")
                    ax.plot(time_converted[i:], variable['temporal_RMSE'], linestyle='dashed', marker='o')
            if not found:
                raise RuntimeError(f"Unable to determine the results for {target} in model {Model.label}")

        plt.tight_layout()
        plt.show()

    def plot_spatial_RMSE(self, target, lon, lat):
        num_models = len(self.output)
        l = None
        fig, axes = plt.subplots(num_models, 1, figsize=(8, 5 * num_models), subplot_kw={'projection': ccrs.PlateCarree()})
        for idx, Model in enumerate(self.output):
            found = False

            for variable in Model.RMSE:
                if variable['target'] == target:
                    found = True
                    units = variable['units']
                    if l is None:
                        l = min(variable['temporal_RMSE'])
                        u = max(variable['temporal_RMSE'])

                    data = variable['spatial_RMSE']
                    title = f"Spatial RMSE - {Model.label}"
                    colorbar_label = f"RMSE ({units})"
                    plot_subplot(axes[idx], lon, lat, data, title, colorbar_label, l, u)


        plt.tight_layout()
        plt.show()



