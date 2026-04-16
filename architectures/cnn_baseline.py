import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from architectures.Models import get_model_archirecture, construct_model, define_optimizer
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import r2_score
import itertools
import time

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
    def __init__(self, data, num_epochs=50, batch_size=32, lr=1e-3, case=0,
                 filters=(16, 32, 32), weight_decay=0, optimizer="Adam",
                 label="baseline", lats=None, debug=True):
        super(CNN2D, self).__init__()

        self.logger = logging.getLogger(label)
        if debug:
           self.logger.setLevel(logging.DEBUG)

        self.in_channels = data.x_train.shape[1]
        self.out_channels = data.Y_train.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.debug(f"You are running this code on {device}")

        if lats is not None:
            self.use_latitude_weights = True
            lats = torch.tensor(lats.values, dtype=torch.float32).to(device)
            weights = torch.cos(torch.deg2rad(lats))
            weights = weights / weights.mean()
            self.weights = weights.view(1, -1, 1)

        if case > 3:
            self.model = construct_model(self.in_channels, self.out_channels, filters)            
        else:
            self.model = get_model_archirecture(case, self.in_channels, self.out_channels)

        self.label = label
        self.out_path = f".results/{label}.pt"
        self.data = data
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.checkpoint = None
        self.targets = self.data.y_description

        self.optimizer = define_optimizer(self.model, optimizer, lr, weight_decay)

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

    def get_val_loss(self, target):
        return self.val_losses[target][-1]

    def train(self):
        # Check if a model already exists, load it and move on to testing:
        path = Path(self.out_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if path.exists():
            self.checkpoint = torch.load(self.out_path, map_location=device, weights_only=False)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            self.val_losses = self.checkpoint['val_losses']
            self.train_losses = self.checkpoint['train_losses']
            return

        self.logger.info(f"/n --- Starting Model Training ---")
        start_time = time.perf_counter()

        self.model.to(device)

        criterion = nn.MSELoss(reduction='none')

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
                    pixel_mse = criterion(preds[:, i], Y_batch[:, i])
                    weighted_mse = (pixel_mse * self.weights).mean()
                    total_loss += weighted_mse
                    current_batch_losses.append(weighted_mse.item())

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
                        pixel_mse = criterion(val_preds[:, i], y_val[:, i])
                        weighted_val_mse = (pixel_mse * self.weights).mean()
                        batch_val_losses[target] += weighted_val_mse.item() * x_val.size(0)

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
        end_time = time.perf_counter()
        training_time = end_time - start_time
        self.logger.info(f"Training completed in {training_time:.4f} seconds")

    def plot_loss_over_epochs(self, figname=None):
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

        min_train_loss = min(min(self.train_losses[t]) for t in self.targets)
        min_val_loss = min(min(self.val_losses[t]) for t in self.targets)

        min_loss = min(min_train_loss, min_val_loss)

        lower_limit = min_loss * 0.8
        upper_limit = max(1.0,
                  max(max(self.train_losses[t]) for t in self.targets),
                  max(max(self.val_losses[t]) for t in self.targets)) * 1.2
        lower_limit = max(lower_limit, 1e-10)

        plt.ylim(bottom=lower_limit, top=upper_limit)
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend(loc='upper right', frameon=True)
        plt.tight_layout()
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)

    def predict_with_ground_truth(self, lon, lat, add_y_actual):
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
                            lon, lat, add_y_actual)
    
        return output
    
    def forecast_rollout(self, lon, lat, add_y_actual):
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

        y_prediction = y_prediction.detach().cpu().numpy()
        
        # Convert back to physical units
        y_prediction = self.inverse_transform(y_prediction)
        y_test = self.inverse_transform(self.data.Y_test)
        label = f"{self.label}"
        output = ModelOutput(label, y_test, y_prediction,
                            self.targets, self.data.standardization_info,
                            lon, lat, add_y_actual)
    
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
                 lon, lat, add_y_actual=True, debug=True):
        self.logger = logging.getLogger(f"{label}-output")
        if debug:
           self.logger.setLevel(logging.DEBUG)

        self.label = label
        self.Y_test = Y_test
        self.Y_pred = Y_pred
        self.targets = targets
        self.var_info = var_info
        self.RMSE = []

        self.calculate_RMSE(lon, lat, add_y_actual)

    def calculate_RMSE(self, lon, lat, add_y_actual=False):
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

            zonal_mean_bias = target_err.mean(axis=(0, 2))

            units = self.get_var_units(target)

            out = {'target': target,
                 'global_RMSE': global_RMSE,
                 'spatial_RMSE': spatial_RMSE,
                 'temporal_RMSE': temporal_RMSE,
                 'zonal_mean_bias': zonal_mean_bias,
                 'prediction': self.Y_pred[:, i, :, :],
                 'regional_RMSE': self.calculate_regional_rmse(sq_err, lat, lon),
                 'units': units}

            if add_y_actual:
                out['y_actual'] = self.Y_test[:, i, :, :]

            self.RMSE.append(out)
            self.logger.debug(f"Global RMSE: {target} :: {global_RMSE}")

    def calculate_regional_rmse(self, sq_err, lattitude, longitude):
        regions = {
            "Global":   {"lat": (-90, 90)},
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
        add_y_actual = True
        if self.output is None:
            self.output = []
        self.output.append(Model.forecast_rollout(self.lon, self.lat, add_y_actual))

    def predict_with_groud_truth(self, Model):
        add_y_actual = False
        if self.output is None:
            self.output = []
            self.output.append(Model.get_persistence_baseline(self.lon, self.lat))
        self.output.append(Model.predict_with_ground_truth(self.lon, self.lat, add_y_actual))

    def plot_regional_RMSE(self, target, figname=None):
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
            bars = ax.bar(
                x + i * width,
                values[i],
                width,
              label=label
            )
            ax.bar_label(bars, padding=3, fmt='%.3f', fontweight='bold')

        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(regions)
        ax.set_ylabel("RMSE")
        ax.set_title(f"Regional RMSE — {target}")
        ax.legend()

        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)

    def plot_global_RMSE(self, target, output_statics=False):
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

        if output_statics:
            mean = np.mean(rmse_values)
            std = np.std(rmse_values)
            plt.text(
                0.95, 0.95,  # position in axes fraction (top-right corner)
                f"Mean: {mean:.3f}\nStd: {std:.3f}",
                horizontalalignment='right',
                verticalalignment='top',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        plt.show()

    def plot_temporal_RMSE(self, target, time_test, nlags=3, figname=None):
        time_converted = time_test.to_index().to_datetimeindex()

        l = float('inf')
        u = float('-inf')

        num_models = len(self.output)
        # First pass: find min and max across all models for consistent axis
        for Model in self.output:
            for variable in Model.RMSE:
                 if variable['target'] == target:
                     prediction = variable['prediction'].mean(axis=(1, 2))
                     rmse = variable['temporal_RMSE']
                     
                     upper_bound = np.max(prediction + rmse)
                     lower_bound = np.min(prediction - rmse)
                     
                     u = max(u, upper_bound)
                     l = min(l, lower_bound)

        fig, axes = plt.subplots(nrows=num_models, ncols=1, figsize=(12, 3 * num_models), sharex=True)
        if num_models == 1:
            axes = [axes]

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

                    ax.set_title(f"{target}: Global RMSE for {Model.label} Model")
                    ax.set_ylim(l, u)
                    ax.grid(True, alpha=0.3)
                    ax.set_ylabel(f"{target} ({units})")
                    x_num = np.array([d.toordinal() for d in time_converted[i:]])
                    slope, intercept = np.polyfit(x_num, variable['temporal_RMSE'], 1)
                    slope_text = f"Slope = {slope:.3e}"

                    ax.text(0.95, 0.95,
                            slope_text,
                            horizontalalignment='right',
                            verticalalignment='top',
                            transform=ax.transAxes,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5')
                    )

                    prediction = variable['prediction'].mean(axis=(1, 2))
                    rmse = variable['temporal_RMSE']

                    actual = variable['y_actual'].mean(axis=(1, 2))

                    ax.plot(time_converted[i:], prediction,
                            color='black', label=f'Avg {target} (Predicted)', linewidth=1.5)
                    
                    ax.plot(time_converted[i:], actual,
                            color='red', linestyle='dashed', label=f'Avg {target} (Actual)', linewidth=1.5)

                    ax.fill_between(time_converted[i:], prediction - rmse,
                                    prediction + rmse, color='blue', alpha=0.2, label='± RMSE')
                    ax.legend()
            if not found:
                raise RuntimeError(f"Unable to determine the results for {target} in model {Model.label}")

        plt.tight_layout()
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)

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

            if not found:
                raise RuntimeError(f"Unable to determine the results for {target} in model {Model.label}")

        plt.tight_layout()
        plt.show()

    def plot_zonal_bias(self, target, lat):
        num_models = len(self.output)
        l = 0
        for Model in self.output:
            for variable in Model.RMSE:
                if variable['target'] == target:
                    if max(abs(variable['zonal_mean_bias'])) > l:
                        l = max(abs(variable['zonal_mean_bias']))

        fig, axes = plt.subplots(nrows=num_models, ncols=1, figsize=(12, 3 * num_models), sharex=True)
        for idx, Model in enumerate(self.output):
            ax = axes[idx]

            found = False
            for variable in Model.RMSE:
                if variable['target'] == target:
                    found = True
                    units = variable['units']

                    ax.set_title(f"{target}: Zonal Mean Bias for {Model.label} Model")
                    ax.set_ylim(-l, l)
                    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
                    ax.grid(True, alpha=0.3)
                    ax.set_ylabel(f"Zonal Mean Bias ({units})")
                    ax.plot(lat, variable['zonal_mean_bias'] , linestyle='dashed', marker='o')

            if not found:
                raise RuntimeError(f"Unable to determine the results for {target} in model {Model.label}")

        plt.tight_layout()
        plt.show()

    def plot_scatter_y_pred_vs_y_actual(self, target, n=1000, random_seed=42, figname=None):
        num_models = len(self.output)
        y_actual = None
        np.random.seed(random_seed)

        fig, axes = plt.subplots(nrows=num_models, ncols=1, figsize=(12, 3 * num_models), sharex=True)
        if num_models == 1:
            axes = [axes]

        for idx, Model in enumerate(self.output):
            ax = axes[idx]
            found = False

            for variable in Model.RMSE:
                if variable['target'] == target:
                    found = True
                    units = variable['units']
                    y_pred = variable['prediction']
                    if y_actual is None:
                        y_actual = variable.get('y_actual')

                    if y_actual is None:
                        raise RuntimeError(f"Error getting the y_actual")

                    y_actual_flat = y_actual.flatten()
                    y_pred_flat = y_pred.flatten()
                    n_points = min(n, y_actual_flat.size)
                    indices = np.random.choice(y_actual_flat.size, size=n_points, replace=False)

                    ax.set_title(f"{target}: y_pred vs y_actual for {Model.label} Model")
                    y_actual_sample = y_actual_flat[indices]
                    y_pred_sample = y_pred_flat[indices]
                    r2 = r2_score(y_actual_sample, y_pred_sample)

                    ax.scatter(y_actual_sample, y_pred_sample)
                    ax.plot([y_actual_sample.min(), y_actual_sample.max()], [y_actual_sample.min(), y_actual_sample.max()], 'r--', label='1:1 line')
                    ax.set_ylim(min(y_actual_flat), max(y_actual_flat))
                    ax.grid(True, alpha=0.3)

                    ax.set_xlabel(f'Actual {target} ({units})')
                    ax.set_ylabel(f'Predicted {target} ({units})')
                    ax.text(0.05, 0.95, f"$R^2$ = {r2:.2f}", transform=ax.transAxes,
                        fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))

            if not found:
                raise RuntimeError(f"Unable to determine the results for {target} in model {Model.label}")

        plt.tight_layout()
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)


def run_configuration_ens(configurations, data, nepochs=20, nruns=20,
                          label=""):
    keys = configurations.keys()
    values = configurations.values()
    all_configs = [
        dict(zip(keys, v))
        for v in itertools.product(*values)
    ]

    print(f"Total number of configurations to test {len(all_configs)}")

    sample_configurations = random.sample(all_configs, nruns)
    print(f"Selecting {len(sample_configurations)} configurations to test")

    results = []
    for run_id, config in enumerate(sample_configurations):
        output = {}
        print(f"run id: {run_id}: {config}")

        learning_rate = config.get("learning_rate")
        batch_size = config.get("batch_size")
        filters = config.get("filters")
        optimizer = config.get("optimizers")
        weight_decay = config.get("weight_decay")

        Model = CNN2D(data,
                  num_epochs=nepochs,
                  batch_size=batch_size,
                  lr = learning_rate,
                  filters = filters,
                  weight_decay = weight_decay,
                  optimizer=optimizer,
                  case = 4,
                  label = f"run_{label}{run_id:02d}",
                  debug = False)
        Model.train()

        output['run_id'] = f"run_{run_id:02d}"
        output['validation_loss'] = Model.get_val_loss("t_ref")
        output.update(config)
        results.append(output)
    return results
