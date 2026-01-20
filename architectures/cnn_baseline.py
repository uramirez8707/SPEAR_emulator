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
        self.label = f".results/{label}.pt"
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

    def train(self):
        # Check if a model already exists, load it and move on to testing:
        path = Path(self.label)
        if path.exists():
            self.checkpoint = torch.load(self.label, weights_only=False)
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

    def test_model(self):
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        x_test = torch.tensor(self.data.x_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred = self.model(x_test).cpu().numpy()

        self.y_pred = self.inverse_transform(y_pred)
        self.y_test = self.inverse_transform(self.data.Y_test)

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

    def calculate_RMSE(self, y_pred=None):
        RMSE = []
        if y_pred is None:
            y_predicted = self.y_pred
        else:
            y_predicted = y_pred

        err = y_predicted - self.y_test
        for i, target in enumerate(self.targets):
            target_err = err[:, i, :, :]
            target_rmse = {}
            target_rmse['Target'] = target
            target_rmse['Global'] = np.sqrt(np.mean(target_err**2))
            target_rmse['Per grid point'] = np.sqrt(np.mean(target_err ** 2, axis=0))
            target_rmse['Over time'] = np.sqrt(np.mean(target_err ** 2, axis=(1,2)))
            target_rmse['Variable units'] = self.get_var_units(target)

            RMSE.append(target_rmse)
            self.logger.debug(f"Global RMSE: {target} :: {target_rmse['Global']}")

        if y_pred is None:
            self.RMSE = RMSE

        return RMSE


    def inference(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device)

        y_test = torch.tensor(self.y_test, dtype=torch.float32).to(device) # (physical units)
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

        return y_prediction.numpy()

    def get_mappings(self):
        mappings = []
        for target in self.targets:
            indices = [i for i, s in enumerate(self.data.x_description) if target in s]
            mappings.append(indices)
        return mappings

    def get_persistence_RMSE(self):
        persistence_results = []
        for i, target in enumerate(self.targets):
            y_actual = self.y_test[1:, i, :, :]
            y_persistence = self.y_test[:-1, i, :, :]

            global_persistence_rmse = np.sqrt(np.mean((y_actual - y_persistence)**2))
            time_persistence_rmse = np.sqrt(np.mean((y_actual - y_persistence)**2, axis=(1, 2)))
            grid_persistence_rmse = np.sqrt(np.mean((y_actual - y_persistence)**2, axis=0))

            persistence_results.append({
                'Target': target,
                'Global': global_persistence_rmse,
                'Over time': time_persistence_rmse,
                'Per grid point': grid_persistence_rmse
            })

            print(f"Persistence - Global RMSE: {target} :: {global_persistence_rmse}")
        return persistence_results

    def plot_global_RMSE(self, persistence_RMSE):
        targets = [f"{d['Target']}\n({d.get('Variable units', 'no units')})" for d in self.RMSE]
        model_rmse = [d['Global'] for d in self.RMSE]
        pers_rmse = [d['Global'] for d in persistence_RMSE]

        x = np.arange(len(targets))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, model_rmse, width, label='CNN Model', color='#1f77b4')
        rects2 = ax.bar(x + width/2, pers_rmse, width, label='Persistence', color='#7f7f7f', alpha=0.7)

        ax.set_ylabel('Global RMSE')
        ax.set_xlabel('Target Variable')
        ax.set_title('Global RMSE Comparison: Model vs. Persistence')
        ax.set_xticks(x)
        ax.set_xticklabels(targets)
        ax.legend()

        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        plt.show()


    def plot_RMSE_per_grid_point(self, lon, lat, persistence_RMSE):
        for i, target_rmse in enumerate(self.RMSE):
            target = target_rmse['Target']
            y_test_map = self.y_test[:, i, :, :]
            y_pred_map = self.y_pred[:, i, :, :]
            rmse_map = target_rmse['Per grid point']

            fig, axes = plt.subplots(1, 2, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})

            data = y_test_map.mean(axis=0)
            vmin = np.min(data)
            vmax = np.max(data)
            title = f"True Values - {target}"
            colorbar_label = target_rmse['Variable units']
            plot_subplot(axes[0], lon, lat, data, title, colorbar_label, vmin, vmax)

            data = y_pred_map.mean(axis=0)
            title = f"Predictions - {target}"
            plot_subplot(axes[1], lon, lat, data, title, colorbar_label, vmin, vmax)

            plt.tight_layout()
            plt.show()

            fig, axes = plt.subplots(1, 2, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})

            target_persistence = persistence_RMSE[i]
            data = target_persistence['Per grid point']
            vmin = np.min(data)
            vmax = np.max(data)
            title = f"RMSE per grid point - {target}"
            colorbar_label = f"RMSE ({target_rmse['Variable units']})"
            plot_subplot(axes[0], lon, lat, data, title, colorbar_label, vmin, vmax)

            data = rmse_map
            title = f"RMSE per grid point - {target}"
            plot_subplot(axes[1], lon, lat, data, title, colorbar_label, vmin, vmax)


            plt.tight_layout()
            plt.show()

    def plot_RMSE_per_time(self, time, nlags=3):
        for i, target_rmse in enumerate(self.RMSE):
            target = target_rmse['Target']
            rmse_time = target_rmse['Over time']
            actual_series = self.y_test[:, i, :, :].mean(axis=(1, 2))
            pred_series = self.y_pred[:, i, :, :].mean(axis=(1, 2))
            time_converted = time[nlags:].to_index().to_datetimeindex()

            fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            ax1.plot(time_converted, rmse_time, color='tab:red', linewidth=1.5)
            ax1.set_ylabel(f"RMSE ({target_rmse['Variable units']})")
            ax1.set_title(f'Global RMSE over Time - {target}')
            ax1.grid(True, alpha=0.3)

            ax2.plot(time_converted, actual_series, label='Actual (Global Mean)',
                 color='black', linestyle='--', alpha=0.7)
            ax2.plot(time_converted, pred_series, label='Predicted (Global Mean)',
                 color='tab:blue', linewidth=1.5)
            ax2.set_title(f'Global {target} over Time')
            ax2.set_ylabel(f"{target} ({target_rmse['Variable units']})")
            ax2.set_xlabel('Time')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()


    def save_checkpoint(self, train_losses, val_losses):
        self.train_losses = train_losses
        self.val_losses = val_losses
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        torch.save(checkpoint, self.label)
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
