import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from architectures.Models import get_model_archirecture
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True  # force deterministic algorithms
    torch.backends.cudnn.benchmark = False     # disable auto-tuner for convolution



class CNN2D_Baseline(nn.Module):
    def __init__(self, in_channels, case=0, label="baseline", out_channels=1, out_var_maps=None):
        super(CNN2D_Baseline, self).__init__()

        self.model = get_model_archirecture(case, in_channels, out_channels=out_channels)

        print("Model Architecture:")
        print(self.model)

        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.y_train = None
        self.checkpoint = None
        self.label = f".results/{label}.pt"
        self.out_var_maps = out_var_maps

    def forward(self, x):
        return self.model(x)

    def create_data_loaders(self, X_train, y_train, X_test, y_test, batch_size=32):

        g = torch.Generator()
        g.manual_seed(42)

        g2 = torch.Generator()
        g2.manual_seed(42)

        def worker_init_fn(worker_id):
            np.random.seed(42 + worker_id)
            random.seed(42 + worker_id)

        # Save the data to use it later:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)

        train_ds = TensorDataset(X_train_t, y_train_t)
        test_ds = TensorDataset(X_test_t, y_test_t)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  generator=g,
                                  worker_init_fn=worker_init_fn,
                                  num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 generator=g2,
                                 worker_init_fn=worker_init_fn,
                                 num_workers=0)

        return train_loader, test_loader

    def train_model(self, train_loader, test_loader, num_epochs=50, lr=1e-3):
        path = Path(self.label)
        if path.exists():
            self.checkpoint = torch.load(self.label, weights_only=False)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        training_loss = []
        validation_loss = []
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                preds = self.model(X_batch)
                if self.out_var_maps is None:
                    loss = criterion(preds, y_batch)
                else:
                    loss = self.get_combined_loss(preds, y_batch, criterion)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= len(train_loader.dataset)
            training_loss.append(epoch_loss)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_val, y_val in test_loader:
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)
                    preds = self.model(X_val)
                    if self.out_var_maps is None:
                        val_loss += criterion(preds, y_val).item() * X_val.size(0)
                    else:
                        val_loss += self.get_combined_loss(preds, y_val, criterion) * X_val.size(0)

            val_loss /= len(test_loader.dataset)
            validation_loss.append(val_loss)

            if epoch % 10 == 0 or epoch == num_epochs-1:
                print(f"Epoch {epoch}/{num_epochs}, Training Loss: {epoch_loss:.6f} Validation Loss: {val_loss:.6f}")

        # Save the model output to avoid having to learn again
        self.checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'num_epochs': num_epochs,
            'mappings': self.out_var_maps
        }

        torch.save(self.checkpoint, self.label)

    def plot_loss_over_epochs(self):
        training_loss = self.checkpoint['training_loss']
        validation_loss = self.checkpoint['validation_loss']
        epochs = range(1, len(training_loss) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, training_loss, label='Training Loss')
        plt.plot(epochs, validation_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, 30000)
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_global_rmse_over_time(self, Tas, X_test_scaled, y_test_scaled):
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_scaled = self.model(X_test_tensor).cpu().numpy()

        # Inverse transform predictions and targets
        y_pred = Tas.scaler.inverse_transform(y_pred_scaled)
        y_test = Tas.scaler.inverse_transform(y_test_scaled)

        rmse =  np.sqrt(np.mean((y_pred - y_test)**2, axis=(1,2,3)))
        return rmse
    
    def get_rmse(self, Tas, X_test_scaled, y_test_scaled, ranges=None):
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_scaled = self.model(X_test_tensor).cpu().numpy()

        # Inverse transform predictions and targets
        y_pred = Tas.scaler.inverse_transform(y_pred_scaled)
        y_test = Tas.scaler.inverse_transform(y_test_scaled)

        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        print(f"Global RMSE (K): {rmse:.6f}")

        # Get a persistense baseline:
        persistence_pred_scaled = X_test_scaled[:, -1, :, :]
        persistence_scaled_4d = persistence_pred_scaled[:, None, :, :]
        persistence_pred = Tas.scaler.inverse_transform(persistence_scaled_4d)
        persistence_rmse = np.sqrt(np.mean((persistence_pred - y_test)**2))

        print(f"Global Persistence RMSE (K): {persistence_rmse:.6f}")

        # Calculate grid-wise RMSE (axis=0 averages over the time dimension)
        rmse_model = np.sqrt(np.mean((y_pred - y_test)**2, axis=0)).squeeze()
        rmse_persistence = np.sqrt(np.mean((persistence_pred - y_test)**2, axis=0)).squeeze()

        lon = Tas.varData.lon # e.g., numpy array of longitudes
        lat = Tas.varData.lat # e.g., numpy array of latitudes

        combined_min = min(rmse_model.min(), rmse_persistence.min())
        combined_max = max(rmse_model.max(), rmse_persistence.max())
        vmin = combined_min
        vmax = combined_max

        if ranges is not None:
            vmin = ranges[0]
            vmax = ranges[1]
        else:
            ranges = (vmin, vmax)

        # Setup the figure and map projection
        fig = plt.figure(figsize=(14, 6))

        # Define the map projection (e.g., PlateCarree is a common choice)
        projection = ccrs.PlateCarree()
        
        # --- Model RMSE Plot ---
        ax0 = fig.add_subplot(1, 2, 1, projection=projection)

        # Plot the data using pcolormesh
        im0 = ax0.pcolormesh(
            lon, lat, rmse_model,
            transform=projection,
            cmap='viridis',
            vmin=vmin, vmax=vmax
        )

        ax0.set_title("CNN Model RMSE (K)")
        ax0.coastlines(resolution='50m') # Add coastlines
        ax0.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False) # Add gridlines

        # Add a colorbar
        divider0 = make_axes_locatable(ax0)
        cax0 = divider0.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
        plt.colorbar(im0, cax=cax0, orientation='vertical', label='RMSE (K)')

        # --- Persistence RMSE Plot ---
        ax1 = fig.add_subplot(1, 2, 2, projection=projection)

        # Plot the data using pcolormesh
        im1 = ax1.pcolormesh(
            lon, lat, rmse_persistence,
            transform=projection,
            cmap='viridis',
            vmin=vmin, vmax=vmax # Use the shared vmin/vmax
        )

        ax1.set_title("Persistence RMSE (K)")
        ax1.coastlines(resolution='50m') # Add coastlines
        ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False) # Add gridlines

        # Add a colorbar
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
        plt.colorbar(im1, cax=cax1, orientation='vertical', label='RMSE (K)')

        # Tight layout adjusted for subplots with colorbars
        fig.tight_layout(rect=[0, 0, 1, 1])

        plt.show()
        return ranges
    
    def get_combined_loss(self, pred, target, criterion):
        loss = 0
        for variable in self.out_var_maps:
            var_pred = pred[:, variable['out_channel'], :, :]
            var_target = target[:, variable['out_channel'], :, :]
            var_loss = criterion(var_pred, var_target)
            variable['loss'] += var_loss
            loss += var_loss
        return loss