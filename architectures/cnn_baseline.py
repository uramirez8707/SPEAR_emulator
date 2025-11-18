import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import random
import numpy as np
from architectures.Models import get_model_archirecture

class CNN2D_Baseline(nn.Module):
    def __init__(self, in_channels, case=0, label="baseline"):
        super(CNN2D_Baseline, self).__init__()

        self.model = get_model_archirecture(case, in_channels)

        print("Model Architecture:")
        print(self.model)

        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.y_train = None
        self.checkpoint = None
        self.label = f".results/{label}.pt"

    def forward(self, x):
        return self.model(x)

    def create_data_loaders(self, X_train, y_train, X_test, y_test, batch_size=32):

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

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def train_model(self, train_loader, test_loader, num_epochs=50, lr=1e-3, seed=42):
        path = Path(self.label)
        if path.exists():
            self.checkpoint = torch.load(self.label, weights_only=False)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            return

        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
                loss = criterion(preds, y_batch)
                loss.backward()
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
                    val_loss += criterion(preds, y_val).item() * X_val.size(0)

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
            'num_epochs': num_epochs
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
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_rmse(self, Tas, X_test_scaled, y_test_scaled):
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

        rmse_model = np.sqrt(np.mean((y_pred - y_test)**2, axis=0)).squeeze()
        rmse_persistence = np.sqrt(np.mean((persistence_pred - y_test)**2, axis=0)).squeeze()

        vmin = min(rmse_model.min(), rmse_persistence.min())
        vmax = max(rmse_model.max(), rmse_persistence.max())

        fig, axs = plt.subplots(1, 2, figsize=(14,5), constrained_layout=True)

        # Model RMSE
        im0 = axs[0].imshow(rmse_model, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0].set_title("CNN Model RMSE")
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        # Persistence RMSE
        im1 = axs[1].imshow(rmse_persistence, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[1].set_title("Persistence RMSE")
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        plt.show()