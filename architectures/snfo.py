import torch
import torch.nn as nn
from architectures.Models import define_optimizer
from architectures.cnn_baseline import CNN2D
from torch_harmonics import RealSHT, InverseRealSHT
import logging
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(message)s"
)

class SNFO(nn.Module):
    def __init__(self, data, num_epochs=50, batch_size=32, optimizer="Adam", 
                 lr=0.001, weight_decay=0.0, label='SFNO', debug=True):
        super().__init__()
        self.logger = logging.getLogger(label)
        if debug:
           self.logger.setLevel(logging.DEBUG)

        self.label = label
        self.out_path = f".results/{label}.pt"
        self.data = data
        self.targets = self.data.y_description
        self.in_channels = self.data.x_train.shape[1]
        self.out_channels = self.data.Y_train.shape[1]
        self.nlat = self.data.x_train.shape[2]
        self.nlon = self.data.x_train.shape[3]
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.sht = RealSHT(self.nlat, self.nlon)
        self.isht = InverseRealSHT(self.nlat, self.nlon)
        self.weights = nn.Parameter(torch.randn(
            self.in_channels, self.out_channels, self.nlat, self.nlon // 2 + 1,
            dtype=torch.complex64))
        self.optimizer = define_optimizer(self, optimizer, lr, weight_decay)

        self.log()

    def log(self):
        self.logger.debug(f"Model name: {self.label}")
        self.logger.debug(f"Number of input channels: {self.in_channels}")
        self.logger.debug(f"Number of output channels: {self.out_channels}")
        self.logger.debug(f"Targets: {self.targets}")
        self.logger.debug(f"Number of latitudes: {self.nlat}")
        self.logger.debug(f"Number of longitudes: {self.nlon}")

    def forward(self, x):
        self.logger.debug(f"x has shape {x.shape}")
        x_spec = self.sht(x)
        self.logger.debug(f"x has shape {x_spec.shape}")
        out_spec = torch.einsum("bclm, cdlm -> bdlm", x_spec, self.weights)
        return self.isht(out_spec)
    
    def save_checkpoint(self, train_losses, val_losses):
        self.train_losses = train_losses
        self.val_losses = val_losses
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        torch.save(checkpoint, self.out_path)
        self.logger.info(f"Checkpoint saved at epoch {self.num_epochs}.")


def train_model(model: SNFO):
    model.logger.debug(f"Training Model {model.label}")
    model.logger.debug(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    train_loader, validate_loader = create_data_tensors(model)

    epoch_train_losses = {target: [] for target in model.targets}
    epoch_val_losses = {target: [] for target in model.targets}

    for epoch in range(model.num_epochs):
        model.train()
        batch_train_losses = {target: 0.0 for target in model.targets}
        total_train_samples = 0

        for x_batch, Y_batch in train_loader:
            x_batch = x_batch.to(device)
            Y_batch = Y_batch.to(device)

            model.optimizer.zero_grad()
            preds = model(x_batch)

            total_loss = 0.0
            current_batch_losses = []

            for i, target in enumerate(model.targets):
                target_loss = criterion(preds[:, i], Y_batch[:, i])
                current_batch_losses.append(target_loss.item())
                total_loss += target_loss

            total_loss.backward()
            model.optimizer.step()

            # Accumulate loss and number of samples for each target
            for i, target in enumerate(model.targets):
                batch_train_losses[target] += current_batch_losses[i] * x_batch.size(0)
            total_train_samples += x_batch.size(0)

        # Calculate average training loss for each target
        for target in model.targets:
            avg_train_loss = batch_train_losses[target] / total_train_samples
            epoch_train_losses[target].append(avg_train_loss)

        # Validation Phase
        model.eval()
        batch_val_losses = {target: 0.0 for target in model.targets}
        total_val_samples = 0

        with torch.no_grad():
            for x_val, y_val in validate_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                val_preds = model(x_val)

                for i, target in enumerate(model.targets):
                    specific_val_loss = criterion(val_preds[:, i], y_val[:, i])
                    batch_val_losses[target] += specific_val_loss.item() * x_val.size(0)

                total_val_samples += x_val.size(0)

        # Calculate average validation loss for each target
        for target in model.targets:
            avg_val_loss = batch_val_losses[target] / total_val_samples
            epoch_val_losses[target].append(avg_val_loss)

        # Logging every 10 epochs
        if (epoch + 1) % 10 == 0:
            for target in model.targets:
                model.logger.info(f"Epoch {epoch + 1}/{model.num_epochs}, "
                                 f"Train Loss ({target}): {epoch_train_losses[target][-1]:.4f}, "
                                 f"Validation Loss ({target}): {epoch_val_losses[target][-1]:.4f}")

        model.save_checkpoint(epoch_train_losses, epoch_val_losses)
        model.logger.info("Training completed.")


def create_data_tensors(model: SNFO):
    g = torch.Generator()
    g.manual_seed(42)

    g2 = torch.Generator()
    g2.manual_seed(42)

    def worker_init_fn(worker_id):
        np.random.seed(42 + worker_id)
        random.seed(42 + worker_id)

    x_train_t = torch.tensor(model.data.x_train, dtype=torch.float32)
    Y_train_t = torch.tensor(model.data.Y_train, dtype=torch.float32)
    x_validate_t = torch.tensor(model.data.x_validate, dtype=torch.float32)
    Y_validate_t = torch.tensor(model.data.Y_validate, dtype=torch.float32)

    train_ds = TensorDataset(x_train_t, Y_train_t)
    validate_ds = TensorDataset(x_validate_t, Y_validate_t)

    train_loader = DataLoader(train_ds, batch_size=model.batch_size, shuffle=True,
                                  generator=g,
                                  worker_init_fn=worker_init_fn,
                                  num_workers=0)
    validate_loader = DataLoader(validate_ds, batch_size=model.batch_size, shuffle=False,
                                 generator=g2,
                                 worker_init_fn=worker_init_fn,
                                 num_workers=0)

    return train_loader, validate_loader