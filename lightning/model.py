import lightning as pl
import torch


class TrainModule(pl.LightningModule):
    """A Lightning auto training model."""

    def __init__(self, model, learning_rate=1e-3, lr_factor=0.5, lr_patience=10):
        """Constructor"""
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

    def training_step(self, batch, _batch_idx):
        """The training step"""
        inputs, targets = batch
        inputs = inputs.float()
        targets = targets.float()
        z = self.model(inputs)
        loss = torch.nn.functional.mse_loss(z, targets)
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, _batch_idx):
        """Validation step"""
        inputs, targets = batch
        inputs = inputs.float()
        targets = targets.float()
        z = self.model(inputs)
        loss = torch.nn.functional.mse_loss(z, targets)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.lr_factor, patience=self.lr_patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
                "name": "reduce_lr_on_plateau",
            },
        }


class SimpleCNN(torch.nn.Module):
    """A simple CNN model with one convolutional layer."""

    def __init__(self, in_channels: int = 3):
        super().__init__()        
        self.cnn1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=2*in_channels,
            kernel_size=3,
            padding=1
        )
        self.relu = torch.nn.ReLU()
        self.cnn2 = torch.nn.Conv2d(
            in_channels=2*in_channels,
            out_channels=1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        y = self.relu(self.cnn1(x))
        y = self.cnn2(y)
        return y.squeeze()


class SimpleLSTM(torch.nn.Module):
    """A simple LSTM model."""

    def __init__(self, input_size: int = 1, hidden_size: int = 10, num_layers: int = 1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out[:, -1, :])  # Use the last output of the LSTM
        return output.squeeze()
