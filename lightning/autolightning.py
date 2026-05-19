from pathlib import Path

import lightning as pl
import numpy as np
import torch

import data_module


class TrainModule(pl.LightningModule):
    """A Lightning auto training model."""

    def __init__(self, 
                model, 
                loss_functional=torch.nn.functional.mse_loss,
                optimizer_cls=torch.optim.Adam,
                use_optim_scheduler=False,
                learning_rate=1e-3,                
                lr_factor=0.5, 
                lr_patience=10):
        """Constructor"""
        super().__init__()
        #self.save_hyperparameters()
        self.model = model
        self.loss_functional = loss_functional
        self.optimizer_cls = optimizer_cls
        self.learning_rate = learning_rate
        self.use_optim_scheduler = use_optim_scheduler
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

    def training_step(self, batch, _batch_idx):
        """The training step"""
        inputs, targets = batch
        z = self.model(inputs)
        loss = self.loss_functional(z, targets)
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, _batch_idx):
        """Validation step"""
        inputs, targets = batch
        z = self.model(inputs)
        loss = self.loss_functional(z, targets)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        """ set optimizer """
        optimizer = self.optimizer_cls(self.parameters(), lr=self.learning_rate)

        if self.use_optim_scheduler:
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

        return optimizer


class AutoDataModule(pl.LightningDataModule):
    """Data module for LSTM training on 1D global-mean time series."""

    def __init__(self, 
        data_dict: dict | None = None,
        sequence_length: int = 3,
        training_size: float = 0.6,
        val_size: float = 0.2,
        batch_size: int = 32
    ):
        super().__init__()

        self.data_dict = data_dict
        self.sequence_length = sequence_length
        self.training_size = training_size
        self.val_size = val_size
        self.batch_size = batch_size

        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        """prepare data"""
        if self.data_dict is None:
            raise ValueError("Set data_dict.")

        for variable, inputfile in self.data_dict.items():
            if not Path(inputfile).expanduser().exists():
                raise ValueError(f"input file {inputfile} for variable {variable} does not exist.")

    def setup(self, stage: str = None):
        """setup data"""

        loaded_data = data_module.load_variable(self.data_dict)

        data_array = np.column_stack(list(loaded_data.values()))
        ntimesteps = data_array.shape[0]
        train_end = int(ntimesteps * self.training_size)
        val_end = int(ntimesteps * (self.training_size + self.val_size))

        train_data = data_array[:train_end]
        val_data = data_array[train_end:val_end]

        self.train_dataset = data_module.TrainingDataset(train_data, sequence_length=self.sequence_length)
        self.val_dataset = data_module.TrainingDataset(val_data, sequence_length=self.sequence_length)

    def train_dataloader(self):
        """Load training data onto DataLoader"""
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        """Load validation data onto DataLoader"""
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

def from_reload(model, saved_chkpt_path):
    """Reload a model from a checkpoint path."""
    if not Path(saved_chkpt_path).exists():
        raise ValueError(f"Checkpoint path {saved_chkpt_path} does not exist.")
    return TrainModule.load_from_checkpoint(saved_chkpt_path, model=model, weights_only=False, map_location="cpu")

