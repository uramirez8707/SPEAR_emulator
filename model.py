import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as L

def get_statistics(config, channel_type="input"):
    expanded_means = []
    expanded_stds = []

    if channel_type == "input":
        channels = config.input_channels
    else:
        channels = config.output_channels

    for channel_name in channels:
        raw_var_name = channel_name.split('(')[0]
        var_idx = config.inputs.index(raw_var_name)

        mean = 0.
        stds = 1
        if config.var_config[raw_var_name]['normalization'] == "z-score":
            mean = config.means[var_idx]
            stds = config.stds[var_idx]

        expanded_means.append(mean)
        expanded_stds.append(stds)

    means_tensor = torch.tensor(expanded_means, dtype=torch.float32).view(1, -1, 1, 1)
    stds_tensor = torch.tensor(expanded_stds, dtype=torch.float32).view(1, -1, 1, 1)

    return means_tensor, stds_tensor

class NormalizeMe(nn.Module):
    def __init__(self, config):
        super().__init__()

        means_tensor, stds_tensor = get_statistics(config, channel_type="input")

        self.register_buffer('means', means_tensor)
        self.register_buffer('stds', stds_tensor)

    def forward(self, x):
        # Applies (x - mean) / std instantly across the batch
        return (x - self.means) / self.stds

class SpearEmulator(L.LightningModule):
    def __init__(self, config):
        input_dim = len(config.input_channels)
        output_dim = len(config.output_channels)

        super().__init__()
        self.normalizer = NormalizeMe(config)

        self.set_target_statistics(config)

        self.model = nn.Sequential(
            self.normalizer,
            nn.Conv2d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=output_dim, kernel_size=3, padding=1)
        )

        self.learning_rate = config.learning_rate
        self.nlat = config.nlat
        self.nlon = config.nlon

        print(self.model)
        print(f"INPUTS: {config.input_channels}")
        print(f"Mean used to normalize: {self.normalizer.means.flatten().tolist()}")
        print(f"Stds used to normalize: {self.normalizer.stds.flatten().tolist()}")

        print(f"OUTPUTS: {config.output_channels}")
        print(f"Mean used to normalize: {self.y_means.flatten().tolist()}")
        print(f"Stds used to normalize: {self.y_stds.flatten().tolist()}")

    def forward(self, x):
        return self.model(x)

    def set_target_statistics(self, config):
        self.y_means, self.y_stds = get_statistics(config, channel_type="output")

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Remove the third empty ensemble dimension for anemoi
        x = x.squeeze(2)
        y = y.squeeze(2)

        x = x.view(x.size(0), x.size(1), self.nlat, self.nlon)
        y = y.view(y.size(0), y.size(1), self.nlat, self.nlon)
        y_norm = (y - self.y_means) / self.y_stds

        preds = self(x)
        loss = F.mse_loss(preds, y_norm)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
