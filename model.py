import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as L
import numpy as np
import torch.optim as optim

def construct_model(in_channels, out_channels, filters):
    layers = []

    # Encoder
    prev_channel = in_channels
    for f in filters[:-1]:
        layers.extend([
            nn.Conv2d(prev_channel, f, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(f),
            nn.ReLU(),
        ])
        prev_channel = f

    # Bottleneck
    bottleneck_channels = filters[-1]
    layers.extend([
        nn.Conv2d(prev_channel, bottleneck_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(bottleneck_channels),
        nn.ReLU(inplace=True)
    ])

    # Decoder
    reversed_filters = list(reversed(filters[:-1]))
    prev_channel = bottleneck_channels

    for i, f in enumerate(reversed_filters):
        layers.extend([
            nn.ConvTranspose2d(
                prev_channel, f,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True)
        ])
        prev_channel = f

    layers.append(
        nn.Conv2d(prev_channel, out_channels, kernel_size=3, padding=1)
    )

    model = nn.Sequential(*layers)

    return model

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
    def __init__(self, config, filters=(16, 32, 32)):
        input_dim = len(config.input_channels)
        output_dim = len(config.output_channels)

        super().__init__()
        self.normalizer = NormalizeMe(config)

        self.set_target_statistics(config)
        self.nlat = config.nlat
        self.nlon = config.nlon

        other_inputs = self.setup_coordinate_channels(config)
        input_dim += len(other_inputs)

        self.model = construct_model(input_dim, output_dim, filters)

        self.learning_rate = config.learning_rate

        self.target_names = config.outputs
        self.setup_area_weights(config)
        self.setup_target_weights(config)
        self.setup_residual_indices(config)

        print(self.model)
        print(f"INPUTS: {config.input_channels}")
        print(f"Mean used to normalize: {self.normalizer.means.flatten().tolist()}")
        print(f"Stds used to normalize: {self.normalizer.stds.flatten().tolist()}")

        if other_inputs is not None:
            print(f"OTHERE INPUTS: {other_inputs}")

        print(f"OUTPUTS: {config.output_channels}")
        print(f"Mean used to normalize: {self.y_means.flatten().tolist()}")
        print(f"Stds used to normalize: {self.y_stds.flatten().tolist()}")

    def forward(self, x):
        # Normalize the targets
        x_norm = self.normalizer(x)

        # Add the cosine/sine of the latitude/longitude as targets
        if self.use_coordinates:
            x_norm = self.return_x_with_coordinates(x_norm)

        # Run the model.
        # This is going to give me the actual targets (use_residual=False) or the delta target (use_residual=True)
        cnn_out =  self.model(x_norm)
        if self.use_residual:
            # Get the value of each target at (t-1)
            t_minus_one = x_norm[:, self.residual_indices, :, :]
            cnn_out = cnn_out + t_minus_one

        return cnn_out

    def set_target_statistics(self, config):
        y_means, y_stds = get_statistics(config, channel_type="output")
        y_means = torch.as_tensor(y_means, dtype=torch.float32)
        y_stds = torch.as_tensor(y_stds, dtype=torch.float32)

        self.register_buffer("y_means", y_means)
        self.register_buffer("y_stds", y_stds)

    def compute_weighted_loss(self, preds, y_norm):
        sq_error = (preds - y_norm) ** 2
        weighted_sq_error = sq_error * self.area_weights * self.target_weights

        global_mse = weighted_sq_error.mean()
        per_target_mse = weighted_sq_error.mean(dim=(0, 2, 3))

        return global_mse, per_target_mse

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = x.squeeze(2)
        y = y.squeeze(2)

        x = x.view(x.size(0), x.size(1), self.nlat, self.nlon)
        y = y.view(y.size(0), y.size(1), self.nlat, self.nlon)
        y_norm = (y - self.y_means) / self.y_stds

        preds = self(x)

        global_mse, per_target_mse = self.compute_weighted_loss(preds, y_norm)
        self.log("train_loss", global_mse, prog_bar=True)

        for i, target in enumerate(self.target_names):
            self.log(f"train_loss_{target}", per_target_mse[i])

        return global_mse

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = x.squeeze(2)
        y = y.squeeze(2)

        x = x.view(x.size(0), x.size(1), self.nlat, self.nlon)
        y = y.view(y.size(0), y.size(1), self.nlat, self.nlon)
        y_norm = (y - self.y_means) / self.y_stds

        preds = self(x)

        global_mse, per_target_mse = self.compute_weighted_loss(preds, y_norm)
        self.log("val_loss", global_mse, prog_bar=True)

        for i, target in enumerate(self.target_names):
            self.log(f"val_loss_{target}", per_target_mse[i])

        return global_mse

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def return_x_with_coordinates(self, x):

        batch_size = x.size(0)
        batch_coords = self.coordinate_channels.expand(batch_size, -1, -1, -1)
        x = torch.cat([x, batch_coords], dim=1)

        return x

    def setup_area_weights(self, config):
        lats, lons = config.get_grid()

        lats_tensor = torch.tensor(lats, dtype=torch.float32)
        area_weights = torch.cos(torch.deg2rad(lats_tensor))
        area_weights = area_weights / area_weights.mean()
        area_weights = area_weights.view(1, 1, self.nlat, self.nlon)
        print(f"Area weights have shape: {area_weights.shape}")

        self.register_buffer("area_weights", area_weights)

    def setup_target_weights(self, config):
        targets = config.outputs
        weights = []
        for target in targets:
            weights.append(config.get_target_weight(target))
            print(f"Using a weight of {weights[-1]} for {target}")

        t_weights_tensor = torch.tensor(weights, dtype=torch.float32)
        t_weights_tensor = t_weights_tensor.view(1, len(weights), 1, 1)
        self.register_buffer("target_weights", t_weights_tensor)
        print(f"Weights have shape: {t_weights_tensor.shape}")

    def setup_coordinate_channels(self, config):
        self.use_coordinates = config.use_coordinates
        if not config.use_coordinates:
            return []

        lats, lons = config.get_grid()

        lats_rad = torch.tensor(np.radians(lats), dtype=torch.float32)
        lons_rad = torch.tensor(np.radians(lons), dtype=torch.float32)

        lat_sin = torch.sin(lats_rad)
        lat_cos = torch.cos(lats_rad)
        lon_sin = torch.sin(lons_rad)
        lon_cos = torch.cos(lons_rad)

        lat_sin = lat_sin.view(self.nlat, self.nlon)
        lat_cos = lat_cos.view(self.nlat, self.nlon)
        lon_sin = lon_sin.view(self.nlat, self.nlon)
        lon_cos = lon_cos.view(self.nlat, self.nlon)

        coords = torch.stack([lat_sin, lat_cos, lon_sin, lon_cos], dim=0)

        coords = coords.unsqueeze(0)
        self.register_buffer("coordinate_channels", coords)

        print(f"Coordinates have shape: {coords.shape}")
        return ["sin(lat)", "cos(lat)", "sin(lon)", "cos(lon)"]

    def setup_residual_indices(self, config):
        self.use_residual = config.use_residual
        temp_indices = []

        targets = config.outputs
        inputs = config.input_channels
        if not self.use_residual:
            return

        for target in targets:
            print(f"Finding the index of target {target} in the y tensor")
            var = f"{target}(t-1)"
            if var in inputs:
                idx = inputs.index(var)
                print(f"Residual mapped: Target '{var}' -> Input Index {idx}")
                temp_indices.append(idx)
            else:
                raise ValueError(f"Could not find {var} in {inputs} for residual connection!")

        indices_tensor = torch.tensor(temp_indices, dtype=torch.long)
        self.register_buffer("residual_indices", indices_tensor)
