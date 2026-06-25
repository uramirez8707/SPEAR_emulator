import torch
from torch import nn
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
import pytorch_lightning as L
import numpy as np
import torch.optim as optim
import logging
from tabulate import tabulate
from architectures.snfo import construct_sfno_model
from architectures.cnn import construct_model, construct_model_better_padding
from architectures.unet import construct_unet_model
from architectures.gnn import construct_gnn_model
from utils import get_indices

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(message)s"
)

logger = logging.getLogger(__name__)

def get_statistics(config, channel_type="input"):
    expanded_means = []
    expanded_stds = []

    if channel_type == "input":
        channels = config.input_channels
    else:
        channels = config.output_channels

    for channel_name in channels:
        raw_var_name = channel_name.split('(')[0]

        # Config.all_vars has all the variables in the order that they are in the
        # database and in the order of config.means and config.stds
        var_idx = config.all_vars.index(raw_var_name)

        mean = 0.
        stds = 1
        if config.var_config[raw_var_name]['normalization'] == "z-score":
            mean = config.means[var_idx]
            stds = config.stds[var_idx]

        expanded_means.append(mean)
        expanded_stds.append(stds)

    means_tensor = torch.tensor(expanded_means, dtype=torch.float32).view(1, -1, 1, 1)
    stds_tensor = torch.tensor(expanded_stds, dtype=torch.float32).view(1, -1, 1, 1)

    # This is just returting the means/stds of all the variables (inputs or outputs)
    return means_tensor, stds_tensor

class NormalizeMe(nn.Module):
    def __init__(self, config):
        super().__init__()

        means_tensor, stds_tensor = get_statistics(config, channel_type="input")

        self.register_buffer('means', means_tensor)
        self.register_buffer('stds', stds_tensor)

    def forward(self, x):
        return (x - self.means) / self.stds

class SpearEmulator(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self._logger = logger
        if config.verbose:
            self._logger.setLevel(logging.DEBUG)

        self._logger.info(f"Initializing {self.__class__.__name__}...")

        input_dim = len(config.input_channels)
        output_dim = len(config.output_channels)

        self.normalizer = NormalizeMe(config)

        self.set_target_statistics(config)
        self.nlat = config.nlat
        self.nlon = config.nlon

        other_inputs = self.setup_coordinate_channels(config)
        input_dim += len(other_inputs)

        self.target_names = config.outputs
        self.setup_area_weights(config)
        self.setup_target_weights(config)

        self.setup_model_architecture(config, input_dim, output_dim)
        self.log_input_channels(config)

        if other_inputs is not None:
            self._logger.info("- Other input channels:\n    " + "\n    ".join(other_inputs))

        self.log_output_channels(config)

        self.setup_input_indices_in_x(config)
        self.setup_residual_indices(config)

        self.input_channels = config.input_channels
        self.output_channels = config.output_channels
        self.dynamic_channels = config.dynamics
        self.shapes_logged = False
        self.optimizer_config = config.optimizer
        self.loss_function = config.loss_function

    def forward(self, x, x_all):
        # Normalize the targets
        if not self.shapes_logged:
            self._logger.debug("Normalizing x")
        x_norm = self.normalizer(x)

        # Add the cosine/sine of the latitude/longitude as targets
        if self.use_coordinates:
            x_norm = self.return_x_with_coordinates(x_norm)
            if not self.shapes_logged:
                self._logger.debug(f"New shape of x {x_norm.shape}")

        # Run the model.
        # This is going to give me the actual targets (use_residual=False) or the delta target (use_residual=True)
        cnn_out =  self.model(x_norm)
        if self.use_residual:
            if not self.shapes_logged:
                self._logger.debug(f"Adding x_all[:, {self.residual_indices}, :, :] to the model prediction")

            # Get the value of each ouput at (t-1) from x_all which has all of the data at t-1
            t_minus_one = x_all[:, self.residual_indices, :, :]
            t_minus_one = (t_minus_one - self.y_means) / self.y_stds

            cnn_out = cnn_out + t_minus_one

        return cnn_out

    def set_target_statistics(self, config):
        y_means, y_stds = get_statistics(config, channel_type="output")
        y_means = torch.as_tensor(y_means, dtype=torch.float32)
        y_stds = torch.as_tensor(y_stds, dtype=torch.float32)

        self.register_buffer("y_means", y_means)
        self.register_buffer("y_stds", y_stds)

    def compute_weighted_loss(self, preds, y_norm):
        if self.loss_function['name'] == "MSE":
            return self.get_MSE_loss(preds, y_norm)
        elif self.loss_function['name'] == "SSIM":
            alpha = self.loss_function.get('alpha', 0.5)
            return self.get_SSIM_loss(preds, y_norm, alpha)
        else:
            raise RuntimeError(f"The loss function {self.loss_function} has not been implemented")

    def get_MSE_loss(self, preds, y_norm):
        sq_error = (preds - y_norm) ** 2
        weighted_sq_error = sq_error * self.area_weights * self.target_weights

        global_mse = weighted_sq_error.mean()
        per_target_mse = weighted_sq_error.mean(dim=(0, 2, 3))

        return global_mse, per_target_mse

    def get_SSIM_loss(self, preds, y_norm, alpha):

        # This computes the MSE (Magnitude loss)
        sq_error = (preds - y_norm) ** 2
        weighted_sq_error = sq_error * self.area_weights * self.target_weights        
        global_mse = weighted_sq_error.mean()
        per_target_mse = weighted_sq_error.mean(dim=(0, 2, 3))

        # This computes the SSIM (Structure loss)
        data_range = y_norm.max() - y_norm.min()
        _, ssim_map = ssim(preds, y_norm, data_range=data_range, return_full_image=True)
        ssim_loss_map = 1.0 - ssim_map
        weighted_ssim_loss = ssim_loss_map * self.area_weights * self.target_weights
        global_ssim_loss = weighted_ssim_loss.mean()

        # This combines the two losses
        combined_loss = (alpha * global_mse) + ((1.0 - alpha) * global_ssim_loss)

        return combined_loss, per_target_mse

    def do_the_training(self, x, y, label):
        x = x.squeeze(2)
        y = y.squeeze(2)

        if not self.shapes_logged:
            self._logger.debug(f"Shape of orignal x: {x.shape}")
            self._logger.debug(f"Shape of orignal y: {y.shape}")

        x = x.view(x.size(0), x.size(1), self.nlat, self.nlon)
        y = y.view(y.size(0), y.size(1), self.nlat, self.nlon)
        y_norm = (y - self.y_means) / self.y_stds

        if not self.shapes_logged:
            self._logger.debug(f"Shape of input x: {x.shape}")
            self._logger.debug(f"Shape of input y: {y.shape}")

        preds = self(x)
        if not self.shapes_logged:
            self._logger.debug(f"Shape of predictions: {preds.shape}")
            self.shapes_logged = True

        global_mse, per_target_mse = self.compute_weighted_loss(preds, y_norm)
        self.log(f"{label}_loss", global_mse, prog_bar=True)

        for i, target in enumerate(self.target_names):
            self.log(f"{label}_loss_{target}", per_target_mse[i])

        return global_mse

    def training_step(self, batch, batch_idx):
        x, y = batch

        return self.do_the_training(x, y, "train")

    def validation_step(self, batch, batch_idx):
        x, y = batch

        return self.do_the_training(x, y, "val")

    def configure_optimizers(self):
        if self.optimizer_config is None:
            raise RuntimeError("Optimizer is not defined in the configuration")

        if self.optimizer_config['name'] == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.optimizer_config['learning_rate'],
                weight_decay=self.optimizer_config['weight_decay']
            )
        elif self.optimizer_config['name'] == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.optimizer_config['learning_rate'],
                weight_decay=self.optimizer_config['weight_decay']
            )
        elif self.optimizer_config['name'] == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.optimizer_config['learning_rate'],
                momentum=self.optimizer_config['momentum'],
                weight_decay=self.optimizer_config['weight_decay']
            )
        else:
            raise RuntimeError(f"Optimizer name is not supported: {self.optimizer_config['name']}")

        self._logger.debug(f"Using optimizer: {optimizer}")

        if "scheduler" in self.optimizer_config:
            scheduler_config = self.optimizer_config['scheduler']
            factor = scheduler_config.get('factor', 0.5)
            patience = scheduler_config.get('patience', 3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=factor,      # Drops the learning rate by factor
                patience=patience,      # Waits for patience epochs of no improvement before dropping
                min_lr=1e-6      # Safety net so the LR doesn't drop to absolute zero
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1
                }
            }

        return optimizer

    def return_x_with_coordinates(self, x):

        batch_size = x.size(0)
        batch_coords = self.coordinate_channels.expand(batch_size, -1, -1, -1)
        x = torch.cat([x, batch_coords], dim=1)

        return x

    def setup_area_weights(self, config):
        """
        Compute latitude-based area weighting for loss normalization.
        - Weights are computed as:
            w(lat) = cos(lat in radians)
        - Then normalized:
              w = w / mean(w)
        - Final shape:
              (1, 1, nlat, nlon)
        """
        lats, lons = config.get_grid()

        lats_tensor = torch.tensor(lats, dtype=torch.float32)
        area_weights = torch.cos(torch.deg2rad(lats_tensor))
        area_weights = area_weights / area_weights.mean()
        area_weights = area_weights.view(1, 1, self.nlat, self.nlon)

        self._logger.debug(f"Calculating the area weights for when calculating the loss \n"
                       f"    Area weights have a shape {list(area_weights.shape)}")

        self.register_buffer("area_weights", area_weights)

    def setup_target_weights(self, config):
        """
        Get the per target weights for loss calculation.
        - The target weights are defined in the configuration yaml
        """
        self._logger.debug("Getting the per target weights for when calculating the loss")

        targets = config.outputs
        weights = []
        for target in targets:
            weights.append(config.get_target_weight(target))
            self._logger.debug(f"Using a weight of {weights[-1]} for {target}")

        t_weights_tensor = torch.tensor(weights, dtype=torch.float32)
        t_weights_tensor = t_weights_tensor.view(1, len(weights), 1, 1)
        self.register_buffer("target_weights", t_weights_tensor)
        self._logger.debug(f"Weights have shape: {list(t_weights_tensor.shape)}")

    def setup_model_architecture(self, config, input_dim, output_dim):
        self._logger.info(f"Using {config.model_type} architecture type")
        if config.model_type == "default":
            filters = (16, 32, 32)
            self.model = construct_model(input_dim, output_dim, filters)
        elif config.model_type == "cnn":
            self.model = construct_model_better_padding(input_dim, output_dim, config)
        elif config.model_type == "sfno":
            self.model = construct_sfno_model(config, input_dim, output_dim, self._logger)
        elif config.model_type == "unet":
            self.model = construct_unet_model(config, input_dim, output_dim)
        elif config.model_type == "gnn":
            self.model = construct_gnn_model(config, input_dim, output_dim)
        else:
            raise RuntimeError(f"{config.model_type} has not been implemented as a model architecture")

        self._logger.info(f"Model architecture \n"
                      f"{self.model}")

    def setup_coordinate_channels(self, config):
        """
        If use_coordinates is set to True in the configuration file,
        calculates the cosine and sine of lattiude and longitude
        and combines them as a tensor so they can be used as
        extra input channels.
        """
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

        channels = ["sin(lat)", "cos(lat)", "sin(lon)", "cos(lon)"]
        self._logger.info(f"Adding additional coordinate input channels: \n"
                      f"    {channels}")

        self._logger.debug(f"Coordinates channels have the shape: {list(coords.shape)}")
        return channels

    def setup_input_indices_in_x(self, config):
        inputs = config.input_channels
        all_vars = config.all_vars
        indices = get_indices(inputs, all_vars)

        indices_tensor = torch.tensor(indices, dtype=torch.long)
        self.register_buffer("input_indices", indices_tensor)

    def setup_residual_indices(self, config):
        self.use_residual = config.use_residual

        if not self.use_residual:
            self._logger.info("Model is predicting the absolute target [y(t)]")
            return

        self._logger.info("Model is predicting the residual target:\n"
                      "   Δy = y(t) - y(t-1) \n"
                      "Model is outputting the abosolute target as:\n"
                      "   y(t) = Δy + y(t-1)")
        temp_indices = []
        targets = config.outputs
        inputs = config.all_vars
        for target in targets:
            if target in inputs:
                idx = inputs.index(target)
                self._logger.debug(f"The index {idx} of X maps to {target}(t-1)")
                temp_indices.append(idx)
            else:
                raise ValueError(f"Could not find {var} in {inputs} for residual connection!")

        indices_tensor = torch.tensor(temp_indices, dtype=torch.long)
        self.register_buffer("residual_indices", indices_tensor)

    def log_input_channels(self, config):
        channels = config.input_channels
        means = self.normalizer.means.flatten().tolist()
        stds = self.normalizer.stds.flatten().tolist()

        rows = [
            [channel, mean, std]
            for channel, mean, std in zip(channels, means, stds)
        ]
        txt = tabulate(rows, headers=["Variable", "Mean", "Std"], tablefmt="github")
        self._logger.info(f"Input channels \n"
                      f"{txt}"
                )

    def log_output_channels(self, config):
        channels = config.output_channels
        means = self.y_means.flatten().tolist()
        stds = self.y_stds.flatten().tolist()

        rows = [
            [channel, mean, std]
            for channel, mean, std in zip(channels, means, stds)
        ]

        txt = tabulate(
            rows,
            headers=["Variable", "Mean", "Std"],
            tablefmt="github"
        )

        self._logger.info(
            f"Output channels\n{txt}"
        )

    def on_before_optimizer_step(self, optimizer):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
        self.log("grad_norm", grad_norm, on_step=False, on_epoch=True, prog_bar=True)

class AutoregressiveSpearEmulator(SpearEmulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y_sequence = batch

        if not self.shapes_logged:
            # X includes all of the inputs + diagnostics at t-1
            self._logger.debug(f"Shape of orignal x: {x.shape}")

            # Y includes just the outputs at t, t+1, ...  t+nlags-1
            self._logger.debug(f"Shape of orignal y: {y_sequence.shape}")

        nsteps = y_sequence.shape[1]

        total_global_mse, total_per_target_mse = self.get_loss(x, y_sequence, nsteps)

        # Get the average across nsteps to log
        avg_global_mse = total_global_mse / nsteps
        avg_per_target_mse = total_per_target_mse / nsteps
        self.log("train_loss", avg_global_mse, prog_bar=True, sync_dist=True)
        for i, target in enumerate(self.target_names):
            self.log(f"train_loss_{target}", avg_per_target_mse[i], sync_dist=True)

        return avg_global_mse

    def validation_step(self,  batch, batch_idx):
        x, y_sequence = batch
        nsteps = y_sequence.shape[1]

        total_global_mse, total_per_target_mse = self.get_loss(x, y_sequence, nsteps)

        # Get the average across nsteps to log
        avg_global_mse = total_global_mse / nsteps
        avg_per_target_mse = total_per_target_mse / nsteps
        self.log("val_loss", avg_global_mse, prog_bar=True, sync_dist=True)
        for i, target in enumerate(self.target_names):
            self.log(f"val_loss_{target}", avg_per_target_mse[i], sync_dist=True)

        return avg_global_mse

    def get_loss(self, x, y_sequence, nsteps):
        x = x.squeeze(2)
        x = x.view(x.size(0), x.size(1), self.nlat, self.nlon)

        x_original = x
        x = x[:, self.input_indices, ...]
        if not self.shapes_logged:
            self._logger.debug(f"Shape of reshaped x original: {x_original.shape}")
            self._logger.debug(f"Shape of reshaped x inputs: {x.shape}")
            self._logger.debug(f"Shape of y_sequence: {y_sequence.shape}")

        current_x = x

        total_global_mse = 0.0
        total_per_target_mse = None

        for step in range(nsteps):
            # Prepare the ground truth
            y_step_all = y_sequence[:, step, ...]
            if not self.shapes_logged:
                 self._logger.debug(f"step: {step} - shape of y_all: {y_step_all.shape}")

            y_step_all = y_step_all.squeeze(2)
            y_step_all = y_step_all.view(y_step_all.size(0), y_step_all.size(1), self.nlat, self.nlon)

            if not self.shapes_logged:
                 self._logger.debug(f"step: {step} - shape of reshape y_all: {y_step_all.shape}")

            # y_step_all contains all of the outputs + dynamic variables, so just get the outputs
            # to compare with the preidiction
            y_step_target = y_step_all[:, 0:len(self.output_channels), ...]
            if not self.shapes_logged:
                 self._logger.debug(f"step: {step} - shape of target y: {y_step_target.shape}")

            y_norm = (y_step_target - self.y_means) / self.y_stds

            # Make the prediction
            preds_norm = self(current_x, x_original)
            if not self.shapes_logged:
                 self._logger.debug(f"step: {step} - shape of prediction: {preds_norm.shape}")

            # Compute the loss
            step_global_mse, step_per_target_mse = self.compute_weighted_loss(preds_norm, y_norm)
            total_global_mse += step_global_mse
            if total_per_target_mse is None:
                total_per_target_mse = step_per_target_mse.clone()
            else:
                total_per_target_mse += step_per_target_mse

            if step == nsteps - 1:
                break

            # Convert the predicitions back to physical units:
            preds = (preds_norm * self.y_stds) + self.y_means

            # Prepare x for the next time step
            # Update the outputs with the predictions
            next_x = current_x.clone()

            for i, input_var in enumerate(self.input_channels):
                idx_in_pred = self.find_output_channel_index(input_var)
                if idx_in_pred >= 0 :
                    if not self.shapes_logged:
                        self._logger.debug(f"step: {step} - replacing {input_var} "
                                        f"next_x[:, {i}, ...] = preds[:, {idx_in_pred}, ...]")

                    next_x[:, i, ...] = preds[:, idx_in_pred, ...]

                else:
                    idx_in_pred = self.find_dynamic_channel_index(input_var)

                    if idx_in_pred == -1:
                        if not self.shapes_logged:
                            self._logger.debug(f"step: {step} - not replacing {input_var} it is a static variable")
                        continue

                    if not self.shapes_logged:
                        self._logger.debug(f"step: {step} - replacing {input_var} "
                                        f"next_x[:, {i}, ...] = y_step_all[:, {idx_in_pred}, ...]")
                    next_x[:, i, ...] = y_step_all[:, idx_in_pred, ...]
            current_x = next_x

        self.shapes_logged = True
        return total_global_mse, total_per_target_mse

    def find_output_channel_index(self, target):
        var_name = target.replace("(t-1)", "")
        for i, output_var in enumerate(self.output_channels):
            if var_name in output_var:
                return i
        return -1
    def find_dynamic_channel_index(self, target):
        var_name = target.replace("(t-1)", "")
        for i, dynamic_var in enumerate(self.dynamic_channels):
            if var_name in dynamic_var:
                return i
        return -1
