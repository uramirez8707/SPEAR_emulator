import re
import yaml
import time
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback

def log_dataset_info(logger, label, dataset):
    logger.debug(f"{label} shape: \n"
                 f"    ntimes: {dataset.shape[0]} \n"
                 f"    nvariables: {dataset.shape[1]} \n"
                 f"    nesembles: {dataset.shape[2]} \n"
                 f"    ngridpoints: {dataset.shape[3]}")
    logger.info(f"{label} time period:\n"
                f"    {dataset.dates[0]} to {dataset.dates[-1]}")

class FortranTracker(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Start the stopwatch when the epoch begins
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed_time = time.time() - self.epoch_start_time
        mins, secs = divmod(elapsed_time, 60)

        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)

        train_loss = trainer.callback_metrics.get('train_loss', 'N/A')
        if isinstance(train_loss, torch.Tensor):
            train_loss = f"{train_loss.item():.4f}"

        val_loss = trainer.callback_metrics.get('val_loss', 'N/A')
        if isinstance(val_loss, torch.Tensor):
            val_loss = f"{val_loss.item():.4f}"

        print(f"\n>>> [Epoch {trainer.current_epoch}] "
              f"Time: {int(mins)}m {int(secs)}s | "
              f"Train Loss: {train_loss} | "
              f"Val Loss: {val_loss} | "
              f"Peak VRAM: {max_mem_gb:.2f} GB <<<")

class configSetUp:
    def __init__(self, config_yaml):

        with open(config_yaml, 'r') as file:
            raw_config = yaml.safe_load(file)

        path = raw_config['paths']
        self.input_dir = path['input_dir']
        self.training = f"{self.input_dir}/{path['training']}"
        self.validating = f"{self.input_dir}/{path['validating']}"
        self.testing = f"{self.input_dir}/{path['testing']}"

        self.set_variable_config(raw_config['variables'])
        self.inputs = [
            var for var, info in self.var_config.items() if info.get('is_input')
        ]
        self.outputs = [
            var for var, info in self.var_config.items() if info.get('is_output')
        ]
        self.statics = [
            var for var, info in self.var_config.items() if info.get('is_static') and info.get('is_input')
        ]
        self.dynamics = [
            var for var, info in self.var_config.items() if not info.get('is_static') and info.get('is_input')
        ]
        self.diagnostics_only = [
            var for var, info in self.var_config.items() if info.get('diagnostic_only')
        ]

        hyperparameters = raw_config['hyperparameters']
        self.batch_size = hyperparameters['batch_size']
        self.learning_rate = hyperparameters['learning_rate']
        self.precision = hyperparameters.get('precision', 'bf16-mixed')

        self.data_config = raw_config['data_config']

        self.seed = raw_config['seed']
        self.use_coordinates = raw_config['use_coordinates']
        self.use_residual = self.set_use_residual()
        self.verbose = raw_config['verbose']
        self.model_type = raw_config['model_type']
        if "sfno" in raw_config:
            self.sfno = raw_config['sfno']
        if "cnn" in raw_config:
            self.cnn = raw_config['cnn']
        if "unet" in raw_config:
            self.unet = raw_config['unet']
        if "gnn" in raw_config:
            self.gnn = raw_config['gnn']

        self.nepochs = raw_config['nepochs']
        self.optimizer = raw_config.get("optimizer")

    def get_nlags(self):
        return self.data_config["method"].get("nlags", 1)

    def set_variable_config(self, var_config):
        out_config = {}
        for var, info in var_config.items():
            vertical_levels = info.get('vertical_levels')
            if vertical_levels is None:
                out_config[var] = info
                continue
            for level in vertical_levels:
                new_var = f"{var}_{level}"
                new_info = info.copy()
                new_info.pop("vertical_levels", None)
                out_config[new_var] = new_info
        self.var_config = out_config

    def set_use_residual(self):
        if self.data_config.get("type") == "residual":
            return True
        return False

    def get_nregressive_steps(self):
        return self.data_config["method"].get("nsteps", 0)

    def get_data_load_method(self):
        method = self.data_config["method"].get("name")
        if method not in ["lags", "autoregressive"]:
            raise RuntimeError(f"The method name must be 'lags' or 'autoregressive', but you specified {method}")
        return method

    def set_channels(self, input_channels, output_channels, diag_channels):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.diag_channels = diag_channels

    def set_grid_size(self, nlat, nlon):
        self.nlat = nlat
        self.nlon = nlon

    def set_normalization_info(self, means, stds):
        self.means = means
        self.stds = stds

    def set_grid(self, database):
        self.lat = database.lat
        self.lon = database.lon

    def get_grid(self):
        if self.lat is None or self.lon is None:
            raise RuntimeError(f"Grid is not set. You need to call self.set_grid() first")

        return self.lat, self.lon

    def find_index(self, var, inputs):
        return inputs.index(var)

    def get_var_units(self, var):
        try:
            units = self.var_config[var]["units"]
        except KeyError as e:
            raise KeyError(
                f"Missing variable or units for '{var}'"
            ) from e

        if not units:
            raise ValueError(f"Invalid units for '{var}': {units!r}")

        return units

    def get_target_weight(self, var):
        try:
            weight = self.var_config[var]["target_weight"]
        except KeyError as e:
            raise KeyError(
                f"Missing variable or target_weight for '{var}'"
            ) from e

        if not weight:
            raise ValueError(f"Invalid target_weight for '{var}': {weight!r}")

        return weight

    def get_mappings(self):
        targets = self.outputs
        inputs = self.input_channels
        nlags = self.get_nlags()

        mappings = []
        for target in targets:
            variable_mapping = {
                "variable_name": target,
                "labels": []
            }

            if nlags > 0:
                for i in range(nlags):
                    label = f"t-{i+1}"
                    variable_name = f"{target}({label})"
                    variable_mapping["labels"].append({
                        "label": label,
                        "index": self.find_index(variable_name, inputs)
                    })
            mappings.append(variable_mapping)

        return mappings

    def split_year(self, string, label):
        years = re.findall(r'\d{4}', string)
        if len(years) == 2:
            return years[0], years[1]
        else:
            raise RuntimeError(f"Could not parse {label} years. Expected YYYY-YYYY but got {string}")

    def get_cnn_filters(self):
        return tuple(self.cnn['filters'])

    def dump_info(self):
        pass
