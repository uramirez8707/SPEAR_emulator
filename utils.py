import re
import yaml

class configSetUp:
    def __init__(self, config_yaml):

        with open(config_yaml, 'r') as file:
            raw_config = yaml.safe_load(file)

        path = raw_config['paths']
        self.input_dir = path['input_dir']
        self.training = f"{self.input_dir}/{path['training']}"
        self.validating = f"{self.input_dir}/{path['validating']}"
        self.testing = f"{self.input_dir}/{path['testing']}"

        var_config = raw_config['variables']
        self.inputs = [
            var for var, info in var_config.items() if info.get('is_input')
        ]
        self.outputs = [
            var for var, info in var_config.items() if info.get('is_output')
        ]
        self.statics = [
            var for var, info in var_config.items() if info.get('is_static') and info.get('is_input')
        ]
        self.dynamics = [
            var for var, info in var_config.items() if not info.get('is_static') and info.get('is_input')
        ]

        self.var_config = var_config
        hyperparameters = raw_config['hyperparameters']
        self.batch_size = hyperparameters['batch_size']
        self.learning_rate = hyperparameters['learning_rate']
        self.data_config = raw_config['data_config']

        self.seed = raw_config['seed']

    def set_channels(self, input_channels, output_channels):
        self.input_channels = input_channels
        self.output_channels = output_channels

    def set_grid_size(self, nlat, nlon):
        self.nlat = nlat
        self.nlon = nlon

    def set_normalization_info(self, means, stds):
        self.means = means
        self.stds = stds

    def split_year(self, string, label):
        years = re.findall(r'\d{4}', string)
        if len(years) == 2:
            return years[0], years[1]
        else:
            raise RuntimeError(f"Could not parse {label} years. Expected YYYY-YYYY but got {string}")

    def dump_info(self):
        pass
