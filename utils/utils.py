import xarray as xr
import numpy as np
import logging
import xesmf as xe
from numpy.lib.stride_tricks import sliding_window_view

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(message)s"
)

class VarSet:
    def __init__(self, var_name, file_name, is_output=False, is_static=False, debug=False,
                 nregressive_steps=1, nlags=3):
        self.logger = logging.getLogger(var_name)
        if debug:
           self.logger.setLevel(logging.DEBUG)

        self.logger.debug(f"Working on {var_name} from file {file_name}")
        self.var_name = var_name
        self.file_name = file_name
        self.file = xr.open_mfdataset(self.file_name, combine='by_coords', decode_timedelta=True)
        self.is_static = is_static
        self.is_output = is_output
        self.x_train = None
        self.x_validate = None
        self.Y_train = None
        self.Y_validate = None
        self.x_test = None
        self.Y_test = None
        self.x_description = ""
        self.y_description = ""
        self.standardization_info = {}
        self.nregressive_steps = nregressive_steps
        self.nlags = nlags

        if self.nregressive_steps != 1 and self.nlags != 1:
            raise RuntimeError("Either nregressive_steps or nlags must be equal to 1!")

        self.procress_variable()

    def get_data_set(self, var_name):
        valid_names = ["training", "validating", "testing"]
        if var_name not in valid_names and not self.is_static:
           raise RuntimeError(f"get_data_set:: {var_name} is not valid. It must be {valid_names}")
        self.logger.debug(f"Getting the {var_name} data")
        out_data = self.file[var_name]

        self.logger.debug(f"{var_name} data has shape {out_data.shape}")
        return out_data
    
    def prepare_data(self, data):
        if self.nlags != 1:
            return self.prepare_lag_dataset(data)
        else:
            return self.prepare_autoregressive_dataset(data)
    
    def prepare_lag_dataset(self, data):
        n_lags = self.nlags
        data_np = data.values

        y = None
        X = sliding_window_view(data_np, window_shape=n_lags, axis=0)
        X = np.moveaxis(X, -1, 1)
        X = X[:-1]
        self.logger.debug(f"Lagged X shape: {X.shape}")
        self.x_description = [f"{self.var_name}(t-{k})" for k in range(n_lags, 0, -1)]

        if self.is_output:
            y = data_np[n_lags:, np.newaxis, :, :]
            self.logger.debug(f"Lagged y shape: {y.shape}")
            self.y_description = [f"{self.var_name}"]

        return X, y
    
    def prepare_autoregressive_dataset(self, data):
        data_np = data.values

        y = None
        X = data_np[:-1, np.newaxis, :, :]
        X = X[:-self.nregressive_steps]
        self.logger.debug(f"Final X shape: {X.shape}")
        self.x_description = [f"{self.var_name}(t)"]

        if self.is_output:
            y = data_np[1:, np.newaxis, :, :]
            y = self.prepare_for_autoregressive(y)
            self.logger.debug(f"Final y shape: {y.shape}")
            self.y_description = [f"{self.var_name}"]

        return X, y

    def prepare_for_autoregressive(self, y):
        self.logger.debug(f"Preparing y with {self.nregressive_steps} regressive steps")
        T, C, H, W = y.shape
        nsteps = self.nregressive_steps

        Y_prime = np.zeros((T - nsteps, nsteps, C, H, W))

        for t in range(T - nsteps):
            for n in range(nsteps):
                Y_prime[t, n, :, :, :] = y[t+n, :, :, :]
        return Y_prime

    def procress_variable(self):
        if self.is_static:
            self.procress_static_variable()
        else:
            self.procress_dynamic_variable()

        self.add_standardization_info()

    def add_standardization_info(self):
        self.logger.debug("Adding info how to get the variable back to physical units")
        #TODO We will likely try out different methods, so the method should be
        # an a variable/attribute in the file

        self.standardization_info['method'] = ""
        if "mean" in self.file:
            self.standardization_info['mean'] = self.file['mean'].values
            self.standardization_info['std'] = self.file['std'].values
        self.standardization_info['var_name'] = self.var_name
        self.standardization_info['original_units'] = self.file.attrs.get('original_variable_units')
        self.logger.debug(self.standardization_info)

    def procress_static_variable(self):
        data = self.get_data_set(self.var_name)
        self.x_train = data
        self.x_validate = data
        self.x_test = data

        if self.var_name == "spatial_features":
            self.x_description = self.file.coords["channel"].values.tolist()
        else:
            self.x_description = [f"{self.var_name}"]

    def procress_dynamic_variable(self):
        training = self.get_data_set("training")
        validating = self.get_data_set("validating")
        testing = self.get_data_set("testing")

        self.x_train, self.Y_train = self.prepare_data(training)
        self.x_validate, self.Y_validate = self.prepare_data(validating)
        self.x_test, self.Y_test = self.prepare_data(testing)

    def get_grid(self):
        return self.file['lon'], self.file['lat']
    
    def get_time(self, time_label):
        return self.file[time_label]

class DataSet:
    def __init__(self, set_name, variables, debug=False):
        self.logger = logging.getLogger(set_name)
        if debug:
           self.logger.setLevel(logging.DEBUG)

        self.x_train = None
        self.x_validate = None
        self.Y_train = None
        self.Y_validate = None
        self.x_test = None
        self.Y_test = None
        self.is_initialized = False
        self.x_description = []
        self.y_description = []
        self.standardization_info = []

        for variable in variables:
            self.logger.debug(f"Adding {variable.var_name} to the dataset")
            if not self.is_initialized:
                if variable.is_static:
                    raise RuntimeError(f"{variable.var_name} is static and cannot be the first variable in the list")

                self.x_train = variable.x_train
                self.x_validate = variable.x_validate
                self.x_test = variable.x_test
                self.Y_train = variable.Y_train
                self.Y_validate = variable.Y_validate
                self.Y_test = variable.Y_test
                self.is_initialized = True
            else:
                if variable.is_static:
                    self.append_static_variable(variable)
                else:
                    self.append_dynamic_variable(variable)

            if variable.is_output:
                self.y_description.extend(variable.y_description)

            self.x_description.extend(variable.x_description)
            self.standardization_info.append(variable.standardization_info)
            if debug: self.log_shape()
            self.logger.debug(variable.standardization_info)

    def append_static_variable(self, variable):
        if 'sample' in variable.x_train.dims:
            return
        
        if len(variable.x_train.dims) == 2:
            self.logger.debug(f"Appending an extra dimension")
            variable.x_train = variable.x_train.expand_dims(channel=1)
            variable.x_validate = variable.x_validate.expand_dims(channel=1)
            variable.x_test = variable.x_test.expand_dims(channel=1)

        nsamples = self.x_train.shape[0]
        variable.x_train = variable.x_train.expand_dims(sample=nsamples)
        self.x_train = np.concatenate([self.x_train, variable.x_train], axis=1)

        nsamples = self.x_validate.shape[0]
        variable.x_validate = variable.x_validate.expand_dims(sample=nsamples)
        self.x_validate = np.concatenate([self.x_validate, variable.x_validate], axis=1)

        nsamples = self.x_test.shape[0]
        variable.x_test = variable.x_test.expand_dims(sample=nsamples)
        self.x_test = np.concatenate([self.x_test, variable.x_test], axis=1)

    def append_dynamic_variable(self, variable):
        self.x_train = np.concatenate([self.x_train, variable.x_train], axis=1)
        self.x_validate = np.concatenate([self.x_validate, variable.x_validate], axis=1)
        self.x_test = np.concatenate([self.x_test, variable.x_test], axis=1)
        if variable.is_output:
            self.Y_train = np.concatenate([self.Y_train, variable.Y_train], axis=1)
            self.Y_validate = np.concatenate([self.Y_validate, variable.Y_validate], axis=1)
            self.Y_test = np.concatenate([self.Y_test, variable.Y_test], axis=1)
    
    def log_shape(self):
        self.logger.info(
          f"Shapes | "
          f"x_train: {self.x_train.shape}, "
          f"Y_train: {self.Y_train.shape}, "
          f"x_val: {self.x_validate.shape}, "
          f"Y_val: {self.Y_validate.shape}, "
          f"x_test: {self.x_test.shape}, "
          f"Y_test: {self.Y_test.shape}"
        )
        self.logger.info(f"Channels: {self.x_description}")
        self.logger.info(f"Targets: {self.y_description}")

def get_variable_info(yaml_data, variable_name):
    for item in yaml_data:
        if item.get('variable_name') == variable_name:
            return item
    return None