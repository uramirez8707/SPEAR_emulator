import xarray as xr
import numpy as np
import logging
import xesmf as xe

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s"
)

class VarData:
  def __init__(self, filename, variable, log_level=logging.INFO, add_spatial=True, lag_data_set=True,
               is_static=False, standardize=True):
    self.fileName = filename
    self.varName = variable
    self.varData = None
    self.file = None
    self.spatial_features = None
    self.scaler = None
    self.add_spatial = add_spatial
    self.lag_data_set = lag_data_set
    self.is_static = is_static
    self.use_mask = False
    self.standardize = standardize

    self.logger = logging.getLogger(self.varName)
    self.logger.setLevel(log_level)  # Set level from flag
    self.logger.info(f"Working on var: {self.varName} from {self.fileName}")

  def load_data(self):
    self.logger.debug(f"Opening the file for {self.fileName}")
    self.file = xr.open_mfdataset(self.fileName, combine='by_coords', decode_timedelta=True)

    self.logger.debug(f"Reading data for variable: {self.varName}")
    self.varData = self.file[self.varName]
    if self.varData.isnull().any():
      self.logger.warning("Input data contains null values. Filling with 0. Need to mask")
      self.use_mask = True

    self.logger.debug(f"Size of the data: {self.varData.shape}")

  def interpolate_data(self, xres=None, yres=None, coarse=False):
    # This is to interpolate data to a coarser grid to make the training faster while developing
    if (xres is None or yres is None) and not coarse:
      target_grid = get_target_grid()
      regridder = xe.Regridder(self.varData, target_grid, method="bilinear")
      self.varData = regridder(self.varData)
      return

    self.logger.debug(f"Coarsing data by: {xres}x{yres}")
    self.varData = self.varData.coarsen(lat=xres, lon=yres, boundary='trim').mean()
    self.logger.debug(f"Size of the data: {self.varData.shape}")

  def add_spatial_features(self):
    if not self.add_spatial:
       self.logger.debug("Not getting spatial features for the dataset")
       return

    self.logger.debug("Getting spatial features for the dataset")

    lat = self.varData.lat.values
    lon = self.varData.lon.values

    # Encode latitude and longitude with sine/cosine
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat_sin, lat_cos = np.sin(lat_rad), np.cos(lat_rad)
    lon_sin, lon_cos = np.sin(lon_rad), np.cos(lon_rad)

    # Create full coordinate grids
    lat_sin_grid, lon_sin_grid = np.meshgrid(lat_sin, lon_sin, indexing="ij")
    lat_cos_grid, lon_cos_grid = np.meshgrid(lat_cos, lon_cos, indexing="ij")

    # Stack features: (lat, lon, 4)
    self.spatial_features = np.stack(
        [lat_sin_grid, lat_cos_grid, lon_sin_grid, lon_cos_grid],
        axis=-1
    )

  def prepare_cnn_input(self):
    self.logger.debug("Combine variable data and spatial encodings")

    data = self.varData.values
    if not self.is_static:
       time = self.varData['time'].values
    else:
       time = None

    if self.add_spatial:
      spatial = np.broadcast_to(
        self.spatial_features,
        data.shape + (self.spatial_features.shape[-1],)
      )
      data = data[..., np.newaxis]
      cnn_input = np.concatenate([data, spatial], axis=-1)
    else:
      cnn_input = data[..., np.newaxis]

    self.logger.debug(f"Size of the full data set {cnn_input.shape}")
    return cnn_input, time

  def create_lagged_samples(self, data, n_lags=3):
    """
    Create lagged samples:
    X: (time - n_lags, n_lags, lat, lon, channels)
    y: (time - n_lags, lat, lon)  (target is the variable only: channel 0)
    """
    self.logger.debug(f"Creating lagged dataset using {n_lags} lags...")

    T, H, W, C = data.shape

    X = []
    y = []

    if self.lag_data_set:
      for t in range(n_lags, T):
        X.append(data[t - n_lags:t])   # stack lagged steps
        y.append(data[t, ..., 0])      # predict channel 0 (the variable)
    else:
       for t in range(n_lags, T):
        X.append(data[t][np.newaxis, ...])   # stack lagged steps
        y.append(data[t, ..., 0])      # predict channel 0 (the variable)


    X = np.stack(X)  # (samples, n_lags, H, W, C)
    y = np.stack(y)  # (samples, H, W)

    self.logger.debug(f"Lagged X shape: {X.shape}")
    self.logger.debug(f"Lagged y shape: {y.shape}")

    return X, y

  def split_data(self, test_frac=0.2, n_lags=3):
    if self.is_static:
       data = self.varData.values
       data = data[np.newaxis, np.newaxis, ...]
       return data

    data, time = self.prepare_cnn_input()

    ntime = data.shape[0]
    split_idx = int(ntime * (1 - test_frac))

    train = data[:split_idx]
    test = data[split_idx:]
    time_test = time[split_idx:]

    self.logger.debug(f"Raw train shape (before scaling & lagging): {train.shape}")
    self.logger.debug(f"Raw test shape  (before scaling & lagging): {test.shape}")

    train_scaled, test_scaled = self.standardize_data(train, test)

    X_train_scaled, y_train = self.create_lagged_samples(train_scaled, n_lags=n_lags)
    X_test_scaled,  y_test  = self.create_lagged_samples(test_scaled,  n_lags=n_lags)

    self.logger.debug(f"Final X_train shape: {X_train_scaled.shape}")
    self.logger.debug(f"Final y_train shape: {y_train.shape}")
    self.logger.debug(f"Final X_test shape:  {X_test_scaled.shape}")
    self.logger.debug(f"Final y_test shape:  {y_test.shape}")

    if self.standardize:
      y_train_scaled = (y_train - self.scaler.mean_) / self.scaler.std_
      y_test_scaled = (y_test - self.scaler.mean_) / self.scaler.std_
    else:
      y_train_scaled = y_train
      y_test_scaled = y_test

    if self.use_mask:
      X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
      y_train_scaled = np.nan_to_num(y_train_scaled, nan=0.0)
      X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)
      y_test_scaled = np.nan_to_num(y_test_scaled, nan=0.0)
      y_train = np.nan_to_num(X_train_scaled, nan=0.0)
      y_test = np.nan_to_num(X_train_scaled, nan=0.0)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,  y_train, y_test, time_test

  def clean_variables(self, variable):
    if self.use_mask:
      variable = np.nan_to_num(variable, nan=0.0)
    return variable

  def standardize_data(self, train_data, test_data):
    """
    Standardize per gridpoint using GridScaler.
    Only applied to channel 0 (the variable).
    """

    # Extract the variable channel (T, H, W)
    var_train = train_data[..., 0]
    var_test = test_data[..., 0]

    # Replace only channel 0 (other channels untouched)
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()

    if self.standardize:
      self.logger.debug("Using GridScaler for per-grid standardization...")

      self.scaler = GridScaler()
      var_train_scaled = self.scaler.fit_transform(var_train)
      var_test_scaled = self.scaler.transform(var_test)

      train_scaled[..., 0] = var_train_scaled
      test_scaled[..., 0] = var_test_scaled
    else:
       self.logger.debug("Assuming data is already standardized...")

    return train_scaled, test_scaled

  def reshape_for_model(self, X, model_type="convlstm"):
    if model_type.lower() in ("convlstm", "3dcnn"):
        return X

    elif model_type.lower() == "2dcnn":
        samples, n_lags, H, W, C = X.shape
        X = X.transpose(0, 2, 3, 1, 4)  # (samples, H, W, n_lags, C)
        X = X.reshape(samples, H, W, n_lags * C)
        X = X.transpose(0, 3, 1, 2)

        self.logger.debug(f"Reshaped size for 2dcnn: {X.shape}")
        return X
    else:
        raise ValueError("Invalid model_type")


class GridScaler:
    """
    Standardize climate data per gridpoint:
    Computes mean & std over the time axis only.
    Works for (time, lat, lon) and (time, lat, lon, channels).
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.eps = 1e-6  # prevent divide-by-zero

    def fit(self, data):
        """
        Fit scaler using training data.
        data shape: (time, lat, lon) or (time, lat, lon, channels)
        """
        if data.ndim == 3:
            # (T, H, W)
            self.mean_ = np.mean(data, axis=0, keepdims=True)
            self.std_ = np.std(data, axis=0, keepdims=True)
        elif data.ndim == 4:
            # (T, H, W, C)
            self.mean_ = np.mean(data, axis=0, keepdims=True)
            self.std_ = np.std(data, axis=0, keepdims=True)
        else:
            raise ValueError("GridScaler only supports 3D or 4D climate data.")

        self.std_[self.std_ < self.eps] = 1.0  # avoid NaNs

    def transform(self, data):
        """Apply scaling to new data."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Must call fit() before transform()")

        return (data - self.mean_) / self.std_

    def fit_transform(self, data):
        """Convenience method."""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        """Convert scaled data back to physical units."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fitted yet.")
        return data * self.std_ + self.mean_


def get_target_grid():
  lat_target = np.arange(-89.5, 90.5, 2.0)
  lon_target = np.arange(0.5, 360.5, 4.0)

  grid_out = xr.Dataset(
    {
        "lat": (["lat"], lat_target),
        "lon": (["lon"], lon_target),
    }
  )
  return grid_out
