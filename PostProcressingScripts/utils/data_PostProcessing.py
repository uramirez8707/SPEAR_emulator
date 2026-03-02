import xarray as xr
import numpy as np
import xesmf as xe
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(name)s - %(message)s"
)

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

class VarData:
    def __init__(self, variable_name, file_name, split_data_info,
                 fill_method='zero-ed', add_spatial_coordinates=False,
                 standardize=True, fill_nans_method='crash', debug=False):
        self.variable_name = variable_name
        self.filename = file_name
        self.data = None
        self.has_null_values = False
        self.fill_method = fill_method
        self.mask = None
        self.add_spatial_coordinates = add_spatial_coordinates
        self.spatial_features = None
        self.logger = logging.getLogger(self.variable_name)
        self.split_data_info = split_data_info
        self.standardize = standardize
        self.fill_nans_method = fill_nans_method
        if debug:
            self.logger.setLevel(logging.DEBUG)

    def load_data(self):
        self.logger.debug(f"Opening the file {self.filename} to load {self.variable_name} data")
        file = xr.open_mfdataset(self.filename, combine='by_coords', decode_timedelta=True)
        self.data = file[self.variable_name]
        self.units = self.data.attrs.get('units', 'None')

    def interpolate_data(self):
        self.logger.debug("Interpolatating to the target grid")
        target_grid = get_target_grid()
        regridder = xe.Regridder(self.data, target_grid, method="bilinear")
        self.data = regridder(self.data)

    def fill_spatial_coordinates(self):
        if not self.add_spatial_coordinates:
            return
        
        lat = self.data.lat.values
        lon = self.data.lon.values

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        lat_sin, lat_cos = np.sin(lat_rad), np.cos(lat_rad)
        lon_sin, lon_cos = np.sin(lon_rad), np.cos(lon_rad)

        # Create full coordinate grids
        lat_sin_grid, lon_sin_grid = np.meshgrid(lat_sin, lon_sin, indexing="ij")
        lat_cos_grid, lon_cos_grid = np.meshgrid(lat_cos, lon_cos, indexing="ij")

        self.spatial_features = np.stack(
            [lat_sin_grid, lat_cos_grid, lon_sin_grid, lon_cos_grid],
            axis=0
            )
        
        self.logger.debug(f"Adding spatial coordinates with size {self.spatial_features.shape}")

    def split_data_set(self):
        if self.data is None:
            raise RuntimeError("Data has not been loaded")

        self.split_data = {}
        self.logger.debug(f"Original shape of data: {self.data.shape}")

        for split_name, year_range in self.split_data_info.items():
            start_year, end_year = year_range.split("-")

            subset = self.data.sel(
                time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
            )

            start_time = subset.time.dt.year.values[0]
            end_time = subset.time.dt.year.values[-1]

            self.split_data[split_name] = subset

            self.logger.debug(
                f"{split_name}: shape={subset.shape}, "
                f"data range=({start_time}-{end_time})"
            )

    def standarize_data_set(self):
        if not self.standardize:
            return
        
        self.logger.debug("Standardizing data using training set statistics.")
        train = self.split_data['training']

        if self.mask is not None:
            self.logger.debug("Excluding masked grid cells from mean/std calculation")
            train_for_stats = train.where(~self.mask)
        else:
            train_for_stats = train

        self._data_mean = train_for_stats.mean(dim=("time", "lat", "lon"))
        self._data_std  = train_for_stats.std(dim=("time", "lat", "lon"))
        self._data_std = self._data_std.where(self._data_std != 0, 1.0)

        for split_name, split_data in self.split_data.items():
            self.split_data[split_name] = (split_data - self._data_mean) / self._data_std

            if self.mask is not None:
                self.logger.debug(
                    f"{split_name} standardized: mean ~ {self.split_data[split_name].where(~self.mask).mean().values.item():.3f}, "
                    f"std ~ {self.split_data[split_name].where(~self.mask).std().values.item():.3f}"
                )
            else:
                self.logger.debug(
                    f"{split_name} standardized: mean ~ {self.split_data[split_name].mean().values.item():.3f}, "
                    f"std ~ {self.split_data[split_name].std().values.item():.3f}"
                )

    def fill_NaNs(self):
        if self.data.isnull().any():
            self.logger.debug("The data contains NaN values")

            if self.fill_nans_method == "ocean_mask":
                self.logger.debug("Assuming that the NaN are values over land")
            elif self.fill_nans_method == "pressure_mask":
                self.logger.debug("Assuming that the NaN are at the surface")
            else:
                raise RuntimeError("Please provide a method to fill the NaN values")
            
            self.mask = self.data.isnull().any(dim=("time"))
            self.data = self.data.fillna(0.0)

    def dump_spatial_features(self):
        if self.spatial_features is None:
            return
        
        spatial_features = xr.DataArray(
            self.spatial_features,
            dims=("channel", "lat", "lon"),
            coords= {
                "channel": ["lat_sin", "lat_cos", "lon_sin", "lon_cos"],
                "lat": self.data.lat,
                "lon": self.data.lon,
            },
            name="spatial_features"
        )

        spatial_features.to_netcdf("PP_DATA/spatial_features.nc")

    def dump_data(self, output_file):
        data_vars = {
            "training":   self.split_data["training"].rename(time="time_train"),
            "validating": self.split_data["validating"].rename(time="time_valid"),
            "testing":    self.split_data["testing"].rename(time="time_test")
        }

        if self.standardize:
            data_vars["mean"] = self._data_mean
            data_vars["std"] = self._data_std

        if self.mask is not None and self.fill_nans_method == "pressure_mask":
            self.logger.debug("Adding a pressure mask to the file")
            data_vars['mask'] = self.mask

        ds = xr.Dataset(data_vars)
        ds.attrs['original_variable_units'] = self.units
        ds.to_netcdf(output_file)

        self.dump_spatial_features()

    def dump_static_output(self, output_file):
        self.logger.debug(f"Dumping static file: {output_file}")
        data_vars = {self.variable_name: self.data}
        ds = xr.Dataset(data_vars)
        ds.to_netcdf(output_file)

def procress_variable(var_info, debug=False):
    variable_name = var_info['variable_name']
    file_name = var_info['file_name']
    split_data_info = var_info['split_data_info']
    output_file_name = var_info['output_file_name']
    add_spatial_coordinates = var_info['add_spatial_coordinates']
    standardize = var_info['standardize']
    fill_nans_method = var_info['fill_nan_method']
    is_static = var_info.get('is_static', False)

    VAR = VarData(variable_name, file_name, split_data_info,
              add_spatial_coordinates=add_spatial_coordinates,
              standardize=standardize,
              fill_nans_method=fill_nans_method, debug=debug)

    VAR.load_data()
    VAR.interpolate_data()
    VAR.fill_NaNs()
    if is_static:
        VAR.dump_static_output(output_file=output_file_name)
    else:
        VAR.fill_spatial_coordinates()
        VAR.split_data_set()
        VAR.standarize_data_set()
        VAR.dump_data(output_file=output_file_name)

def get_sum_over_dimension(file_name, var_name, dim_name):
    data_file = xr.open_mfdataset(file_name, combine='by_coords', decode_timedelta=True)
    var_data = data_file[var_name]

    fill_value = var_data.attrs.get('_FillValue', None)
    if fill_value is not None:
        var_data = var_data.where(var_data != fill_value)  # Mask out fill values before summing

    out_data = var_data.sum(dim=dim_name, skipna=False)

    if dim_name in out_data.coords:
        out_data = out_data.drop_vars(dim_name, errors="ignore")
    if dim_name in out_data.dims:
        out_data = out_data.squeeze(dim_name)

    return out_data


def get_data_at_pressure_level(file_name, var_name, dim_name, dim_value):
    data_file = xr.open_mfdataset(file_name, combine='by_coords', decode_timedelta=True)
    var_data = data_file[var_name]

    out_data = var_data.sel({dim_name: dim_value}, method='nearest')
    return out_data


def output_file(var_data, var_name_out, var_file_out, metadata):
    var_data = var_data.rename(var_name_out)

    for key, value in metadata.items():
        var_data.attrs[key] = value

    var_data.to_netcdf(var_file_out, encoding={var_name_out: {"zlib": True}})