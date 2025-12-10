import xarray as xr

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