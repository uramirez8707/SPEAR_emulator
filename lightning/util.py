import numpy as np
import xarray as xr

#example:  data_dict = {"t_ref": "data/atmos.192101-201012.t_ref.nc"}

def load_data(data_dict: dict, test_with_global_average: bool = False) -> tuple[np.ndarray, int]:
    """Open NetCDF files and return a mapping of variable names to arrays."""
    
    data = {}
    for variable, datafile in data_dict.items():
        with xr.open_dataset(datafile, decode_timedelta=True) as ds:
            data[variable] = ds[variable].values
        
    key = list(data.keys())[0]
    ntimes = data[key].shape[0]
        
    datalist = []
    if test_with_global_average:
        for itime in range(ntimes):
            datalist.append([data[variable][itime].mean() for variable in data.keys()])
    else:
        for itime in range(ntimes):
            datalist.append([data[variable][itime] for variable in data.keys()])
    
    return np.array(datalist), ntimes
