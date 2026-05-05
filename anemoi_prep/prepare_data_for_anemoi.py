from pathlib import Path
import xarray as xr
import pandas as pd
import yaml

def get_variable_list(data_dir:Path):
    '''
        Go to the data_dir
        - Get all of the static variables (i.e they are netcdf files and not directories)
        - Get all of the dynamic variables (i.e they are folders)
    '''
    static_variables = []
    dynamic_variables = []
    for entry in raw_data_dir.iterdir():
        if entry.is_dir():
            dynamic_variables.append(entry.name)
        elif entry.is_file():
            static_variables.append(entry.name)

    return static_variables, dynamic_variables

def procress_static_variables(static_variables, in_dir, out_dir):
    out = []
    for static in static_variables:
        var_name = Path(static).stem
        print(f"Procressing {var_name}")
        var_file_in = in_dir / f"{var_name}.nc"
        var_file_out = out_dir / f"{var_name}.nc"

        var = {}
        var['repeated-dates'] = {
                "mode": "constant",
                "source": {
                    "netcdf": {
                        "path": str(var_file_out),
                        "variable": var_name
                        }
                    }
                }
        out.append(var)
        if var_file_out.is_file():
            print(f"--> Skipping {str(var_file_out)} has already been created")
            continue

        print(f"--> Updating file: {str(var_file_in)}")
        fix_GFDL_file(var_file_in, var_file_out, is_static=True)
    return out

def fix_GFDL_file(input_file, output_file, is_static=False):
    ds = xr.open_dataset(input_file, chunks={'time': 'auto'})

    rename_dict = {}
    if 'grid_yt' in ds.dims or 'grid_yt' in ds.variables:
        rename_dict['grid_yt'] = 'latitude'
    if 'grid_xt' in ds.dims or 'grid_xt' in ds.variables:
        rename_dict['grid_xt'] = 'longitude'

    if rename_dict:
        ds = ds.rename(rename_dict)

    if is_static:
        ds.to_netcdf(output_file)
        ds.close()

    else:
        ds_standard = ds.convert_calendar('standard', align_on='date')
        ds_standard = ds_standard.drop_vars('time_bnds', errors='ignore')
        if 'bounds' in ds_standard.time.attrs:
            del ds_standard.time.attrs['bounds']
        ds_standard.to_netcdf(output_file)
        ds.close()
        ds_standard.close()

def get_start_end_time(filename):
    start_year = int(filename[:4])
    end_year = start_year +  1

    start_year_str = f"{start_year}-01-01T06"
    end_year_str = f"{end_year}-01-01"

    return start_year_str, end_year_str

def procress_dynamic_variables(dynamic_variables, in_dir:Path, out_dir:Path):
    out = []
    frequency = "6h"
    for variable in dynamic_variables:
        print(f"--> Procressing {variable}")
        var_dir = in_dir / variable
        new_var_dir = out_dir / variable
        new_var_dir.mkdir(parents=True, exist_ok=True)
        var = {}
        var['concat'] = []
        for file in sorted(var_dir.iterdir()):
            if "ens_01" not in str(file):
                continue

            new_file = new_var_dir / file.name
            start_time, end_time = get_start_end_time(file.name)
            file_dict = {
                    'dates': {'start': start_time, 'end': end_time, 'frequency': frequency},
                    'netcdf': {'path': str(new_file), 'variable': variable}
                    }
            var['concat'].append(file_dict)
            if new_file.is_file():
                print(f"Skipping {str(new_file)} has already been created")
                continue

            print(f"  Updating file: {str(file)}")
            fix_GFDL_file(file, new_file)
        out.append(var)
    return out

out_data_dir = Path("/scratch4/GFDL/gfdlscr/Uriel.Ramirez/archive_260429/ANEMOI-READY")
raw_data_dir = Path("/scratch4/GFDL/gfdlscr/Uriel.Ramirez/archive_260429/RAW")

static_variables, dynamic_variables = get_variable_list(out_data_dir)

dynamic_config = procress_dynamic_variables(dynamic_variables, raw_data_dir, out_data_dir)
static_config = procress_static_variables(static_variables, raw_data_dir, out_data_dir)

out_config = {}
out_config['dates'] = {
    'start': '1851-01-01T06:00:00',
    'end': '1861-01-01T00:00:00',  # Adjust these to match your full dataset span
    'frequency': '6h'
}

out_config['input'] = {}
out_config['input']['join'] = dynamic_config + static_config
with open('anemoi_recipe.yaml', 'w') as f:
    yaml.dump(out_config, f, sort_keys=False, default_flow_style=False)

