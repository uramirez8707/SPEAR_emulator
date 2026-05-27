from pathlib import Path
import xarray as xr
import pandas as pd
import yaml

def get_variable_list(data_dir:Path, coarse_dir:Path):
    '''
        Go to the coarse_dir
        - Get all of the coarsed variables (i.e they are directory)
        Go to the data_dir
        - Get all of the static variables (i.e they are netcdf files and not directories).
          This does not include the coarse variables
        - Get all of the dynamic variables (i.e they are folders)
          This does not include "phalf.nc"
    '''
    static_variables = []
    dynamic_variables = []
    coarsed_variables = []

    for entry in coarse_dir.iterdir():
        if entry.is_dir():
            coarsed_variables.append(entry.name)
    coarse_set = set(coarsed_variables)

    for entry in raw_data_dir.iterdir():
        name = entry.name
        if entry.is_dir():
            if name not in coarse_set:
                dynamic_variables.append(name)
        elif entry.is_file():
            if name != "phalf.nc" and name != "18510101.full_state.ens_01.tile1.nc":
                static_variables.append(name)

    return static_variables, dynamic_variables, coarsed_variables

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

    coord_mapping = {
        'grid_yt': 'latitude',
        'grid_xt': 'longitude',
        'yh': 'latitude',
        'xh': 'longitude'
    }

    rename_dict = {}
    for old_name, new_name in coord_mapping.items():
        if old_name in ds.dims or old_name in ds.variables:
            rename_dict[old_name] = new_name

    if rename_dict:
        ds = ds.rename(rename_dict)

    bounds_vars_to_drop = []
    for var in ds.variables:
        if 'bounds' in ds[var].attrs:
            bounds_vars_to_drop.append(ds[var].attrs['bounds'])
            del ds[var].attrs['bounds']

    if bounds_vars_to_drop:
        ds = ds.drop_vars(bounds_vars_to_drop, errors='ignore')

    if is_static:
        ds.to_netcdf(output_file)
        ds.close()
    else:
        ds_standard = ds.convert_calendar('standard', align_on='date')
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

def procress_coarsed_variables(coarse_variables, in_dir:Path, out_dir:Path,
        level_labels):
    out = []
    frequency = "6h"
    for variable in coarse_variables:
        print(f"--> Procressing {variable}")
        var_dir = in_dir / variable
        new_var_dir = out_dir / variable
        new_var_dir.mkdir(parents=True, exist_ok=True)

        concat_block = []
        for file in sorted(var_dir.iterdir()):
            if "ens_01" not in str(file):
                continue
            new_file = new_var_dir / file.name
            start_time, end_time = get_start_end_time(file.name)

            concat_block.append({
                "dates": {
                    "start": start_time,
                    "end": end_time,
                    "frequency": frequency,
                },
                "netcdf": {
                    "path": str(new_file),
                    "variable": variable,
                },
            })

            if new_file.is_file():
                print(f"Skipping {str(new_file)} has already been created")
                continue

            print(f"  Updating file: {str(file)}")
            fix_GFDL_file(file, new_file)

        rename_mapping = {
                            f"{variable}_{level}": f"{variable}_level{i}"
                            for i, level in enumerate(pressure_level_label, start=1)
                         }
        pipe = {
                "pipe": [
                    {
                        "concat": concat_block
                    },
                    {
                        "rename": rename_mapping
                    }
                 ]
            }

        out.append(pipe)

    return out

def get_units(file, variable):
    with xr.open_dataset(file) as ds:
        units = ds[variable].attrs.get("units")

    return {
        "variable": variable,
        "units": units,
    }

def dump_variable_metadata(variables, in_dir):
    out = []
    print("")
    for variable in variables[0]:
        print(f"Getting Metadata for {variable}")

        var_dir = in_dir / variable
        if var_dir.is_file():
            first_file = var_dir
            variable = variable.removesuffix(".nc")
        else:
            first_file = sorted( f for f in var_dir.iterdir() if f.is_file())[0]
        out.append(get_units(first_file, variable))
    return out

out_data_dir = Path("/scratch4/GFDL/gfdlscr/Uriel.Ramirez/archive_260429/ANEMOI-READY")
raw_data_dir = Path("/scratch4/GFDL/gfdlscr/Uriel.Ramirez/archive_260429/RAW")
coarse_data_dir = Path("/scratch4/GFDL/gfdlscr/Uriel.Ramirez/archive_260429/COARSE")

pressure_level_label = [25, 96, 203, 345, 517, 695, 847, 963]

static_variables, dynamic_variables, coarsed_variables = get_variable_list(out_data_dir, coarse_data_dir)

print(f"Coarsed variables {coarsed_variables}")
print(f"Dynamic variables {dynamic_variables}")
print(f"Static variables {static_variables}")

metadata_path = Path("metadata.yaml")
if not metadata_path.exists():
    metadata = dump_variable_metadata([static_variables + dynamic_variables + coarsed_variables], raw_data_dir)
    with open(metadata_path, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

coarsed_config = procress_coarsed_variables(coarsed_variables, coarse_data_dir, out_data_dir, pressure_level_label)
dynamic_config = procress_dynamic_variables(dynamic_variables, raw_data_dir, out_data_dir)
static_config = procress_static_variables(static_variables, raw_data_dir, out_data_dir)

out_config = {}
out_config['dates'] = {
    'start': '1851-01-01T06:00:00',
    'end': '1861-01-01T00:00:00',
    'frequency': '6h'
}

out_config['input'] = {}
out_config['input']['join'] = coarsed_config + dynamic_config + static_config
with open('wut.yaml', 'w') as f:
    yaml.dump(out_config, f, sort_keys=False, default_flow_style=False)

