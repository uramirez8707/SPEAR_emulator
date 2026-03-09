import yaml
import xarray as xr
import os

working_path = "/work/unr/SPEAR_emulator"
yaml_file = f"{working_path}/RawData/data_summary.yaml"
levels = [1000, 925, 850, 700, 500, 300, 250, 200, 100, 50, 10]
dim_name = "level"

with open(yaml_file, 'r') as yaml_file:
    variables_info = yaml.safe_load(yaml_file)

out_yaml = []
for variable in variables_info:
    variable_name = variable['variable_name']
    print(f"Working on {variable_name}")
    print(variable['files'])

    for file in variable['files']:
        print(f"Opening {file['path']}")
        ensemble_label = file['ensemble_label']
        basename = file['basename']

        if variable_name == "static":
            print("Skipping because variable is static!")
            var = {'var_name': variable_name,
                   'path': file['path'],
                   'basename': basename,
                   'original_path': file['path'],
                   'ensemble_label': ensemble_label}
            out_yaml.append(var)
            continue

        ds = xr.open_dataset(file['path'], decode_timedelta=True)
        var_data = ds[variable_name]

        if len(var_data.dims) != 4:
            print("Skipping because variable is not 3D!")
            var = {'var_name': variable_name,
                   'path': file['path'],
                   'basename': basename,
                   'original_path': file['path'],
                   'ensemble_label': ensemble_label}
            out_yaml.append(var)
            continue

        for level in levels:
            print(f"Working on the level {level}hPa ... ")

            var = {}
            new_var_name = f"{variable_name}{level}"
            new_long_name = f"{var_data.attrs['long_name']} at {level}hPa"
            new_file_name = basename.replace(variable_name, new_var_name)
            new_file_path = f"{working_path}/DATA/{ensemble_label}/{new_file_name}"

            var['var_name'] = new_var_name
            var['path'] = new_file_path
            var['basename'] = new_file_name
            var['original_path'] = file['path']
            var['ensemble_label'] = ensemble_label
            out_yaml.append(var)

            os.makedirs(f"{working_path}/DATA/{ensemble_label}", exist_ok=True)

            if os.path.isfile(new_file_path):
                print(f"Skiping {new_file_path} because it alread exists!")
            else:
                level_data = var_data.sel({dim_name: level}, method='nearest').copy()

                print(f"Setting up {new_var_name} which is {new_long_name} to file {new_file_path}")
                level_data = level_data.rename(new_var_name)
                level_data.attrs['long_name'] = new_long_name
                level_data.to_netcdf(new_file_path, encoding={new_var_name: {"zlib": True}})
                print(f"{new_file_path} has been created !!!")

with open(f"{working_path}/DATA/data_summary.yaml", 'w') as yaml_file:
    yaml.dump(out_yaml, yaml_file, default_flow_style=False)
