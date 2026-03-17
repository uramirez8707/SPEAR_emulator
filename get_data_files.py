import pandas as pd
import re
import os
import shutil
import xarray as xr
import yaml

def dmgetmagic(x):
    cmd = 'dmget %s'% str(x) 
    return os.system(cmd)

frequency = "mon"
realm = "atmos"
chunk_freq = "90yr"
variables = ["temp", "t_surf", "ucomp", "vcomp", "omega", "hght", "sphum", "precip", "t_ref", "ps", "swdn_toa"]

working_path = "/work/unr/SPEAR_emulator/RawData"
pattern = r"(pp_ens_\d{2})"

df = pd.read_csv("RawData/data_catalog.csv")
df_vars = df[(df["frequency"] == frequency) &
                 (df["realm"] == realm) & 
                 (df["chunk_freq"] == chunk_freq)]

# Move all of the data from archive to a local location
variable_data = []
for variable in variables:
    df_var = df_vars[(df_vars["variable_id"] == variable)]
    paths = df_var["path"].tolist()

    var = {}
    var["variable_name"] = variable
    files = []
    for path in paths:
        ensemble_path = re.search(pattern, path).group(1)
        destination_path = f"{working_path}/{ensemble_path}/"
        basename = os.path.basename(path)
        new_path = f"{destination_path}{basename}"

        if not os.path.isfile(new_path):
            dmgetmagic(path)
            os.makedirs(destination_path, exist_ok=True)
            shutil.copy(path, destination_path)

        file = {}
        file["original_path"] = path
        file["path"] = new_path
        file["ensemble_label"] = ensemble_path
        file["basename"] = basename
        files.append(file)
    var["files"] = files
    variable_data.append(var)

var = {}
var["variable_name"] = "static"
var["files"] = []

atmos = df[(df["frequency"] == "fx") & (df["realm"] == "atmos")]
path = atmos['path'].tolist()[0]
basename = os.path.basename(path)
new_path = f"{working_path}/{basename}"
var["files"].append({'original_path': path,
                     'basename': basename,
                     'path': new_path,
                     'ensemble_label': ""})

ocean = df[(df["frequency"] == "fx") & (df["realm"] == "ocean")]
path = ocean['path'].tolist()[0]
basename = os.path.basename(path)
new_path = f"{working_path}/{basename}"
var["files"].append({'original_path': path,
                     'basename': basename,
                     'path': new_path,
                     'ensemble_label': ""})

for file in var["files"]:
    if not os.path.isfile(file['path']):
        dmgetmagic(file['original_path'])
        shutil.copy(file['original_path'], file['path'])
variable_data.append(var)

with open(f"{working_path}/data_summary.yaml", 'w') as yaml_file:
    yaml.dump(variable_data, yaml_file, default_flow_style=False)

print(f"Summary file located in {working_path}/data_summary.yaml")