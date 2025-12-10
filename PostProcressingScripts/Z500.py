'''
This script can be used to extract the gets the geopotential height at 500hPa

The geopotential height can be obtained from: 
/archive/wfc/SPEAR/SPEAR_c192_o1_Hist_AllForc_IC1921_K50_ens_01_03/pp_ens_01/atmos/ts/monthly/90yr/atmos.192101-201012.hght.nc
'''

import xarray as xr
from utils.data_PostProcessing import get_data_at_pressure_level, output_file

variable = "hght"
varname_out = 'Z500'
dim_name = 'level'
dim_value = 500

metadata = {'long_name':"Geopotential Height at the 500 hPa pressure level"}
files = [
   {'file_name_in':"RawData/atmos.192101-201012.hght.nc",
    'output_file': "DATA/atmos.192101-201012.Z500.nc",
    'metadata': metadata}
]

for file in files:
    file_name = file.get("file_name_in")
    out_name = file.get("output_file")
    metadata = file.get('metadata')

    out_var = get_data_at_pressure_level(file_name, variable, dim_name, dim_value)
    output_file(out_var, varname_out, out_name, metadata)