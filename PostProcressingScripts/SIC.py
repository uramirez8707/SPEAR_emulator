'''
This script calculates the sea ice concentration by summing the ice concentration at each thickness category

The `CN` (ice concentration per thickness category) data was obtained from:
/archive/wfc/SPEAR/SPEAR_c192_o1_Hist_AllForc_IC1921_K50_ens_01_03/pp_ens_01/ice/ts/monthly/90yr/ice.192101-201012.CN.nc
'''

from utils.data_PostProcessing import output_file, get_sum_over_dimension

variable = "CN"
varname_out = "sic"
dim_name = "ct"

metadata = {'long_name': "Sea Ice Concentration",
           'units': "fraction",
           "note": "Summed over ice categories (ct)"}
files = [
    {'file_name_in': 'RawData/ice.192101-201012.CN.nc',
     'output_file': 'DATA/ice.192101-201012.sic.nc',
     'metadata': metadata}
]

for file in files:
    file_name = file.get("file_name_in")
    out_name = file.get("output_file")
    metadata = file.get('metadata')

    out_var = get_sum_over_dimension(file_name,variable, dim_name)

    output_file(out_var, varname_out, out_name, metadata)