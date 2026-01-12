from PostProcressingScripts.utils.data_PostProcessing import procress_variable

split_data_info ={'training': "1921-1990",
                  'validating': "1991-2000",
                  'testing': "2001-2010"}

var_name = 't_ref'
T_REF = {'variable_name': var_name,
            'file_name': f'DATA/atmos.192101-201012.{var_name}.nc',
            'split_data_info': split_data_info,
            'output_file_name': f"PP_DATA/{var_name}.192101-201012.nc",
            'add_spatial_coordinates': True,
            'standardize': True,
            'fill_nan_method': 'crash'
        }

var_name = 't_surf'
T_SURF = {'variable_name': var_name,
            'file_name': f'DATA/atmos.192101-201012.{var_name}.nc',
            'split_data_info': split_data_info,
            'output_file_name': f"PP_DATA/{var_name}.192101-201012.nc",
            'add_spatial_coordinates': False,
            'standardize': True,
            'fill_nan_method': 'crash'
            }

var_name = 'T850'
T850 = {'variable_name': var_name,
            'file_name': f'DATA/atmos.192101-201012.{var_name}.nc',
            'split_data_info': split_data_info,
            'output_file_name': f"PP_DATA/{var_name}.192101-201012.nc",
            'add_spatial_coordinates': False,
            'standardize': True,
            'fill_nan_method': 'pressure_mask'
            }

var_name = 'slp'
SLP = {'variable_name': var_name,
            'file_name': f'DATA/atmos.192101-201012.{var_name}.nc',
            'split_data_info': split_data_info,
            'output_file_name': f"PP_DATA/{var_name}.192101-201012.nc",
            'add_spatial_coordinates': False,
            'standardize': True,
            'fill_nan_method': 'crash'
            }

var_name = 'Z500'
Z500 = {'variable_name': var_name,
            'file_name': f'DATA/atmos.192101-201012.{var_name}.nc',
            'split_data_info': split_data_info,
            'output_file_name': f"PP_DATA/{var_name}.192101-201012.nc",
            'add_spatial_coordinates': False,
            'standardize': True,
            'fill_nan_method': 'crash'
            }

var_name = 'SST'
SST = {'variable_name': var_name,
            'file_name': f'DATA/ocean.192101-201012.{var_name}.nc',
            'split_data_info': split_data_info,
            'output_file_name': f"PP_DATA/{var_name}.192101-201012.nc",
            'add_spatial_coordinates': False,
            'standardize': True,
            'fill_nan_method': 'ocean_mask'
            }

var_name = 'sic'
SIC = {'variable_name': var_name,
            'file_name': f'DATA/ice.192101-201012.{var_name}.nc',
            'split_data_info': split_data_info,
            'output_file_name': f"PP_DATA/{var_name}.192101-201012.nc",
            'add_spatial_coordinates': False,
            'standardize': False,
            'fill_nan_method': 'ocean_mask'
            }

VARS = [T_REF, T_SURF, T850, SLP, Z500, SST, SIC]

for var_info in VARS:
    procress_variable(var_info)