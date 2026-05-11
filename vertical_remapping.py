import xarray as xr
from pathlib import Path

data_path = "/scratch4/GFDL/gfdlscr/Uriel.Ramirez/archive_260429"

class ConfigClass:
    def __init__(self, phalf_data_path, in_data_path, out_data_path, tmp_path,
                 variables,
                 coarse_mapping):
        self.phalf_data_path = phalf_data_path
        self.in_data_path = in_data_path
        self.out_data_path = out_data_path
        self.tmp_path = tmp_path
        self.variables = variables
        self.coarse_mapping = coarse_mapping

    def get_list_of_files(self, variable):
        variable_directory = Path(self.in_data_path) / variable
        filtered_files = [f for f in variable_directory.iterdir() if "ens_01" in f.name]
        return sorted(filtered_files)

    def set_pressure_thickness(self):
        '''
        Calculates the pressure thickness (dp) for each full pressure level.

        The pressure thickness is calculated at each pressure level by computing 
        the difference between adjacent half-pressure levels (phalf). This thickness 
        represents the mass weight of each layer and is required for mass-conserving 
        vertical coarsening or integration.
        '''

        print(f"[1] Reading the pressure thickness from the file {str(self.phalf_data_path)}")
        ds = xr.open_dataset(self.phalf_data_path)
        ds['dp'] = ('pfull', ds['phalf'].diff('phalf').values)
        ds.close()

        self.dp = ds['dp']
    
    def remap_file(self, file, variable):
        '''
        Reads a high-resolution 3D atmospheric NetCDF file and performs a 
        mass-weighted vertical coarsening to 8 vertical lelvels

        This method groups the native vertical levels into 8 coarse bins and 
        calculates a pressure-thickness-weighted average for the target variable. 
        This ensures that mass and energy conservation.

        $$\bar{X} = \frac{\sum (X_i \Delta p_i)}{\sum \Delta p_i}$$

        '''
        file_name = file.name
        new_var_dir = Path(out_data_path) / variable
        new_var_dir.mkdir(exist_ok=True)
        new_file = new_var_dir / file_name
        if new_file.exists():
            print("Skipping ... file was already remapped")
            return

        print(f"----> Remapping into {str(new_file)}")

        ds = xr.open_dataset(file)
        ds.coords['coarse_level'] = ('pfull', self.coarse_mapping)
        ds['dp'] = ('pfull', self.dp.values)

        ds_weighted_sum = (ds[variable] * ds['dp']).groupby('coarse_level').sum(dim='pfull')
        dp_sum = ds['dp'].groupby('coarse_level').sum(dim='pfull')

        ds_coarse = ds_weighted_sum / dp_sum
        ds_coarse.name = variable
        ds_coarse = ds_coarse.to_dataset()

        ds_coarse = ds_coarse.rename({'coarse_level': 'pfull'})
        ace_pressures = [25, 96, 203, 345, 517, 695, 847, 963]

        ds_coarse['pfull'] = ace_pressures
        ds_coarse['pfull'].attrs['axis'] = 'Z'
        ds_coarse['pfull'].attrs['units'] = 'hPa' # standard pressure units
        ds_coarse['pfull'].attrs['long_name'] = 'pressure'
        ds_coarse['pfull'].attrs['standard_name'] = 'air_pressure'

        ds_coarse.to_netcdf(new_file)

        ds.close()
        print("----> Done")

    def remap_3D_variables(self):
        self.set_pressure_thickness()
        for variable in self.variables:
            print(f"[2] Working on remapping the variable {variable}")
            files = self.get_list_of_files(variable)
            for file in files:
                print(f"--> Working on the file {file}")
                self.remap_file(file, variable)

phalf_data_path = f"{data_path}/RAW/phalf.nc"
junk_directory = f"{data_path}/tmp"
in_data_path = f"{data_path}/RAW"
out_data_path = f"{data_path}/COARSE"
variables = ["air_temperature"]
coarse_mapping = (
    [0]*7 +
    [1]*4 +
    [2]*3 +
    [3]*2 +
    [4]*3 +
    [5]*3 + 
    [6]*3 +
    [7]*8
)

config = ConfigClass(phalf_data_path, in_data_path, out_data_path, junk_directory,
                     variables,
                     coarse_mapping)

config.remap_3D_variables()
