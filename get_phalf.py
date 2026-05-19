import xarray as xr

file = "/scratch4/GFDL/gfdlscr/Uriel.Ramirez/archive_260429/RAW/18510101.full_state.ens_01.tile1.nc"
new_file = "/scratch4/GFDL/gfdlscr/Uriel.Ramirez/archive_260429/RAW/phalf.nc"

ds = xr.open_dataset(file)
phalf = ds[["phalf"]] 
phalf.encoding.pop("unlimited_dims", None)

phalf.to_netcdf(new_file)
ds.close()
