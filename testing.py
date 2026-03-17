from architectures.snfo import SNFO, train_model
from architectures.cnn_baseline import set_seed
from utils.utils import VarSet, DataSet

set_seed()
t_ref = VarSet(
    var_name="t_ref",
    file_name="PP_DATA_low_res/pp_ens_01/atmos.192101-201012.t_ref.nc",
    is_output=True
)
spatial_features = VarSet(
    var_name="spatial_features",
    file_name="PP_DATA_low_res/spatial_features.nc",
    is_static=True
)
variables = [t_ref] #, spatial_features]
data = DataSet(set_name="set1", variables=variables)
lon, lat = t_ref.get_grid()
time_test = t_ref.get_time('time_test')

model = SNFO(data)
train_model(model)