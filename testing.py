from utils.utils import VarSet, DataSet, get_variable_info
from architectures.cnn_baseline import CNN2D, set_seed
import yaml
from architectures.cnn_baseline import Results

def gather_data_set(variables_info, vars, outputs, static_vars, spatial_features):
    VARIABLES = []
    for variable in vars:
        var_info = get_variable_info(variables_info, variable)
        is_output = variable in outputs
        is_static = variable in static_vars

        VarData = VarSet( var_name=var_info['variable_name'],
                          file_name=var_info['output_file_name'],
                          is_output=is_output, 
                          is_static=is_static,
                          nregressive_steps = 2,
                          nlags=1,
                          debug=False
                        )

        VARIABLES.append(VarData)

        if variable == "t_ref":
            lon, lat = VarData.get_grid()
            time_test = VarData.get_time('time_test')

    VARIABLES.append(spatial_features)
    return VARIABLES, lon, lat, time_test

nepochs = 50
Learning_Rate = 0.00016536937182824412
Layers = [64, 128, 128]
Batch_Size = 32
Optimizer = "AdamW"
Weight_Decay = 0.00012030178871154674

set_seed()

working_path = "/work/unr/SPEAR_emulator"
yaml_file = f"{working_path}/data_summary.yaml"
with open(yaml_file, 'r') as yaml_file:
    variables_info = yaml.safe_load(yaml_file)

spatial_features = VarSet(
    var_name="spatial_features",
    file_name="PP_DATA/spatial_features.nc",
    is_static=True
)

vars = ["t_ref", "land_mask", "zsurf", "swdn_toa", "SST"]
outputs = ["t_ref"]
static_vars = ["land_mask", "zsurf"]

VARIABLES, lon, lat, time_test = gather_data_set(variables_info, vars, outputs, static_vars, spatial_features)

DATA = DataSet(set_name="set0", variables=VARIABLES)
DATA.log_shape()

model2 = CNN2D(
        DATA,
        num_epochs=nepochs,
        batch_size=Batch_Size,
        lr=Learning_Rate,
        filters=Layers,
        weight_decay=Weight_Decay,
        optimizer=Optimizer,
        case=4,
        label=f"EncoderDecoder.phase2-multi-step-loss-dev",
        debug=True,
        nregressive_steps=2
    )

model2.train()