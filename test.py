from model import SpearEmulator, AutoregressiveSpearEmulator
from data_load import get_dataloaders, get_updated_channels
from utils import configSetUp, get_indices
import torch
from output import OutputData, ModelResults
from pathlib import Path
from plotting_utils import plot_loss_heat_map, plot_loss
import logging
from inference import do_inference

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(message)s"
)

logger = logging.getLogger(__name__)

def test_model(config, label, fig_dir, working_dir, data_dir):
    logger.info(f"Checking if output has already been created for {label}")
    output_data = data_dir / f"{label}_out.npz"

    if output_data.exists():
        logger.info("Output already exists, loading it up")
        output = load_output()
    else:
        logger.info("Output does not exists, generating ...")
        output = do_inference(config, label, working_dir, output_data, fig_dir)

        variables = config.outputs
        for var in variables:
            output.plot_spatial_maps(var, label=f"{label}")

    return output

##################################3

logger.setLevel(logging.DEBUG)

working_dir = "/scratch4/GFDL/gfdlscr/Uriel.Ramirez/SPEAR_TRAINING_JOBS/architecture_comparison"
data_dir = Path(f"{working_dir}/output/results")
data_dir.mkdir(parents=True, exist_ok=True)

run_cnn = True
run_unet = True
run_sfno = True
if run_cnn:
    fig_dir = Path(f"{working_dir}/output/figs_cnn")
    fig_dir.mkdir(parents=True, exist_ok=True)
    Results_cnn = ModelResults(fig_dir)

    config = configSetUp(config_yaml=f"{working_dir}/config_cnn-1.yaml")
    output = test_model(config, "cnn-candidate-1", fig_dir, working_dir, data_dir)
    Results_cnn.add_model(output)

    config = configSetUp(config_yaml=f"{working_dir}/config_cnn-2.yaml")
    output = test_model(config, "cnn-candidate-2", fig_dir, working_dir, data_dir)
    Results_cnn.add_model(output)

    Results_cnn.create_var_plots(config)

if run_unet:
    fig_dir = Path(f"{working_dir}/output/figs_unet")
    fig_dir.mkdir(parents=True, exist_ok=True)
    Results_unet = ModelResults(fig_dir)

    config = configSetUp(config_yaml=f"{working_dir}/config_unet-1.yaml")
    output = test_model(config, "unet-candidate-1", fig_dir, working_dir, data_dir)
    Results_unet.add_model(output)

    config = configSetUp(config_yaml=f"{working_dir}/config_unet-2.yaml")
    output = test_model(config, "unet-candidate-2", fig_dir, working_dir, data_dir)
    Results_unet.add_model(output)

    Results_unet.create_var_plots(config)

if run_sfno:
    fig_dir = Path(f"{working_dir}/output/figs_sfno")
    fig_dir.mkdir(parents=True, exist_ok=True)
    Results_sfno = ModelResults(fig_dir)

    config = configSetUp(config_yaml=f"{working_dir}/config_sfno-1.yaml")
    output = test_model(config, "sfno-candidate-1", fig_dir, working_dir, data_dir)
    Results_sfno.add_model(output)

    config = configSetUp(config_yaml=f"{working_dir}/config_sfno-2.yaml")
    output = test_model(config, "sfno-candidate-2", fig_dir, working_dir, data_dir)
    Results_sfno.add_model(output)

    Results_sfno.create_var_plots(config)


