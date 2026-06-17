from model import SpearEmulator, AutoregressiveSpearEmulator
from data_load import get_dataloaders, get_updated_channels
from utils import configSetUp, get_indices
import torch
from output import OutputData, ModelResults
from pathlib import Path
from plotting_utils import plot_loss_heat_map, plot_loss
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(message)s"
)

logger = logging.getLogger(__name__)

def get_variable_mappings(mappings, variable):
    for mapping in mappings:
        if mapping["variable_name"] == variable:
            labels = mapping["labels"]
            for label in labels:
                if label['label'] == "t-1":
                    return label['index']
    raise RuntimeError(f"Unable to determine the index for {variable} in {mappings}")

def do_forecast_rollout(label, model, testing, config, device, working_dir, fig_dir):
    model.eval()
    
    logger.info(f"\n ------------------------------- \n🏃 Starting rollout on {device}...")

    input_names = config.input_channels
    all_vars = config.all_vars
    indices = get_indices(input_names, all_vars)

    target_names = config.outputs
    dynamics_variables = config.dynamics

    nlags = config.get_nlags()
    method = config.get_data_load_method()

    y_means = model.y_means
    y_stds = model.y_stds

    # y_batch has dimensions (nsamples, nlags, ndynamicvariables, nensembles, npoints)
    # target_indices_y corresponds to the indices of ndynamicvariables for the output variables only
    target_indices_in_y = [dynamics_variables.index(v) for v in target_names]
    predictions = []
    ground_truth = []
    physical_preds_buffer = []

    with torch.no_grad():
        for step, (x_batch, y_batch) in enumerate(testing.tensor):
            logger.debug(f"x batch shape {x_batch.shape} - y batch shape {y_batch.shape} - {target_indices_in_y}")

            if method == "autoregressive":
                y_step = y_batch[:, 0, target_indices_in_y, ...]
            else:
                y_step = y_batch

            logger.debug(f"y step shape {y_step.shape}")

            # x_batch contains all of the variables, so get only the acutal_inputs
            x = x_batch.squeeze(2)
            x = x.view(x.size(0), x.size(1), config.nlat, config.nlon)

            # All of the data at time (t-1), including the diagnostics
            x_all = x

            x = x[:, indices, :, :]
            logger.debug(f"x shape {x.shape} (inputs only!)")

            # Reshape to the format expected by the model
            y = y_step.to(device).squeeze(2).view(-1, len(target_names), config.nlat, config.nlon)
            logger.debug(f"y shape {y.shape}")

            logger.debug(f"Working on step {step}")
            if step > 0:
                x_next = x.clone()

                if method == "autoregressive":
                    past_pred = physical_preds_buffer[-1]

                    for i, input_var in enumerate(input_names):
                        base_name = input_var.split('(')[0]
                        logger.debug(f"Working on ... {base_name}")
                        for j, output in enumerate(target_names):
                            base_out = output.split('(')[0]
                            if base_name == base_out:
                                logger.debug(f"--- {base_name}: x_next[:, {i}, :, :] = past_pred[:, {j}, :, :]")
                                x_next[:, i, :, :] = past_pred[:, j, :, :]
                                break
                else:
                    raise RuntimeError("This is no longer supported --- sorry!")
                    for target_idx, target_name in enumerate(target_names):
                        target_lag_indx = get_variable_mappings(mappings, target_name)
                        for past_pred_phys in reversed(physical_preds_buffer):
                            x_next[:, target_lag_indx, :, :] = past_pred_phys[:, target_idx, :, :]
                            target_lag_indx += 1

                x = x_next

            preds_norm = model(x, x_all)
            preds_physical = (preds_norm * y_stds) + y_means
            physical_preds_buffer.append(preds_physical.clone())
            if len(physical_preds_buffer) > nlags:
                physical_preds_buffer.pop(0)

            predictions.append(preds_physical.detach().cpu())
            ground_truth.append(y.detach().cpu())
            if step > 100:
                break

            if step == 1:
                logger.debug("Turning off debug logs")
                logger.setLevel(logging.INFO)

    start_idx = config.get_nlags()
    end_idx = start_idx + len(predictions)
    rollout_dates = testing.times[start_idx : end_idx]

    return OutputData(label, predictions, ground_truth,
            rollout_dates, testing.lat, config.nlat, testing.lon, config.nlon,
            config.outputs, working_dir, fig_dir)

def test_model(config, label, fig_dir, working_dir):
    training, validating, testing  = get_dataloaders(config)
    input_channels, out_channels = get_updated_channels(config)

    config.set_channels(input_channels, out_channels)
    config.set_grid(training)

    checkpoint_path = f"{working_dir}/output/{label}/checkpoints/last.ckpt"
    logger.info(f"Getting checkpoint from: {checkpoint_path}")

    # Set up the correct class
    method = config.get_data_load_method()
    if method == "autoregressive":
        ModelClass = AutoregressiveSpearEmulator
    else:
        ModelClass = SpearEmulator

    model = ModelClass.load_from_checkpoint(checkpoint_path, config=config)
    model.shapes_logged = True

    # Plot the training/validation losses
    logs_dir = Path(working_dir) / "output" / label / "logs"
    version_dirs = [d for d in logs_dir.glob("version_*") if d.is_dir()]
    latest_version = max(
        version_dirs,
        key=lambda p: int(p.name.split("_")[1])
    )
    log_file = latest_version / "metrics.csv"
    output_file = f"{fig_dir}/losses.{label}.png"
    plot_loss_heat_map(log_file,
              fig_dir=fig_dir,
              label=label,
              output_channels=config.outputs)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.debug("🚀 GPU detected! Running on:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.debug("🐢 No GPU available. Falling back to CPU for inference.")

    # Test with rollout
    model.to(device)
    output = do_forecast_rollout(label, model, testing, config, device,
             working_dir, fig_dir)

    print("Finished with the rollout ... ")

    # Plot spatial plots
    variables = config.outputs
    for var in variables:
        output.plot_spatial_maps(var, label=f"{label}")

    return output

##################################3

logger.setLevel(logging.DEBUG)

working_dir = "/scratch4/GFDL/gfdlscr/Uriel.Ramirez/SPEAR_TRAINING_JOBS/architecture_comparison"

run_cnn = True
run_unet = True
run_sfno = True
if run_cnn:
    fig_dir = Path(f"{working_dir}/output/figs_cnn")
    fig_dir.mkdir(parents=True, exist_ok=True)
    Results_cnn = ModelResults(fig_dir)

    config = configSetUp(config_yaml=f"{working_dir}/config_cnn-1.yaml")
    output = test_model(config, "cnn-candidate-1", fig_dir, working_dir)
    Results_cnn.add_model(output)

    config = configSetUp(config_yaml=f"{working_dir}/config_cnn-2.yaml")
    output = test_model(config, "cnn-candidate-2", fig_dir, working_dir)
    Results_cnn.add_model(output)

    Results_cnn.create_var_plots(config)

if run_unet:
    fig_dir = Path(f"{working_dir}/output/figs_unet")
    fig_dir.mkdir(parents=True, exist_ok=True)
    Results_unet = ModelResults(fig_dir)

    config = configSetUp(config_yaml=f"{working_dir}/config_unet-1.yaml")
    output = test_model(config, "unet-candidate-1", fig_dir, working_dir)
    Results_unet.add_model(output)

    config = configSetUp(config_yaml=f"{working_dir}/config_unet-2.yaml")
    output = test_model(config, "unet-candidate-2", fig_dir, working_dir)
    Results_unet.add_model(output)

    Results_unet.create_var_plots(config)

if run_sfno:
    fig_dir = Path(f"{working_dir}/output/figs_sfno")
    fig_dir.mkdir(parents=True, exist_ok=True)
    Results_sfno = ModelResults(fig_dir)

    config = configSetUp(config_yaml=f"{working_dir}/config_sfno-1.yaml")
    output = test_model(config, "sfno-candidate-1", fig_dir, working_dir)
    Results_sfno.add_model(output)

    config = configSetUp(config_yaml=f"{working_dir}/config_sfno-2.yaml")
    output = test_model(config, "sfno-candidate-2", fig_dir, working_dir)
    Results_sfno.add_model(output)

    Results_sfno.create_var_plots(config)


