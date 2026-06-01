from model import SpearEmulator, AutoregressiveSpearEmulator
from data_load import get_dataloaders, get_updated_channels
from utils import configSetUp
import torch
from output import OutputData, ModelResults
from pathlib import Path
from plotting_utils import plot_loss
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
    target_names = config.outputs
    dynamics_variables = config.dynamics
    diagnostic_variables = config.diagnostics_only

    nlags = config.get_nlags()
    method = config.get_data_load_method()

    logger.info(f"Input channels {input_names}")
    logger.info(f"Targets {target_names}")
    logger.info(f"Dynamic variables {dynamics_variables}")
    logger.info(f"Diagnostic variables {diagnostic_variables}")

    y_means = model.y_means
    y_stds = model.y_stds
    logger.debug(
        "Mean and stds used for normalizing\n"
        f"Means:\n{y_means.cpu().numpy()}\n\n"
        f"Stds:\n{y_stds.cpu().numpy()}"
    )
    mappings = config.get_mappings()

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

            # Reshape to the format expected by the model
            x = x_batch.to(device).squeeze(2).view(-1, len(input_names), config.nlat, config.nlon)
            y = y_step.to(device).squeeze(2).view(-1, len(target_names), config.nlat, config.nlon)
            logger.debug(f"x shape {x.shape} - y shape {y.shape}")

            logger.debug(f"Working on step {step}")
            if step > 0:
                x_next = x.clone()

                if method == "autoregressive":
                    past_pred = physical_preds_buffer[-1]

                    for i, target_name in enumerate(target_names):
                        base_name = target_name.split('(')[0]

                        if base_name in diagnostic_variables:
                            logger.debug(f"Skipping {target_name} because it is a diagnostic only")
                            continue

                        idx_in_x = next(j for j, name in enumerate(input_names) if name.startswith(base_name))
                        logger.debug(f"--- x_next[:, {idx_in_x}, :, :] = past_pred[:, {i}, :, :]")
                        x_next[:, idx_in_x, :, :] = past_pred[:, i, :, :]
                else:
                    for target_idx, target_name in enumerate(target_names):
                        target_lag_indx = get_variable_mappings(mappings, target_name)
                        for past_pred_phys in reversed(physical_preds_buffer):
                            x_next[:, target_lag_indx, :, :] = past_pred_phys[:, target_idx, :, :]
                            target_lag_indx += 1

                x = x_next

            preds_norm = model(x)
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

    input_channels, out_channels, diag_channels = get_updated_channels(config)
    config.set_channels(input_channels, out_channels, diag_channels)
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
    log_file = f"{working_dir}/output/{label}/logs/version_0/metrics.csv"
    output_file = f"{fig_dir}/losses.{label}.png"
    plot_loss(log_file,
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

    # Plot spatial plots
    variables = config.outputs
    for var in variables:
        output.plot_spatial_maps(var, label=f"{label}")

    return output

##################################3

logger.setLevel(logging.DEBUG)

working_dir = "/scratch4/GFDL/gfdlscr/Uriel.Ramirez/SPEAR_TRAINING_JOBS/run2"

fig_dir = Path(f"{working_dir}/output/figs")
fig_dir.mkdir(parents=True, exist_ok=True)

Results = ModelResults(fig_dir)

#config = configSetUp(config_yaml=f"{working_dir}/config_autoregressive.yaml")
#output = test_model(config, "autoregressive_nsteps_3", fig_dir, working_dir)
#Results.add_model(output)

config = configSetUp(config_yaml=f"{working_dir}/config_autoregressive_padding.yaml")
output = test_model(config, "autoregressive_nsteps_3_padding", fig_dir, working_dir)
Results.add_model(output)

#config = configSetUp(config_yaml=f"{working_dir}/config_autoregressive_sfno.yaml")
#output = test_model(config, "autoregressive_nsteps_3_sfno", fig_dir, working_dir)
#Results.add_model(output)

Results.create_var_plots(config)
