from model import SpearEmulator, AutoregressiveSpearEmulator
from data_load import get_dataloaders, get_updated_channels
from utils import configSetUp
import torch
from output import OutputData, ModelResults
from pathlib import Path
from plotting_utils import plot_loss

def get_variable_mappings(mappings, variable):
    for mapping in mappings:
        if mapping["variable_name"] == variable:
            labels = mapping["labels"]
            for label in labels:
                if label['label'] == "t-1":
                    return label['index']
    raise RuntimeError(f"Unable to determine the index for {variable} in {mappings}")


def do_forecast_rollout(label, model, testing, config, device, fig_dir):
    model.eval()
    
    print(f"\n ------------------------------- \n🏃 Starting rollout on {device}...")

    input_names = config.input_channels
    target_names = config.outputs
    dynamics_variables = config.dynamics

    nlags = config.get_nlags()
    method = config.get_data_load_method()

    print(f"Input channels {input_names}")
    print(f"Targets {target_names}")
    print(f"Dynamic variables {dynamics_variables}")

    y_means = model.y_means
    y_stds = model.y_stds
    print(
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
            print(f"x batch shape {x_batch.shape} - y batch shape {y_batch.shape} - {target_indices_in_y}")

            if method == "autoregressive":
                y_step = y_batch[:, 0, target_indices_in_y, ...]
                print(f"y_step: {y_step[:,:,:,1]}")
            else:
                y_step = y_batch

            print(f"y step shape {y_step.shape}")

            # Reshape to the format expected by the model
            x = x_batch.to(device).squeeze(2).view(-1, len(input_names), config.nlat, config.nlon)
            y = y_step.to(device).squeeze(2).view(-1, len(target_names), config.nlat, config.nlon)
            print(f"x shape {x.shape} - y shape {y.shape}")

            print(f"Working on step {step}")
            if step > 0:
                x_next = x.clone()

                if method == "autoregressive":
                    past_pred = physical_preds_buffer[-1]

                    for i, target_name in enumerate(target_names):
                        base_name = target_name.split('(')[0]
                        idx_in_x = next(j for j, name in enumerate(input_names) if name.startswith(base_name))
                        print(f"--- x_next[:, {idx_in_x}, :, :] = past_pred[:, {i}, :, :]")
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

    start_idx = config.get_nlags()
    end_idx = start_idx + len(predictions)
    rollout_dates = testing.times[start_idx : end_idx]
    print(rollout_dates)

    return OutputData(label, predictions, ground_truth,
            rollout_dates, testing.lat, config.nlat, testing.lon, config.nlon,
            config.outputs, fig_dir)

def test_model(config, label, fig_dir):
    training, validating, testing  = get_dataloaders(config)

    input_channels, out_channels = get_updated_channels(config)
    config.set_channels(input_channels, out_channels)
    config.set_grid(training)

    checkpoint_path = f"logs/spear_emulator/{label}/checkpoints/epoch=49-step=18300.ckpt"

    # Set up the correct class
    method = config.get_data_load_method()
    if method == "autoregressive":
        ModelClass = AutoregressiveSpearEmulator
    else:
        ModelClass = SpearEmulator

    model = ModelClass.load_from_checkpoint(checkpoint_path, config=config)

    # Plot the training/validation losses
    log_file = f"logs/spear_emulator/{label}/metrics.csv"
    output_file = f"{fig_dir}/losses.{label}.png"
#    plot_loss(log_file, output_file)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 GPU detected! Running on:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("🐢 No GPU available. Falling back to CPU for inference.")

    # Test with rollout
    model.to(device)
    output = do_forecast_rollout(label, model, testing, config, device, fig_dir)

    # Plot spatial plots
    var = "air_temperature_at_two_meters"
    output.plot_spatial_maps(var, label=f"{label}")

    return output

##################################3
fig_dir = Path("figs_phase2a")
fig_dir.mkdir(parents=True, exist_ok=True)

Results = ModelResults(fig_dir)

#config = configSetUp(config_yaml="examples/config_default.yaml")
#output = test_model(config, "nlag_3", fig_dir)
#Results.add_model(output)
#
#config = configSetUp(config_yaml="examples/config_residual.yaml")
#output = test_model(config, "residual_nlag_3", fig_dir)
#Results.add_model(output)
#
config = configSetUp(config_yaml="examples/config_autoregressive.yaml")
output = test_model(config, "autoregressive_nsteps_3_upsample", fig_dir)
Results.add_model(output)

var = "air_temperature_at_two_meters"
Results.plot_RMSE(var)
Results.plot_temporal_evolution(var, config)
Results.plot_scatter_pred_vs_actual(var, config)
