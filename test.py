from model import SpearEmulator
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


def do_forecast_rollout(label, model, testing, config, device):
    model.eval()
    
    print(f"\n ------------------------------- \n🏃 Starting rollout on {device}...")

    input_names = config.input_channels
    target_names = config.outputs
    nlags = config.get_nlags()

    print(f"Input channels {input_names}")
    print(f"Targets {target_names}")

    y_means = model.y_means
    y_stds = model.y_stds
    print(
        "Mean and stds used for normalizing\n"
        f"Means:\n{y_means.cpu().numpy()}\n\n"
        f"Stds:\n{y_stds.cpu().numpy()}"
    )
    mappings = config.get_mappings()

    predictions = []
    ground_truth = []
    physical_preds_buffer = []

    with torch.no_grad():
        for step, (x_batch, y_batch) in enumerate(testing.tensor):

            # Reshape to the format expected by the model
            x = x_batch.to(device).squeeze(2).view(-1, len(config.input_channels), config.nlat, config.nlon)
            y = y_batch.to(device).squeeze(2).view(-1, len(config.output_channels), config.nlat, config.nlon)

            print(f"Working on step {step}")
            if step > 0:
                x_next = x.clone()

                for target_idx, target_name in enumerate(target_names):
                    target_lag_indx = get_variable_mappings(mappings, target_name)
                    for past_pred_phys in reversed(physical_preds_buffer):
                        print(f"Replacing x_next[:, {target_lag_indx}, :, :] with previous prediction")
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

            if step > 10:
                break

    start_idx = config.get_nlags()
    end_idx = start_idx + len(predictions)
    rollout_dates = testing.times[start_idx : end_idx]
    print(rollout_dates)

    return OutputData(label, predictions, ground_truth,
            rollout_dates, testing.lat, config.nlat, testing.lon, config.nlon,
            config.outputs)


config = configSetUp(config_yaml="examples/config.yaml")
config.dump_info()

training, validating, testing  = get_dataloaders(config)

input_channels, out_channels = get_updated_channels(config)
config.set_channels(input_channels, out_channels)
config.set_grid(training)

version = "version_24"
checkpoint_path = f"logs/spear_emulator/{version}/checkpoints/epoch=49-step=18300.ckpt"
model = SpearEmulator.load_from_checkpoint(checkpoint_path, config=config)

fig_dir = Path("figs")
fig_dir.mkdir(parents=True, exist_ok=True)
log_file = f"logs/spear_emulator/{version}/metrics.csv"
output_file = f"figs/losses.{version}.png"

plot_loss(log_file, output_file)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 GPU detected! Running on:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("🐢 No GPU available. Falling back to CPU for inference.")

model.to(device)

Results = ModelResults()
output = do_forecast_rollout(version, model, testing, config, device)

var = "air_temperature_at_two_meters"

output.plot_spatial_maps(var, label=f"{version}")
Results.add_model(output)
Results.plot_RMSE(var)
Results.plot_temporal_evolution(var, config)
Results.plot_scatter_pred_vs_actual(var, config)
