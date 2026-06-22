from pathlib import Path
from inference import load_output
from output import OutputData, ModelResults

def create_plots(candidates, Results, label):
    for candidate in candidates:
        print(f"Loading the data for {candidate}")
        out_path = output_dir / candidate
        out = load_output(out_path)
        Results.add_model(out)

    var = "air_temperature_at_two_meters"
    Results.plot_spatial_performance_comparison(var, f"{label}_spatial_comparison")
    Results.plot_time_series_comparison(var, f"{label}_time_series_comparison")

    var = "surface_temperature"
    Results.plot_spatial_performance_comparison(var, f"{label}_spatial_comparison")
    Results.plot_time_series_comparison(var, f"{label}_time_series_comparison")

    var = "PRESsfc"
    Results.plot_spatial_performance_comparison(var, f"{label}_spatial_comparison")
    Results.plot_time_series_comparison(var, f"{label}_time_series_comparison")

# Directory where the saved output is located
output_dir = Path("/scratch4/GFDL/gfdlscr/Uriel.Ramirez/SPEAR_TRAINING_JOBS/architecture_comparison/output/results")

# Directory to store the output figures
fig_dir = Path("/scratch4/GFDL/gfdlscr/Uriel.Ramirez/SPEAR_TRAINING_JOBS/architecture_comparison/figs_presentation")
fig_dir.mkdir(parents=True, exist_ok=True)
Results = ModelResults(fig_dir)

# Candidates to compare
candidates = ["unet-candidate-1_out.npz", 
              "unet-candidate-2_out.npz"]
Results = ModelResults(fig_dir)
create_plots(candidates, Results, label="unet")

candidates = ["cnn-candidate-1_out.npz",
              "cnn-candidate-2_out.npz"]
Results = ModelResults(fig_dir)
create_plots(candidates, Results, label="cnn")

candidates = ["sfno-candidate-1_out.npz",
              "sfno-candidate-2_out.npz"]
Results = ModelResults(fig_dir)
create_plots(candidates, Results, label="sfno")
