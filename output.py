import numpy as np
import matplotlib.pyplot as plt
from utils import configSetUp
from scipy.stats import linregress
import gc
import yaml
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import re

class OutputData:
    def __init__(self, model_label, predictions, ground_truth,
                 times, lattitudes, nlat, longitudes, nlon,
                 output_targets, working_dir, fig_dir):
        self.model_label = model_label
        self.times = times
        self.lattitudes = lattitudes
        self.nlat = nlat
        self.longitudes = longitudes
        self.nlon = nlon
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.output_targets = output_targets
        self.fig_dir = fig_dir
        self.working_dir = working_dir
        self.add_variable_metadata()

    def add_variable_metadata(self):
        metadata_path = f"{self.working_dir}/metadata.yaml"

        with open(metadata_path) as f:
            self.metadata = yaml.safe_load(f)

    def get_variable_units(self, variable):
        variable = re.sub(r'_\d+$', '', variable)
        for item in self.metadata:
            if variable in item["variable"]:
                return item["units"]

        raise Exception(f"Unable to get the correct units for {variable}")

    def find_target_index(self, target):
        for i, output_var in enumerate(self.output_targets):
            if target == output_var:
                return i
        raise RuntimeError(f"Unable to find {target} in {self.output_targets}")

    def plot_spatial_maps(self, target, label):
        units = self.get_variable_units(target)

        target_index = self.find_target_index(target)

        truth_start = self.ground_truth[0].detach().numpy()[:, target_index, :, :].squeeze()
        pred_start = self.predictions[0].detach().numpy()[:, target_index, :, :].squeeze()

        truth_end = self.ground_truth[-1].detach().numpy()[:, target_index, :, :].squeeze()
        pred_end = self.predictions[-1].detach().numpy()[:, target_index, :, :].squeeze()

        diff_start = pred_start - truth_start
        diff_end = pred_end - truth_end

        lat = np.unique(self.lattitudes)
        lon = np.unique(self.longitudes)

        lons, lats = np.meshgrid(lon, lat)

        fig, axes = plt.subplots(
            nrows=2, ncols=3, figsize=(18, 10), layout='constrained',
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
        fig.suptitle(f'Spatial Snapshot: {target} \n{self.model_label}', fontsize=16)

        global_vmin = min(truth_start.min(), pred_start.min(), truth_end.min(), pred_end.min())
        global_vmax = max(truth_start.max(), pred_start.max(), truth_end.max(), pred_end.max())
        global_max_diff = max(np.max(np.abs(diff_start)), np.max(np.abs(diff_end)))

        im0 = axes[0, 0].pcolormesh(lons, lats, truth_start, cmap='viridis', vmin=global_vmin, vmax=global_vmax, shading='auto', transform=ccrs.PlateCarree())
        axes[0, 0].set_title(f'Ground Truth (t={self.times[0]})')

        im1 = axes[0, 1].pcolormesh(lons, lats, pred_start, cmap='viridis', vmin=global_vmin, vmax=global_vmax, shading='auto', transform=ccrs.PlateCarree())
        axes[0, 1].set_title(f'Prediction (t={self.times[0]})')

        cbar1 = fig.colorbar(im1, ax=axes[0, :2], orientation='vertical', fraction=0.02, pad=0.04)
        cbar1.set_label(f"[{units}]")

        im2 = axes[0, 2].pcolormesh(lons, lats, diff_start, cmap='RdBu_r', vmin=-global_max_diff, vmax=global_max_diff, shading='auto', transform=ccrs.PlateCarree())
        axes[0, 2].set_title('Difference (Pred - Truth)')

        cbar2 = fig.colorbar(im2, ax=axes[0, 2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.set_label(f"Diff [{units}]")

        im3 = axes[1, 0].pcolormesh(lons, lats, truth_end, cmap='viridis', vmin=global_vmin, vmax=global_vmax, shading='auto', transform=ccrs.PlateCarree())
        axes[1, 0].set_title(f'Ground Truth (t={self.times[-1]})')

        im4 = axes[1, 1].pcolormesh(lons, lats, pred_end, cmap='viridis', vmin=global_vmin, vmax=global_vmax, shading='auto', transform=ccrs.PlateCarree())
        axes[1, 1].set_title(f'Prediction (t={self.times[-1]})')

        cbar3 = fig.colorbar(im4, ax=axes[1, :2], orientation='vertical', fraction=0.02, pad=0.04)
        cbar3.set_label(f"[{units}]")

        im5 = axes[1, 2].pcolormesh(lons, lats, diff_end, cmap='RdBu_r', vmin=-global_max_diff, vmax=global_max_diff, shading='auto', transform=ccrs.PlateCarree())
        axes[1, 2].set_title('Difference (Pred - Truth)')

        cbar4 = fig.colorbar(im5, ax=axes[1, 2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar4.set_label(f"Diff [{units}]")

        for ax in axes.flatten():
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False

        plt.savefig(f"{self.fig_dir}/spatial_snapshot.{target}.{label}.png", dpi=300, bbox_inches='tight')
        plt.close()

class ModelResults:
    def __init__(self, fig_dir):
        self.models = []
        self.fig_dir = fig_dir

    def add_model(self, model:OutputData):
        print(f"Adding model {model.model_label}")
        self.models.append(model)

    def create_var_plots(self, config):
        variables = config.outputs
        RMSE_data = []
        for var in variables:
            RMSE, model_names = self.plot_RMSE(var)
            var_data = {"var_name": var, "RMSE": RMSE, "models":model_names}
            RMSE_data.append(var_data)
            self.plot_temporal_evolution(var, config)
            self.plot_scatter_pred_vs_actual(var, config)

        self.plot_RMSE_scorecard(RMSE_data)

    def plot_RMSE_scorecard(self, RMSE_data):
        out_data, model_names = self.construct_RMSE_scorecard(RMSE_data)
        parsed_data = {}
        variables = []
        regions = []

        for item in out_data:
            for key, winner_idx in item.items():
                var_name, region = key.rsplit('-', 1)

                if var_name not in variables:
                    variables.append(var_name)
                if region not in regions:
                    regions.append(region)

                parsed_data[(var_name, region)] = winner_idx

        grid = np.zeros((len(variables), len(regions)))

        for i, var in enumerate(variables):
            for j, reg in enumerate(regions):
                grid[i, j] = parsed_data.get((var, reg), -1)

        fig_height = max(8, len(variables) * 0.35)
        fig, ax = plt.subplots(figsize=(12, fig_height), layout='constrained')

        num_models = len(model_names)
        cmap = plt.get_cmap('Pastel1', num_models)

        cax = ax.imshow(grid, cmap=cmap, aspect='auto')

        ax.set_xticks(np.arange(len(regions)))
        ax.set_yticks(np.arange(len(variables)))
        ax.set_xticklabels(regions, fontsize=11, fontweight='bold')
        ax.set_yticklabels(variables, fontsize=10)
        ax.xaxis.set_ticks_position('top')

        win_counts = {name: 0 for name in model_names}
        for i in range(len(variables)):
            for j in range(len(regions)):
                winner_idx = parsed_data.get((variables[i], regions[j]))
                if winner_idx is not None and winner_idx != -1:
                    winner_name = model_names[int(winner_idx)]

                    win_counts[winner_name] += 1
                    ax.text(j, i, winner_name,
                            ha="center", va="center", color="black", fontsize=9)

        ax.set_title("Winning Model (Lowest RMSE)", pad=30, fontsize=16, fontweight='bold')

        filename = f"{self.fig_dir}/rmse_scorecard.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved scorecard to {filename}")

        if win_counts:
            overall_winner = max(win_counts, key=win_counts.get)
            max_wins = win_counts[overall_winner]

            # You can print this out, or add it to your plot as a title/caption!
            print(f"Overall Winner: {overall_winner} with {max_wins} wins.")
            print(f"Full Leaderboard: {win_counts}")

    def construct_RMSE_scorecard(self, RMSE_data):
        out_data = []
        for var_data in RMSE_data:
            var_rmse = var_data['RMSE']
            model_names = var_data['models']
            for region, values in var_rmse.items():
                values_arrays = np.array(values)
                winning_model = np.argmin(values_arrays)
                out_data.append({f"{var_data['var_name']}-{region}": winning_model})

        return out_data, model_names

    def plot_RMSE(self, target):
        units = self.models[0].get_variable_units(target)
        print(f"Plotting the area weighted RMSE for {target} with units: {units}")
        regions = {
            'Global': (-90, 90),
            'Tropics': (-20, 20),
            'NH': (0, 90),
            'SH': (-90, 0),
            'NH_mid': (30, 60)
        }

        model_names = []
        regional_rmses = {region: [] for region in regions.keys()}
        var_out = []
        for model in self.models:
            model_names.append(model.model_label)
            target_index = model.find_target_index(target)

            preds = np.array([p.detach().cpu().numpy()[:, target_index, :, :].squeeze() for p in model.predictions])
            truths = np.array([t.detach().cpu().numpy()[:, target_index, :, :].squeeze() for t in model.ground_truth])
            lats = np.array(model.lattitudes)
            weights = np.cos(np.radians(lats))

            preds_flat = preds.reshape(preds.shape[0], -1)
            truths_flat = truths.reshape(truths.shape[0], -1)

            for region_name, (lat_min, lat_max) in regions.items():
                lat_mask = (lats >= lat_min) & (lats <= lat_max)

                region_preds = preds_flat[:, lat_mask]
                region_truths = truths_flat[:, lat_mask]
                region_weights = weights[lat_mask]
                sq_errors = (region_preds - region_truths) ** 2
                weighted_sq_errors = sq_errors * region_weights

                total_weight = np.sum(region_weights) * region_preds.shape[0]
                weighted_mse = np.sum(weighted_sq_errors) / total_weight
                rmse = np.sqrt(weighted_mse)

                regional_rmses[region_name].append(rmse)

        x = np.arange(len(regions))
        num_models = len(self.models)
        width = 0.8 / num_models

        fig, ax = plt.subplots(figsize=(14, 7), layout='constrained')

        # Loop to create grouped side-by-side bars
        for i, model_name in enumerate(model_names):
            rmses_for_this_model = [regional_rmses[reg][i] for reg in regions.keys()]
            offset = (i - num_models / 2 + 0.5) * width
            rects = ax.bar(x + offset, rmses_for_this_model, width, label=model_name)
            ax.bar_label(rects, padding=3, fmt='%.3f', fontweight='bold')

        ax.set_ylabel('RMSE')
        ax.set_title(f'Area-weighted RMSE\n{target} [{units}]', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(regions.keys())
        ax.legend(loc='upper left')

        ax.grid(axis='y', alpha=0.4)

        filename = f"{self.fig_dir}/regional_rmse.{target}.png"
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        return regional_rmses, model_names

    def plot_temporal_evolution(self, target, config:configSetUp):

        units = self.models[0].get_variable_units(target)
        print(f"Plotting the temporal evolution for {target} with units of {units}")

        num_models = len(self.models)
        fig, axes = plt.subplots(nrows=num_models, ncols=1, figsize=(16, 4 * num_models),
                                 sharex=True, layout='constrained')
        if num_models == 1:
            axes = [axes]

        ymin = []
        ymax = []
        for i, model in enumerate(self.models):
            ax = axes[i]

            target_index = model.find_target_index(target)
            preds = np.array([p.detach().cpu().numpy()[:, target_index, :, :].squeeze() for p in model.predictions])
            truths = np.array([t.detach().cpu().numpy()[:, target_index, :, :].squeeze() for t in model.ground_truth])

            times = np.array(model.times)
            time_indices = np.arange(len(times))

            lats = np.array(model.lattitudes)
            weights = np.cos(np.radians(lats))

            preds_flat = preds.reshape(preds.shape[0], -1)
            truths_flat = truths.reshape(truths.shape[0], -1)

            pred_time_series_mean = np.sum(preds_flat * weights, axis=1) / np.sum(weights)
            truth_time_series_mean = np.sum(truths_flat * weights, axis=1) / np.sum(weights)
            sq_errors = (preds_flat - truths_flat) ** 2

            weighted_mse_per_timestep = np.sum(sq_errors * weights, axis=1) / np.sum(weights)
            rmse_time_series = np.sqrt(weighted_mse_per_timestep)
            lower_bound = pred_time_series_mean - rmse_time_series
            upper_bound = pred_time_series_mean + rmse_time_series

            ymin.append(np.min(lower_bound))
            ymax.append(np.max(upper_bound))

            slope, intercept, _, _, _ = linregress(time_indices, pred_time_series_mean)

            ax.plot(times, truth_time_series_mean, color='red', linestyle='--',
                    label=f'Avg {target} (Actual)')

            ax.plot(times, pred_time_series_mean, color='black', linewidth=1.5,
                    label=f'Avg {target} (Predicted)')

            ax.fill_between(times, lower_bound, upper_bound, color='blue', alpha=0.2,
                            edgecolor='none', label='$\pm$ RMSE')

            ax.set_title(f'{target}: Global RMSE for {model.model_label} Model',
                         fontsize=12, fontweight='medium')

            ax.set_ylabel(f'{target} ({units})')
            ax.grid(axis='both', alpha=0.3)

            textstr = f'Slope = {slope:.3e}'
            props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='silver')
            ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', bbox=props)

            if i == num_models - 1:
                ax.legend(loc='upper left', frameon=True, fontsize=10,
                          fancybox=True, shadow=False)
                ax.set_xlabel('Time')

            del preds, truths, preds_flat, truths_flat, sq_errors
            gc.collect()
        for ax in axes:
            ax.set_ylim(min(ymin), max(ymax))

        filename = f"{self.fig_dir}/temporal_evolution.{target}.png"
        plt.savefig(filename, dpi=300)
        plt.close(fig)

    def plot_scatter_pred_vs_actual(self, target, config, sample_size=3000):
        units = self.models[0].get_variable_units(target)
        print(f"Plotting scatter plots for {target} with units of {units}")

        num_models = len(self.models)

        fig, axes = plt.subplots(nrows=num_models, ncols=1, figsize=(12, 4 * num_models),
                                 sharex=True, layout='constrained')

        if num_models == 1:
            axes = [axes]

        lower_bounds = []
        upper_bounds = []
        for i, model in enumerate(self.models):
            ax = axes[i]

            target_index = model.find_target_index(target)
            preds = np.array([p.detach().cpu().numpy()[:, target_index, :, :].squeeze() for p in model.predictions])
            truths = np.array([t.detach().cpu().numpy()[:, target_index, :, :].squeeze() for t in model.ground_truth])

            preds_flat = preds.flatten()
            truths_flat = truths.flatten()

            if len(preds_flat) > sample_size:
                rng = np.random.default_rng(42)
                indices = rng.choice(len(preds_flat), size=sample_size, replace=False)
                preds_sub = preds_flat[indices]
                truths_sub = truths_flat[indices]
            else:
                preds_sub = preds_flat
                truths_sub = truths_flat

            correlation_matrix = np.corrcoef(truths_sub, preds_sub)
            r_squared = correlation_matrix[0, 1] ** 2

            lower_bounds.append(np.min([truths_sub, preds_sub]))
            upper_bounds.append(np.max([truths_sub, preds_sub]))

            ax.scatter(truths_sub, preds_sub, c='tab:blue', alpha=0.9, edgecolors='none')

            min_val = min(np.nanmin(truths_sub), np.nanmin(preds_sub))
            max_val = max(np.nanmax(truths_sub), np.nanmax(preds_sub))
            ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

            ax.set_title(f'Model: {model.model_label} \n{target} - y_pred vs y_actual', fontsize=12)
            ax.set_ylabel(f'Predicted {target} [{units}]')
            ax.grid(alpha=0.3)

            textstr = f'$R^2$ = {r_squared:.2f}'
            props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.03, 0.94, textstr, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', bbox=props)

            if i == num_models - 1:
                ax.set_xlabel(f'Actual {target} ({units})')

            del preds, truths, preds_flat, truths_flat
            gc.collect()

            for ax in axes:
                ax.set_ylim(min(lower_bounds), max(upper_bounds))

        filename = f"{self.fig_dir}/scatter_comparison.{target}.png"
        plt.savefig(filename, dpi=300)
        plt.close(fig)
