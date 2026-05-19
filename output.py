import numpy as np
import matplotlib.pyplot as plt
from utils import configSetUp
from scipy.stats import linregress
import gc

class OutputData:
    def __init__(self, model_label, predictions, ground_truth,
                 times, lattitudes, nlat, longitudes, nlon,
                 output_targets, fig_dir):
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

    def find_target_index(self, target):
        for i, output_var in enumerate(self.output_targets):
            if target == output_var:
                return i
        raise RuntimeError(f"Unable to find {target} in {self.output_targets}")

    def plot_spatial_maps(self, target, label):
        target_index = self.find_target_index(target)

        truth_start = self.ground_truth[0].detach().numpy()[:, target_index, :, :].squeeze()
        pred_start = self.predictions[0].detach().numpy()[:, target_index, :, :].squeeze()

        print(f"SHAPES {truth_start.shape} - {pred_start.shape}")

        truth_end = self.ground_truth[-1].detach().numpy()[:, target_index, :, :].squeeze()
        pred_end = self.predictions[-1].detach().numpy()[:, target_index, :, :].squeeze()

        diff_start = pred_start - truth_start
        diff_end = pred_end - truth_end

        lat = np.unique(self.lattitudes)
        lon = np.unique(self.longitudes)

        lons, lats = np.meshgrid(lon, lat)

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), layout='constrained')
        fig.suptitle(f'Spatial Snapshot: {self.model_label}', fontsize=16)

        global_vmin = min(truth_start.min(), pred_start.min(), truth_end.min(), pred_end.min())
        global_vmax = max(truth_start.max(), pred_start.max(), truth_end.max(), pred_end.max())
        global_max_diff = max(np.max(np.abs(diff_start)), np.max(np.abs(diff_end)))

        im0 = axes[0, 0].pcolormesh(lons, lats, truth_start, cmap='viridis', vmin=global_vmin, vmax=global_vmax, shading='auto')
        axes[0, 0].set_title(f'Ground Truth (t={self.times[0]})')

        im1 = axes[0, 1].pcolormesh(lons, lats, pred_start, cmap='viridis', vmin=global_vmin, vmax=global_vmax, shading='auto')
        axes[0, 1].set_title(f'Prediction (t={self.times[0]})')
        fig.colorbar(im1, ax=axes[0, :2], orientation='vertical', fraction=0.02, pad=0.04)

        im2 = axes[0, 2].pcolormesh(lons, lats, diff_start, cmap='RdBu_r', vmin=-global_max_diff, vmax=global_max_diff, shading='auto')
        axes[0, 2].set_title('Difference (Pred - Truth)')
        fig.colorbar(im2, ax=axes[0, 2], orientation='vertical', fraction=0.046, pad=0.04)

        im3 = axes[1, 0].pcolormesh(lons, lats, truth_end, cmap='viridis', vmin=global_vmin, vmax=global_vmax, shading='auto')
        axes[1, 0].set_title(f'Ground Truth (t={self.times[-1]})')

        im4 = axes[1, 1].pcolormesh(lons, lats, pred_end, cmap='viridis', vmin=global_vmin, vmax=global_vmax, shading='auto')
        axes[1, 1].set_title(f'Prediction (t={self.times[-1]})')
        fig.colorbar(im4, ax=axes[1, :2], orientation='vertical', fraction=0.02, pad=0.04)

        im5 = axes[1, 2].pcolormesh(lons, lats, diff_end, cmap='RdBu_r', vmin=-global_max_diff, vmax=global_max_diff, shading='auto')
        axes[1, 2].set_title('Difference (Pred - Truth)')
        fig.colorbar(im5, ax=axes[1, 2], orientation='vertical', fraction=0.046, pad=0.04)

        for ax in axes.flatten():
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

        plt.savefig(f"{self.fig_dir}/spatial_snapshot.{target}.{label}.png", dpi=300, bbox_inches='tight')

class ModelResults:
    def __init__(self, fig_dir):
        self.models = []
        self.fig_dir = fig_dir

    def add_model(self, model:OutputData):
        print(f"Adding model {model.model_label}")
        self.models.append(model)

    def plot_RMSE(self, target):
        regions = {
            'Global': (-90, 90),
            'Tropics': (-20, 20),
            'NH': (0, 90),
            'SH': (-90, 0),
            'NH_mid': (30, 60)
        }

        model_names = []
        regional_rmses = {region: [] for region in regions.keys()}
        for model in self.models:
            print(f"Working on {model.model_label}")
            print(len(model.predictions))

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
            for i, model_name in enumerate(model_names):
                rmses_for_this_model = [regional_rmses[reg][i] for reg in regions.keys()]
                offset = (i - num_models / 2 + 0.5) * width
                rects = ax.bar(x + offset, rmses_for_this_model, width, label=model_name)
                ax.bar_label(rects, padding=3, fmt='%.3f', fontweight='bold')

        ax.set_ylabel('RMSE')
        ax.set_title(f'Regional RMSE — {target}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(regions.keys())
        ax.legend(loc='upper left')

        ax.grid(axis='y', alpha=0.4)

        filename = f"{self.fig_dir}/regional_rmse.{target}.png"
        plt.savefig(filename, dpi=300)
        plt.close(fig)

    def plot_temporal_evolution(self, target, config:configSetUp):

        units = config.get_var_units(target)
        print(f"Plotting the temporal evolution for {target} with units of {units}")

        num_models = len(self.models)
        fig, axes = plt.subplots(nrows=num_models, ncols=1, figsize=(16, 4 * num_models),
                                 sharex=True, layout='constrained')
        if num_models == 1:
            axes = [axes]

        all_y_values = []
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

            all_y_values.extend([np.min(lower_bound), np.max(upper_bound),
                                 np.min(truth_time_series_mean), np.max(truth_time_series_mean)])

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

        filename = f"{self.fig_dir}/temporal_evolution.{target}.png"
        plt.savefig(filename, dpi=300)
        plt.close(fig)

    def plot_scatter_pred_vs_actual(self, target, config, sample_size=3000):
        units = config.get_var_units(target)
        print(f"Plotting scatter plots for {target} with units of {units}")

        num_models = len(self.models)

        fig, axes = plt.subplots(nrows=num_models, ncols=1, figsize=(12, 4 * num_models),
                                 sharex=True, layout='constrained')

        if num_models == 1:
            axes = [axes]

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

            ax.scatter(truths_sub, preds_sub, c='tab:blue', alpha=0.9, edgecolors='none')

            min_val = min(np.nanmin(truths_sub), np.nanmin(preds_sub))
            max_val = max(np.nanmax(truths_sub), np.nanmax(preds_sub))
            ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

            ax.set_title(f'{target}: y_pred vs y_actual for {model.model_label} Model', fontsize=12)
            ax.set_ylabel(f'Predicted {target} ({units})')
            ax.grid(alpha=0.3)

            textstr = f'$R^2$ = {r_squared:.2f}'
            props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.03, 0.94, textstr, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', bbox=props)

            if i == num_models - 1:
                ax.set_xlabel(f'Actual {target} ({units})')

            del preds, truths, preds_flat, truths_flat
            gc.collect()

        filename = f"{self.fig_dir}/scatter_comparison.{target}.png"
        plt.savefig(filename, dpi=300)
        plt.close(fig)
