import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs

def plot_global_mean(variable_data, variable_name, path, split_data_info):

    os.makedirs(path, exist_ok=True)
    filename = f"{path}/Global_Mean_{variable_name}.png"

    if os.path.isfile(filename):
        return
    
    testing_period = split_data_info['testing']
    training_period = split_data_info['training']
    validating_period = split_data_info['validating']

    latitudes = variable_data.lat
    weights = np.cos(np.deg2rad(latitudes))
    weights.name = "weights"

    var_weighted = variable_data.weighted(weights)
    global_mean_ts = var_weighted.mean(("lat", "lon"))

    plt.figure(figsize=(14, 6))

    train_start, train_end = training_period.split('-')
    val_start, val_end = validating_period.split('-')
    test_start, test_end = testing_period.split('-')

    global_mean_ts.plot(alpha=0.3, color='gray', label='Monthly Mean')
    global_mean_ts.rolling(time=12, center=True).mean().plot(color='red', linewidth=2, label='12-Month Rolling Mean')
    
    DateType = type(global_mean_ts.time.values[0])
    plt.axvspan(DateType(int(train_start), 1, 1), DateType(int(train_end), 12, 31), color='blue', alpha=0.1, label='Training')
    plt.axvspan(DateType(int(val_start), 1, 1), DateType(int(val_end), 12, 31), color='green', alpha=0.1, label='Validating')
    plt.axvspan(DateType(int(test_start), 1, 1), DateType(int(test_end), 12, 31), color='orange', alpha=0.1, label='Testing')

    plt.title(f"90-Year Global Mean ({variable_name})", fontsize=14)
    plt.ylabel(f"{variable_name} ({variable_data.units})")
    plt.xlabel("Year")
    plt.grid(True, alpha=0.3)
    plt.legend()

    ax = plt.gca()
    times = global_mean_ts.time.values
    years = global_mean_ts.time.dt.year.values
    ax.set_xticks(times[::60])
    ax.set_xticklabels(years[::60])

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_animated_map(variable_data, variable_name, path, interval=250):

    os.makedirs(path, exist_ok=True)
    filename = f"{path}/Animated_Map_{variable_name}.gif"

    if os.path.isfile(filename):
        return
    
    plt.ioff()  # Turn off interactive mode
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.coastlines()
    ax.gridlines(draw_labels=False, linestyle='--')
    
    # Initial plot
    mesh = variable_data.isel(time=0).plot(
        ax=ax, 
        transform=ccrs.PlateCarree(), 
        x='lon', y='lat', 
        cmap='RdYlBu_r', 
        add_colorbar=True,
        cbar_kwargs={'label': f'{variable_name} ({variable_data.units})', 'shrink': 0.8}
    )
    
    title = ax.set_title(f"{variable_name} - {str(variable_data.time.values[0])[:10]}")
    
    def update(i):
        mesh.set_array(variable_data.isel(time=i).values.flatten())
        title.set_text(f"{variable_name} - {str(variable_data.time.values[i])[:10]}")
        return mesh, title

    frames = range(0, len(variable_data.time), 12)
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    ani.save(filename, writer='imagemagick', fps=5)
    plt.close(fig)