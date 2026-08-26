  
#%%
from pathlib import Path
import json
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import tensorflow as tf
from tensorflow.keras import Model
import xarray as xr

#matplotlib.use("WebAgg") 
print("TensorFlow version:", tf.__version__)
print("TensorFlow keras version:", tf.keras.__version__)

train = True
load = True
learning_rate = 1e-3
epochs = 1000
filters = 32

train_fraction = 0.6
val_fraction = 0.2

sequence_length = 5
batch_size = 1

plot_train = False
plot_val = True
plot_test = True
plot_rollout = True

# Model and history artifacts.
artifact_dir = Path(__file__).resolve().parent / "artifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)
model_path = artifact_dir / f"convlstm_tref_{filters}_seqlength_{sequence_length}.keras"
history_path = artifact_dir / f"convlstm_tref_{filters}_seqlength_{sequence_length}_history.json"

# load data
data_dir = "/home/Mikyung.Lee/spear-emulator-data"
t_ref = Path(data_dir)/"atmos.192101-201012.t_ref.nc"
with xr.open_dataset(t_ref, decode_timedelta=True) as ds:
    t_ref = ds["t_ref"].values
    lat = ds["lat"].values
    lon = ds["lon"].values
    ntimes = t_ref.shape[0]

# compute time
train_start, train_end = 0, int(train_fraction * ntimes)
val_start, val_end = train_end, int((train_fraction + val_fraction) * ntimes)
test_start, test_end = val_end, ntimes

# split data
t_ref_dict = {
    "train": t_ref[train_start:train_end],
    "val": t_ref[val_start:val_end],
    "test": t_ref[test_start:test_end],
}

# normalize data
normalized_tref_dict = {}
for datatype, data in t_ref_dict.items():
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(data)
    normalized_tref_dict[datatype] = tf.expand_dims(normalizer(data), axis=-1)

# dataset
share_kwargs = {"sequence_length": sequence_length, "sequence_stride": 1, "shuffle": False, "batch_size": batch_size}
ds_dict = {
    datatype: tf.keras.utils.timeseries_dataset_from_array(
        data=data[:-1], targets=data[sequence_length:], **share_kwargs,
    )
    for datatype, data in normalized_tref_dict.items()
}

history_dict = {}
if history_path.exists():
    history_dict = json.loads(history_path.read_text()) 

# prepare model
if load:
    if not model_path.exists(): raise FileNotFoundError(f"Saved model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
else:
    input_shape = next(iter(ds_dict["train"]))[0].shape[1:]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.ConvLSTM2D(filters=filters, kernel_size=(3, 3), padding="same", activation="tanh"),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same"),
    ])

# compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
model.summary()
print(f"load model={load}, Learning rate: {learning_rate}")

# train
if train:    
    history = model.fit(
        ds_dict["train"],
        epochs=epochs,
        verbose=2,
    )
    model.save(model_path)
    for key, values in history.history.items():
        history_dict.setdefault(key, [])
        history_dict[key].extend(values)
    history_path.write_text(json.dumps(history_dict, indent=2))
    print(f"Saved trained model to: {model_path}")
    print(f"Saved training history to: {history_path}")

# evaluate model fitting
predicted = {}
for label in ["train", "val", "test"]:
    loss, mae = model.evaluate(ds_dict[label], verbose=0)
    predicted[label] = model.predict(ds_dict[label])
    print(f"{label} loss: {loss}, mae: {mae}")

#test roll out
forecast_tf = normalized_tref_dict["test"][:sequence_length]
for itime in range(test_start+sequence_length, test_end):
    next_pred = model(tf.expand_dims(forecast_tf[-sequence_length:, :, :, :], axis=0))
    forecast_tf = tf.concat([forecast_tf, next_pred], axis=0)
predicted["forecast"] = forecast_tf.numpy()[sequence_length:]

#hack for plotting
normalized_tref_dict["forecast"] = normalized_tref_dict["test"]   

# Plot training loss from the saved history artifact.
if history_path.exists():
    loss = history_from_file = json.loads(history_path.read_text()).get("loss")
    fig, ax = plt.subplots(figsize=(10, 4))    
    ax.plot(loss, label="loss", color="tab:blue")
    ax.set_title("Training Loss")
    ax.legend()

# plotting means
time = np.arange(ntimes)
plot_config = {
    "train": SimpleNamespace(plot=plot_train, time=time[train_start:train_end]),
    "val": SimpleNamespace(plot=plot_val, time=time[val_start:val_end]),
    "test": SimpleNamespace(plot=plot_test, time=time[test_start:test_end]),
    "forecast": SimpleNamespace(plot=plot_rollout, time=time[test_start:test_end]),
}
fig, ax = plt.subplots(figsize=(14, 10))
for label, config in plot_config.items():
    if config.plot:
        answer_mean = np.mean(normalized_tref_dict[label].numpy(), axis=(1, 2, 3))
        pred_mean = np.mean(predicted[label], axis=(1, 2, 3))
        ax.plot(config.time, answer_mean, label=f"{label} answers", color="black")
        ax.plot(config.time[sequence_length:], pred_mean, label=f"{label} mean", alpha=0.5)
ax.legend()

# contour map for the last timestep of t_ref.
timestep = -1
map_stride = 2

fig, axes = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()})

lon2d, lat2d = np.meshgrid(lon, lat)
lon2d_map = lon2d[::map_stride, ::map_stride]
lat2d_map = lat2d[::map_stride, ::map_stride]
forecast_map = predicted["forecast"][timestep, ::map_stride, ::map_stride, 0]
t_ref_map = predicted["test"][timestep+sequence_length, ::map_stride, ::map_stride, 0]

plot_kwargs = {
    "levels": 10, 
    "cmap": "coolwarm", 
    "vmin": min(np.min(t_ref_map), np.min(forecast_map)),
    "vmax": max(np.max(t_ref_map), np.max(forecast_map)), 
    "transform": ccrs.PlateCarree()
}

contour_ref = axes[0].contourf(lon2d_map, lat2d_map, t_ref_map, **plot_kwargs)
contour_fcst = axes[1].contourf(lon2d_map, lat2d_map, forecast_map, **plot_kwargs)

for label, ax in zip(["t_ref", "forecast"], axes):
    ax.coastlines()
    ax.set_title(f"{label} at timestep {timestep}")
    
plt.colorbar(contour_ref, ax=axes[0], shrink=0.8, pad=0.05, label="t_ref")
plt.colorbar(contour_fcst, ax=axes[1], shrink=0.8, pad=0.05, label="forecast")

plt.show()

#https://www.tensorflow.org/tutorials/structured_data/time_series
# %%
