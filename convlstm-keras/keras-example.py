
#%%
from pathlib import Path
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import xarray as xr

#matplotlib.use("WebAgg") 
print("TensorFlow version:", tf.__version__)
print("TensorFlow keras version:", tf.keras.__version__)

data_dir = "/home/Mikyung.Lee/spear-emulator-data"
t_ref = Path(data_dir)/"atmos.192101-201012.t_ref.nc"

#load and normalize
with xr.open_dataset(t_ref, decode_timedelta=True) as ds:
    t_ref = ds["t_ref"].values    
    ntimes = t_ref.shape[0]

normalizer = tf.keras.layers.Normalization(axis=0)
normalizer.adapt(t_ref)
mean = normalizer.mean.numpy()
std = normalizer.variance.numpy() ** 0.5

#normalize and add features layer, so far one feature
normalized_tref = normalizer(t_ref)
normalized_tref = tf.expand_dims(normalized_tref, axis=-1)

sequence_length = 3
batch_size = 1

series_ds = tf.keras.utils.timeseries_dataset_from_array(
    data=normalized_tref[:-1],
    targets=normalized_tref[sequence_length:],
    sequence_length=sequence_length,
    sequence_stride=1,
    shuffle=False,
    batch_size=batch_size,
)

input_shape = next(iter(series_ds))[0].shape[1:]

train = True
load = True

# Model and history artifacts.
artifact_dir = Path(__file__).resolve().parent / "artifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)
model_path = artifact_dir / "convlstm_tref_1.keras"
history_path = artifact_dir / "convlstm_tref_history_1.json"

if load:
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from: {model_path}")
    history_dict = json.loads(history_path.read_text()) if history_path.exists() else {}
    if history_path.exists():
        print(f"Loaded training history from: {history_path}")
else:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.ConvLSTM2D(filters=8, kernel_size=(3, 3), padding="same", activation="tanh"),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    history_dict = {"loss": [], "mae": []}
    print("Using a new model instance")

if train:
    model.summary()
    history = model.fit(
        series_ds,
        epochs=100,
        verbose=2,
    )
    for key, values in history.history.items():
        history_dict.setdefault(key, [])
        history_dict[key].extend(values)

    model.save(model_path)
    history_path.write_text(json.dumps(history_dict, indent=2))
    print(f"Saved trained model to: {model_path}")
    print(f"Saved training history to: {history_path}")

val_loss, val_mae = model.evaluate(series_ds, verbose=0)
y_pred = model.predict(series_ds)
print(val_loss, val_mae)

# Plot normalized_tref and y_pred averages on the same figure.
normalized_tref_time_mean = np.mean(normalized_tref.numpy(), axis=(1, 2, 3))
t_ref_time_index = np.arange(ntimes)

# y_pred has shape (batch, lat, lon, channel), so average over spatial/channel dims.
y_pred_time_mean = np.mean(y_pred, axis=(1, 2, 3))
y_pred_time_index = np.arange(sequence_length, sequence_length + y_pred_time_mean.shape[0])

plt.figure(figsize=(10, 4))
plt.plot(t_ref_time_index, normalized_tref_time_mean, label="answers")
plt.plot(y_pred_time_index, y_pred_time_mean, label="y_pred mean")
plt.legend()
plt.tight_layout()
plt.show()


# %%
