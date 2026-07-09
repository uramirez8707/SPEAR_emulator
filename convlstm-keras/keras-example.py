
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

# Split into train/val/test: 60% / 20% / 20%.
train_end = int(0.6 * ntimes)
val_end = int(0.8 * ntimes)

t_ref_train = t_ref[:train_end]
t_ref_val = t_ref[train_end:val_end]
t_ref_test = t_ref[val_end:]

splits = {
    "train": t_ref_train,
    "val": t_ref_val,
    "test": t_ref_test,
}

normalizers, normalized_tref = {}, {}
for split_name, split_values in splits.items():
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(split_values)
    normalizers[split_name] = normalizer
    normalized_tref[split_name] = tf.expand_dims(normalizer(split_values), axis=-1)


sequence_length = 3
batch_size = 1

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    data=normalized_tref["train"][:-1],
    targets=normalized_tref["train"][sequence_length:],
    sequence_length=sequence_length,
    sequence_stride=1,
    shuffle=False,
    batch_size=batch_size,
)

val_ds = tf.keras.utils.timeseries_dataset_from_array(
    data=normalized_tref["val"][:-1],
    targets=normalized_tref["val"][sequence_length:],
    sequence_length=sequence_length,
    sequence_stride=1,
    shuffle=False,
    batch_size=batch_size,
)

input_shape = next(iter(train_ds))[0].shape[1:]

train = True
load = True
learning_rate = 1e-3

# Model and history artifacts.
artifact_dir = Path(__file__).resolve().parent / "artifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)
model_path = artifact_dir / "convlstm_tref.keras"
history_path = artifact_dir / "convlstm_tref_history.json"

history_dict = {"loss": [], "mae": []}
if history_path.exists():
    loaded = json.loads(history_path.read_text())
    if isinstance(loaded, dict):
        history_dict.update({k: v if isinstance(v, list) else [v] for k, v in loaded.items()})

if load:
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    print(f"Loaded model from: {model_path}")
    print(f"Recompiled loaded model with learning rate: {learning_rate}")
    if history_path.exists():
        print(f"Loaded training history from: {history_path}")
else:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.ConvLSTM2D(filters=8, kernel_size=(3, 3), padding="same", activation="tanh"),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    print("Using a new model instance")

if train:
    model.summary()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5000,
        verbose=2,
    )
    for key, values in history.history.items():
        history_dict.setdefault(key, [])
        history_dict[key].extend(float(v) for v in values)

    model.save(model_path)
    history_path.write_text(json.dumps(history_dict, indent=2))
    print(f"Saved trained model to: {model_path}")
    print(f"Saved training history to: {history_path}")

train_loss, train_mae = model.evaluate(train_ds, verbose=0)
train_pred = model.predict(train_ds)

val_loss, val_mae = model.evaluate(val_ds, verbose=0)
val_pred = model.predict(val_ds)

print("train", train_loss, train_mae)
print("val", val_loss, val_mae)


#test roll out
forecast = normalized_tref["test"][:sequence_length]
for itime in range(val_end+sequence_length, ntimes):
    predicted = model(tf.expand_dims(forecast[-sequence_length:, :, :, :], axis=0))
    forecast = tf.concat([forecast, predicted], axis=0)

test_time_index = np.arange(val_end, ntimes)
test_answer_mean = np.mean(normalized_tref["test"].numpy(), axis=(1, 2, 3))
forecast_mean = np.mean(forecast.numpy(), axis=(1, 2, 3))

# Plot train answers against train predictions on the global timeline.
train_answer_mean = np.mean(normalized_tref["train"].numpy(), axis=(1, 2, 3))
train_time_index = np.arange(0, train_end)
train_pred_mean = np.mean(train_pred, axis=(1, 2, 3))

# Plot validation answers against validation predictions on the global timeline.
val_answer_mean = np.mean(normalized_tref["val"].numpy(), axis=(1, 2, 3))
val_time_index = np.arange(train_end, val_end)
val_pred_mean = np.mean(val_pred, axis=(1, 2, 3))


plt.figure(figsize=(10, 4))
#train
plt.plot(train_time_index, train_answer_mean, label="train answers", color="black")
plt.plot(train_time_index[sequence_length:], train_pred_mean, label="train_pred mean", alpha=0.3)
#val
plt.plot(val_time_index, val_answer_mean, label="val answers", color="black")
plt.plot(val_time_index[sequence_length:], val_pred_mean, label="val_pred mean", alpha=0.3)
#predict
plt.plot(test_time_index, test_answer_mean, label="test answers", color="black")
plt.plot(test_time_index, forecast_mean, label="forecast mean", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#https://www.tensorflow.org/tutorials/structured_data/time_series
# %%
