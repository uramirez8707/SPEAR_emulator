import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
import numpy as np
import torch

import data_module
from autolightning import AutoDataModule, TrainModule
from model import SimpleLSTM

# read data
data_dict = {
    "t_ref": "data/atmos.192101-201012.t_ref.nc",
    "swdn_toa": "data/atmos.192101-201012.swdn_toa.nc",
}
data = data_module.load_variable(data_dict)

tref = data["t_ref"]
tref_mean = np.array([datum.mean() for datum in tref], dtype=np.float32)
tref_mean = tref_mean / tref_mean.mean()

swdn_toa = data["swdn_toa"]
swdn_toa_mean = np.array([datum.mean() for datum in swdn_toa], dtype=np.float32)
swdn_toa_mean = swdn_toa_mean / swdn_toa_mean.mean()

labels = [(0, 'tref'), (1, 'swdn_toa')]
inputs = np.column_stack((tref_mean, swdn_toa_mean))

#lstm parameters
input_size = 2
sequence_length = 5
lstm = SimpleLSTM(input_size=input_size, output_size=2)

reload = True
train = False

max_epochs = 5000
learning_rate = 1e-3
saved_chkpt_path = "/home/Mikyung.Lee/spear-emulator-me/lstm-2variable-2outputs/version_0/checkpoints/epoch=4999-step=55000.ckpt"


if reload:
    if saved_chkpt_path is None:
        raise ValueError("Set saved_chkpt_path before using reload=True")
    model = TrainModule.load_from_checkpoint(saved_chkpt_path, weights_only=False, map_location="cpu")
else:
    model = TrainModule(lstm, learning_rate=learning_rate)

if train:
    tb_logger = TensorBoardLogger(save_dir="lstm-2variable-2outputs", name="")
    trainer = pl.Trainer(max_epochs=max_epochs, logger=tb_logger)    
    datamodule = AutoDataModule(data=inputs, sequence_length=sequence_length)
    
    trainer.fit(model=model, datamodule=datamodule)

    #plot training plot    
    inputs_, targets = next(iter(datamodule.train_dataloader()))
    with torch.no_grad():
        z = model.model(inputs_)
    
    for column, label in labels:
        fig, ax = plt.subplots()
        ax.plot(targets[:, column].detach(), color='black', label=f'actual {label}')
        ax.plot(z[:, column].detach(), label=f'training fit {label}')
        ax.legend()
        trainer.logger.experiment.add_figure(f"training_{label}", fig, global_step=trainer.global_step)
    
    #  validation plot
    inputs_, targets = next(iter(datamodule.val_dataloader()))
    with torch.no_grad():
        z = model.model(inputs_)
    
    for column, label in labels:
        fig, ax = plt.subplots()
        ax.plot(targets[:, column].detach(), color='black', label=f'actual {label}')
        ax.plot(z[:, column].detach(), label=f'validation fit {label}')
        ax.legend()
        trainer.logger.experiment.add_figure(f"validation_{label}", fig, global_step=trainer.global_step)
    trainer.logger.experiment.flush()

# evaluate
model.model.eval()
model.model.cpu()
datamodule = data_module.PredictLSTMDataset(inputs, sequence_length=sequence_length)

with torch.no_grad():
    for itime in range(sequence_length, datamodule.ntimes):
        model_inputs = datamodule.get_inputs()
        z = model.model(model_inputs)
        datamodule.add(z)
        
for column, label in labels:
    fig, ax = plt.subplots()
    ax.plot(datamodule.data[:, column], color='black', label=f'actual {label}')
    ax.plot(datamodule.predictions[:, column].detach(), label=f'predicted {label}')
    ax.set_title(label)
    ax.legend()
    if train:
        trainer.logger.experiment.add_figure(f"evaluation_{label}", fig, global_step=trainer.global_step)
        trainer.logger.experiment.flush()

if not train:
    plt.show()


