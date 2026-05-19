import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
import numpy as np
import torch

import data_module
from autolightning import AutoDataModule, TrainModule, from_reload
from models import SimpleLSTM

# read data
data_dict = {
    "t_ref": "~/spear-emulator-data/atmos.192101-201012.t_ref.nc",
    "swdn_toa": "~/spear-emulator-data/atmos.192101-201012.swdn_toa.nc",
}
labels = [(0, "t_ref"), (1, "swdn_toa")]

#lstm parameters
input_size = len(data_dict)
sequence_length = 5

learning_rate = 0.001
max_epochs = 5000

lstm = SimpleLSTM(input_size=input_size, output_size=input_size)

reload_ = False
train = True
save_dir = "lstm-2variable-2outputs"
name = ""
saved_chkpt_path = "/home/Mikyung.Lee/spear-emulator-me/lstm-2variable-2outputs/version_0/checkpoints/epoch=4999-step=55000.ckpt"

#initialize model
if reload_:
    model = from_reload(lstm, saved_chkpt_path)
else:
    model = TrainModule(lstm, learning_rate=learning_rate)

#tensorboard
tb_logger = TensorBoardLogger(save_dir=save_dir, name=name)

#train
if train:
    trainer = pl.Trainer(max_epochs=max_epochs, logger=tb_logger)    
    datamodule = AutoDataModule(data_dict=data_dict, sequence_length=sequence_length)
    
    trainer.fit(model=model, datamodule=datamodule)

    #plot training plot    
    #inputs_, targets = next(iter(datamodule.train_dataloader()))
    #with torch.no_grad():
    #    z = model.model(inputs_)
    
    #for column, label in labels:
    #    fig, ax = plt.subplots()
    #    ax.plot(targets[:, column].detach(), color='black', label=f'actual {label}')
    #    ax.plot(z[:, column].detach(), label=f'training fit {label}')
    #    ax.legend()
    #    trainer.logger.experiment.add_figure(f"training_{label}", fig, global_step=trainer.global_step)
    
    #  validation plot
    #inputs_, targets = next(iter(datamodule.val_dataloader()))
    #with torch.no_grad():
    #    z = model.model(inputs_)
    
    #for column, label in labels:
    #    fig, ax = plt.subplots()
    #    ax.plot(targets[:, column].detach(), color='black', label=f'actual {label}')
    #    ax.plot(z[:, column].detach(), label=f'validation fit {label}')
    #    ax.legend()
    #    trainer.logger.experiment.add_figure(f"validation_{label}", fig, global_step=trainer.global_step)
    #trainer.logger.experiment.flush()

exit()
# evaluate
model.model.eval()
model.model.cpu()
loaded = data_module.load_variable(data_dict)
inputs = np.column_stack(list(loaded.values()))
datamodule = data_module.PredictDataset(inputs, sequence_length=sequence_length)

with torch.no_grad():
    for itime in range(sequence_length, datamodule.ntimes):
        model_inputs = datamodule.get_inputs().unsqueeze(0)
        z = model.model(model_inputs)
        datamodule.add(z.squeeze(0))
        
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


