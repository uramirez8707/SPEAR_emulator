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

#lstm parameters
input_size = len(data_dict)
sequence_length = 5

#training parameters
learning_rate = 0.001
max_epochs = 5

#control
reload_ = False
train = False
predict = True
save_dir = "lstm-2variable-2outputs"
name = ""
saved_chkpt_path = "/home/Mikyung.Lee/spear-emulator-me/lstm-2variable-2outputs/version_0/checkpoints/epoch=4999-step=55000.ckpt"

#NN model
lstm = SimpleLSTM(input_size=input_size, output_size=input_size)

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
    datamodule = AutoDataModule(data_dict=data_dict, sequence_length=sequence_length, test_with_global_average=True)
    
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

# predict:
if predict:
    model.model.eval()
    model.model.cpu()
    predict = data_module.PredictDataset(data_dict, sequence_length=sequence_length, test_with_global_average=True)
    with torch.no_grad():
        for itime in range(sequence_length, predict.ntimes):
            inputs = predict.get_inputs()
            z = model.model(inputs)
            predict.add(z)
        predict.plot(tb_logger=tb_logger)


