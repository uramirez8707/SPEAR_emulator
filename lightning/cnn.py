import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch

import cnn_module
from autolightning import AutoDataModule, TrainModule, from_reload, set_trainingdataset

# read data
data_dict = {
    "t_ref": "/home/Mikyung.Lee/spear-emulator-data/atmos.192101-201012.t_ref.nc",
    "swdn_toa": "/home/Mikyung.Lee/spear-emulator-data/atmos.192101-201012.swdn_toa.nc",
}

# cnn parameters
sequence_length = 3

# training parameters
learning_rate = 0.001
max_epochs = 5000

# control
reload_ = False
train = True
predict = True
save_dir = "cnn-2variable-2output"
name = ""
saved_chkpt_path = ""

# NN model
cnn = cnn_module.SimpleCNN(nfeatures=len(data_dict), sequence_length=sequence_length)

# initialize model
if reload_:
    model = from_reload(cnn, saved_chkpt_path)
else:
    model = TrainModule(cnn, learning_rate=learning_rate)

# tensorboard
tb_logger = TensorBoardLogger(save_dir=save_dir, name=name)

# train
if train:
    set_trainingdataset(cnn_module.TrainingDataset)
    trainer = pl.Trainer(max_epochs=max_epochs, logger=tb_logger)
    datamodule = AutoDataModule(data_dict=data_dict, sequence_length=sequence_length)

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

# predict:
if predict:
    model.model.eval()
    model.model.cpu()
    predictor = cnn_module.PredictDataset(data_dict, sequence_length=sequence_length)
    with torch.no_grad():
        for itime in range(sequence_length, predictor.ntimes):
            inputs = predictor.get_inputs()
            z = model.model(inputs)
            predictor.add(z)
    predictor.plot(tb_logger=tb_logger)


