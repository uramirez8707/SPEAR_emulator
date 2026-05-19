from model import SpearEmulator, AutoregressiveSpearEmulator
from data_load import get_dataloaders, get_updated_channels
import pytorch_lightning as L
from utils import configSetUp
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

def train_model(config, label, working_dir):
    config.dump_info()

    training, validating, testing  = get_dataloaders(config)
    input_channels, out_channels = get_updated_channels(config)
    config.set_channels(input_channels, out_channels)
    config.set_grid(training)

    L.seed_everything(config.seed, workers=True, verbose=False)

    ckpt_dir = f"{working_dir}/output/{label}/checkpoints"
    last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
    )

    logger = CSVLogger(f"{working_dir}/output", name="spear_emulator", version=label)

    method = config.get_data_load_method()
    if method == "autoregressive":
        SPEAR = AutoregressiveSpearEmulator(config)
    else:
        SPEAR = SpearEmulator(config)

    trainer = L.Trainer(
        max_epochs=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=1,
        deterministic=True,
        benchmark=False
    )

    resume_ckpt = last_ckpt_path if os.path.exists(last_ckpt_path) else None

    if resume_ckpt:
        print(f"Resuming training from checkpoint: {resume_ckpt}")

    trainer.fit(
        model=SPEAR,
        train_dataloaders=training.tensor,
        val_dataloaders=validating.tensor,
        ckpt_path=resume_ckpt
    )

    print("ALL ABOARD THE CHU-CHU-SPEAR-TRAIN!")

#config = configSetUp(config_yaml="examples/config_default.yaml")
#train_model(config, "nlag_3")
#
#print("------------------------------------------")
#print("------------------------------------------")
#
#config = configSetUp(config_yaml="examples/config_residual.yaml")
#train_model(config, "residual_nlag_3")
#
#print("------------------------------------------")
#print("------------------------------------------")
#

working_dir = "/scratch4/GFDL/gfdlscr/Uriel.Ramirez/SPEAR_TRAINING_JOBS/dev"

config = configSetUp(config_yaml=f"{working_dir}/config_autoregressive.yaml")
train_model(config, "autoregressive_nsteps_3", working_dir)

config = configSetUp(config_yaml=f"{working_dir}/config_autoregressive_padding.yaml")
train_model(config, "autoregressive_nsteps_3_padding", working_dir)

