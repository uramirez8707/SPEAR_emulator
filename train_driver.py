from model import SpearEmulator, AutoregressiveSpearEmulator
from data_load import get_dataloaders, get_updated_channels
import pytorch_lightning as L
from utils import configSetUp, FortranTracker
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

def train_model(config, label, working_dir):
    config.dump_info()

    training, validating, testing  = get_dataloaders(config)
    input_channels, out_channels, diag_channels = get_updated_channels(config)
    config.set_channels(input_channels, out_channels, diag_channels)
    config.set_grid(training)

    L.seed_everything(config.seed, workers=True, verbose=False)

    run_dir = f"{working_dir}/output/{label}"
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
    )

    logger = CSVLogger(
            save_dir=run_dir,
            name="logs")

    method = config.get_data_load_method()
    if method == "autoregressive":
        SPEAR = AutoregressiveSpearEmulator(config)
    else:
        SPEAR = SpearEmulator(config)

    trainer = L.Trainer(
        max_epochs=config.nepochs,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            FortranTracker()
            ],
        accelerator="auto",
        devices=1,
        deterministic=True,
        benchmark=False,
        precision=config.precision
    )

#    resume_ckpt = last_ckpt_path if os.path.exists(last_ckpt_path) else None
#
#    if resume_ckpt:
#        print(f"Resuming training from checkpoint: {resume_ckpt}")
#
#    trainer.fit(
#        model=SPEAR,
#        train_dataloaders=training.tensor,
#        val_dataloaders=validating.tensor,
#        ckpt_path=resume_ckpt
#    )
#
    print("ALL ABOARD THE CHU-CHU-SPEAR-TRAIN!")

working_dir = "/scratch4/GFDL/gfdlscr/Uriel.Ramirez/SPEAR_TRAINING_JOBS/run3"

# Test with encoder/decoder CNN
config = configSetUp(config_yaml=f"{working_dir}/config_cnn.yaml")
train_model(config, "autoregressive_cnn", working_dir)

# Test with SFNO
config = configSetUp(config_yaml=f"{working_dir}/config_sfno.yaml")
train_model(config, "autoregressive_sfno", working_dir)

# Test with UNet
config = configSetUp(config_yaml=f"{working_dir}/config_unet.yaml")
train_model(config, "autoregressive_unet", working_dir)

