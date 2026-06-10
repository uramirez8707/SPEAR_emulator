from model import SpearEmulator, AutoregressiveSpearEmulator
from data_load import get_dataloaders, get_updated_channels
import pytorch_lightning as L
from utils import configSetUp, FortranTracker
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import click

def train_model(config, label, working_dir):
    config.dump_info()

    training, validating, testing  = get_dataloaders(config)
    input_channels, out_channels = get_updated_channels(config)

    config.set_channels(input_channels, out_channels)
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


@click.command()
@click.option('--working_dir', required=True, type=str, help='Base working directory for jobs')
@click.option('--yaml_name', required=True, type=str, help='Name of the YAML config file')
@click.option('--label', required=True, type=str, help='Label for the training run')
def cli(working_dir, yaml_name, label):
    # Construct the full path to the YAML file
    config_path = os.path.join(working_dir, yaml_name)

    # Initialize config and run training
    config = configSetUp(config_yaml=config_path)
    train_model(config, label, working_dir)


if __name__ == "__main__":
    cli()

