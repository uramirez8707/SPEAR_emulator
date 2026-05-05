from model import SpearEmulator
from data_load import get_dataloaders, get_updated_channels
import pytorch_lightning as L
from utils import configSetUp
from pytorch_lightning.loggers import CSVLogger

config = configSetUp(config_yaml="config.yaml")
config.dump_info()

training, validating, testing  = get_dataloaders(config)
input_channels, out_channels = get_updated_channels(config)
config.set_channels(input_channels, out_channels)

L.seed_everything(config.seed, workers=True)

logger = CSVLogger("logs", name="spear_emulator")

SPEAR = SpearEmulator(config)
trainer = L.Trainer(
    max_epochs=10,
    logger=logger,
    accelerator="auto", 
    devices=1,
    deterministic=True,
    benchmark=False
)

trainer.fit(SPEAR, training)
#print("ALL ABOARD THE CHU-CHU-SPEAR-TRAIN!")
