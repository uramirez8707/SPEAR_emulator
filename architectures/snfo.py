from makani.models.networks.sfnonet import SphericalFourierNeuralOperatorNet

def construct_sfno_model(config, in_channels, out_channels, logger):
    logger.info("Constructing the snfo architecture")
    logger.info(f"--> emded_dim: {config.sfno['emded_dim']}")
    logger.info(f"--> num_layers: {config.sfno['num_layers']}")
    return SphericalFourierNeuralOperatorNet(
        inp_chans=in_channels,
        out_chans=out_channels,
        inp_shape=(config.nlat, config.nlon),
        out_shape=(config.nlat, config.nlon),
        spectral_transform="sht",
        embed_dim=config.sfno['emded_dim'],
        num_layers=config.sfno['num_layers']
    )
