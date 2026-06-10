from makani.models.networks.sfnonet import SphericalFourierNeuralOperatorNet

def construct_sfno_model(config, in_channels, out_channels, logger):
    logger.info("Constructing the snfo architecture")

    embed_dim = config.sfno.get('embed_dim', 64)
    num_layers = config.sfno.get('num_layers', 4)
    filter_type = config.sfno.get('filter_type', 'linear')
    operator_type = config.sfno.get('operator_type', 'dhconv')
    scale_factor = config.sfno.get('scale_factor', 1)
    spectral_layers = config.sfno.get('spectral_layers', 3)
    mlp_ratio = config.sfno.get('mlp_ratio', 2.0)
    drop_path_rate = config.sfno.get('drop_path_rate', 0.0)

    logger.info(f"--> embed_dim: {embed_dim}")
    logger.info(f"--> num_layers: {num_layers}")
    logger.info(f"--> operator_type: {operator_type}")
    logger.info(f"--> spectral_layers: {spectral_layers}")

    return SphericalFourierNeuralOperatorNet(
        inp_chans=in_channels,
        out_chans=out_channels,
        inp_shape=(config.nlat, config.nlon),
        out_shape=(config.nlat, config.nlon),
        spectral_transform="sht",
        embed_dim=embed_dim,
        num_layers=num_layers,
        filter_type=filter_type,
        operator_type=operator_type,
        scale_factor=scale_factor,
        spectral_layers=spectral_layers,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate
    )
