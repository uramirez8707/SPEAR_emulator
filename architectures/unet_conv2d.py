import torch
import torch.nn as nn
from architectures.cnn import GlobalGridPad2d, get_activation_function

def get_normalization_function(config, out_channels):
    normalization = config.unet['normalization']
    norm_type = normalization['norm_type']
    if norm_type == "batch":
        return nn.BatchNorm2d(out_channels)
    elif norm_type == "group":
        ngroups = normalization['num_groups']
        return nn.GroupNorm(num_groups=ngroups, num_channels=out_channels)

class DoubleConv(nn.Module):
    """ Defines a double convolution block used in the UNet architecture. """
    def __init__(self, in_channels, out_channels, config, dilation):
        super().__init__()

        use_normalization = False
        if "use_batchnorm" in config.unet:
            use_normalization = config.unet['use_batchnorm']
            if use_normalization:
                config.unet['normalization'] = {"norm_type": "batch"}

        if "normalization" in config.unet:
            use_normalization = True

        padding = dilation
        if use_normalization:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
                get_normalization_function(config, out_channels),
                get_activation_function(config.unet),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
                get_normalization_function(config, out_channels),
                get_activation_function(config.unet)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
                get_activation_function(config.unet),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
                get_activation_function(config.unet)
            )

    def forward(self, x):
        return self.double_conv(x)

class UNetModel_conv2d(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # --- Encoder (Downsampling) ---
        # Loops through the features list to build downward layers
        encoder_layers = config.unet['encoder']['filters']
        dilation_rates = config.unet['encoder'].get('dilation')
        if not dilation_rates:
            dilation_rates = [1] * len(encoder_layers)

        curr_in = in_channels
        for i, (feature, d) in enumerate(zip(encoder_layers, dilation_rates)):
            self.downs.append(DoubleConv(curr_in, feature, config, dilation=d))
            curr_in = feature

            # We need a downsample layer after every block EXCEPT the bottleneck
            if i < len(encoder_layers) - 1:
                self.downsamples.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=feature, out_channels=feature, kernel_size=3, stride=2, padding=1, bias=False),
                        get_normalization_function(config, feature),
                        get_activation_function(config.unet)
                    )
                )

        # --- Decoder (Upsampling) ---
        # Loops through the features list to build upward layers
        decoder_layers = config.unet['decoder']['filters']
        upsampling_mode = config.unet.get('upsampling', 'transpose')

        for i in range(len(decoder_layers) - 1):
            in_feat = decoder_layers[i]      # e.g., 256
            out_feat = decoder_layers[i+1]   # e.g., 128

            # Upsampling layer
            if upsampling_mode == "bilinear":
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1) # Smooths the interpolation
                    )
                )
            else:
                self.ups.append(
                    nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
                )

            # Double convolution after concatenating skip connections
            self.ups.append(
                DoubleConv(in_feat, out_feat, config, dilation=1)
            )

        # Last layer maps the final feature size (e.g., 64) back to target channels
        self.out_layer = nn.Conv2d(encoder_layers[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # --- Downward Path ---
        # Run through all down layers EXCEPT the last one (which is the bottleneck)
        for i in range(len(self.downs) - 1):
            x = self.downs[i](x)
            skip_connections.append(x) # Save for skip connection
            x = self.downsamples[i](x)

        # Bottleneck
        x = self.downs[-1](x)

        # --- Upward Path ---
        # Reverse skip connections for easy access during upward path
        skip_connections = skip_connections[::-1]

        # The ups list contains pairs of [ConvTranspose2d, DoubleConv]
        for i in range(0, len(self.ups), 2):
            transpose_conv = self.ups[i]
            double_conv = self.ups[i+1]

            x = transpose_conv(x)
            skip = skip_connections[i//2]

            # Pad spatial dimensions to match skip connection if necessary
            if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
                pad_h = skip.shape[2] - x.shape[2]
                pad_w = skip.shape[3] - x.shape[3]
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

            x = torch.cat([x, skip], dim=1) # Concatenate along channel dimension
            x = double_conv(x)

        return self.out_layer(x)
