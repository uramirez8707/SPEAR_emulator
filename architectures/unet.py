import torch
import torch.nn as nn
from architectures.cnn import GlobalGridPad2d, get_activation_function

class DoubleConv(nn.Module):
    """ Defines a double convolution block used in the UNet architecture. """
    def __init__(self, in_channels, out_channels, config, dilation):
        super().__init__()
        use_batchnorm = config.unet['use_batchnorm']
        padding = dilation
        if use_batchnorm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                get_activation_function(config.unet),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
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

class UNetModel(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Encoder (Downsampling) ---
        # Loops through the features list to build downward layers
        encoder_layers = config.unet['encoder']['filters']
        dilation_rates = config.unet['encoder'].get('dilation')
        if not dilation_rates:
            dilation_rates = [1] * len(encoder_layers)

        curr_in = in_channels
        for feature, d in zip(encoder_layers, dilation_rates):
            self.downs.append(DoubleConv(curr_in, feature, config, dilation=d))
            curr_in = feature

        # --- Decoder (Upsampling) ---
        # Loops through the features list to build upward layers
        decoder_layers = config.unet['decoder']['filters']
        for i in range(len(decoder_layers) - 1):
            in_feat = decoder_layers[i]      # e.g., 256
            out_feat = decoder_layers[i+1]   # e.g., 128

            # Upsampling layer
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
            x = self.pool(x)

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

def construct_unet_model(config, input_dim, output_dim):
    return UNetModel(config=config, in_channels=input_dim, out_channels=output_dim)
