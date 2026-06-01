import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """ Defines a double convolution block used in the UNet architecture. """
    def __init__(self, in_channels, out_channels, use_batchnorm=False):
        super().__init__()
        if use_batchnorm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)

class UNetModel(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256], use_batchnorm=False):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Encoder (Downsampling) ---
        # Loops through the features list to build downward layers
        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature, use_batchnorm))
            curr_in = feature

        # --- Decoder (Upsampling) ---
        # Iterate backwards to match the encoder layers
        reversed_features = list(reversed(features))

        for i in range(len(reversed_features) - 1):
            in_feat = reversed_features[i]      # e.g., 256
            out_feat = reversed_features[i+1]   # e.g., 128

            # Upsampling layer
            self.ups.append(
                nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
            )
            # Double convolution after concatenating skip connections
            self.ups.append(
                DoubleConv(in_feat, out_feat, use_batchnorm)
            )

        # Last layer maps the final feature size (e.g., 64) back to target channels
        self.out_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

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
    """
    Factory function to build the UNet. 
    Checks the config for a use_batchnorm flag, defaulting to False.
    """
    use_batchnorm = getattr(config.unet, 'use_batchnorm', False)

    return UNetModel(in_channels=input_dim, out_channels=output_dim, use_batchnorm=use_batchnorm)
