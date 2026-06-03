import torch
import torch.nn.functional as F
from torch import nn

class GlobalGridPad2d(nn.Module):
    """
    Pads the longitude (width) circularly,
    and the latitude (height) by replicating the edge values.
    """
    def __init__(self, padding=1):
        super().__init__()
        self.p = padding

    def forward(self, x):
        # F.pad takes boundaries as: (Left, Right, Top, Bottom)
        # 1. Pad longitude circularly
        x = F.pad(x, (self.p, self.p, 0, 0), mode='circular')

        # 2. Pad latitude by replicating the poles
        x = F.pad(x, (0, 0, self.p, self.p), mode='replicate')
        return x

def construct_model(in_channels, out_channels, filters):
    layers = []

    # Encoder
    prev_channel = in_channels
    for f in filters[:-1]:
        layers.extend([
            nn.Conv2d(prev_channel, f, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(f),
            nn.ReLU(),
        ])
        prev_channel = f

    # Bottleneck
    bottleneck_channels = filters[-1]
    layers.extend([
        nn.Conv2d(prev_channel, bottleneck_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(bottleneck_channels),
        nn.ReLU(inplace=True)
    ])

    # Decoder
    reversed_filters = list(reversed(filters[:-1]))
    prev_channel = bottleneck_channels

    for i, f in enumerate(reversed_filters):
        layers.extend([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(prev_channel, f, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True)
        ])
        prev_channel = f

    # Final Output Layer
    layers.append(
        nn.Conv2d(prev_channel, out_channels, kernel_size=3, padding=1)
    )

    model = nn.Sequential(*layers)

    return model


def get_activation_function(config):
    activation_function = config.cnn['activation_function']
    if activation_function == "relu":
        return nn.ReLU(inplace=True)

    elif activation_function == "leaky_relu":
        return nn.LeakyReLU(
            negative_slope=config.cnn.get("negative_slope", 0.01),
            inplace=True
        )

    elif activation_function == "gelu":
        return nn.GELU()

    else:
        raise ValueError(
            f"Unsupported activation function: {activation_function}"
        )

def construct_model_better_padding(in_channels, out_channels, config):
    layers = []

    # Encoder
    prev_channel = in_channels
    filters = config.cnn['encoder']['filters']
    for f in filters:
        layers.extend([
            GlobalGridPad2d(padding=1),
            nn.Conv2d(prev_channel, f, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(f),
            get_activation_function(config),
        ])
        prev_channel = f

    # Bottleneck
    filters = config.cnn['bottleneck']['filters']
    for f in filters:
        layers.extend([
            GlobalGridPad2d(padding=1),
            nn.Conv2d(prev_channel, f, kernel_size=3, padding=0),
            nn.BatchNorm2d(f),
            get_activation_function(config)
        ])
        prev_channel = f

    # Decoder
    filters = config.cnn['decoder']['filters']
    for f in filters:
        layers.extend([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            GlobalGridPad2d(padding=1),
            nn.Conv2d(prev_channel, f, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(f),
            get_activation_function(config)
        ])
        prev_channel = f

    # Final Output Layer
    layers.extend([
        GlobalGridPad2d(padding=1),
        nn.Conv2d(prev_channel, out_channels, kernel_size=3, padding=0)
    ])

    model = nn.Sequential(*layers)

    return model
