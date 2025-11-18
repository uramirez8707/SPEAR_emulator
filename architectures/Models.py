import torch.nn as nn

def get_model_archirecture(case, in_channels):
    if case == 0:
        model = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)  # output 1 channel, same HxW
        )
    elif case == 1:
        model = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)  # output 1 channel, same HxW
        )
    elif case == 2:
        model = nn.Sequential(
            # Block 1: Conv -> BN -> ReLU -> Downsample
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),  # downsample: HxW -> H/2 x W/2

            # Block 2: Conv -> BN -> ReLU -> Downsample
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),  # downsample again: H/2 x W/2 -> H/4 x W/4

            # Bottleneck
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Upsample back to original size
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # upsample: H/4 x W/4 -> H/2 x W/2
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # output 1 channel, original HxW
        )

    return model