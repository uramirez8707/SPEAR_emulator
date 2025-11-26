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
        nn.MaxPool2d(2, stride=2),  # H: 90 -> 45, W: 144 -> 72

        # Block 2: Conv -> BN -> ReLU -> Downsample
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),  # H: 45 -> 22, W: 72 -> 36

        # Bottleneck
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        # Upsample: H: 22 -> 45, W: 36 -> 72
        nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, output_padding=(1, 0)),  # output_padding=(H_extra, W_extra)
        nn.Conv2d(16, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),

        # Upsample: H: 45 -> 90, W: 72 -> 144
        nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, output_padding=(0, 0)),
        nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    return model