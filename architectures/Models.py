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
            # ----- Down 1/2 -----
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # ----- Down 1/2 -----
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Bottleneck
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # ----- Up 2 -----
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2,
                padding=1, output_padding=0
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # ----- Up 2 -----
            nn.ConvTranspose2d(
                16, 16, kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Final output
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    return model