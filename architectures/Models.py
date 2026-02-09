import torch.nn as nn
import torch.optim as optim

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
            nn.ConvTranspose2d(
                prev_channel, f,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1 if i == len(reversed_filters) - 1 else 0
            ),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True)
        ])
        prev_channel = f

    layers.append(
        nn.Conv2d(prev_channel, out_channels, kernel_size=3, padding=1)
    )

    model = nn.Sequential(*layers)

    return model

def get_model_archirecture(case, in_channels, out_channels):
    if case == 0:
        model = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)  # output 1 channel, same HxW
        )
    elif case == 1:
        model = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)  # output 1 channel, same HxW
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
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    return model

def define_optimizer(model, optimizer_name, lr, weight_decay):
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise RuntimeError(f"The optimizer {optimizer_name} is not supported.")
    
    return optimizer