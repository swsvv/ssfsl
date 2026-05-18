from torch import nn


def convnet(num_blocks: int, in_channel: int, out_dim: int) -> nn.Sequential:
    layers = []
    # for _ in range(4):
    for _ in range(num_blocks):
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channel, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        )
        in_channel = 64
    layers.append(nn.Flatten())
    net = nn.Sequential(*layers)
    # net.out_dim = 1600
    net.out_dim = out_dim
    return net


def convnet4() -> nn.Sequential:
    return convnet(num_blocks=4, in_channel=3, out_dim=1600)


def convnet5() -> nn.Sequential:
    return convnet(num_blocks=5, in_channel=3, out_dim=1600)
