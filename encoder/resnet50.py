import torch.nn as nn

from encoder.resnet import Bottleneck, ResNet


class ResNet50(ResNet):
    def __init__(self) -> None:
        super().__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
        )
        self.fc = nn.Identity()
