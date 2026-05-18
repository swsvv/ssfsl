import torch.nn as nn

from encoder.resnet import BasicBlock, ResNet


class ResNet34(ResNet):
    def __init__(self) -> None:
        super().__init__(
            block=BasicBlock,
            layers=[3, 4, 6, 3],
        )
        self.fc = nn.Identity()
