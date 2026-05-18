import torch.nn as nn

from encoder.resnet import BasicBlock, ResNet


class ResNet18(ResNet):
    def __init__(self) -> None:
        super().__init__(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
        )
        self.fc = nn.Identity()
