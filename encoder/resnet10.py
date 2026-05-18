import torch.nn as nn

from encoder.resnet import BasicBlock, ResNet


class ResNet10(ResNet):
    def __init__(self) -> None:
        super().__init__(
            block=BasicBlock,
            layers=[1, 1, 1, 1],
        )
        self.fc = nn.Identity()
