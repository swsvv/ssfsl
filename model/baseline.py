import torch.nn.functional as F
from torch import nn, Tensor

from interface import Sup
from config import Config


class Baseline(Sup):
    def __init__(self, config: Config, encoder: nn.Module) -> None:
        super().__init__(config, encoder)
        self.encoder = encoder
        self.classifier = nn.Linear(config.proj.in_dim, config.model.num_classes)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        pred = self.classifier(self.encoder(x))

        loss = F.cross_entropy(pred, y)

        return loss
