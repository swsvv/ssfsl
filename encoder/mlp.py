import torch
from torch import nn, Tensor


def layer(in_dim: int, out_dim: int, **kwargs) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
    )


def mlp(size: int, in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    layer_blocks = []
    layer_blocks.append(layer(in_dim, hidden_dim))

    for _ in range(size - 2):
        layer_blocks.append(layer(hidden_dim, hidden_dim))

    layer_blocks.append(nn.Linear(hidden_dim, out_dim))

    encoder = nn.Sequential(*layer_blocks)

    return encoder


class MLP(nn.Module):
    def __init__(
        self,
        name: str,
        size: int,
        in_dim: int = 1024,
        hidden_dim: int = 512,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = mlp(size, in_dim, hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)
        return feat
