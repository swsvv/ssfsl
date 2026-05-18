# ref: https://github.com/facebookresearch/barlowtwins/blob/main/main.py

import torch
from torch import nn, Tensor

from interface import UnSup
from config import Config
from util.fro_norm import get_fro_norm, off_diagonal


class BarlowTwins(UnSup):
    def __init__(self, config: Config, encoder: nn.Module) -> None:
        super().__init__(config, encoder)
        self.encoder = encoder
        self.proj = self._build_proj(**config.proj)

        self.lambd = config.model.lambd

        # normalization layer for the representations z0 and z1
        self.bn = nn.BatchNorm1d(self.proj_out_dim, affine=False)

    def forward(self, x: Tensor) -> tuple[Tensor, float]:
        z0 = self.proj(self.encoder(x[0]))
        z1 = self.proj(self.encoder(x[1]))

        # empirical cross-correlation matrix
        c = self.bn(z0).T @ self.bn(z1)
        c.div_(x[0].shape[0])

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.lambd * off_diag

        corr_diff = get_fro_norm(z0, z1)

        return loss, corr_diff
