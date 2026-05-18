# ref : https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from interface import UnSup
from config import Config
from util.fro_norm import get_fro_norm, off_diagonal


class VICReg(UnSup):
    def __init__(self, config: Config, encoder: nn.Module) -> None:
        super().__init__(config, encoder)
        self.encoder = encoder
        self.proj = self._build_proj(**config.proj)
        self.batch_size = config.batch_size
        self.num_features = self.proj_out_dim

        self.eps = config.model.eps
        self.lambd = config.model.lambd
        self.mu = config.model.mu
        self.nu = config.model.nu

        self.config = config
        self.corr_diff = 0.0

    def forward(self, x: Tensor) -> tuple[Tensor, float]:
        z0 = self.proj(self.encoder(x[0]))
        z1 = self.proj(self.encoder(x[1]))

        inv = self._get_inv(z0, z1)

        z0 = z0 - z0.mean(dim=0)
        z1 = z1 - z1.mean(dim=0)

        var = self._get_var(z0, z1)
        cov = self._get_cov(z0, z1)

        loss = self.mu * var + self.lambd * inv + self.nu * cov

        corr_diff = get_fro_norm(z0, z1)

        return loss, corr_diff

    def _get_inv(self, z0: Tensor, z1: Tensor) -> Tensor:
        sim_loss = F.mse_loss(z0, z1)
        return sim_loss

    def _get_var(self, z0: Tensor, z1: Tensor) -> Tensor:
        std_z0 = torch.sqrt(z0.var(dim=0) + self.eps)
        std_z1 = torch.sqrt(z1.var(dim=0) + self.eps)
        std_loss = (
            torch.mean(F.relu(1 - std_z0)) / 2 + torch.mean(F.relu(1 - std_z1)) / 2
        )
        return std_loss

    def _get_cov(self, z0: Tensor, z1: Tensor) -> Tensor:
        cov_z0 = z0.T @ z0 / (self.batch_size - 1)
        cov_z1 = z1.T @ z1 / (self.batch_size - 1)

        cov_loss = off_diagonal(cov_z0).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_z1).pow_(2).sum().div(self.num_features)
        return cov_loss
