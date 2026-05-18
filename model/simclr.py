# ref: https://github.com/alinlab/PsCo/blob/main/models.py#L206

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from interface import UnSup
from config import Config
from util.fro_norm import get_fro_norm


class SimCLR(UnSup):
    def __init__(self, config: Config, encoder: nn.Module) -> None:
        super().__init__(config, encoder)
        self.encoder = encoder

        self.proj = self._build_proj(**config.proj)
        self.proj = nn.Sequential(
            self.proj, nn.BatchNorm1d(self.proj_out_dim, affine=False)
        )

        self.temperature = config.model.temperature

    def forward(self, x: Tensor) -> tuple[Tensor, float]:
        n = x[0].shape[0]

        z0 = self.proj(self.encoder(x[0]))
        z0 = F.normalize(z0)

        z1 = self.proj(self.encoder(x[1]))
        z1 = F.normalize(z1)

        z = torch.cat([z0, z1])

        logits = torch.einsum("ij,jk->ik", [z, z.T]).div(self.temperature)
        logits.fill_diagonal_(float("-inf"))

        labels = torch.tensor(
            list(range(n, 2 * n)) + list(range(n)), device=logits.device
        )

        loss = F.cross_entropy(logits, labels)

        corr_diff = get_fro_norm(z0, z1)

        return loss, corr_diff
