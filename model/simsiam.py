import torch.nn.functional as F
from torch import nn, Tensor

from interface import UnSup
from config import Config
from util.fro_norm import get_fro_norm


class SimSiam(UnSup):
    def __init__(self, config: Config, encoder: nn.Module) -> None:
        super().__init__(config, encoder)
        self.encoder = encoder

        self.proj = self._build_proj(**config.proj)
        self.proj = nn.Sequential(
            self.proj, nn.BatchNorm1d(self.proj_out_dim, affine=False)
        )

        self.pred = self._build_pred(**config.pred)

    def forward(self, x: Tensor) -> tuple[Tensor, float]:
        z0 = self.proj(self.encoder(x[0]))
        z1 = self.proj(self.encoder(x[1]))

        p0 = self.pred(z0)
        p1 = self.pred(z1)

        loss = 0.5 * (
            self._negative_cosine_similarity(p0, z1.detach())
            + self._negative_cosine_similarity(p1, z0.detach())
        )

        # NOTE: z or p?
        corr_diff = get_fro_norm(z0, z1)

        return loss, corr_diff

    def _negative_cosine_similarity(self, p: Tensor, z: Tensor) -> Tensor:
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()
