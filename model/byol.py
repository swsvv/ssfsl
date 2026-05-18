import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from interface import UnSup
from config import Config
from util.fro_norm import get_fro_norm


class BYOL(UnSup):
    def __init__(self, config: Config, encoder: nn.Module) -> None:
        super().__init__(config, encoder)
        self.tau = config.model.tau

        self.encoder = encoder
        self.proj = self._build_proj(**config.proj)
        self.pred = self._build_pred(**config.pred)

        self.encoder_target = copy.deepcopy(encoder)
        self.proj_target = copy.deepcopy(self.proj)

        self._initializes_target_network(self.encoder_target)
        self._initializes_target_network(self.proj_target)

    def forward(self, x: Tensor) -> tuple[Tensor, float]:
        p0 = self.pred(self.proj(self.encoder(x[0])))
        p1 = self.pred(self.proj(self.encoder(x[1])))

        with torch.no_grad():
            self._update_encoder_target(self.encoder, self.encoder_target)
            self._update_encoder_target(self.proj, self.proj_target)

            z0_target = self.proj_target(self.encoder_target(x[0]))
            z1_target = self.proj_target(self.encoder_target(x[1]))
            z0_target = z0_target.detach()
            z1_target = z1_target.detach()

        loss = self._negative_cosine_similarity(
            p0, z1_target
        ) + self._negative_cosine_similarity(p1, z0_target)
        loss = loss.mean()

        corr_diff = get_fro_norm(p0, p1)

        return loss, corr_diff

    def _initializes_target_network(self, target: nn.Module) -> None:
        # init momentum network as encoder net
        for param in target.parameters():
            param.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _update_encoder_target(self, source: nn.Module, target: nn.Module) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data = (
                self.tau * target_param.data + (1.0 - self.tau) * source_param.data
            )

    def _negative_cosine_similarity(self, p: Tensor, z: Tensor) -> Tensor:
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return 2 - 2 * (p * z).sum(dim=-1)
