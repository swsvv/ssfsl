import torch
import torch.nn.functional as F
from torch import nn, Tensor

from interface import UnSup
from config import Config
from util.fro_norm import get_fro_norm


class SwAV(UnSup):
    def __init__(self, config: Config, encoder: nn.Module) -> None:
        super().__init__(config, encoder)
        self.encoder = encoder

        self.proj = self._build_proj(**config.proj)

        self.prototypes = nn.Linear(
            self.proj_out_dim, config.model.num_prototypes, bias=False
        )

        self.eps = config.model.eps
        self.temperature = config.model.temperature
        self.sinkhorn_iter = config.model.sinkhorn_iter

    def forward(self, x: Tensor) -> tuple[Tensor, float]:
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = F.normalize(w)
            self.prototypes.weight.copy_(w)

        z0 = self.proj(self.encoder(x[0]))
        z0 = F.normalize(z0)

        z1 = self.proj(self.encoder(x[1]))
        z1 = F.normalize(z1)

        scores_t = self.prototypes(z0)
        scores_s = self.prototypes(z1)

        z0 = z0.detach()
        z1 = z1.detach()

        q_t = self._sinkhorn(scores_t.detach())
        q_s = self._sinkhorn(scores_s.detach())

        # p_t = F.log_softmax(scores_t / self.temperature, dim=1)
        # p_s = F.log_softmax(scores_s / self.temperature, dim=1)
        #
        # loss = -0.5 * torch.mean(q_t * p_s + q_s * p_t)

        p_t = F.log_softmax(scores_t.div(self.temperature), dim=1)
        p_s = F.log_softmax(scores_s.div(self.temperature), dim=1)

        loss = -(torch.sum(q_t.mul(p_s), dim=1) + torch.sum(q_s.mul(p_t), dim=1)).mean()

        corr_diff = get_fro_norm(z0, z1)

        return loss, corr_diff

    @torch.no_grad()
    def _sinkhorn(self, out: Tensor) -> Tensor:
        """
        ref: https://github.com/facebookresearch/swav/blob/main/main_swav.py#L354
        """
        Q = torch.exp(
            out / self.eps
        ).t()  # Q is K-by-B for consistency with notations from our paper
        # B = Q.shape[1] * self.world_size  # number of samples to assign
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iter):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            # dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()
