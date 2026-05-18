# ref: https://github.com/mwritescode/nnclr-cifar100/blob/main/src/models/nnclr.py
# ref: https://arxiv.org/pdf/2104.14548#page=11

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from interface import UnSup
from config import Config
from util.fro_norm import get_fro_norm


class NNCLR(UnSup):
    def __init__(self, config: Config, encoder: nn.Module) -> None:
        super().__init__(config, encoder)
        self.encoder = encoder
        self.proj = self._build_proj(**config.proj)
        self.temperature = config.model.temperature
        self.queue_size = config.model.queue_size

        # referred to moco code
        self.register_buffer("queue", torch.randn(self.queue_size, self.proj_out_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.predictor = self._build_pred(**config.pred)

    def forward(self, x: Tensor) -> tuple[Tensor, float]:
        n = x[0].shape[0]

        # each of x[0], x[1] is augmented x
        z0 = self.proj(self.encoder(x[0]))
        z0 = F.normalize(z0)

        z1 = self.proj(self.encoder(x[1]))
        z1 = F.normalize(z1)

        p0 = self.predictor(z0)
        p1 = self.predictor(z1)

        with torch.no_grad():
            # dequeue enqueue and nn(nearest neighbor)
            nn_z0 = self.compute_similarity(z0)
            nn_z1 = self.compute_similarity(z1)

        loss = (
            self.compute_loss(nn_z0, p1) * 0.5 + self.compute_loss(nn_z1, p0) * 0.5
        ) / 2

        self.dequeue_enqueue(z0)

        # didn't change corr_diff
        corr_diff = get_fro_norm(z0, z1)

        return loss, corr_diff

    @torch.no_grad()
    # reference code: nnclr-cifar100
    def dequeue_enqueue(self, z: Tensor) -> None:
        batch_size = z.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size >= self.queue_size:
            self.queue[ptr:, :] = z[: self.queue_size - ptr].detach()
            self.queue_ptr[0] = 0
        else:
            self.queue[ptr : ptr + batch_size, :] = z.detach()
            self.queue_ptr[0] = ptr + batch_size

    def compute_similarity(self, z: Tensor):
        # similarity
        z = F.normalize(z, dim=1)
        queue_norm = F.normalize(self.queue.clone().detach(), dim=1)

        sim = torch.einsum("nc,ck->nk", [z, queue_norm.T])

        _, idxs = torch.max(sim, dim=1)
        nn_z = self.queue[idxs]

        return nn_z

    def compute_loss(self, nn_z: Tensor, p: Tensor):
        nn_z = F.normalize(nn_z, dim=1)
        p = F.normalize(p, dim=1)

        # logit
        logits = torch.mm(nn_z.T, p).div(self.temperature)

        # CEloss
        labels = torch.arange(p.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss
