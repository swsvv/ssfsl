# ref: https://github.com/facebookresearch/moco/blob/main/moco/builder.py

import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from interface import UnSup
from config import Config
from util.fro_norm import get_fro_norm


class MoCo(UnSup):
    def __init__(self, config: Config, encoder: nn.Module) -> None:
        super().__init__(config, encoder)
        self.encoder = encoder
        self.encodoer_k = copy.deepcopy(self.encoder)

        self.proj_q = self._build_proj(**config.proj)
        self.proj_k = copy.deepcopy(self.proj_q)

        self.moco_m = config.model.momentum
        self.temperature = config.model.temperature
        self.queue_size = config.model.queue_size

        self.register_buffer(
            "queue",
            torch.randn(config.model.queue_size, self.proj_out_dim),
        )

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, x: Tensor) -> tuple[Tensor, float]:
        q = self.proj_q(self.encoder(x[0]))
        q = F.normalize(q)

        with torch.no_grad():
            self._momentum_update_encoder_k()

            k = self.proj_k(self.encodoer_k(x[1]))
            k = F.normalize(k)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().T.detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = logits / self.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        self._dequeue_enqueue(k)

        corr_diff = get_fro_norm(q, k)

        return loss, corr_diff

    @torch.no_grad()
    def _momentum_update_encoder_k(self) -> None:
        for param_q, param_k in zip(
            self.encoder.parameters(), self.encodoer_k.parameters()
        ):
            param_k.data = (
                self.moco_m * param_k.data + (1.0 - self.moco_m) * param_q.data
            )

    @torch.no_grad()
    def _dequeue_enqueue(self, k) -> None:
        batch_size = k.shape[0]
        ptr = self.queue_ptr.item()

        self.queue[ptr : ptr + batch_size] = k
        self.queue_ptr[0] = (ptr + batch_size) % self.queue.shape[0]
