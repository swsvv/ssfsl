import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def get_fro_norm(z1: Tensor, z2: Tensor) -> Tensor:
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)

    c = z1.T @ z2

    I = torch.eye(c.shape[0])
    I = I.to(c.device)
    corr_diff = torch.norm(c - I, "fro")
    return corr_diff


def off_diagonal(x: Tensor) -> Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
