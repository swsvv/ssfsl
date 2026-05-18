import torch
from torch import nn, Tensor
import torch.nn.functional as F

from interface import UnSup
from config import Config
from util.fro_norm import get_fro_norm


class WMSE(UnSup):
    def __init__(self, config: Config, encoder: nn.Module) -> None:
        super().__init__(config, encoder)
        self.encoder = encoder
        self.proj = self._build_proj(**config.proj)

        self.loss_f = self._norm_mse_loss if config.model.norm else F.mse_loss

        self.num_features = config.model.w_emb_size
        self.momentum = config.model.momentum
        self.track_running_stats = config.model.track_running_stats
        self.eps = config.model.w_eps

        self.w_iter = config.model.w_iter
        self.w_size = config.model.w_size

        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros([1, self.num_features, 1, 1])
            )
            self.register_buffer("running_variance", torch.eye(self.num_features))

    def forward(self, x: Tensor) -> tuple[Tensor, float]:
        n = x[0].shape[0]

        h = [self.encoder(x_i) for x_i in x]
        h = self.proj(torch.cat(h))

        loss = 0
        for _ in range(self.w_iter):
            z = torch.empty_like(h)
            perm = torch.randperm(n).view(-1, self.w_size)

            for idx in perm:
                for i in range(len(x)):
                    z[idx + i * n] = self._whitening(h[idx + i * n])

            for i in range(len(x) - 1):
                for j in range(i + 1, len(x)):
                    x0 = z[i * n : (i + 1) * n]
                    x1 = z[j * n : (j + 1) * n]
                    loss += self.loss_f(x0, x1)

        loss /= self.w_iter * self.num_features

        # NOTE: WMSE use number of samples. _corr_diff use two samples.
        corr_diff = get_fro_norm(h, h)

        return loss, corr_diff

    def _whitening(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(2).unsqueeze(3)

        if not self.training and self.track_running_stats:
            m = self.running_mean
        else:
            m = x.mean(0).view(self.num_features, -1).mean(-1).view(1, -1, 1, 1)

        xn = x - m

        T = xn.permute(1, 0, 2, 3).contiguous().view(self.num_features, -1)
        f_cov = torch.mm(T, T.permute(1, 0)) / (T.shape[-1] - 1)

        eye = torch.eye(self.num_features).type(f_cov.type())
        eye = eye.to(x.device)

        if not self.training and self.track_running_stats:
            f_cov = self.running_variance

        f_cov_shrinked = (1 - self.eps) * f_cov + self.eps * eye

        inv_sqrt = torch.linalg.solve_triangular(
            torch.linalg.cholesky(f_cov_shrinked), eye, upper=False
        )

        inv_sqrt = inv_sqrt.contiguous().view(
            self.num_features, self.num_features, 1, 1
        )

        decorrelated = F.conv2d(xn, inv_sqrt)
        if self.training and self.track_running_stats:
            self.running_mean = torch.add(
                self.momentum * m.detach(),
                (1 - self.momentum) * self.running_mean,
                out=self.running_mean,
            )
            self.running_variance = torch.add(
                self.momentum * f_cov.detach(),
                (1 - self.momentum) * self.running_variance,
                out=self.running_variance,
            )

        return decorrelated.squeeze(2).squeeze(2)

    def _norm_mse_loss(self, x0, x1) -> Tensor:
        x0 = F.normalize(x0)
        x1 = F.normalize(x1)

        return 2 - 2 * (x0 * x1).sum(dim=-1).mean()
