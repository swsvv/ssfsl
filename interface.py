from abc import ABC, abstractmethod

from torch import nn, Tensor

from config import Config


class Sup(nn.Module):
    def __init__(self, config: Config, encoder: nn.Module) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor, y: Tensor) -> Tensor: ...


class IUnSupModel(nn.Module, ABC):
    @abstractmethod
    def _build_proj(
        self,
        proj_size: str,
        activate_fn: str,
    ) -> nn.Module: ...

    @abstractmethod
    def forward(self, x: Tensor) -> tuple[Tensor, float]: ...


class UnSup(IUnSupModel):
    def __init__(self, config: Config, encoder: nn.Module) -> None:
        super().__init__()
        self.model = config.model

    @staticmethod
    def _get_activation(activate_fn: str) -> nn.Module:
        activations = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "LeakyReLU": nn.LeakyReLU,
        }
        if activate_fn not in activations:
            raise ValueError(f"Unknown activation function: {activate_fn}")
        return activations[activate_fn]()

    @staticmethod
    def _build_mlp(dim_sizes: list[int], activate: nn.Module) -> nn.Sequential:
        layers = []
        for i in range(len(dim_sizes) - 2):
            layers.append(nn.Linear(dim_sizes[i], dim_sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(dim_sizes[i + 1]))
            layers.append(activate)
        layers.append(nn.Linear(dim_sizes[-2], dim_sizes[-1], bias=False))
        return nn.Sequential(*layers)

    def _build_proj(
        self,
        proj_size: str,
        activate_fn: str,
    ) -> nn.Sequential:
        dim_sizes = list(map(int, proj_size.split("-")))
        self.proj_out_dim = dim_sizes[-1]
        activate = self._get_activation(activate_fn)
        return self._build_mlp(dim_sizes, activate)

    def _build_pred(
        self,
        pred_size: str,
        activate_fn: str,
    ) -> nn.Sequential:
        dim_sizes = list(pred_size.split("-"))
        dim_sizes = (
            [self.proj_out_dim]
            + [int(x) for x in dim_sizes[1:-1]]
            + [self.proj_out_dim]
        )
        activate = self._get_activation(activate_fn)
        return self._build_mlp(dim_sizes, activate)

    def forward(self, x: Tensor) -> tuple[Tensor, float]: ...
