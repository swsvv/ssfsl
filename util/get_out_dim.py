from torch import nn
from torch.utils.data import DataLoader


def get_out_dim(encoder: nn.Module, dataloader: DataLoader) -> int:
    x = next(iter(dataloader))
    output = encoder(x)
    output_dim = output.size(1)
    return output_dim
