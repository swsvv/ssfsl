from typing import List
from torch.utils.data import DataLoader


def get_data_dim(data_loader: DataLoader, model: str) -> List:
    x, _ = next(iter(data_loader))
    if model != "Baseline":
        x = x[0]
    return list(x.shape)
