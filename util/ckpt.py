import os

import torch
from torch import nn, optim

from config import Config


def save(
    config: Config, epoch: int, model: nn.Module, optimizer: optim.Optimizer
) -> None:
    file_name = f"{config.exp_path}/ckpt-{epoch}.pkl"

    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        file_name,
    )


def get_ckpt_list(dir: str) -> list:
    # List all files in the directory
    files = os.listdir(dir)
    ckpt_list = []

    # Filter files with the specified pattern and extract the number
    files_with_pattern = [
        file for file in files if file.startswith("ckpt-") and file.endswith(".pkl")
    ]

    # Define a custom sorting function to sort files based on the number in the pattern
    def sort_by_number(filename):
        return int(filename.split("-")[1].split(".")[0])

    # Sort the files based on the number in the pattern
    ckpt_list = sorted(files_with_pattern, key=sort_by_number)

    return ckpt_list
