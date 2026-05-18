from typing import Callable

import tempfile

from pathlib import Path

import pickle

from torch import nn, optim

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler

from config import Config
from util.ray_tune_space import tune_search_space


def ray_tune_run(train_fn: Callable[[Config], None], config: Config) -> None:
    cpus_per_trial = 2
    gpus_per_trial = 1
    max_num_epochs = 100
    num_samples = 50

    tune_param = tune_search_space(config)

    tune_scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2,
    )

    tune_result = tune.run(
        train_fn,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=tune_param.model_dump(),
        num_samples=num_samples,
        scheduler=tune_scheduler,
    )

    best_trial = tune_result.get_best_trial("val_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")


def ray_tune_ckpt(model: nn.Module, optimizer: optim.Optimizer) -> int:
    checkpoint = get_checkpoint()

    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["model_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    return start_epoch


def ray_tune_report(
    epoch: int, model: nn.Module, optimizer: optim.Optimizer, val_loss: float
) -> None:
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "wb") as fp:
            pickle.dump(checkpoint_data, fp)

        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        train.report(
            {"loss": val_loss},
            checkpoint=checkpoint,
        )
