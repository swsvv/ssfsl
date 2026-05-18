import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from expedantic import ConfigBase
from faker import Faker


def flatten_uniques(nested: dict) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for k, v in nested.items():
        if isinstance(v, dict):
            inner = flatten_uniques(v)
            duplicates = set(flattened.keys()) & set(inner.keys())
            if len(duplicates) > 0:
                warnings.warn(
                    f"[flatten_uniques] key{'s' if len(duplicates) > 1 else ''} {duplicates} are not unique; omitted."
                )
                for key in duplicates:
                    inner.pop(key)
                    flattened.pop(key)
            flattened.update(inner)
        else:
            flattened[k] = v

    return flattened


class ProjConfig(ConfigBase):
    proj_size: str
    activate_fn: str


class PredConfig(ConfigBase):
    pred_size: str
    activate_fn: str


class ClassifierConfig(ConfigBase):
    in_dim: int
    num_classes: int


class BaselineConfig(ConfigBase):
    name: str
    num_classes: int


class BarlowTwinsConfig(ConfigBase):
    name: str
    lambd: float


class VICRegConfig(ConfigBase):
    name: str
    eps: float
    lambd: float
    mu: float
    nu: float


class WMSEConfig(ConfigBase):
    name: str
    w_iter: int
    w_size: int
    w_eps: int
    w_emb_size: int
    momentum: float
    track_running_stats: bool
    norm: bool


class BYOLConfig(ConfigBase):
    name: str
    tau: float


class MoCoConfig(ConfigBase):
    name: str
    queue_size: int
    temperature: float
    momentum: float


class SwAVConfig(ConfigBase):
    name: str
    eps: float
    temperature: float
    sinkhorn_iter: int
    num_prototypes: int
    multicrop: bool
    size_crops: list[int]
    nmb_crops: list[int]
    min_scale_crops: list[float]
    max_scale_crops: list[float]


class SimCLRConfig(ConfigBase):
    name: str
    temperature: float


class SimSiamConfig(ConfigBase):
    name: str


class DINOv2Config(ConfigBase):
    name: str
    student_temp: float
    teacher_center_temp: float
    momentum: float


class NNCLRConfig(ConfigBase):
    name: str
    queue_size: int
    temperature: float


class Config(ConfigBase):
    project: str = "shfsl"
    mode: Literal["train", "test", "debug", "tune", "ray_tune", "tune_loss_acc"]
    device: str
    num_workers: int

    dataset: Literal[
        "miniImageNet",
        "tieredImageNet",
        "tinyImageNet",
        "CIFAR-FS",
        "FC100",
        "Omniglot",
        "MNIST",
        "CIFAR10",
    ]
    eval_dataset: Literal[
        "miniImageNet",
        "tieredImageNet",
        "tinyImageNet",
        "CIFAR-FS",
        "FC100",
        "Omniglot",
        "MNIST",
        "CIFAR10",
        "ChestX",
        "CropDisease",
        "EuroSAT",
        "ISIC",
        "CUB",
    ]
    dataset_path: str
    encoder: Literal[
        "ConvNet4",
        "ConvNet5",
        "ResNet10",
        "ResNet12",
        "ResNet18",
        "ResNet34",
        "ResNet50",
        "ViTTiny",
        "ViTSmall",
    ]
    proj: ProjConfig | None = None
    pred: PredConfig | None = None
    model: (
        SimCLRConfig
        | MoCoConfig
        | SwAVConfig
        | BYOLConfig
        | SimSiamConfig
        | BarlowTwinsConfig
        | VICRegConfig
        | WMSEConfig
        | BaselineConfig
        | DINOv2Config
        | NNCLRConfig
    )

    # Train
    num_epoch: int
    batch_size: int
    lr: float
    momentum: float
    weight_decay: float
    save_freq: int
    save_path: str

    # Test
    num_episode: int | None = None
    n_way: int | None = None
    num_spt: int | None = None
    num_qry: int | None = None
    classifier: str | None = None

    # post inits
    seed: int | None = None
    exp_name: str | None = None
    exp_path: Path | None = None
    group: str | None = None
    wandb_run_id: str | None = None

    # ablation study
    batch_size_factor: int | None = None
    grad_avg_test: bool | None = None

    def model_post_init(self, _context) -> None:
        super().model_post_init(_context)

        if self.group:
            flattened = flatten_uniques(self.model_dump())
            self.group = self.group.format_map(flattened)

        if self.exp_name:
            fake = Faker()
            word = fake.word()

            now = datetime.now()
            current_time = now.strftime("%Y%m%d_%H_%M")

            if self.seed is None:
                self.seed = random.randint(0, 2**31 - 1)

            flattened = flatten_uniques(self.model_dump())
            flattened["word"] = word
            flattened["current_time"] = current_time
            flattened["seed"] = self.seed

            self.exp_name = self.exp_name.format_map(flattened)

            self.exp_path = Path(
                self.save_path,
                self.exp_name,
            )


if __name__ == "__main__":
    config = Config.load_from_yaml("./config/deep_lda.yaml")
    print(config)
