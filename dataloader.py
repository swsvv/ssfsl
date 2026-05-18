from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import Config
from dataset.multicrop import MultiCropDataset
from dataset.sampler import EpisodeSampler
from util.get_module import get_transform


def _make_train_loader(dataset, config: Config) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def _make_episode_loader(dataset, config: Config) -> DataLoader:
    sampler = EpisodeSampler(
        dataset_samples=dataset.samples,
        n_way=config.n_way,
        num_spt=config.num_spt,
        num_qry=config.num_qry,
    )
    return DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
    )


def get_dataloader(config: Config, step: str, val_metric: str = "") -> DataLoader:
    if (step == "train") or (step == "val" and val_metric == "loss"):
        transform_name = config.model.name
    elif (step == "test") or (step == "val" and val_metric == "acc"):
        transform_name = "EpisodeTest"
    else:
        raise ValueError("Unknown step!")

    transform = get_transform(config.dataset, transform_name)

    if config.model == "SwAV" and config.model.multicrop is True:
        dataset = MultiCropDataset(
            config.dataset_path,
            config.model.size_crops,
            config.model.nmb_crops,
            config.model.min_scale_crops,
            config.model.max_scale_crops,
        )
    else:
        dataset = ImageFolder(root=Path(config.dataset_path, step), transform=transform)

    if step == "train" or (step == "val" and val_metric == "loss"):
        return _make_train_loader(dataset, config)
    elif step == "test" or (step == "val" and val_metric == "acc"):
        return _make_episode_loader(dataset, config)
    else:
        raise ValueError("Unknown step!")
