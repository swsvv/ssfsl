import numpy as np


class EpisodeSampler:
    def __init__(
        self,
        dataset_samples: list,
        n_way: int,
        num_spt: int,
        num_qry: int,
    ) -> None:
        self.n_way = n_way
        self.num_spt = num_spt
        self.num_qry = num_qry
        self.num_samples = n_way * (num_spt + num_qry)

        label_list = np.array(dataset_samples)[:, 1]
        self.labels = np.unique(label_list)
        self.label_idx_dict = {label: [] for label in self.labels}

        for idx, label in enumerate(label_list):
            self.label_idx_dict[label].append(idx)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        self.episode = []
        samples = []
        ways = np.random.choice(self.labels, size=self.n_way, replace=False)

        for label in ways:
            samples.append(
                np.random.choice(
                    self.label_idx_dict[label],
                    size=self.num_spt + self.num_qry,
                    replace=False,
                )
            )

        self.episode.append(np.concatenate(samples))

        return iter(self.episode)

if __name__=='__main__':
    from pathlib import Path
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torch import Tensor
    from torchvision import transforms

    image_size = 84
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    step = "test"
    dataset_path = "/mnt/e/bak/common_dataset/CDFSL/CropDisease"
    num_workers = 8

    n_way = 5
    num_spt = 20
    num_qry = 15

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    dataset = ImageFolder(root=Path(dataset_path, step), transform=transform)

    sampler = EpisodeSampler(
        dataset_samples=dataset.samples,
        n_way=n_way,
        num_spt=num_spt,
        num_qry=num_qry,
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
    )

    for x, y in data_loader:
        print(len(x))
        print(len(y))
