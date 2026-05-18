import numpy as np


class EpisodeLowShotSampler:
    def __init__(
        self,
        base_dataset_samples: list,
        novel_dataset_samples: list,
        n_way: int,
        num_spt: int,
        num_qry: int,
        base_ratio: float,
    ) -> None:
        self.n_way = n_way
        self.num_spt = num_spt
        self.num_qry = num_qry
        self.num_samples = n_way * (num_spt + num_qry)

        self.base_ratio = base_ratio

        base_label_list = np.array(base_dataset_samples)[:, 1]
        self.base_labels = np.unique(base_label_list)
        self.base_label_idx_dict = {base_label: [] for base_label in self.base_labels}

        for idx, label in enumerate(base_label_list):
            self.base_label_idx_dict[label].append(idx)

        novel_label_list = np.array(novel_dataset_samples)[:, 1]
        self.novel_labels = np.unique(novel_label_list)
        self.novel_label_idx_dict = {
            novel_label: [] for novel_label in self.novel_labels
        }

        for idx, label in enumerate(novel_label_list):
            self.novel_label_idx_dict[label].append(idx)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        self.episode = []
        base_samples, novel_samples = [], []

        num_base = int(round(self.n_way * self.base_ratio))
        num_novel = self.n_way - num_base

        base_ways = np.random.choice(self.base_labels, size=num_base, replace=False)
        novel_ways = np.random.choice(self.novel_labels, size=num_novel, replace=False)

        for label in base_ways:
            base_samples.append(
                np.random.choice(
                    self.base_label_idx_dict[label],
                    size=self.num_spt + self.num_qry,
                    replace=False,
                )
            )

        for label in novel_ways:
            novel_samples.append(
                np.random.choice(
                    self.novel_label_idx_dict[label],
                    size=self.num_spt + self.num_qry,
                    replace=False,
                )
            )

        samples = base_samples + novel_samples

        self.episode.append(np.concatenate(samples))

        return iter(self.episode)


if __name__ == "__main__":
    import numpy as np
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    import torchvision
    import torchvision.transforms as transforms
    from PIL import Image

    image_size = 224
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=Image.Resampling.BICUBIC
            ),
            torchvision.transforms.ToTensor(),
        ]
    )
    dataset = ImageFolder(
        root="/path/to/dataset/miniImageNet/test", transform=transform
    )
    dataset_info = np.genfromtxt(
        f"./script/miniImageNet/test.csv", skip_header=1, delimiter=",", dtype=str
    )

    num_episode = 100
    episode_stat = []
    for _ in range(num_episode):
        s = EpisodeSampler(
            dataset_samples=dataset.samples,
            n_way=3,
            num_spt=3,
            num_qry=7,
            batch_episode=5,
        )
        # dl = DataLoader(dataset, batch_sampler=s)
        dl = DataLoader(dataset, sampler=s, batch_size=3 * 10)
        for x, y in dl:
            # print(x)
            print(y)
            episode_stat.append(np.unique(y.numpy()))

    unique_elements, counts = np.unique(
        np.concatenate(np.array(episode_stat)), return_counts=True
    )

    for element, count in zip(unique_elements, counts):
        print(f"{element}: {count}")
