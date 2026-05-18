from torchvision import transforms

from transform.augmentations import GaussianBlurSigmaRange


class WMSETransform:
    def __init__(self, image_size: int, mean: list, std: list, n=2) -> None:
        self.transform = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.2, 1.0),
                    ratio=(0.75, 4 / 3),
                    interpolation=3,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([GaussianBlurSigmaRange([0.1, 2.0])], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.num = n

    def __call__(self, x) -> tuple:
        return tuple(self.transform(x) for _ in range(self.num))
