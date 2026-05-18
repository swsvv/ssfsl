from PIL import Image

from torch import Tensor
from torchvision import transforms

from transform.augmentations import GaussianBlur, Solarization


class BarlowTwinsTransform:
    def __init__(
        self,
        image_size: int,
        mean: list,
        std: list,
    ) -> None:
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, interpolation=Image.Resampling.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, interpolation=Image.Resampling.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, x) -> tuple[Tensor, Tensor]:
        x1 = self.transform(x)
        x2 = self.transform_prime(x)
        return x1, x2
