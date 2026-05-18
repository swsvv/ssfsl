from torch import Tensor
from torchvision import transforms

from transform.augmentations import GaussianBlur


class SimCLRTransform:
    def __init__(self, image_size: int, mean: list, std: list, s: float = 1.0) -> None:
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2
