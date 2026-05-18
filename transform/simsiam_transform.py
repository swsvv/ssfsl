from torch import Tensor
from torchvision import transforms

from transform.augmentations import GaussianBlurSigmaRange


class SimSiamTransform:
    def __init__(self, image_size: int, mean: list, std: list) -> None:
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlurSigmaRange([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2
