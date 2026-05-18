from torch import Tensor
from torchvision import transforms


class EpisodeTestTransform:
    def __init__(self, image_size: int, mean: list, std: list) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, x: Tensor) -> Tensor:
        x = self.transform(x)
        return x
