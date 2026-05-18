import torch
from torchvision import transforms

class DINOv2Transform:
    def __init__(self, image_size: int, mean: list, std: list):
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.global_transform1(x), self.global_transform2(x)
