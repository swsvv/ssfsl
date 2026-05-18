def get_dataset_stats(dataset: str) -> tuple[int, list, list]:
    if dataset == "Omniglot":
        mean = [0.9218]
        std = [0.2685]
        image_size = 28
    elif dataset in ("miniImageNet", "tieredImageNet"):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_size = 84
    elif dataset == "tinyImageNet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_size = 64
    elif dataset in ("CIFAR10", "CIFAR100", "FC100", "CIFAR-FS"):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        image_size = 32
    elif dataset in (
        "CUB",
        "CropDiseases",
        "EuroSAT",
        "ISIC",
        "ChestX",
        "Places",
        "Cars",
        "Plantae",
    ):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_size = 84
    else:
        raise ValueError("Unknwon dataset!")

    return image_size, mean, std
