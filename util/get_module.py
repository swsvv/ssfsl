from torch import nn

from config import Config
from dataset.dataset_norm import get_dataset_stats
from encoder import ENCODER_REGISTRY
from interface import Sup, UnSup
from model import MODEL_REGISTRY
from transform import TRANSFORM_REGISTRY


def get_model(model_name: str, config: Config, encoder: nn.Module) -> Sup | UnSup:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](config, encoder)


def get_encoder(encoder_name: str) -> nn.Module:
    if encoder_name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder: {encoder_name}. Available: {list(ENCODER_REGISTRY.keys())}"
        )
    return ENCODER_REGISTRY[encoder_name]()


def get_transform(dataset_name: str, model_name: str) -> object:
    if model_name not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform: {model_name}. Available: {list(TRANSFORM_REGISTRY.keys())}"
        )
    image_size, mean, std = get_dataset_stats(dataset_name)
    return TRANSFORM_REGISTRY[model_name](image_size, mean, std)
