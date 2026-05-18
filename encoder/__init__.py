from encoder.resnet10 import ResNet10
from encoder.resnet18 import ResNet18
from encoder.resnet34 import ResNet34
from encoder.resnet50 import ResNet50
from encoder.convnet import convnet4, convnet5
from encoder.vittiny import ViTTiny
from encoder.vitsmall import ViTSmall

ENCODER_REGISTRY: dict[str, type] = {
    "ResNet10": ResNet10,
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ConvNet4": convnet4,
    "ConvNet5": convnet5,
    "ViTTiny": ViTTiny,
    "ViTSmall": ViTSmall,
}
