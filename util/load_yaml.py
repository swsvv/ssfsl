import yaml
from pathlib import Path


def load_yaml(path: str) -> dict:
    # Load YAML from a file
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data
