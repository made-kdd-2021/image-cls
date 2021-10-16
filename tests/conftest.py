import os

import pytest
import torch

from model import PneumoniaMobileNetV3

CONFIG_PATH = os.path.join("..", "configs")


@pytest.fixture(scope="session")
def image_batch():
    generator = torch.Generator().manual_seed(122)
    return torch.rand((2, 3, 64, 64), generator=generator)


@pytest.fixture(scope="session")
def image_batch_info():
    generator = torch.Generator().manual_seed(122)
    return {"images": torch.rand((2, 3, 64, 64), generator=generator), "labels": torch.ones(2, dtype=torch.long)}


@ pytest.fixture(scope="session")
def model():
    return PneumoniaMobileNetV3(num_classes=2)
