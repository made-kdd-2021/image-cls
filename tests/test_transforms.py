import pytest
from hydra import compose, initialize, utils
import torch

from transforms import ResizeWithPadding

from .conftest import CONFIG_PATH


@pytest.mark.parametrize("max_size", [100, 300, 400])
@pytest.mark.parametrize("aspect_ratio", [0.5, 1.5])
def test_transform(max_size, aspect_ratio, rgb_image_batch):
    tr = ResizeWithPadding(max_size=max_size, aspect_ratio=aspect_ratio)

    for image in rgb_image_batch:
        resized = tr(image)
        assert resized.shape[-1] == resized.shape[-2] == max_size


@pytest.mark.parametrize("transform_type", ["train_transform", "test_transform"])
def test_config_transform(transform_type, rgb_image_batch):
    with initialize(config_path=CONFIG_PATH):
        cfg = compose(config_name="training")
        transform = utils.instantiate(getattr(cfg.transforms, transform_type))
        transformed = transform(rgb_image_batch)
        assert transformed.shape[-1] == transformed.shape[-2]
        assert transformed.shape[-3] == rgb_image_batch.shape[-3]
        assert transformed.dtype == torch.float
