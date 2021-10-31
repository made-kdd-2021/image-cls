import pytest
from hydra import compose, initialize, utils

from training import ChestXrayDataModule

from .conftest import CONFIG_PATH, TEST_DATA_DIR


def test_datamodule():
    with initialize(config_path=CONFIG_PATH):
        cfg = compose(config_name="training", overrides=[f"datamodule.data_dir={TEST_DATA_DIR}"])
        train_transform = utils.instantiate(cfg.transforms.train_transform)
        test_tranform = utils.instantiate(cfg.transforms.test_transform)
        datamodule: ChestXrayDataModule = utils.instantiate(cfg.datamodule, train_transforms=train_transform,
                                                            test_transforms=test_tranform, val_transforms=test_tranform)
        datamodule.setup()

        for get_dataloader in (datamodule.train_dataloader, datamodule.test_dataloader, datamodule.val_dataloader):
            for batch in get_dataloader():
                assert "image" in batch and "label" in batch
                assert batch["image"].ndim == 4
