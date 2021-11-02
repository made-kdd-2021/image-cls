from typing import Dict, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from dataloader import ChestXrayDataset


class ChestXrayDataModule(LightningDataModule):
    def __init__(self, *,
                 data_dir: str,
                 train_transforms,
                 test_transforms,
                 val_transforms,
                 train_batch_size: int,
                 test_batch_size: int,
                 val_batch_size: int,
                 train_load_workers: int,
                 test_load_workers: int,
                 val_load_workers: int,
                 class_mapping: Dict[str, int]):
        super().__init__(train_transforms=train_transforms,
                         val_transforms=val_transforms, test_transforms=test_transforms)
        self._data_dir = data_dir
        self._train_batch_size = train_batch_size
        self._test_batch_size = test_batch_size
        self._val_batch_size = val_batch_size
        self._train_load_workers = train_load_workers
        self._test_load_workers = test_load_workers
        self._val_load_workers = val_load_workers
        self._class_mapping = class_mapping

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ChestXrayDataset(data_dir=self._data_dir,
                                              transform=self.train_transforms, data_type="train", class_mapping=self._class_mapping)
        self.test_dataset = ChestXrayDataset(data_dir=self._data_dir,
                                             transform=self._test_transforms, data_type="val", class_mapping=self._class_mapping)
        # In the datset test is validation in the PyTorch Lighting
        self.val_dataset = ChestXrayDataset(data_dir=self._data_dir,
                                            transform=self._val_transforms, data_type="test", class_mapping=self._class_mapping)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self._train_batch_size,
                          shuffle=True, drop_last=True, pin_memory=True, num_workers=self._train_load_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self._val_batch_size,
                          shuffle=False, drop_last=False, pin_memory=True, num_workers=self._val_load_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self._test_batch_size,
                          shuffle=False, drop_last=False, pin_memory=True, num_workers=self._test_load_workers)
