from typing import Dict, List
import pathlib

from torch.utils import data
from torch import nn
from torchvision import io
from torchvision.io.image import ImageReadMode


class ChestXrayDataset(data.Dataset):
    def __init__(self, data_dir: str,
                 transform: nn.Sequential,
                 data_type: str,
                 class_mapping: Dict[str, int],
                 image_ext: List[str] = [".jpeg"]):
        assert data_type in ("train", "test", "val")
        self.data_dir = data_dir
        self.transform = transform
        self.data_type = data_type
        self.class_mapping = class_mapping
        self._image_ext = set(image_ext)
        self._images_path = []
        self._class_labels = []
        self._fill_image_list()

    def _fill_image_list(self):
        image_dir = pathlib.Path(self.data_dir, self.data_type)
        assert image_dir.is_dir(), f"{image_dir} is not a directory"

        for entry in image_dir.rglob("*"):
            if entry.is_file() and entry.suffix.lower() in self._image_ext and entry.parent.name in self.class_mapping:
                self._images_path.append(entry)
                self._class_labels.append(self.class_mapping[entry.parent.name])

    def __len__(self) -> int:
        return len(self._images_path)

    def __getitem__(self, index) -> dict:
        image = io.read_image(str(self._images_path[index]), mode=ImageReadMode.GRAY)
        image_path = self._images_path[index].relative_to(self.data_dir).as_posix()
        return {"image": self.transform(image), "label": self._class_labels[index], "image_path": image_path}
