#!/usr/bin/env python


import os
import pathlib
import numpy as np
import zipfile
import numbers
import torch
from torch import nn
from torch import utils
from torchvision import io
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms.functional import pad
from torch.utils.data import DataLoader, random_split

CONSTANT_MEAN = [0.485, 0.456, 0.406]
CONSTANT_STD = [0.229, 0.224, 0.225]

def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

class NewPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, get_padding(img), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip()
    ]),
    'general': transforms.Compose([
        #NewPad(),
        transforms.Resize(size=500),
        transforms.ConvertImageDtype(dtype=torch.float32), 
        transforms.Grayscale(num_output_channels=3),
        #transforms.ToTensor(),
        transforms.Normalize(mean=CONSTANT_MEAN, std=CONSTANT_STD)
    ]),
}

class ChestXray(utils.data.Dataset):
    def __init__(self, path_to_zip, set_name, class_names = {"NORMAL": 0, "PNEUMONIA": 1}):
        self._path_to_zip = path_to_zip
        self._class_mapping = class_names
        self._image_lists = []
        self._image_classes = []
        self._set_name = set_name
        assert set_name in ['train', 'test', 'val']
        if set_name == 'train':
          self._transform = transforms.Compose([data_transforms['general'], data_transforms['train']])
        elif set_name == 'test' or set_name == 'val':
          self._transform = transforms.Compose([data_transforms['general']]);

        
        with zipfile.ZipFile(path_to_zip) as zip_file:
            for zip_info in zip_file.infolist():
                if not zip_info.is_dir():
                    image_path = pathlib.Path(zip_info.filename)
                    if self._set_name == image_path.parent.parent.name:
                        self._image_classes.append(self._class_mapping[image_path.parent.name])
                        self._image_lists.append(zip_info)

    def __len__(self):
        return len(self._image_lists)
    
    def __getitem__(self, index):
        with zipfile.ZipFile(self._path_to_zip) as zip_file:
            with zip_file.open(self._image_lists[index], "r") as image_file:
                image = np.frombuffer(image_file.read(), dtype=np.uint8)
                image = io.decode_jpeg(torch.from_numpy(image.copy()), mode=io.ImageReadMode.RGB)

        return {"image": self._transform(image), "label": self._image_classes[index]}


class XrayDataLoader(utils.data.DataLoader):
    def __init__(self, path_to_zip, set_name, batch_size=64,
                 pin_memory=True, shuffle=False, drop_last=True, num_workers=1):
        self._path_to_zip = path_to_zip
        self._set_name = set_name
        self._batch_size = batch_size
        self._pin_memory = pin_memory
        #self._shuffle = shuffle
        self._drop_last = drop_last
        self._num_workers = num_workers

        assert set_name in ["train", "test", "val"]
        if set_name == 'train':
          self._shuffle = False
        elif set_name == 'test' or set_name == 'val':
          self._shuffle = True;

    def __getitem__(self):
        return utils.data.DataLoader(ChestXray(path_to_zip, set_name), batch_size=self._batch_size, pin_memory = self._pin_memory,
                              shuffle = self._shuffle, drop_last = self._drop_last, num_workers = self._num_workers)

#dataloader = XrayDataLoader("/content/image-cls/data/raw/chest_xray.zip", "test", batch_size=64, 
#                                   pin_memory=True, 
#                                   shuffle=False, 
#                                   drop_last=True, 
#                                   num_workers=2)
