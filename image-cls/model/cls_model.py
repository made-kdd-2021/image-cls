from torch import nn
from typing import Tuple, Union
import torch

KernelType = Union[int, Tuple[int, int]]


def get_conv_block(in_channels: int, out_channels: int, conv_kernel_size: KernelType, pool_kernel_size: KernelType):
    return [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel_size),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=pool_kernel_size)
    ]


class PneumoniaMobileNetV3(nn.Module):
    def __init__(self, dropout: float = 0.4):
        super().__init__()

        in_channels = 1
        out_channels = 16

        feature_extractor = nn.Sequential()
        feature_extractor.add_module("conv1", nn.Sequential(
            *get_conv_block(in_channels, out_channels, (3, 3), (2, 2))))

        in_channels = out_channels
        out_channels = 32
        feature_extractor.add_module("conv2",
                                     nn.Sequential(
                                         *get_conv_block(in_channels, out_channels, (3, 3), (2, 2))))

        in_channels = out_channels
        out_channels = 64

        feature_extractor.add_module("conv3", nn.Sequential(
            *(get_conv_block(in_channels, out_channels, (3, 3), (2, 2)) + [nn.BatchNorm2d(num_features=out_channels)])))

        in_channels = out_channels
        out_channels = 128

        feature_extractor.add_module("conv4", nn.Sequential(
            *get_conv_block(in_channels, out_channels, (3, 3), (2, 2))))

        adaptive_pool_size = 2
        feature_extractor.add_module("final_pooling", nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(adaptive_pool_size, adaptive_pool_size)),
            nn.Flatten()))

        self._num_classes = 1

        classifier = nn.Sequential(
            nn.Linear(out_channels * adaptive_pool_size * adaptive_pool_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=dropout),
            nn.Linear(64, self._num_classes)
        )

        self._backbone = nn.Sequential(feature_extractor, classifier)
        self._num_classes = 1

    def forward(self, images):
        """images is [B x C x H x W]
        """
        return self._backbone.forward(images)

    def predict_logits(self, images):
        return self.forward(images).view(-1)

    def predict_proba(self, predicted_logits):
        """Predict probability of class
        """
        return torch.sigmoid(predicted_logits)

    def predict_class(self, predicted_logits, threshold: float = 0.5):
        """Predict class label
        """
        return torch.LongTensor(self.predict_proba(predicted_logits) > threshold)

    def num_classes(self) -> int:
        return self._num_classes
