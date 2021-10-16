from torch import nn
from torchvision import models
from torch.nn import functional as F


class PneumoniaMobileNetV3(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self._backbone = models.mobilenet.mobilenet_v3_small(pretrained=True)

        last_cls_layer = self._backbone.classifier[-1]
        self._num_classes = num_classes
        self._backbone.classifier[-1] = nn.Linear(last_cls_layer.in_features, num_classes)

    def freeze(self):
        self._backbone.eval()
        self._backbone.requires_grad_(False)

    def unfreeze_model(self):
        self._backbone.train()
        self._backbone.requires_grad_(True)

    def unfreeze_cls(self):
        self._backbone.classifier.train()
        self._backbone.classifier.requires_grad_(True)

    def forward(self, images):
        """images is [B x C x H x W]
        """
        return self._backbone.forward(images)

    def predict_logits(self, images):
        return self.forward(images)

    def predict_proba(self, images):
        """Predict probability of class
        """
        return F.softmax(self.forward(images), dim=-1)

    def predict_class(self, images):
        """Predict class label
        """
        return self.forward(images).argmax(dim=-1)

    def num_classes(self) -> int:
        return self._num_classes
