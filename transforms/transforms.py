import torch
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

def get_dtype():
    return torch.float


class ResizeWithPadding(torch.nn.Module):

    def __init__(self, max_size: int, aspect_ratio: float, fill_value: int = 0,
                 padding_mode: str = "constant"):
        """aspect_ratio is width / height
        """
        assert padding_mode in ("constant", "edge", "reflect", "symmetric")
        self.max_size = max_size
        self.aspect_ratio = aspect_ratio
        self.fill_value = fill_value
        self.padding_mode = padding_mode

    def __call__(self, image: torch.Tensor):
        height, width = image.shape[-2:]
        if height > width:
            new_height = self.max_size
            new_width = round(self.aspect_ratio * height)
        else:
            new_width = self.max_size
            new_height = round(new_width / self.aspect_ratio)

        resized = F.resize(image, size=[new_height, new_width],
                           interpolation=InterpolationMode.BILINEAR)

        top_bottom_pad, left_right_pad = self.max_size - new_height, self.max_size - new_width
        left_right_half_pad, right_reminder = divmod(left_right_pad, 2)
        top_half_pad, bottom_reminder = divmod(top_bottom_pad, 2)

        return F.pad(resized,
                     padding=[left_right_half_pad, top_half_pad, left_right_half_pad +
                              right_reminder, top_half_pad + bottom_reminder],
                     padding_mode=self.padding_mode, fill=self.fill_value)
