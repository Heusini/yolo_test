import math
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class RGBPhotometricTransformer(Dataset):
    """Photometric jitter on the RGB channels only (uint8 in, uint8 out). Events and boxes untouched. Train only.

    Per sample, each effect is applied independently with its own probability:
    brightness/contrast, gamma, exposure scaling (under/over-exposure with clipping), Gaussian noise, motion blur.
    """

    def __init__(self, dataset: Dataset, p: float = 0.8, c_rgb: int = 3, blur_max: int = 15):
        self.dataset = dataset
        self.p, self.c_rgb, self.blur_max = p, c_rgb, blur_max

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def motion_blur(x: torch.Tensor, length: int, angle: float) -> torch.Tensor:
        """x [C, H, W] float; straight-line kernel of `length` px at `angle` radians."""
        k = torch.zeros(length, length)
        c = (length - 1) / 2
        for t in torch.linspace(-c, c, 2 * length):
            i, j = int(round(c + t.item() * math.sin(angle))), int(round(c + t.item() * math.cos(angle)))
            k[i, j] = 1.0
        k = (k / k.sum()).expand(x.shape[0], 1, length, length)
        return F.conv2d(x.unsqueeze(0), k, padding=length // 2, groups=x.shape[0]).squeeze(0)

    def jitter(self, rgb: torch.Tensor) -> torch.Tensor:
        x = rgb.float()
        if random.random() < 0.5:  # brightness / contrast
            mean = x.mean()
            x = (x - mean) * random.uniform(0.7, 1.3) + mean + random.uniform(-25, 25)
        if random.random() < 0.3:  # gamma
            x = 255.0 * (x.clamp(0, 255) / 255.0) ** random.uniform(0.7, 1.5)
        if random.random() < 0.3:  # exposure: under (dark) or over (clipped highlights)
            x = x * random.choice([random.uniform(0.3, 0.6), random.uniform(1.5, 2.5)])
        if random.random() < 0.3:  # sensor noise
            x = x + torch.randn_like(x) * random.uniform(2, 12)
        if random.random() < 0.3 and self.blur_max >= 3:  # motion blur (RGB smears, events do not)
            x = self.motion_blur(x, random.randrange(3, self.blur_max + 1, 2), random.uniform(0, math.pi))
        return x.clamp(0, 255).round().to(rgb.dtype)

    def __getitem__(self, index: int):
        data = self.dataset.__getitem__(index)
        if random.random() < self.p:
            img = data["img"].clone()
            img[: self.c_rgb] = self.jitter(img[: self.c_rgb])
            data["img"] = img
        return data
