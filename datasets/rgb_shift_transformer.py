import random

import torch
from torch.utils.data import Dataset


class RGBShiftTransformer(Dataset):
    """Translate only the RGB channels by a random integer offset (zero fill). Events and boxes stay as they are.

    Simulates residual misalignment between the two cameras (parallax, timing). Train only.
    """

    def __init__(self, dataset: Dataset, max_dx: int, max_dy: int, p: float = 0.5, c_rgb: int = 3):
        self.dataset = dataset
        self.max_dx, self.max_dy, self.p, self.c_rgb = max_dx, max_dy, p, c_rgb

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def shift(rgb: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        out = torch.zeros_like(rgb)
        h, w = rgb.shape[-2:]
        ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
        xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
        out[..., yd, xd] = rgb[..., ys, xs]
        return out

    def __getitem__(self, index: int):
        data = self.dataset.__getitem__(index)
        if random.random() < self.p:
            dx = random.randint(-self.max_dx, self.max_dx)
            dy = random.randint(-self.max_dy, self.max_dy)
            if dx or dy:
                img = data["img"].clone()
                img[: self.c_rgb] = self.shift(img[: self.c_rgb], dx, dy)
                data["img"] = img
        return data
