import random

import torch
from torch.utils.data import Dataset


class HFlipTransformer(Dataset):
    """Horizontal flip of all channels (RGB + events) and the xyxy pixel boxes. Train only."""

    def __init__(self, dataset: Dataset, p: float = 0.5):
        self.dataset = dataset
        self.p = p

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        data = self.dataset.__getitem__(index)
        if random.random() < self.p:
            img = data["img"]
            w = img.shape[-1]
            data["img"] = torch.flip(img, dims=[-1])
            boxes = data["bboxes"]
            if len(boxes):
                boxes = boxes.clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]  # x1' = w - x2, x2' = w - x1
                data["bboxes"] = boxes
        return data
