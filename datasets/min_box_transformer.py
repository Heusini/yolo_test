import torch
from torch.utils.data import Dataset


class MinBoxTransformer(Dataset):
    """Grow every xyxy pixel box to at least `size` px per side, around its center, clipped to the image.
    A label definition, not an augmentation: apply it to train and val alike, so a prediction counts as right
    when it is centered on a small drone, whatever the exact extent of the drone."""

    def __init__(self, dataset: Dataset, size: float):
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def grow(boxes: torch.Tensor, size: float, h: int, w: int) -> torch.Tensor:
        if not len(boxes):
            return boxes
        b = boxes.clone()
        c = (b[:, :2] + b[:, 2:]) / 2
        half = ((b[:, 2:] - b[:, :2]).clamp(min=size)) / 2
        b[:, :2], b[:, 2:] = c - half, c + half
        b[:, [0, 2]] = b[:, [0, 2]].clamp(0, w)
        b[:, [1, 3]] = b[:, [1, 3]].clamp(0, h)
        return b

    def __getitem__(self, index: int):
        data = self.dataset.__getitem__(index)
        h, w = data["img"].shape[-2:]
        data["bboxes"] = self.grow(data["bboxes"], self.size, h, w)
        return data
