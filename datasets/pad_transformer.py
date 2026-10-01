from typing import List, Tuple
import torch
import random
import torch.nn.functional as F
from torch.utils.data import Dataset


class PadTransformer(Dataset):
    def __init__(self, dataset: Dataset, padding: Tuple[float, float, float, float]):
        self.dataset = dataset
        self.pad = padding

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        data = self.dataset.__getitem__(index)
        img = data["img"]
        bboxes = data["bboxes"]

        pad_left = int(self.pad[2])
        pad_top = int(self.pad[0])

        final_img = F.pad(img, self.pad, mode="constant", value=0)

        if len(bboxes) > 0:
            bboxes[:, 0] += pad_left  # shift x1
            bboxes[:, 1] += pad_top  # shift y1
            bboxes[:, 2] += pad_left  # shift x2
            bboxes[:, 3] += pad_top  # shift y2

        data["img"] = final_img
        data["bboxes"] = bboxes

        return data
