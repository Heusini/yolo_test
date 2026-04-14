from typing import Any, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from datasets.basedataset import BaseDataset


class RGBDataset(BaseDataset):
    def __init__(self, path: str, transform=None):
        super().__init__(path)
        self.path = Path(path)
        self.transform = transform
        assert self.path.is_dir()
        self.im_height, self.im_width = None, None

    def load_image(self, i, rect_mode: bool = True):
        match = self.match_list[i]
        im = np.load(match.frame_path)["arr_0"]
        im = np.transpose(im, (1, 2, 0))

        return im, im.shape[:2], im.shape[:2]

    def __getitem__(self, index: int):
        match = self.match_list[index]

        label = np.load(match.label_path)
        labels = label[list(label.keys())[0]]

        frame = np.load(match.frame_path)
        frame = frame[list(frame.keys())[0]]
        frame = torch.from_numpy(frame)
        frame = frame.permute(-1, 0, 1)

        cls = torch.from_numpy(labels["class_id"]).unsqueeze(1)
        padded_shape = self.get_im_padded_shape()
        bboxes = self.convert_boxes(labels, padded_shape[0], padded_shape[1])

        padded_img = F.pad(frame, self.padding, mode="constant", value=0)
        if self.transform is not None:
            img_np = padded_img.numpy().transpose(1, 2, 0)
            bboxes_np = bboxes.numpy()
            cls_np = cls.numpy().flatten()
            transformed = self.transform(
                image=img_np, bboxes=bboxes_np, class_labels=cls_np
            )

            padded_img = torch.from_numpy(transformed["image"].transpose(2, 0, 1))

            if len(transformed["bboxes"]) > 0:
                bboxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
                cls = torch.tensor(
                    transformed["class_labels"], dtype=torch.float32
                ).unsqueeze(1)
            else:
                bboxes = torch.empty((0, 4))
                cls = torch.empty((0, 1))

        return {
            "img": padded_img,
            "image_id": index,
            "cls": cls,
            "bboxes": bboxes,
            "im_file": str(match.frame_path),
            "ori_shape": padded_shape,
            "resized_shape": padded_shape,
            "normalized": True,
            "bbox_format": "xywh",
        }
