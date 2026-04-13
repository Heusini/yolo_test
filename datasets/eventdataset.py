from typing import Any, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from datasets.basedataset import BaseDataset


def get_image_size(path):
    test_img = np.load(path)["arr_0"]
    return test_img.shape[-1] * test_img.shape[-2]


class EventDataset(BaseDataset):
    def __init__(self, path: str, transform=None):
        super().__init__(path)
        self.transform = transform
        self.im_height, self.im_width = None, None
        self.im_height_padded, self.im_width_padded = None, None

    def load_image(self, i, rect_mode: bool = True):
        match = self.match_list[i]
        im = np.load(match.event_path)["arr_0"]
        im = np.transpose(im, (1, 2, 0))

        return im, im.shape[:2], im.shape[:2]

    def __getitem__(self, index: int):
        match = self.match_list[index]

        event = np.load(match.event_path)
        event = event[list(event.keys())[0]]
        event = torch.from_numpy(event)

        label = np.load(match.label_path)
        labels = label[list(label.keys())[0]]

        cls = torch.from_numpy(labels["class_id"]).unsqueeze(1)
        padded_shape = self.get_im_padded_shape()
        bboxes = self.convert_boxes(labels, padded_shape[0], padded_shape[1])

        padded_img = F.pad(event, self.padding, mode="constant", value=0)

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
            "im_file": str(match.event_path),
            "ori_shape": padded_shape,
            "resized_shape": padded_shape,
            "normalized": True,
            "bbox_format": "xywh",
        }
