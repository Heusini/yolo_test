import sys
from typing import Any, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from datasets.basedataset import BaseDataset


class EventRGBDataset(BaseDataset):
    def __init__(self, path: str, transform=None):
        super().__init__(path)
        self.transform = transform
        self.im_height, self.im_width = None, None
        self.im_height_padded, self.im_width_padded = None, None

    def __getitem__(self, index: int):
        match = self.match_list[index]

        event = np.load(match.event_path)
        event = event[list(event.keys())[0]]
        event = torch.from_numpy(event).float()

        frame = np.load(match.frame_path)
        frame = frame[list(frame.keys())[0]]
        frame = torch.from_numpy(frame)
        frame = frame.permute(-1, 0, 1)
        frame = frame / 255

        label = np.load(match.label_path)
        labels = label[list(label.keys())[0]]

        cls = torch.from_numpy(labels["class_id"]).unsqueeze(1)
        padded_shape = self.get_im_padded_shape()
        bboxes = torch.from_numpy(
            self.convert_boxes(labels, padded_shape[0], padded_shape[1])
        ).float()

        padded_event = F.pad(event, self.padding, mode="constant", value=0)
        padded_frame = F.pad(frame, self.padding, mode="constant", value=0)

        padded_img = torch.vstack([padded_frame, padded_event])
        # print(f"{padded_img.shape=}")
        # sys.exit(0)

        # No transformations yet
        # if self.transform is not None:
        #     img_np = padded_img.numpy().transpose(1, 2, 0)
        #     bboxes_np = bboxes.numpy()
        #     cls_np = cls.numpy().flatten()
        #     transformed = self.transform(
        #         image=img_np, bboxes=bboxes_np, class_labels=cls_np
        #     )

        #     padded_img = torch.from_numpy(transformed["image"].transpose(2, 0, 1))

        #     if len(transformed["bboxes"]) > 0:
        #         bboxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        #         cls = torch.tensor(
        #             transformed["class_labels"], dtype=torch.float32
        #         ).unsqueeze(1)
        #     else:
        #         bboxes = torch.empty((0, 4))
        #         cls = torch.empty((0, 1))

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
