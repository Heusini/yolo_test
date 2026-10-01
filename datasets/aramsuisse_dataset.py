import torch
import numpy as np
from datasets.basedataset import BaseDataset


def get_boxes_from_labels_xyxy(labels):
    if len(labels) == 0:
        return np.empty((0, 4))
    x, y, w, h = labels["x"], labels["y"], labels["w"], labels["h"]
    return np.stack([x, y, x + w, y + h], axis=-1)


class ArmasuisseDataset(BaseDataset):
    def __init__(self, path: str, load_rgbs: bool, load_events: bool):
        super().__init__(path)

        self.load_events = load_events
        self.load_rgbs = load_rgbs

    def __getitem__(self, index: int):
        match = self.match_list[index]

        event = None
        frame = None
        if self.load_events:
            event = np.load(match.event_path)
            event = torch.from_numpy(event)

        if self.load_rgbs:
            frame = np.load(match.frame_path)
            frame = torch.from_numpy(frame)
            frame = frame.permute(-1, 0, 1)

        out_img = None
        if self.load_events and self.load_rgbs:
            out_img = torch.vstack([frame, event])
        elif self.load_rgbs:
            out_img = frame
        elif self.load_events:
            out_img = event
        else:
            raise ValueError(
                "at least load_events or load_rgbs must be set to train on something"
            )

        labels = np.load(match.label_path)

        bboxes = get_boxes_from_labels_xyxy(labels)
        bboxes = torch.from_numpy(bboxes).float()
        cls = torch.from_numpy(labels["class_id"]).unsqueeze(1).float()

        height, width = self.get_im_shape()
        return {
            "img": out_img,
            "image_id": index,
            "cls": cls,
            "bboxes": bboxes,
            "im_file": str(match.event_path),
            "ori_shape": [height, width],
            "resized_shape": [height, width],
            "normalized": False,
            "bbox_format": "xyxy",
        }

    def get_im_shape(self):
        if self.im_width is None:
            if self.load_rgbs:
                rgb = np.load(self.match_list[0].frame_path)  # HWC
                self.im_height = rgb.shape[0]
                self.im_width = rgb.shape[1]
            else:
                event = np.load(self.match_list[0].event_path)
                self.im_width = event.shape[-1]
                self.im_height = event.shape[-2]

        return self.im_height, self.im_width
