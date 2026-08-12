import random
from typing import Any, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from datasets.basedataset import BaseDataset


class EventRGBDatasetSpliter(BaseDataset):
    def __init__(self, path: str, crop_h=384, crop_w=640, transform=None):
        super().__init__(path)
        self.transform = transform
        self.im_height, self.im_width = None, None
        self.im_height_padded, self.im_width_padded = None, None
        self.crop_h = crop_h
        self.crop_w = crop_w

    def get_boxes_from_labels_xyxy(self, labels):
        if len(labels) == 0:
            return np.empty((0, 4))
        x, y, w, h = labels["x"], labels["y"], labels["w"], labels["h"]
        return np.stack([x, y, x + w, y + h], axis=-1)

    def __getitem__(self, index: int):
        match = self.match_list[index]

        event = np.load(match.event_path)
        event = event[list(event.keys())[0]]
        event = torch.from_numpy(event)

        frame = np.load(match.frame_path)
        frame = frame[list(frame.keys())[0]]
        frame = torch.from_numpy(frame)
        frame = frame.permute(-1, 0, 1)

        label = np.load(match.label_path)
        labels = label[list(label.keys())[0]]

        cls = torch.from_numpy(labels["class_id"]).unsqueeze(1).float()
        padded_shape = self.get_im_padded_shape()
        
        padded_event = F.pad(event, self.padding, mode="constant", value=0)
        padded_frame = F.pad(frame, self.padding, mode="constant", value=0)

        padded_img = torch.vstack([padded_frame, padded_event])
        img_h, img_w = padded_img.shape[-2], padded_img.shape[-1]

        bboxes = self.get_boxes_from_labels_xyxy(labels)

        # 50/50 Chance between Crop or Rescale
        if random.random() < 0.5:
            # --- METHOD 1: RANDOM CROP ---
            start_x = random.randint(0, img_w - self.crop_w)
            start_y = random.randint(0, img_h - self.crop_h)

            final_img = padded_img[
                :, start_y : start_y + self.crop_h, start_x : start_x + self.crop_w
            ]

            if len(bboxes) > 0:
                shift = np.array([start_x, start_y, start_x, start_y])
                bboxes = bboxes - shift
                bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, self.crop_w)
                bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, self.crop_h)
        else:
            # --- METHOD 2: RESCALE AND PAD ---
            scale = min(self.crop_w / img_w, self.crop_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            # Resize the 6-channel tensor (must cast to float for interpolate)
            resized_img = F.interpolate(
                padded_img.unsqueeze(0).float(), 
                size=(new_h, new_w), 
                mode='area' 
            ).squeeze(0).to(padded_img.dtype)

            # Pad to final crop size (adds padding to bottom and right)
            pad_w = self.crop_w - new_w
            pad_h = self.crop_h - new_h
            final_img = F.pad(resized_img, (0, pad_w, 0, pad_h), mode="constant", value=0)

            if len(bboxes) > 0:
                bboxes = bboxes * scale
                # No shifting needed since padding is on bottom/right
                bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, self.crop_w)
                bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, self.crop_h)

        # --- COMMON POST-PROCESSING (Filter invalid and convert to YOLO format) ---
        if len(bboxes) > 0:
            bbox_widths = bboxes[:, 2] - bboxes[:, 0]
            bbox_heights = bboxes[:, 3] - bboxes[:, 1]

            # Filter out boxes that became too small after cropping/scaling
            valid_mask = (bbox_widths > 5) & (bbox_heights > 5)
            bboxes = bboxes[valid_mask]
            cls = cls[valid_mask]

            bboxes_yolo = np.empty_like(bboxes, dtype=np.float32)

            bboxes_yolo[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2.0 / self.crop_w
            bboxes_yolo[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2.0 / self.crop_h
            bboxes_yolo[:, 2] = (bboxes[:, 2] - bboxes[:, 0]) / self.crop_w
            bboxes_yolo[:, 3] = (bboxes[:, 3] - bboxes[:, 1]) / self.crop_h

            bboxes = bboxes_yolo

        bboxes = torch.from_numpy(bboxes).float()
        
        return {
            "img": final_img,
            "image_id": index,
            "cls": cls,
            "bboxes": bboxes,
            "im_file": str(match.event_path),
            "ori_shape": [self.crop_h, self.crop_w],
            "resized_shape": [self.crop_h, self.crop_w],
            "normalized": True,
            "bbox_format": "xywh",
        }
