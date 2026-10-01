import torch
import random
import torch.nn.functional as F
from torch.utils.data import Dataset


class SplitTransformer(Dataset):
    def __init__(self, dataset: Dataset, crop_h, crop_w):
        self.dataset = dataset
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        data = self.dataset.__getitem__(index)
        img = data["img"]
        bboxes = data["bboxes"]
        cls = data["cls"]
        img_h, img_w = img.shape[-2], img.shape[-1]
        if random.random() < 0.5:
            # --- METHOD 1: RANDOM CROP ---
            start_x = random.randint(0, img_w - self.crop_w)
            start_y = random.randint(0, img_h - self.crop_h)

            final_img = img[
                :, start_y : start_y + self.crop_h, start_x : start_x + self.crop_w
            ]

            if len(bboxes) > 0:
                shift = torch.tensor([start_x, start_y, start_x, start_y])
                bboxes = bboxes - shift
                bboxes[:, [0, 2]] = torch.clip(bboxes[:, [0, 2]], 0, self.crop_w)
                bboxes[:, [1, 3]] = torch.clip(bboxes[:, [1, 3]], 0, self.crop_h)
        else:
            # --- METHOD 2: RESCALE AND PAD ---
            scale = min(self.crop_w / img_w, self.crop_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            # Resize the 6-channel tensor (must cast to float for interpolate)
            resized_img = (
                F.interpolate(
                    img.unsqueeze(0).float(), size=(new_h, new_w), mode="area"
                )
                .squeeze(0)
                .to(img.dtype)
            )

            # Pad to final crop size (adds padding to bottom and right)
            pad_w = self.crop_w - new_w
            pad_h = self.crop_h - new_h
            final_img = F.pad(
                resized_img, (0, pad_w, 0, pad_h), mode="constant", value=0
            )

            if len(bboxes) > 0:
                bboxes = bboxes * scale
                # No shifting needed since padding is on bottom/right
                bboxes[:, [0, 2]] = torch.clip(bboxes[:, [0, 2]], 0, self.crop_w)
                bboxes[:, [1, 3]] = torch.clip(bboxes[:, [1, 3]], 0, self.crop_h)

        # --- COMMON POST-PROCESSING (Filter invalid and convert to YOLO format) ---
        if len(bboxes) > 0:
            bbox_widths = bboxes[:, 2] - bboxes[:, 0]
            bbox_heights = bboxes[:, 3] - bboxes[:, 1]

            # Filter out boxes that became too small after cropping/scaling
            valid_mask = (bbox_widths > 5) & (bbox_heights > 5)
            bboxes = bboxes[valid_mask]
            cls = cls[valid_mask]

        data["img"] = final_img
        data["bboxes"] = bboxes
        data["cls"] = cls
        return data
