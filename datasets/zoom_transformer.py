import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ZoomTransformer(Dataset):
    """Random zoom of all channels and the xyxy pixel boxes. Train only. One draw per frame: with probability p
    the frame is zoomed, then out with probability p_out (factor in [1, zout]) else in (factor in [1, zin]).

    zoom-in: crop a window of 1/s and resize to full size (RGB bilinear, events nearest -> counts stay integers).
    zoom-out: shrink by s and paste on a zero canvas at a random position (RGB bilinear, events area-summed so
    the event mass is kept, rounded, clipped at cutoff). Boxes are rescaled, clipped, and dropped
    when less than min_vis of their area is left.
    """

    def __init__(self, dataset: Dataset, p: float = 0.5, zin: float = 1.5, zout: float = 1.0, p_out: float = 1 / 3,
                 c_rgb: int = 3, min_vis: float = 0.5, cutoff: int = 10):
        self.dataset = dataset
        self.p, self.zin, self.zout, self.p_out, self.c_rgb, self.min_vis, self.cutoff = p, zin, zout, p_out, c_rgb, min_vis, cutoff

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _boxes(boxes, x0, y0, sx, sy, w, h, min_vis):
        if not len(boxes):
            return boxes, torch.zeros(0, dtype=torch.bool)
        b = boxes.clone()
        b[:, [0, 2]] = (b[:, [0, 2]] - x0) * sx
        b[:, [1, 3]] = (b[:, [1, 3]] - y0) * sy
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        b[:, [0, 2]] = b[:, [0, 2]].clamp(0, w)
        b[:, [1, 3]] = b[:, [1, 3]].clamp(0, h)
        vis = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        keep = (vis >= min_vis * area) & (b[:, 2] - b[:, 0] >= 1) & (b[:, 3] - b[:, 1] >= 1)
        return b[keep], keep

    @staticmethod
    def zoom_in(img, boxes, y0, x0, ch, cw, c_rgb, min_vis):
        """Crop [y0:y0+ch, x0:x0+cw] and resize back to the full size."""
        h, w = img.shape[-2:]
        crop = img[:, y0 : y0 + ch, x0 : x0 + cw].float().unsqueeze(0)
        rgb = F.interpolate(crop[:, :c_rgb], (h, w), mode="bilinear", align_corners=False).round()
        evt = F.interpolate(crop[:, c_rgb:], (h, w), mode="nearest")
        out = torch.cat([rgb, evt], 1)[0].clamp(min=0).to(img.dtype)
        boxes, keep = ZoomTransformer._boxes(boxes, x0, y0, w / cw, h / ch, w, h, min_vis)
        return out, boxes, keep

    @staticmethod
    def zoom_out(img, boxes, y0, x0, sh, sw, c_rgb, min_vis, cutoff=10):
        """Shrink to (sh, sw) and paste at [y0, x0] on a zero canvas of the original size."""
        h, w = img.shape[-2:]
        x = img.float().unsqueeze(0)
        rgb = F.interpolate(x[:, :c_rgb], (sh, sw), mode="bilinear", align_corners=False).round()
        evt = (F.interpolate(x[:, c_rgb:], (sh, sw), mode="area") * (h / sh) * (w / sw)).round().clamp(max=cutoff)
        out = torch.zeros_like(img)
        out[:, y0 : y0 + sh, x0 : x0 + sw] = torch.cat([rgb, evt], 1)[0].clamp(min=0).to(img.dtype)
        boxes, keep = ZoomTransformer._boxes(boxes, -x0 * w / sw, -y0 * h / sh, sw / w, sh / h, w, h, min_vis)
        return out, boxes, keep

    def __getitem__(self, index: int):
        data = self.dataset.__getitem__(index)
        rng = random.Random(data["aug_key"] * 8 + 4) if data.get("aug_key") else random  # one draw per clip
        if rng.random() >= self.p:
            return data
        h, w = data["img"].shape[-2:]
        if self.zout > 1.0 and rng.random() < self.p_out:
            s = rng.uniform(1.0, self.zout)
            sh, sw = max(round(h / s), 1), max(round(w / s), 1)
            y0, x0 = rng.randint(0, h - sh), rng.randint(0, w - sw)
            img, boxes, keep = self.zoom_out(data["img"], data["bboxes"], y0, x0, sh, sw, self.c_rgb, self.min_vis, self.cutoff)
        else:
            s = rng.uniform(1.0, self.zin)
            ch, cw = max(round(h / s), 1), max(round(w / s), 1)
            y0, x0 = rng.randint(0, h - ch), rng.randint(0, w - cw)
            img, boxes, keep = self.zoom_in(data["img"], data["bboxes"], y0, x0, ch, cw, self.c_rgb, self.min_vis)
        data["img"], data["bboxes"] = img, boxes
        if len(keep):
            data["cls"] = data["cls"][keep]
        return data
