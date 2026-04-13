from typing import Any
import torch
import numpy as np
from ultralytics.utils import ops
from ultralytics.models.yolo.detect import DetectionValidator
from pathlib import Path

from ultralytics.utils.metrics import DetMetrics
from einops import rearrange, reduce
from ultralytics.utils.plotting import plot_images


def ev_repr_to_img(x: np.ndarray):
    ch, ht, wd = x.shape[-3:]
    assert ch > 1 and ch % 2 == 0
    ev_repr_reshaped = rearrange(x, "(posneg C) H W -> posneg C H W", posneg=2)
    img_neg = np.asarray(
        reduce(ev_repr_reshaped[0], "C H W -> H W", "sum"), dtype="int32"
    )
    img_pos = np.asarray(
        reduce(ev_repr_reshaped[1], "C H W -> H W", "sum"), dtype="int32"
    )
    img_diff = img_pos - img_neg
    img = 127 * np.ones((3, ht, wd), dtype=np.uint8)
    img[:, img_diff > 0] = 255
    img[:, img_diff < 0] = 0
    return img


class EventValidator(DetectionValidator):
    def __init__(
        self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None
    ) -> None:
        """Initialize detection validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict[str, Any], optional): Arguments for the validator.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()

    def pred_to_json(
        self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]
    ) -> None:
        path = Path(pbatch["im_file"])
        # image_id = "/".join(path.parts[-3:])
        image_id = int(pbatch["image_id"])
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": str(path),
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch of images and annotations for validation.

        Args:
            si (int): Sample index within the batch.
            batch (dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (dict[str, Any]): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox = (
                ops.xywh2xyxy(bbox)
                * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]
            )  # target boxes
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
            "image_id": batch["image_id"][si],
        }

    def plot_val_samples(self, batch, ni):
        """Plots the validation ground truth labels"""
        images = batch["img"].clone().detach()
        imagei = np.stack([ev_repr_to_img(img.cpu().numpy()) for img in images])

        new_batch = batch.copy()
        new_batch["img"] = imagei

        plot_images(
            labels=new_batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self,
        batch: dict[str, Any],
        preds: list[dict[str, torch.Tensor]],
        ni: int,
        max_det: int | None = None,
    ) -> None:
        if not preds:
            return

        """Plots the model's actual predictions"""
        images = batch["img"].clone().detach()
        imagei = np.stack([ev_repr_to_img(img.cpu().numpy()) for img in images])

        for i, pred in enumerate(preds):
            pred["batch_idx"] = (
                torch.ones_like(pred["conf"]) * i
            )  # add batch index to predictions
        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {
            k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys
        }
        batched_preds["bboxes"] = ops.xyxy2xywh(
            batched_preds["bboxes"]
        )  # convert to xywh format
        plot_images(
            images=imagei,
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred
