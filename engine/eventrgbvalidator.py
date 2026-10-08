from typing import Any
import torch
import numpy as np
from ultralytics.utils import ops
from ultralytics.models.yolo.detect import DetectionValidator
from pathlib import Path

from ultralytics.utils.metrics import DetMetrics
from einops import rearrange, reduce
from ultralytics.utils.plotting import plot_images


class EventRGBValidator(DetectionValidator):
    N_RGB = 3  # leading RGB channels to scale by 1/255 (0 for event-only models)

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

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch of images for YOLO validation.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.

        Returns:
            (dict[str, Any]): Preprocessed batch.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()

        # Same normalization as the trainers' preprocess_batch
        if self.N_RGB:
            batch["img"][:, : self.N_RGB] /= 255  # RGB to [0, 1]; event counts stay raw (as in RVT)

        return batch

    def plot_val_samples(self, batch, ni):
        """Plots the validation ground truth labels"""
        images = batch["img"].clone().detach()

        new_batch = batch.copy()
        new_batch["img"] = images[:, :3, :, :]

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
        imagei = images[:, :3, :, :]

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

    def on_plot(self, name, data=None):
        """Overrides the default on_plot to perfectly synchronize W&B uploads."""
        super().on_plot(name, data)
        import wandb
        from pathlib import Path
        
        if wandb.run:
            fname = Path(name)
            if fname.exists() and "val_batch" in fname.stem:
                # Using commit=False attaches the image to the exact current epoch step!
                wandb.log({fname.stem: wandb.Image(str(fname))}, commit=False)
