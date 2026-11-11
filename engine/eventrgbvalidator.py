from typing import Any
import torch
import numpy as np
from ultralytics.utils import ops
from ultralytics.models.yolo.detect import DetectionValidator
from pathlib import Path

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import DetMetrics, ap_per_class
from einops import rearrange, reduce
from ultralytics.utils.plotting import plot_images

from evrgb.temporal import step_memory


class EventRGBValidator(DetectionValidator):
    N_RGB = 3  # leading RGB channels to scale by 1/255 (0 for event-only models)
    CLIP_LEN = 1  # > 1: the dataloader yields clip batches (datasets/clip_sampler.py); recurrent blocks get reset flags
    # What best.pt is picked by: "map" (mAP50-95, Ultralytics default), "map50", or "center" (cAP, see center_hits).
    FITNESS = "map"

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
        self._mem_model = None

    def init_metrics(self, model) -> None:
        super().init_metrics(model)
        self._mem_model = model  # DetectionModel in training, AutoBackend standalone; both contain the blocks
        self.metrics.stats["tp_c"] = []  # center hits, next to the IoU-based tp

    @staticmethod
    def center_hits(pred_cls, true_cls, pred_boxes, true_boxes) -> np.ndarray:
        """(N,) bool: a prediction hits when its center lies inside a true box of its class that no other prediction
        took yet (closest center wins). Size and overlap of the predicted box play no role."""
        pc = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
        tc = (true_boxes[:, :2] + true_boxes[:, 2:]) / 2
        inside = (pc[:, None] >= true_boxes[None, :, :2]).all(-1) & (pc[:, None] <= true_boxes[None, :, 2:]).all(-1)
        inside &= pred_cls[:, None] == true_cls[None]
        hit = np.zeros(len(pred_boxes), dtype=bool)
        pairs = inside.nonzero()
        if len(pairs):
            dist = (pc[pairs[:, 0]] - tc[pairs[:, 1]]).norm(dim=-1)
            used_p, used_t = set(), set()
            for i, j in pairs[dist.argsort()].tolist():
                if i not in used_p and j not in used_t:
                    used_p.add(i), used_t.add(j)
                    hit[i] = True
        return hit

    def _process_batch(self, preds, batch):
        out = super()._process_batch(preds, batch)
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            out["tp_c"] = np.zeros(preds["cls"].shape[0], dtype=bool)
        else:
            out["tp_c"] = self.center_hits(preds["cls"], batch["cls"], preds["bboxes"], batch["bboxes"])
        return out

    def get_stats(self):
        st = self.metrics.stats
        cap = float("nan")
        if len(st["tp"]) and sum(len(c) for c in st["tp_c"]):
            tp_c, conf, pred_cls, target_cls = (np.concatenate(st[k], 0) for k in ("tp_c", "conf", "pred_cls", "target_cls"))
            cap = float(ap_per_class(tp_c[:, None], conf, pred_cls, target_cls)[5].mean())
        results = super().get_stats()
        results["val/cAP(B)"] = cap  # not "metrics/": plot_results has a fixed grid for those
        self.metrics.center_ap = cap  # reachable from model.val() as metrics.center_ap
        if self.FITNESS == "map50":
            results["fitness"] = results["metrics/mAP50(B)"]
        elif self.FITNESS == "center":
            results["fitness"] = cap
        return results

    def print_results(self) -> None:
        super().print_results()
        LOGGER.info(f"{'cAP (center inside true box)':>40s} {self.metrics.center_ap:11.3g}")

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
        if self._mem_model is not None:
            step_memory(self._mem_model, batch["frame_idx"], self.CLIP_LEN)

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
