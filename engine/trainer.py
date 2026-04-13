import torch
import numpy as np

from typing import Any

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.utils.plotting import plot_images

from copy import copy
from einops import rearrange, reduce

from datasets.eventdataset import EventDataset
from engine.validator import EventValidator


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


def collate_fn(batch):
    imgs = torch.stack([b["img"] for b in batch], 0)
    ori_shapes = torch.stack([torch.tensor(b["ori_shape"]) for b in batch], 0)
    resized_shapes = torch.stack([torch.tensor(b["resized_shape"]) for b in batch], 0)
    image_ids = torch.stack([torch.tensor(b["image_id"]) for b in batch], 0)
    cls = torch.vstack([b["cls"] for b in batch])
    im_files = [b["im_file"] for b in batch]
    bboxes = torch.vstack([b["bboxes"] for b in batch])
    batch_idx = []
    for i, b in enumerate(batch):
        n = b["cls"].shape[0]
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
    batch_idx = torch.cat(batch_idx)
    return {
        "img": imgs,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx,
        "im_file": im_files,
        "ori_shape": ori_shapes,
        "resized_shape": resized_shapes,
        "ratio_pad": [((1.0, 1.0), (0.0, 0.0))] * len(imgs),
        "image_id": image_ids,
    }


class EventTrainer(DetectionTrainer):
    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
    ):
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        return EventDataset(
            path=img_path,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float()
        return batch

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        dataset = self.build_dataset(dataset_path, mode)
        return InfiniteDataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.args.workers,
        )

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        print(type(self.test_loader))
        return DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        images = batch["img"].clone().detach()
        imagei = np.stack([ev_repr_to_img(img.cpu().numpy()) for img in images])

        new_batch = batch.copy()
        new_batch["img"] = imagei

        plot_images(
            labels=new_batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
