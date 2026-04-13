import torch
import numpy as np

import albumentations as A

from typing import Any

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.utils.plotting import plot_images

from copy import copy
from einops import rearrange, reduce

from datasets.eventdataset import EventDataset
from engine.basetrainer import collate_fn
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


class EventTrainer(DetectionTrainer):
    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
    ):
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        transform = None
        if mode == "train":
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Affine(
                        scale=(0.9, 1.1),
                        translate_percent=(-0.0625, 0.0625),
                        rotate=(-15, 15),
                        p=0.5,
                    ),
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )
        return EventDataset(
            path=img_path,
            transform=transform,
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
            shuffle=True,
            num_workers=self.args.workers,
        )

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return EventValidator(
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
