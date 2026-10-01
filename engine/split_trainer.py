import torch
import numpy as np

import albumentations as A

from typing import Any

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.utils.plotting import plot_images

from copy import copy

from datasets.aramsuisse_dataset import ArmasuisseDataset
from datasets.split_transformer import SplitTransformer
from datasets.yolo_converter import YoloConverter

from engine.basetrainer import collate_fn
from engine.eventrgbvalidator import EventRGBValidator


class SplitTrainer(DetectionTrainer):
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
        arma = ArmasuisseDataset(img_path, True, True)
        split = SplitTransformer(arma, 384, 640)
        yolo = YoloConverter(img_path, split)
        return yolo

    def preprocess_batch(self, batch: dict) -> dict:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float()
        batch["img"][:, :3, :, :] /= 255
        batch["img"][:, 3:, :, :] = torch.log1p(batch["img"][:, 3:, :, :])
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
        return EventRGBValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def validate(self):
        """Forces plotting to be enabled during intermediate epochs."""
        # Ultralytics normally disables plots during training to save time.
        # We explicitly enable it here so W&B gets the images every epoch!
        self.validator.args.plots = True
        return super().validate()

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        images = batch["img"].clone().detach()

        new_batch = batch.copy()
        new_batch["img"] = images[:, :3, :, :]

        plot_images(
            labels=new_batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
