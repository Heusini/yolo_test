import torch

from typing import Any

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.utils.plotting import plot_images

from copy import copy

from datasets.aramsuisse_dataset import ArmasuisseDataset
from datasets.flip_transformer import HFlipTransformer
from datasets.pad_transformer import PadTransformer
from datasets.rgb_photometric_transformer import RGBPhotometricTransformer
from datasets.rgb_shift_transformer import RGBShiftTransformer
from datasets.yolo_converter import YoloConverter
from engine.basetrainer import collate_fn
from engine.eventrgbvalidator import EventRGBValidator


class EventRGBTrainer(DetectionTrainer):
    # RGB-only translation jitter (train only), max pixels in x / y. 0 = off.
    RGB_SHIFT_PX = (0, 0)
    # RGB-only photometric jitter (train only): probability per sample. 0 = off. KW: blur_max, exposure=(lo, hi).
    RGB_PHOTOMETRIC_P = 0.0
    RGB_PHOTOMETRIC_KW = {}
    # Horizontal flip of both modalities + boxes (train only): probability. 0 = off.
    HFLIP_P = 0.0

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
    ):
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        ds = ArmasuisseDataset(img_path, True, True)
        if mode == "train" and any(self.RGB_SHIFT_PX):
            ds = RGBShiftTransformer(ds, *self.RGB_SHIFT_PX)
        if mode == "train" and self.RGB_PHOTOMETRIC_P > 0:
            ds = RGBPhotometricTransformer(ds, p=self.RGB_PHOTOMETRIC_P, **self.RGB_PHOTOMETRIC_KW)
        if mode == "train" and self.HFLIP_P > 0:
            ds = HFlipTransformer(ds, p=self.HFLIP_P)
        padded = PadTransformer(ds, (0, 0, 0, 24))  # 360 -> 384 rows
        yolo = YoloConverter(img_path, padded)
        return yolo

    def preprocess_batch(self, batch: dict) -> dict:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float()
        batch["img"][:, :3, :, :] /= 255  # RGB to [0, 1]; event counts stay raw (as in RVT)
        return batch

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        dataset = self.build_dataset(dataset_path, mode)
        return InfiniteDataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=mode == "train",
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
        self.validator.args.plots = True  # keep val plots every epoch (W&B)
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
