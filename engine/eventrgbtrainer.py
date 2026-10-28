import torch

from typing import Any

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.utils.plotting import plot_images

from copy import copy

from datasets.aramsuisse_dataset import ArmasuisseDataset
from datasets.clip_sampler import ClipBatchSampler
from datasets.flip_transformer import HFlipTransformer
from datasets.pad_transformer import PadTransformer
from datasets.rgb_photometric_transformer import RGBPhotometricTransformer
from datasets.rgb_shift_transformer import RGBShiftTransformer
from datasets.zoom_transformer import ZoomTransformer
from datasets.yolo_converter import YoloConverter
from engine.basetrainer import collate_fn
from engine.eventrgbvalidator import EventRGBValidator
from evrgb.temporal import clear_memory, step_memory


class EventRGBTrainer(DetectionTrainer):
    # RGB-only translation jitter (train only), max pixels in x / y. 0 = off.
    RGB_SHIFT_PX = (0, 0)
    # RGB-only photometric jitter (train only): probability per sample. 0 = off. KW: blur_max, exposure=(lo, hi).
    RGB_PHOTOMETRIC_P = 0.0
    RGB_PHOTOMETRIC_KW = {}
    # Horizontal flip of both modalities + boxes (train only): probability. 0 = off.
    HFLIP_P = 0.0
    # Zoom of both modalities + boxes (train only): probability, max zoom-in factor, max zoom-out factor (1 = off).
    ZOOM_P = 0.0
    ZOOM_IN = 1.5
    ZOOM_OUT = 1.0
    # "both" (13 ch), "rgb" (3 ch) or "event" (10 ch). Single-modality runs are the reference rows for the fusion.
    MODALITY = "both"
    # Recurrent training: a batch = (batch // CLIP_LEN) sequence slots x CLIP_LEN consecutive frames; the first
    # RANDOM_SLOTS slots draw random clips (RVT "mixed" sampling). 1 = ordinary shuffled single frames.
    CLIP_LEN = 1
    RANDOM_SLOTS = 0

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
    ):
        super().__init__(cfg, overrides, _callbacks)
        assert self.MODALITY in ("both", "rgb", "event"), self.MODALITY
        self.load_rgb = self.MODALITY != "event"
        self.load_evt = self.MODALITY != "rgb"
        self.n_rgb = 3 if self.load_rgb else 0

    def get_model(self, cfg=None, weights=None, verbose=True):
        self.data["channels"] = 3 * self.load_rgb + 10 * self.load_evt
        return super().get_model(cfg, weights, verbose)

    def build_dataset(self, img_path, mode="train", batch=None):
        arma = ds = ArmasuisseDataset(img_path, self.load_rgb, self.load_evt)
        if mode == "train" and self.ZOOM_P > 0:
            ds = ZoomTransformer(ds, p=self.ZOOM_P, zin=self.ZOOM_IN, zout=self.ZOOM_OUT, c_rgb=3 if self.load_rgb else 0)
        if mode == "train" and self.load_rgb and any(self.RGB_SHIFT_PX):
            ds = RGBShiftTransformer(ds, *self.RGB_SHIFT_PX)
        if mode == "train" and self.load_rgb and self.RGB_PHOTOMETRIC_P > 0:
            ds = RGBPhotometricTransformer(ds, p=self.RGB_PHOTOMETRIC_P, **self.RGB_PHOTOMETRIC_KW)
        if mode == "train" and self.HFLIP_P > 0:
            ds = HFlipTransformer(ds, p=self.HFLIP_P)
        h, w = arma.get_im_shape()
        padded = PadTransformer(ds, (0, -w % 32, 0, -h % 32))  # pad right/bottom to a multiple of 32 (360 -> 384, 720 -> 736)
        yolo = YoloConverter(img_path, padded)
        return yolo

    def preprocess_batch(self, batch: dict) -> dict:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float()
        if self.n_rgb:
            batch["img"][:, : self.n_rgb] /= 255  # RGB to [0, 1]; event counts stay raw (as in RVT)
        step_memory(unwrap_model(self.model), batch["frame_idx"], self.CLIP_LEN, self.RANDOM_SLOTS)
        return batch

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        dataset = self.build_dataset(dataset_path, mode)
        if self.CLIP_LEN > 1:
            assert batch_size % self.CLIP_LEN == 0, f"batch {batch_size} must be a multiple of CLIP_LEN {self.CLIP_LEN}"
            sampler = ClipBatchSampler(dataset.match_list, batch_size // self.CLIP_LEN, self.CLIP_LEN,
                                       shuffle=mode == "train", random_slots=self.RANDOM_SLOTS if mode == "train" else 0)
            return InfiniteDataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=self.args.workers)
        return InfiniteDataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=mode == "train",
            num_workers=self.args.workers,
        )

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        EventRGBValidator.N_RGB = self.n_rgb
        EventRGBValidator.CLIP_LEN = self.CLIP_LEN
        return EventRGBValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def validate(self):
        self.validator.args.plots = True  # keep val plots every epoch (W&B)
        return super().validate()

    def save_model(self):
        clear_memory(unwrap_model(self.model), self.ema.ema)  # recurrent state must not end up in the checkpoint
        super().save_model()

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
