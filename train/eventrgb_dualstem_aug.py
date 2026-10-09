"""Dual-stem run with selectable training-time augmentations (all off by default = train/eventrgb_dualstem.py).

examples:
  python -m train.eventrgb_dualstem_aug --name flip --flip 0.5 --device 0
  python -m train.eventrgb_dualstem_aug --name flip_drop --flip 0.5 --drop-rgb 0.15 --drop-evt 0.05
  python -m train.eventrgb_dualstem_aug --name flip_photo --flip 0.5 --photo 0.8 --blur 0 --exposure 0.6 1.6
  python -m train.eventrgb_dualstem_aug --name flip_shift --flip 0.5 --shift 2 1
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
import torch

from engine.dualstem_trainer import DualStemTrainer
from engine.eventrgbtrainer import EventRGBTrainer


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="run name suffix: eventrgb_dualstem_<name>_yolo26n")
    p.add_argument("--device", type=int, default=1)
    p.add_argument("--batch", type=int, default=16, help="train batch (val uses 2x); 8 halves GPU memory, effective batch stays 64")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--p2", action="store_true", help="use the P2-head variant (conf/yolo26n_evrgb_dualstem_p2.yaml)")
    p.add_argument("--baseline", action="store_true", help="plain 13-channel yolo26n (conf/yolo26n_evrgb.yaml) instead of the dual stem")
    p.add_argument("--modality", choices=["both", "rgb", "event"], default="both", help="single-modality reference runs (implies --baseline)")
    p.add_argument("--data", default="./conf/eventrgb_data.yaml", help="data yaml")
    p.add_argument("--imgsz", type=int, default=640, help="long side of the images (640 or 1280); the loader pads to /32")
    p.add_argument("--flip", type=float, default=0.0, help="horizontal flip probability")
    p.add_argument("--photo", type=float, default=0.0, help="RGB photometric jitter probability")
    p.add_argument("--blur", type=int, default=15, help="max motion-blur kernel in px for --photo (0 = no blur)")
    p.add_argument("--exposure", type=float, nargs=2, default=(0.3, 2.5), metavar=("LO", "HI"), help="exposure factor range for --photo")
    p.add_argument("--shift", type=int, nargs=2, default=(0, 0), metavar=("DX", "DY"), help="RGB-only shift in px")
    p.add_argument("--drop-rgb", type=float, default=0.0, help="modality dropout probability for RGB")
    p.add_argument("--drop-evt", type=float, default=0.0, help="modality dropout probability for events")
    return p.parse_args()


def data_yaml_for_modality(data: str, modality: str) -> str:
    """Copy of the data yaml with `channels` matching the modality (the standalone final validation reads it)."""
    import yaml

    d = yaml.safe_load(Path(data).read_text())
    d["channels"] = {"rgb": 3, "event": 10}[modality]
    out = Path(data).with_name(f"{Path(data).stem}_{modality}.yaml")
    out.write_text(yaml.safe_dump(d, sort_keys=False))
    return str(out)


def main():
    a = parse()
    matplotlib.use("Agg")
    cv2.setNumThreads(0)
    torch.set_num_threads(16)

    EventRGBTrainer.HFLIP_P = a.flip  # DualStemTrainer inherits these
    EventRGBTrainer.RGB_PHOTOMETRIC_P = a.photo
    EventRGBTrainer.RGB_PHOTOMETRIC_KW = dict(blur_max=a.blur, exposure=tuple(a.exposure))
    EventRGBTrainer.RGB_SHIFT_PX = tuple(a.shift)
    DualStemTrainer.P_DROP_RGB = a.drop_rgb
    DualStemTrainer.P_DROP_EVT = a.drop_evt
    if a.modality != "both":
        a.baseline = True
        a.data = data_yaml_for_modality(a.data, a.modality)
    EventRGBTrainer.MODALITY = a.modality
    if a.baseline and (a.drop_rgb or a.drop_evt):
        raise SystemExit("modality dropout lives in DualStemFuse; not available with --baseline / single modality")
    print(f"augmentation: flip={a.flip} photo={a.photo} (blur<={a.blur}, exposure={a.exposure}) shift={a.shift} "
          f"drop_rgb={a.drop_rgb} drop_evt={a.drop_evt}")

    if a.baseline:
        prefix = {"both": "eventrgb_baseline", "rgb": "rgbonly", "event": "eventonly"}[a.modality]
        trainer_cls, model = EventRGBTrainer, "./conf/yolo26n_evrgb.yaml"
    else:
        variant = "_p2" if a.p2 else ""
        trainer_cls, model, prefix = DualStemTrainer, f"./conf/yolo26n_evrgb_dualstem{variant}.yaml", f"eventrgb_dualstem{variant}"
    trainer = trainer_cls(
        overrides=dict(
            model=model,
            pretrained="./yolo26n.pt",
            data=a.data,
            epochs=a.epochs,
            workers=8,
            batch=a.batch,
            project="yolo",
            name=f"{prefix}_{a.name}_yolo26n",
            device=[a.device],
            imgsz=a.imgsz,
            rect=True,
            save_json=True,
            # Ultralytics' own augmentations are NOT applied by our dataset chain; set to 0 so args.yaml is honest.
            hsv_h=0,
            hsv_s=0,
            hsv_v=0,
            mosaic=0,
        )
    )
    trainer.train()


if __name__ == "__main__":
    main()
