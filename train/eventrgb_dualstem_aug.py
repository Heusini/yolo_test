"""Dual-stem run with selectable training-time augmentations (all off by default = train/eventrgb_dualstem.py).

examples:
  python -m train.eventrgb_dualstem_aug --name flip --flip 0.5 --device 0
  python -m train.eventrgb_dualstem_aug --name flip_drop --flip 0.5 --drop-rgb 0.15 --drop-evt 0.05
  python -m train.eventrgb_dualstem_aug --name flip_photo --flip 0.5 --photo 0.8 --blur 0 --exposure 0.6 1.6
  python -m train.eventrgb_dualstem_aug --name flip_shift --flip 0.5 --shift 2 1
"""

import argparse

import cv2
import matplotlib
import torch

from engine.dualstem_trainer import DualStemTrainer


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="run name suffix: eventrgb_dualstem_<name>_yolo26n")
    p.add_argument("--device", type=int, default=1)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--p2", action="store_true", help="use the P2-head variant (conf/yolo26n_evrgb_dualstem_p2.yaml)")
    p.add_argument("--flip", type=float, default=0.0, help="horizontal flip probability")
    p.add_argument("--photo", type=float, default=0.0, help="RGB photometric jitter probability")
    p.add_argument("--blur", type=int, default=15, help="max motion-blur kernel in px for --photo (0 = no blur)")
    p.add_argument("--exposure", type=float, nargs=2, default=(0.3, 2.5), metavar=("LO", "HI"), help="exposure factor range for --photo")
    p.add_argument("--shift", type=int, nargs=2, default=(0, 0), metavar=("DX", "DY"), help="RGB-only shift in px")
    p.add_argument("--drop-rgb", type=float, default=0.0, help="modality dropout probability for RGB")
    p.add_argument("--drop-evt", type=float, default=0.0, help="modality dropout probability for events")
    return p.parse_args()


def main():
    a = parse()
    matplotlib.use("Agg")
    cv2.setNumThreads(0)
    torch.set_num_threads(16)

    DualStemTrainer.HFLIP_P = a.flip
    DualStemTrainer.RGB_PHOTOMETRIC_P = a.photo
    DualStemTrainer.RGB_PHOTOMETRIC_KW = dict(blur_max=a.blur, exposure=tuple(a.exposure))
    DualStemTrainer.RGB_SHIFT_PX = tuple(a.shift)
    DualStemTrainer.P_DROP_RGB = a.drop_rgb
    DualStemTrainer.P_DROP_EVT = a.drop_evt
    print(f"augmentation: flip={a.flip} photo={a.photo} (blur<={a.blur}, exposure={a.exposure}) shift={a.shift} "
          f"drop_rgb={a.drop_rgb} drop_evt={a.drop_evt}")

    variant = "_p2" if a.p2 else ""
    trainer = DualStemTrainer(
        overrides=dict(
            model=f"./conf/yolo26n_evrgb_dualstem{variant}.yaml",
            pretrained="./yolo26n.pt",
            data="./conf/eventrgb_data.yaml",
            epochs=a.epochs,
            workers=8,
            project="yolo",
            name=f"eventrgb_dualstem{variant}_{a.name}_yolo26n",
            device=[a.device],
            imgsz=640,
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
