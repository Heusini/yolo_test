"""Dual-stem run with selectable training-time augmentations (all off by default = train/eventrgb_dualstem.py).

examples:
  python -m train.eventrgb_dualstem_aug --name flip --flip 0.5 --device 0
  python -m train.eventrgb_dualstem_aug --name flip_drop --flip 0.5 --drop-rgb 0.15 --drop-evt 0.05
  python -m train.eventrgb_dualstem_aug --name flip_photo --flip 0.5 --photo 0.8 --blur 0 --exposure 0.6 1.6
  python -m train.eventrgb_dualstem_aug --name flip_shift --flip 0.5 --shift 2 1
  python -m train.eventrgb_dualstem_aug --name zoom --flip 0.5 --zoom 0.5 1.5 --zoom-out 1.2
  # ConvLSTM fine-tune from the 1280 checkpoint (RVT "mixed" sampling: 2 random + 2 stream slots of 4 frames)
  python -m train.eventrgb_dualstem_aug --name lstm1280 --data conf/eventrgb_data_720.yaml --imgsz 1280 \
      --mem lstm --clip-len 4 --random-slots 2 --batch 16 --pretrained <best.pt> --epochs 4 --lr0 2e-4 --warmup 0 \
      --flip 0.5 --photo 0.5 --blur 0 --exposure 0.7 1.4 --shift 2 1 --drop-rgb 0.1 --drop-evt 0.05
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
    p.add_argument("--device", default="1", help="GPU index or cpu")
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
    p.add_argument("--zoom", type=float, nargs=2, default=(0.0, 1.5), metavar=("P", "ZIN"), help="zoom probability and max zoom-in factor (crop 1/ZIN, resize back)")
    p.add_argument("--zoom-out", type=float, default=1.0, metavar="ZOUT", help="max zoom-out factor (shrink, zero canvas); taken in 1/3 of the zoomed frames. 1 = off")
    p.add_argument("--shift", type=int, nargs=2, default=(0, 0), metavar=("DX", "DY"), help="RGB-only shift in px")
    p.add_argument("--drop-rgb", type=float, default=0.0, help="modality dropout probability for RGB")
    p.add_argument("--drop-evt", type=float, default=0.0, help="modality dropout probability for events")
    p.add_argument("--pretrained", default="./yolo26n.pt", help="single-stem (remapped) or dual-stem checkpoint")
    p.add_argument("--mem", choices=["none", "lstm"], default="none", help="temporal block in the event stem")
    p.add_argument("--clip-len", type=int, default=1, help="frames per sequence slot (batch = slots x clip-len)")
    p.add_argument("--random-slots", type=int, default=0, help="slots that draw random clips instead of streaming")
    p.add_argument("--lr0", type=float, default=None, help="fixed initial lr (sets optimizer AdamW; default: Ultralytics auto)")
    p.add_argument("--warmup", type=float, default=None, help="warmup epochs (default: Ultralytics 3)")
    p.add_argument("--freeze", choices=["none", "mem"], default="none",
                   help="mem: train only the temporal block and the gate (everything else frozen, BN stats fixed)")
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
    EventRGBTrainer.ZOOM_P, EventRGBTrainer.ZOOM_IN = a.zoom
    EventRGBTrainer.ZOOM_OUT = a.zoom_out
    DualStemTrainer.P_DROP_RGB = a.drop_rgb
    DualStemTrainer.P_DROP_EVT = a.drop_evt
    if a.modality != "both":
        a.baseline = True
        a.data = data_yaml_for_modality(a.data, a.modality)
    EventRGBTrainer.MODALITY = a.modality
    if a.baseline and (a.drop_rgb or a.drop_evt):
        raise SystemExit("modality dropout lives in DualStemFuse; not available with --baseline / single modality")
    if a.mem != "none" and (a.baseline or a.clip_len < 2):
        raise SystemExit("--mem needs the dual stem and --clip-len >= 2")
    if a.freeze == "mem" and a.mem == "none":
        raise SystemExit("--freeze mem needs --mem lstm")
    if a.clip_len > 1 and (a.batch % a.clip_len or a.random_slots > a.batch // a.clip_len):
        raise SystemExit(f"--batch {a.batch} must be a multiple of --clip-len {a.clip_len}, --random-slots <= slots")
    EventRGBTrainer.CLIP_LEN = a.clip_len
    EventRGBTrainer.RANDOM_SLOTS = a.random_slots
    DualStemTrainer.MEM = a.mem
    print(f"augmentation: flip={a.flip} photo={a.photo} (blur<={a.blur}, exposure={a.exposure}) shift={a.shift} zoom={a.zoom} zoom_out={a.zoom_out} "
          f"drop_rgb={a.drop_rgb} drop_evt={a.drop_evt}")
    print(f"memory: {a.mem} clip_len={a.clip_len} random_slots={a.random_slots} pretrained={a.pretrained}")

    if a.baseline:
        prefix = {"both": "eventrgb_baseline", "rgb": "rgbonly", "event": "eventonly"}[a.modality]
        trainer_cls, model = EventRGBTrainer, "./conf/yolo26n_evrgb.yaml"
    else:
        variant = "_p2" if a.p2 else ""
        trainer_cls, model, prefix = DualStemTrainer, f"./conf/yolo26n_evrgb_dualstem{variant}.yaml", f"eventrgb_dualstem{variant}"
    trainer = trainer_cls(
        overrides=dict(
            model=model,
            pretrained=a.pretrained,
            data=a.data,
            epochs=a.epochs,
            workers=8,
            batch=a.batch,
            project="yolo",
            name=f"{prefix}_{a.name}_yolo26n",
            device="cpu" if a.device == "cpu" else [int(a.device)],
            imgsz=a.imgsz,
            rect=True,
            save_json=True,
            # Ultralytics' own augmentations are NOT applied by our dataset chain; set to 0 so args.yaml is honest.
            hsv_h=0,
            hsv_s=0,
            hsv_v=0,
            mosaic=0,
            **({"optimizer": "AdamW", "lr0": a.lr0} if a.lr0 is not None else {}),  # optimizer=auto ignores lr0
            **({"warmup_epochs": a.warmup} if a.warmup is not None else {}),
            # layer 0 = DualStemFuse: both stems frozen, evt_mem + fuse train; 1 = Index, 2..20 trunk + neck, 21 = Detect
            **({"freeze": ["0.rgb_stem", "0.evt_stem"] + list(range(1, 22))} if a.freeze == "mem" else {}),
        )
    )
    trainer.train()


if __name__ == "__main__":
    main()
