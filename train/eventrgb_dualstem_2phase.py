"""Two-phase dual-stem training (A/B against train/eventrgb_dualstem.py).

phase A: RGB stem + trunk + neck frozen (COCO weights), only event stem, gate and Detect head train.
phase B: everything trains from the phase-A checkpoint at a lower learning rate.
"""

import argparse

import cv2
import matplotlib
import torch

from engine.dualstem_trainer import DualStemTrainer

# layer 0 = DualStemFuse (rgb_stem frozen, evt_stem + fuse trainable), 1 = Index, 2..20 = trunk + neck, 21 = Detect
FREEZE = ["0.rgb_stem"] + list(range(2, 21))


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="run name suffix: eventrgb_dualstem_2phase{A,B}_<name>_yolo26n")
    p.add_argument("--data", default="./conf/eventrgb_data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=int, default=1)
    p.add_argument("--batch", type=int, default=16, help="train batch (val uses 2x); 8 halves GPU memory, effective batch stays 64")
    p.add_argument("--epochs-a", type=int, default=4, help="frozen phase")
    p.add_argument("--epochs-b", type=int, default=8, help="full fine-tune phase")
    return p.parse_args()


def main():
    a = parse()
    matplotlib.use("Agg")
    cv2.setNumThreads(0)
    torch.set_num_threads(16)

    common = dict(
        data=a.data,
        workers=8,
        batch=a.batch,
        project="yolo",
        device=[a.device],
        imgsz=a.imgsz,
        rect=True,
        save_json=True,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        mosaic=0,
    )

    phase_a = DualStemTrainer(
        overrides=dict(
            common,
            model="./conf/yolo26n_evrgb_dualstem.yaml",
            pretrained="./yolo26n.pt",
            freeze=FREEZE,
            epochs=a.epochs_a,
            optimizer="AdamW",
            lr0=1e-3,
            warmup_epochs=1,
            name=f"eventrgb_dualstem_2phaseA_{a.name}_yolo26n",
        )
    )
    phase_a.train()

    phase_b = DualStemTrainer(
        overrides=dict(
            common,
            model=str(phase_a.last),  # dual-stem checkpoint: loaded as is, no remap
            epochs=a.epochs_b,
            optimizer="AdamW",
            lr0=5e-4,
            cos_lr=True,
            warmup_epochs=1,
            name=f"eventrgb_dualstem_2phaseB_{a.name}_yolo26n",
        )
    )
    phase_b.train()


if __name__ == "__main__":
    main()
