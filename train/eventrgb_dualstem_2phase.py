"""Two-phase dual-stem training (A/B against train/eventrgb_dualstem.py).

phase A: RGB stem + trunk + neck frozen (COCO weights), only event stem, gate and Detect head train.
phase B: everything trains from the phase-A checkpoint at a lower learning rate.
"""

import cv2
import matplotlib
import torch

from engine.dualstem_trainer import DualStemTrainer

PHASE_A_EPOCHS = 5
PHASE_B_EPOCHS = 15
# layer 0 = DualStemFuse (rgb_stem frozen, evt_stem + fuse trainable), 1 = Index, 2..20 = trunk + neck, 21 = Detect
FREEZE = ["0.rgb_stem"] + list(range(2, 21))

COMMON = dict(
    data="./conf/eventrgb_data.yaml",
    workers=8,
    project="yolo",
    device=[1],
    imgsz=640,
    rect=True,
    save_json=True,
    hsv_h=0,
    hsv_s=0,
    hsv_v=0,
    mosaic=0,
)


def main():
    matplotlib.use("Agg")
    cv2.setNumThreads(0)
    torch.set_num_threads(16)

    phase_a = DualStemTrainer(
        overrides=dict(
            COMMON,
            model="./conf/yolo26n_evrgb_dualstem.yaml",
            pretrained="./yolo26n.pt",
            freeze=FREEZE,
            epochs=PHASE_A_EPOCHS,
            optimizer="AdamW",
            lr0=1e-3,
            warmup_epochs=1,
            name="eventrgb_dualstem_2phaseA_yolo26n_10_10000",
        )
    )
    phase_a.train()

    phase_b = DualStemTrainer(
        overrides=dict(
            COMMON,
            model=str(phase_a.last),  # dual-stem checkpoint: loaded as is, no remap
            epochs=PHASE_B_EPOCHS,
            optimizer="AdamW",
            lr0=5e-4,
            cos_lr=True,
            warmup_epochs=1,
            name="eventrgb_dualstem_2phaseB_yolo26n_10_10000",
        )
    )
    phase_b.train()


if __name__ == "__main__":
    main()
