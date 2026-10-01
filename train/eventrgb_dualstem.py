import cv2
import matplotlib
import torch

from engine.dualstem_trainer import DualStemTrainer


def main():
    matplotlib.use("Agg")
    cv2.setNumThreads(0)
    torch.set_num_threads(16)

    # The trainer is used directly (not YOLO(...).train) so the dual-stem yaml is resolved for the scale
    # before the model is built and yolo26n.pt is remapped into both stems.
    trainer = DualStemTrainer(
        overrides=dict(
            model="./conf/yolo26n_evrgb_dualstem.yaml",
            pretrained="./yolo26n.pt",
            data="./conf/eventrgb_data.yaml",
            epochs=15,
            workers=8,
            project="yolo",
            name="eventrgb_dualstem_yolo26n_10_10000",
            device=[1],
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
