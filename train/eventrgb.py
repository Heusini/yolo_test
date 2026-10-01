import torch
from ultralytics import YOLO
from engine.eventrgbtrainer import EventRGBTrainer
import matplotlib
import cv2


def main():
    matplotlib.use("Agg")

    model = YOLO("./conf/yolo26n_evrgb.yaml", task="detect").load("./yolo26n.pt")

    cv2.setNumThreads(0)
    torch.set_num_threads(16)

    model.train(
        trainer=EventRGBTrainer,
        data="./conf/eventrgb_data.yaml",
        epochs=15,
        workers=8,
        project="yolo",
        name="eventrgb_yolo26n_10_10000",
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


if __name__ == "__main__":
    main()
