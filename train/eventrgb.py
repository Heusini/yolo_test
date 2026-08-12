import torch
from ultralytics import YOLO
from engine.eventrgbtrainer import EventRGBTrainer
import matplotlib
import cv2


def main():
    matplotlib.use("Agg")

    model = YOLO("./conf/eventrgb_conf.yaml", task="detect").load("./yolo26s.pt")

    cv2.setNumThreads(0)
    torch.set_num_threads(16)

    model.train(
        trainer=EventRGBTrainer,
        data="./conf/eventrgb_data.yaml",
        epochs=15,
        workers=8,
        project="yolo",
        name="eventrgb_yolo_10_10000",
        device=[1],
        imgsz=640,
        rect=True,
        save_json=True,
    )


if __name__ == "__main__":
    main()
