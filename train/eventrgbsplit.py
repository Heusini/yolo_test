import torch
from ultralytics import YOLO
from engine.eventrgbsplittrainer import EventRGBSplitTrainer
from engine.split_trainer import SplitTrainer
import matplotlib
import cv2


def main():
    matplotlib.use("Agg")

    model = YOLO("./conf/eventrgb_conf.yaml", task="detect").load(
        "./pretrained_weights/yolo_eventrgb_best.pt"
    )

    cv2.setNumThreads(0)
    torch.set_num_threads(16)

    model.train(
        trainer=SplitTrainer,
        data="./conf/eventrgb_big_image_data.yaml",
        epochs=15,
        workers=8,
        project="yolo",
        name="eventrgbsplit_yolo_10_10000",
        device=[1],
        imgsz=640,
        rect=True,
        save_json=False,
    )


if __name__ == "__main__":
    main()
