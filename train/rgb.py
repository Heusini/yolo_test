import cv2
import torch
from ultralytics import YOLO
from engine.rgbtrainer import RGBTrainer
import matplotlib


def main():
    matplotlib.use("Agg")

    model = YOLO("./conf/rgb_conf.yaml", task="detect").load(
        "./pretrained_weights/yolo26s.pt"
    )

    cv2.setNumThreads(0)
    torch.set_num_threads(16)
    model.train(
        trainer=RGBTrainer,
        data="./conf/rgb_data.yaml",
        epochs=15,
        workers=8,
        project="yolo",
        name="rgb_yolo",
        device=[0],
        imgsz=640,
        rect=True,
        save_json=True,
    )


if __name__ == "__main__":
    main()
