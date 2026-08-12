import cv2
import torch
from ultralytics import YOLO
from engine.trainer import EventTrainer
import matplotlib


def main():
    matplotlib.use("Agg")

    model = YOLO("./conf/event_conf.yaml", task="detect").load("./yolo26s.pt")

    cv2.setNumThreads(0)
    torch.set_num_threads(16)
    model.train(
        trainer=EventTrainer,
        data="./conf/event_data.yaml",
        epochs=15,
        workers=8,
        project="yolo",
        name="event_yolo",
        device=[0],
        imgsz=640,
        rect=True,
        save_json=True,
    )


if __name__ == "__main__":
    main()
