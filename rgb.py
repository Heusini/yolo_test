from ultralytics import YOLO
from engine.rgbtrainer import RGBTrainer
import matplotlib

matplotlib.use("Agg")

model = YOLO("rgb_conf.yaml", task="detect").load("./yolo26s.pt")

model.train(
    trainer=RGBTrainer,
    data="rgb_data.yaml",
    epochs=5,
    workers=0,
    project="yolo",
    name="rgb_yolo",
    device=[0],
    imgsz=640,
    rect=True,
    save_json=True,
)
