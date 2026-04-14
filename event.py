from ultralytics import YOLO
from engine.trainer import EventTrainer
import matplotlib

matplotlib.use("Agg")

model = YOLO("event_conf.yaml", task="detect").load("./yolo26s.pt")

model.train(
    trainer=EventTrainer,
    data="event_data.yaml",
    epochs=5,
    workers=0,
    project="yolo",
    name="event_yolo",
    device=[0],
    imgsz=640,
    rect=True,
    save_json=True,
)
