import torch
import numpy as np
from ultralytics import YOLO
import torch.nn.functional as F

from utils.format_tensors import get_event_tensor_padded, get_rgb_tensor_padded


checkpoint = "/home/sheusinger/repos/yolo_test/runs/detect/yolo/eventrgb_yolo_10_100004/weights/last.pt"
model = YOLO(checkpoint)
# model.model.args["imgsz"] = [384, 640]
# model.model.args["ch"] = 20
model.export(format="onnx", imgsz=[384, 640], rect=True)
# event_path = "/scratch/sheusinger/st_stephan_360_640_20/train/2024_01_10_160626_back_drive_013/events/event_0.npz"
# rgb_path = "/scratch/sheusinger/st_stephan_360_640_20/train/2024_01_10_160626_back_drive_013/rgbs/rgb_0.npz"
#
event_path = "/scratch/sheusinger/st_stephan_360_640_10_10000/train/2024_01_10_112814_drone_000/events/event_200.npz"
rgb_path = "/scratch/sheusinger/st_stephan_360_640_10_10000/train/2024_01_10_112814_drone_000/rgbs/rgb_200.npz"

t_rgb_data = get_rgb_tensor_padded(rgb_path, 24)
t_ev_data = get_event_tensor_padded(event_path, 24)

combined = torch.vstack((t_rgb_data, t_ev_data))
combined = combined.unsqueeze(0)
# model(combined)

with torch.no_grad():
    ka = model.model(combined)
    print(len(ka))
    # for i in range(len(ka)):
    out = ka[0].cpu().numpy()
    print(out.shape)
    print(np.max(out[:, 4, :]))
