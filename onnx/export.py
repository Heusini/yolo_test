import sys
import torch
import numpy as np
from ultralytics import YOLO
import torch.nn.functional as F

sys.path.append("./")
from utils.format_tensors import get_event_tensor_padded, get_rgb_tensor_padded

if len(sys.argv) < 2:
    print("provide input weights")
    sys.exit(1)

checkpoint = sys.argv[1]
model = YOLO(checkpoint)
# model.model.args["imgsz"] = [384, 640]
# model.model.args["ch"] = 20
# model.export(format="onnx", imgsz=[384, 640], rect=True, nms=False, half=True)
model.export(format="onnx", imgsz=[736, 1280], rect=True, nms=False, half=True)

# rgb = True
# event = False

# event_path = "/scratch/sheusinger/st_stephan_360_640_20000_10/train/2024_01_10_112814_drone_000/events/event_200.npy"
# rgb_path = "/scratch/sheusinger/st_stephan_360_640_20000_10/train/2024_01_10_112814_drone_000/rgbs/rgb_200.npy"

# t_rgb_data = get_rgb_tensor_padded(rgb_path, 24)
# t_ev_data = get_event_tensor_padded(event_path, 24)
# if rgb and event:
#     input_data = torch.vstack((t_rgb_data, t_ev_data))
#     input_data = input_data.unsqueeze(0)
# elif event:
#     input_data = t_ev_data
# else:
#     input_data = t_rgb_data
# # model(combined)input_data

# with torch.no_grad():
#     ka = model.model(input_data)
#     # for i in range(len(ka)):
#     out = ka[0].cpu().numpy()
#     print(out.shape)
#     print(np.max(out[:, 4, :]))
