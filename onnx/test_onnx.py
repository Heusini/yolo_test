import sys
import torch
import math
import onnxruntime as ort
import numpy as np
import torch.nn.functional as F
import random

from utils.format_tensors import get_event_tensor_padded, get_rgb_tensor_padded

random.seed(42)


event_path = "/scratch/sheusinger/st_stephan_360_640_20/train/2024_01_10_160626_back_drive_013/events/event_0.npz"
rgb_path = "/scratch/sheusinger/st_stephan_360_640_20/train/2024_01_10_160626_back_drive_013/rgbs/rgb_0.npz"

t_ev_data = get_event_tensor_padded(event_path, 24)
t_rgb_data = get_rgb_tensor_padded(rgb_path, 24)
combined = torch.vstack((t_rgb_data, t_ev_data))
combined = combined.unsqueeze(0)

if len(sys.argv) < 2:
    print("Provide path to onnx file as argument")
    sys.exit()

session = ort.InferenceSession(
    sys.argv[1],
    providers=["CPUExecutionProvider"],
)
input_names = session.get_inputs()
output_names = session.get_outputs()
for name in output_names:
    print(name)
inputs = {}
np.random.seed(42)
for name in input_names:
    print(name.shape)
    inputs[name.name] = np.random.random(name.shape).astype(np.float32)

inputs[input_names[0].name] = combined.numpy()
# inputs[input_names[1].name] = t_rgb_data.numpy()
# print(t_rgb_data.numpy().shape)

# print(inputs)
outputs = session.run(["output0"], inputs)
print(len(outputs))
print(outputs[0].shape)

print(np.max(outputs[0][:, 4, :]))
