"""usage: python onnx/test_onnx_evrgb.py <model.onnx> <event.npy> <rgb.npy>

Runs one padded event+RGB sample through the ONNX model. YOLO26 (end2end) output is [B, max_det, 6] = x1 y1 x2 y2 score cls.
"""

import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.append("./")
from utils.format_tensors import get_event_tensor_padded, get_rgb_tensor_padded

if len(sys.argv) < 4:
    print(__doc__)
    sys.exit(1)

onnx_path, event_path, rgb_path = sys.argv[1:4]
combined = torch.vstack((get_rgb_tensor_padded(rgb_path, 24), get_event_tensor_padded(event_path, 24))).unsqueeze(0)

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
inp = session.get_inputs()[0]
print("input ", inp.name, inp.shape, "| sample", tuple(combined.shape))
dtype = np.float16 if "float16" in inp.type else np.float32
out = session.run(None, {inp.name: combined.numpy().astype(dtype)})[0]
print("output", session.get_outputs()[0].name, out.shape)
print("max score", float(out[0, :, 4].max()), "| detections > 0.25:", int((out[0, :, 4] > 0.25).sum()))
