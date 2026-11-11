"""usage: python onnx/test_onnx_evrgb.py <model.onnx> <event.npy|npz> <rgb.npy|jpg> [<best.pt>]

Runs one padded event+RGB sample through the ONNX model. YOLO26 (end2end) output is [B, max_det, 6] = x1 y1 x2 y2 score cls.
With the .pt given, runs the PyTorch model on the same sample and prints the difference of the detections.
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
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
inp = session.get_inputs()[0]
rgb = get_rgb_tensor_padded(rgb_path, 0)
pad = inp.shape[2] - rgb.shape[1]  # bottom padding to the model's input height (e.g. 720 -> 736)
combined = torch.vstack((get_rgb_tensor_padded(rgb_path, pad), get_event_tensor_padded(event_path, pad))).unsqueeze(0)
print("input ", inp.name, inp.shape, "| sample", tuple(combined.shape), f"(padded {pad} rows)")
dtype = np.float16 if "float16" in inp.type else np.float32
out = session.run(None, {inp.name: combined.numpy().astype(dtype)})[0]
print("output", session.get_outputs()[0].name, out.shape)
print("max score", float(out[0, :, 4].max()), "| detections > 0.25:", int((out[0, :, 4] > 0.25).sum()))

if len(sys.argv) > 4:
    import evrgb  # noqa: F401  (registers the fusion modules for checkpoint loading)
    from ultralytics import YOLO

    model = YOLO(sys.argv[4]).model.float().eval()
    with torch.no_grad():
        ref = model(combined)
    ref = (ref[0] if isinstance(ref, (tuple, list)) else ref).numpy()
    print("pytorch max score", float(ref[0, :, 4].max()), "| detections > 0.25:", int((ref[0, :, 4] > 0.25).sum()))
    k = 20  # for each of the top ONNX candidates, the closest PyTorch candidate (ties in score may reorder the lists)
    d = np.abs(out[0, :k, None, :5].astype(np.float32) - ref[0, None, :, :5]).max(-1).min(-1)
    print(f"onnx vs pytorch, top {k}: worst match {d.max():.4f} (max abs diff over x1 y1 x2 y2 score)")
    for name, arr in (("onnx   ", out), ("pytorch", ref)):
        top = arr[0, arr[0, :, 4].argsort()[::-1][:3]]
        print(name, "top 3:", np.round(top.astype(np.float32), 2).tolist())
