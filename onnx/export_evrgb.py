"""usage: python onnx/export_evrgb.py <weights.pt> [height width]   (default 384 640)

Works for the plain yolo26*_evrgb and the dual-stem checkpoints (import evrgb registers the custom module).
"""

import sys

sys.path.append("./")
import evrgb  # noqa: F401
from ultralytics import YOLO

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

imgsz = [int(sys.argv[2]), int(sys.argv[3])] if len(sys.argv) >= 4 else [384, 640]
YOLO(sys.argv[1]).export(format="onnx", imgsz=imgsz, rect=True, nms=False, half=True)
