"""Tracker baseline: per-frame detections vs. ByteTrack-associated boxes of the same checkpoint, same metrics.

usage (repo root):
  python test/tracker_eval.py <ckpt.pt> --data conf/<data>.yaml --imgsz 1280 --device 0 [--tracker bytetrack.yaml]

Frames are processed one by one in dataset order (= sequence order); the tracker is re-created at every sequence
start (frame_idx == 0). "tracked" boxes are the detections ByteTrack kept and associated (low-score detections that
continue a track survive, unmatched low-score ones are dropped), scored with the track score.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append("./")
import evrgb  # noqa: F401
from datasets.aramsuisse_dataset import ArmasuisseDataset
from datasets.pad_transformer import PadTransformer
from datasets.yolo_converter import YoloConverter
from engine.basetrainer import collate_fn
from engine.eventrgbvalidator import EventRGBValidator
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml


def make_validator(model, data, device, imgsz, save_dir):
    v = EventRGBValidator(args=dict(data=data["yaml_file"] if "yaml_file" in data else None, imgsz=imgsz, batch=1,
                                    plots=False, half=False, verbose=False, task="detect", mode="val"), save_dir=save_dir)
    v.data, v.device, v.training = data, device, False
    v.init_metrics(model)
    return v


def to_pred(bboxes, conf, cls, device):
    return {"bboxes": torch.as_tensor(bboxes, dtype=torch.float32, device=device).reshape(-1, 4),
            "conf": torch.as_tensor(conf, dtype=torch.float32, device=device).reshape(-1),
            "cls": torch.as_tensor(cls, dtype=torch.float32, device=device).reshape(-1)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tracker", default="bytetrack.yaml")
    a = p.parse_args()

    device = torch.device("cpu" if a.device == "cpu" else f"cuda:{a.device}")
    model = YOLO(a.ckpt).model.to(device).eval()
    data = check_det_dataset(a.data)
    tracker_cfg = IterableSimpleNamespace(**YAML.load(check_yaml(a.tracker)))

    arma = ArmasuisseDataset(data["val"], True, True)
    h, w = arma.get_im_shape()
    ds = YoloConverter(data["val"], PadTransformer(arma, (0, -w % 32, 0, -h % 32)))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.workers, collate_fn=collate_fn)

    save_dir = Path("runs/tracker_eval") / Path(a.ckpt).resolve().parent.parent.name
    v_raw = make_validator(model, data, device, a.imgsz, save_dir / "raw")
    v_trk = make_validator(model, data, device, a.imgsz, save_dir / "tracked")

    tracker, n_raw, n_trk, n_seq = None, 0, 0, 0
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if batch["frame_idx"].item() == 0:
                tracker, n_seq = BYTETracker(tracker_cfg, frame_rate=30), n_seq + 1
            batch = v_raw.preprocess(batch)
            preds = v_raw.postprocess(model(batch["img"]))
            v_raw.update_metrics(preds, batch)
            d = preds[0]
            n_raw += len(d["conf"])

            tracks = np.zeros((0, 8), dtype=np.float32)
            if len(d["conf"]):
                det = Boxes(torch.cat([d["bboxes"], d["conf"][:, None], d["cls"][:, None]], 1).cpu(), orig_shape=tuple(batch["img"].shape[2:]))
                tracks = tracker.update(det.numpy()).reshape(-1, 8)  # rows: x1 y1 x2 y2 id score cls idx
            n_trk += len(tracks)
            v_trk.update_metrics([to_pred(tracks[:, :4], tracks[:, 5], tracks[:, 6], device)], batch)
            if i % 500 == 0:
                print(f"frame {i}/{len(loader)}  sequences {n_seq}", flush=True)

    print(f"\n{len(loader)} frames, {n_seq} sequences, boxes: raw {n_raw}  tracked {n_trk}")
    print(f"{'':9s}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for name, v in [("raw", v_raw), ("tracked", v_trk)]:
        v.get_stats()
        b = v.metrics.box
        print(f"{name:9s}{b.mp:8.3f}{b.mr:8.3f}{b.map50:8.3f}{b.map:10.3f}")


if __name__ == "__main__":
    main()
