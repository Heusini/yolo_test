"""CPU check of the event+RGB pad pipeline on the fake dataset.

usage (from repo root): PYTHONPATH=. python test/test_pad_pipeline.py <fake_dataset_dir>
"""

import sys

import torch

from datasets.aramsuisse_dataset import ArmasuisseDataset
from datasets.pad_transformer import PadTransformer
from datasets.yolo_converter import YoloConverter
from engine.basetrainer import collate_fn


def main(root: str):
    ds = YoloConverter(f"{root}/train", PadTransformer(ArmasuisseDataset(f"{root}/train", True, True), (0, 0, 0, 24)))
    batch = collate_fn([ds[i] for i in range(4)])
    img, boxes = batch["img"], batch["bboxes"]
    assert img.shape == (4, 13, 384, 640), img.shape
    assert img.dtype == torch.uint8, img.dtype
    assert img[:, 3:, 360:].abs().sum() == 0, "event padding rows must be zero"
    assert boxes.shape[1] == 4 and (boxes >= 0).all() and (boxes <= 1).all(), "boxes must be normalized xywh"
    assert len(batch["batch_idx"]) == len(boxes)
    matches = ds.dataset.dataset.match_list
    for seq_id in {m.seq for m in matches}:
        pos = [m.pos for m in matches if m.seq == seq_id]
        assert pos == list(range(len(pos))), f"frames of sequence {seq_id} must be numbered 0..n-1 in order"
    assert batch["seq_id"].shape == batch["frame_idx"].shape == (4,)
    print(f"ok: img {tuple(img.shape)} {img.dtype}, {len(boxes)} boxes, rgb max {img[:, :3].max().item()}, event max {img[:, 3:].max().item()}")


if __name__ == "__main__":
    main(sys.argv[1])
