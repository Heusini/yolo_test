"""Stressed validation: mAP of a checkpoint under controlled input perturbations (no training).

usage (repo root):
  python test/robustness_eval.py <ckpt.pt> [<ckpt2.pt> ...] --data conf/<data>.yaml --imgsz 1280 --device 0
  optional: --conditions clean rgb_zero evt_zero rgb_dark rgb_bright rgb_noise rgb_shift2 rgb_shift4 rgb_shift8 rgb_prev
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append("./")
import evrgb  # noqa: F401  (registers DualStemFuse for checkpoint loading)
from datasets.aramsuisse_dataset import ArmasuisseDataset
from datasets.pad_transformer import PadTransformer
from datasets.rgb_shift_transformer import RGBShiftTransformer
from datasets.yolo_converter import YoloConverter
from engine.basetrainer import collate_fn
from engine.eventrgbvalidator import EventRGBValidator
from ultralytics import YOLO

CONDITIONS = ["clean", "rgb_zero", "evt_zero", "rgb_dark", "rgb_bright", "rgb_noise", "rgb_shift2", "rgb_shift4", "rgb_shift8", "rgb_prev"]


class PrevRGB(Dataset):
    """Replace the RGB channels with the previous frame's RGB (same sequence; first frame keeps its own)."""

    def __init__(self, ds: ArmasuisseDataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        d = self.ds[i]
        m = self.ds.match_list[i]
        if m.pos > 0:
            prev = np.load(self.ds.match_list[i - 1].frame_path)
            d["img"] = d["img"].clone()
            d["img"][:3] = torch.from_numpy(prev).permute(2, 0, 1)
        return d


def perturb(img: torch.Tensor, cond: str) -> torch.Tensor:
    """img: float batch after preprocess (RGB in [0,1], events raw counts)."""
    rgb, evt = img[:, :3], img[:, 3:]
    if cond == "rgb_zero":
        rgb = torch.zeros_like(rgb)
    elif cond == "evt_zero":
        evt = torch.zeros_like(evt)
    elif cond == "rgb_dark":
        rgb = rgb * 0.4
    elif cond == "rgb_bright":
        rgb = (rgb * 2.0).clamp(0, 1)
    elif cond == "rgb_noise":
        rgb = (rgb + torch.randn_like(rgb) * 0.05).clamp(0, 1)
    elif cond.startswith("rgb_shift"):
        rgb = RGBShiftTransformer.shift(rgb, int(cond[9:]), 0)
    return torch.cat([rgb, evt], 1)


class RobustValidator(EventRGBValidator):
    condition = "clean"

    def get_dataloader(self, dataset_path, batch_size):
        arma = ArmasuisseDataset(dataset_path, True, True)
        ds = PrevRGB(arma) if self.condition == "rgb_prev" else arma
        h, w = arma.get_im_shape()
        ds = YoloConverter(dataset_path, PadTransformer(ds, (0, -w % 32, 0, -h % 32)))
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=self.args.workers, collate_fn=collate_fn)

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        batch["img"] = perturb(batch["img"], self.condition)
        return batch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    a = p.parse_args()

    torch.manual_seed(0)
    rows = {}
    for ckpt in a.ckpts:
        name = Path(ckpt).resolve().parent.parent.name
        model = YOLO(ckpt)
        for cond in a.conditions:
            RobustValidator.condition = cond
            m = model.val(data=a.data, imgsz=a.imgsz, device=a.device, batch=a.batch, workers=a.workers, rect=True,
                          plots=False, verbose=False, validator=RobustValidator, project="runs/robustness", name=f"{name}_{cond}", exist_ok=True)
            rows[(name, cond)] = (m.box.mp, m.box.mr, m.box.map50, m.box.map)
            print(f"{name:45s} {cond:11s} P {m.box.mp:.3f}  R {m.box.mr:.3f}  mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}", flush=True)

    print("\n=== mAP50-95 per condition ===")
    names = list(dict.fromkeys(n for n, _ in rows))
    print(f"{'condition':11s}" + "".join(f"{n[-28:]:>30s}" for n in names))
    for cond in a.conditions:
        print(f"{cond:11s}" + "".join(f"{rows[(n, cond)][3]:30.3f}" for n in names))


if __name__ == "__main__":
    main()
