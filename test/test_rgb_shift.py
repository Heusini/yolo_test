"""CPU check of RGBShiftTransformer. usage (repo root): PYTHONPATH=. python test/test_rgb_shift.py <fake_dataset_dir>"""

import sys

import torch

from datasets.aramsuisse_dataset import ArmasuisseDataset
from datasets.rgb_shift_transformer import RGBShiftTransformer


def main(root: str):
    base = ArmasuisseDataset(f"{root}/train", True, True)
    x = base[0]["img"]

    # exact shift semantics: pixel (y, x) moves to (y+dy, x+dx), zero fill, events untouched
    for dx, dy in [(3, 2), (-4, 0), (0, -5), (7, -3)]:
        s = RGBShiftTransformer.shift(x[:3], dx, dy)
        y0, x0 = 100, 200
        assert torch.equal(s[:, y0 + dy, x0 + dx], x[:3, y0, x0]), (dx, dy)
        if dx > 0:
            assert s[:, :, :dx].sum() == 0
        if dy > 0:
            assert s[:, :dy, :].sum() == 0

    # wrapper: always shifts with p=1, never with p=0; events and boxes identical in both cases
    torch.manual_seed(0)
    on = RGBShiftTransformer(base, 8, 2, p=1.0)[0]
    off = RGBShiftTransformer(base, 8, 2, p=0.0)[0]
    assert torch.equal(on["img"][3:], x[3:]) and torch.equal(off["img"], x)
    assert torch.equal(on["bboxes"], base[0]["bboxes"])
    print("ok: shift semantics, zero fill, events/boxes untouched, p=0 is identity")


if __name__ == "__main__":
    main(sys.argv[1])
