"""CPU check of RGBPhotometricTransformer. usage (repo root): PYTHONPATH=. python test/test_rgb_photometric.py <fake_dataset_dir>"""

import random
import sys

import torch

from datasets.aramsuisse_dataset import ArmasuisseDataset
from datasets.rgb_photometric_transformer import RGBPhotometricTransformer


def main(root: str):
    base = ArmasuisseDataset(f"{root}/train", True, True)
    x = base[0]["img"]
    random.seed(0)
    torch.manual_seed(0)

    changed = 0
    for _ in range(20):
        s = RGBPhotometricTransformer(base, p=1.0)[0]
        img = s["img"]
        assert img.dtype == x.dtype and img.shape == x.shape
        assert torch.equal(img[3:], x[3:]), "events must be untouched"
        assert torch.equal(s["bboxes"], base[0]["bboxes"])
        changed += int(not torch.equal(img[:3], x[:3]))
    assert changed >= 15, f"jitter rarely changed the image ({changed}/20)"
    assert torch.equal(RGBPhotometricTransformer(base, p=0.0)[0]["img"], x), "p=0 must be identity"

    # motion blur keeps the mean (normalized kernel) and smooths
    rgb = x[:3].float()
    b = RGBPhotometricTransformer.motion_blur(rgb, 9, 0.3)
    assert abs(b.mean() - rgb.mean()) < 1.0 and b.std() < rgb.std()
    print(f"ok: uint8 kept, events/boxes untouched, {changed}/20 samples changed, blur normalized")


if __name__ == "__main__":
    main(sys.argv[1])
