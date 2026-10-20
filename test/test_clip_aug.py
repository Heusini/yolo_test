"""Per-clip augmentation: with the clip sampler's tokens, flip and shift decisions are constant inside a clip.

usage: PYTHONPATH=. python test/test_clip_aug.py [<dataset root>]
"""

import random
import sys
from pathlib import Path

import torch

from datasets.aramsuisse_dataset import ArmasuisseDataset
from datasets.clip_sampler import ClipBatchSampler
from datasets.flip_transformer import HFlipTransformer
from datasets.rgb_photometric_transformer import RGBPhotometricTransformer
from datasets.rgb_shift_transformer import RGBShiftTransformer


def expected_shift(key, max_dx, max_dy, p):  # mirrors RGBShiftTransformer.__getitem__
    rng = random.Random(key * 8 + 2)
    if rng.random() < p:
        return rng.randint(-max_dx, max_dx), rng.randint(-max_dy, max_dy)
    return 0, 0


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / ".cache/yolo_test_cpu/fake") / "train"
    raw = ArmasuisseDataset(str(root), True, True)
    n = len(raw)
    shift = RGBShiftTransformer(raw, 8, 8, p=0.5)
    aug = HFlipTransformer(RGBPhotometricTransformer(shift, p=0.5, blur_max=0), p=0.5)
    L = 3
    sampler = ClipBatchSampler(raw.match_list, 2, L, shuffle=True, seed=3)
    flips, moved = [], 0
    for _ in range(4):  # several epochs -> many clips with different tokens
        for batch in sampler:
            for b in range(2):
                idx = batch[b * L:(b + 1) * L]
                key = idx[0] // n
                dx, dy = expected_shift(key, 8, 8, 0.5)
                f = set()
                for i in idx:
                    d, r = aug[i], raw[i % n]
                    assert d["aug_key"] == key
                    evt, evt_raw = d["img"][3:], r["img"][3:]
                    assert torch.equal(evt, evt_raw) or torch.equal(evt, evt_raw.flip(-1)), "events must only be flipped"
                    f.add(not torch.equal(evt, evt_raw))
                    s = shift[i]["img"][:3]  # shift only: must equal the raw RGB moved by the clip's (dx, dy)
                    assert torch.equal(s, RGBShiftTransformer.shift(r["img"][:3], dx, dy)), "shift differs inside a clip"
                assert len(f) == 1, f"flip changed inside a clip: {f}"
                flips.append(f.pop())
                moved += bool(dx or dy)
    assert True in flips and False in flips, "flip never varied across clips"
    assert moved, "shift never varied across clips"
    print(f"{len(flips)} clips: {sum(flips)} flipped, {moved} shifted, decisions constant inside every clip; ok")


if __name__ == "__main__":
    main()
